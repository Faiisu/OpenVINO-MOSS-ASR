"""Serve the live output dashboard and reviewed-dataset API on port 18000."""

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, unquote, urlsplit

from .data import MonitorStore
from .live_audio import LiveAudioHub
from .labels import LabelStore
from .mqtt import LiveTranscriptHub, MqttTranscriptSubscriber
from ..config import load_settings


STATIC_ROOT = Path(__file__).with_name("static")


class MonitorHandler(BaseHTTPRequestHandler):
    server_version = "MossMonitor/1.0"
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self.respond(False)

    def do_HEAD(self):
        self.respond(True)

    def do_POST(self):
        request = urlsplit(self.path)
        parts = [unquote(part) for part in request.path.strip("/").split("/")]
        try:
            if request.path == "/api/live-audio/publish":
                if not ipaddress.ip_address(self.client_address[0]).is_loopback:
                    self.send_error(403, "Live audio publishing is local-only")
                    return
                length = int(self.headers.get("Content-Length", "0"))
                sample_rate = int(self.headers.get("X-Audio-Sample-Rate", "0"))
                if length < 2 or length > 65536 or length % 2:
                    self.send_error(413, "Live audio chunk must contain 1 to 32768 PCM16 samples")
                    return
                self.server.live_audio.publish(self.rfile.read(length), sample_rate)
                self.send_response(204)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
            elif request.path == "/api/dataset/export":
                with self.server.store.lock:
                    result = self.server.labels.export()
                self.send_json(result)
            elif (len(parts) == 6 and parts[:2] == ["api", "runs"]
                  and parts[3] == "labels"):
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 65536:
                    self.send_error(413, "Label request must be between 1 and 65536 bytes")
                    return
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                with self.server.store.lock:
                    result = self.server.labels.save(parts[2], parts[4], parts[5], payload)
                self.send_json(result)
            else:
                self.send_error(404)
        except FileNotFoundError:
            self.send_error(404, "Run or output file not found")
        except (ValueError, TypeError, UnicodeDecodeError):
            self.send_error(400, "Invalid label request")
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        except OSError:
            self.send_error(503, "Dataset files are temporarily unavailable")

    def respond(self, head):
        self.connection.settimeout(15)
        request = urlsplit(self.path)
        parts = [unquote(part) for part in request.path.strip("/").split("/")]
        try:
            if request.path == "/api/live-audio/stream":
                if head:
                    self.send_common_headers(200, "text/event-stream; charset=utf-8", 0)
                    self.end_headers()
                else:
                    self.stream_live_audio()
            elif request.path == "/api/live-transcript/stream":
                if head:
                    self.send_common_headers(200, "text/event-stream; charset=utf-8", 0)
                    self.end_headers()
                else:
                    self.stream_live_transcripts()
            elif request.path == "/api/settings":
                self.send_json({"save_audio": self.server.settings["save_audio"],
                                "save_transcript": self.server.settings["save_transcript"],
                                "mqtt_enabled": self.server.settings["mqtt_enabled"]}, head)
            elif request.path in ("/", "/app.js", "/styles.css"):
                name = {"/": "index.html", "/app.js": "app.js", "/styles.css": "styles.css"}[request.path]
                mime = {"index.html": "text/html", "app.js": "text/javascript", "styles.css": "text/css"}[name]
                self.send_bytes((STATIC_ROOT / name).read_bytes(), mime + "; charset=utf-8", head)
            elif parts == ["api", "runs"]:
                self.send_json({"runs": self.server.store.runs(), "refresh_ms": 1000}, head)
            elif len(parts) == 3 and parts[:2] == ["api", "runs"]:
                query = parse_qs(request.query)
                limit = max(1, min(50, int(query.get("limit", ["15"])[0])))
                before = int(query["before"][0]) if "before" in query else None
                self.send_json(self.server.store.snapshot(parts[2], limit, before), head)
            elif len(parts) == 6 and parts[:2] == ["api", "runs"] and parts[3] == "audio":
                self.send_audio(self.server.store.audio_path(parts[2], parts[4], parts[5]), head)
            else:
                self.send_error(404)
        except FileNotFoundError:
            self.send_error(404, "Run or output file not found")
        except (ValueError, TypeError):
            self.send_error(400, "Invalid request")
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        except OSError:
            self.send_error(503, "Output files are temporarily unavailable")

    def send_json(self, value, head=False):
        self.send_bytes(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                        "application/json; charset=utf-8", head)

    def send_common_headers(self, status, mime, length):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")

    def send_bytes(self, data, mime, head=False):
        self.send_common_headers(200, mime, len(data))
        self.end_headers()
        if not head:
            self.wfile.write(data)

    def send_audio(self, path, head=False):
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        value = self.headers.get("Range")
        if value:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
            if not match or not any(match.groups()):
                return self.range_error(size)
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(size - 1, int(last)) if last else size - 1
            else:
                start = max(0, size - int(last))
            if start > end or start >= size or not size:
                return self.range_error(size)
            status = 206
        self.send_common_headers(status, "audio/wav", end - start + 1)
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not head:
            with path.open("rb") as file:
                file.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    data = file.read(min(65536, remaining))
                    if not data:
                        break
                    self.wfile.write(data)
                    remaining -= len(data)

    def stream_live_audio(self):
        value = self.headers.get("Last-Event-ID")
        try:
            sequence = int(value) if value else self.server.live_audio.latest_sequence
        except ValueError:
            self.send_error(400, "Invalid live audio event ID")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        while True:
            chunks = self.server.live_audio.after(sequence, timeout=10.0)
            if not chunks:
                self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
                continue
            for chunk in chunks:
                payload = json.dumps({"sample_rate": chunk.sample_rate,
                                      "audio": base64.b64encode(chunk.pcm16).decode("ascii")},
                                     separators=(",", ":"))
                self.wfile.write(f"id: {chunk.sequence}\ndata: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
                sequence = chunk.sequence

    def stream_live_transcripts(self):
        sequence = self.server.live_transcripts.latest_sequence
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        while True:
            events = self.server.live_transcripts.after(sequence, timeout=10.0)
            if not events:
                self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
                continue
            for event in events:
                sequence = event["sequence"]
                payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                self.wfile.write(f"id: {sequence}\ndata: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()

    def range_error(self, size):
        self.send_common_headers(416, "text/plain", 0)
        self.send_header("Content-Range", f"bytes */{size}")
        self.end_headers()

    def log_message(self, format, *args):
        # Polling is intentionally quiet; startup and fatal errors remain visible.
        pass


class MonitorServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, output_base, settings=None):
        self.settings = settings or load_settings()
        self.store = MonitorStore(output_base)
        self.labels = LabelStore(self.store.root)
        self.live_audio = LiveAudioHub()
        self.live_transcripts = LiveTranscriptHub()
        super().__init__(address, MonitorHandler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--output-base", type=Path)
    args = parser.parse_args(argv)
    settings = load_settings()
    output_base = args.output_base or Path(settings["output_base"])
    subscriber = None
    try:
        with MonitorServer((args.host, args.port), output_base, settings=settings) as server:
            if settings["mqtt_enabled"] and not settings["save_transcript"]:
                subscriber = MqttTranscriptSubscriber(
                    hub=server.live_transcripts, host=settings["mqtt_host"],
                    port=settings["mqtt_port"], topic=settings["mqtt_topic"],
                    qos=settings["mqtt_qos"],
                    keepalive_seconds=settings["mqtt_keepalive_seconds"])
                subscriber.start()
            print(f"MOSS monitor: http://{args.host}:{server.server_port}/", flush=True)
            print("Output directory:", server.store.root, flush=True)
            try:
                server.serve_forever(poll_interval=0.5)
            except KeyboardInterrupt:
                pass
    finally:
        if subscriber is not None:
            subscriber.close()


if __name__ == "__main__":
    main()
