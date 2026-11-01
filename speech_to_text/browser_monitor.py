"""Send live microphone PCM to a browser through an SSH-forwarded HTTP port."""

import atexit
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import socket
import threading
from urllib.parse import parse_qs, urlsplit

import numpy as np
import sounddevice as sd


class BrowserMicrophoneMonitor:
    """Serve one live listener on loopback; capture only while it is connected."""

    def __init__(self):
        self.server = None
        self.thread = None
        self.stopping = threading.Event()
        self.capture_lock = threading.Lock()
        self.last_status = ""
        self.peak = 0.0
        self.url = ""

    @property
    def active(self):
        return self.server is not None and not self.stopping.is_set()

    @property
    def listening(self):
        return self.capture_lock.locked()

    def start(self, input_device, port=8765, samplerate=48000):
        """Start the local server without opening a playback device or microphone."""
        self.close()
        sd.check_input_settings(device=input_device, channels=1,
                                dtype="int16", samplerate=samplerate)
        self.input_device = input_device
        self.samplerate = samplerate
        self.token = secrets.token_urlsafe(24)
        self.peak = 0.0
        self.last_status = ""
        self.stopping.clear()
        monitor = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.connection.settimeout(2)
                request = urlsplit(self.path)
                token = parse_qs(request.query).get("token", [""])[0]
                if not hmac.compare_digest(token, monitor.token):
                    self.send_error(403, "Use the link printed by the notebook.")
                    return
                if request.path == "/":
                    page = Path(__file__).with_name("browser_monitor.html").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(page)
                elif request.path == "/audio":
                    monitor._serve_audio(self)
                else:
                    self.send_error(404)

            def log_message(self, format, *args):
                pass

        try:
            self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            # Allow atexit to run before Python waits for request threads.
            self.server.daemon_threads = True
            actual_port = self.server.server_port
            self.url = f"http://127.0.0.1:{actual_port}/?token={self.token}"
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            atexit.register(self.close)
        except BaseException:
            self.close()
            raise
        return self.url

    def _serve_audio(self, handler):
        if not self.capture_lock.acquire(blocking=False):
            handler.send_error(409, "Another listener is connected. Stop it first.")
            return
        headers_sent = False
        try:
            if self.stopping.is_set():
                handler.send_error(503, "The notebook server has stopped.")
                return
            with sd.RawInputStream(device=self.input_device, samplerate=self.samplerate,
                                   channels=1, dtype="int16") as stream:
                handler.send_response(200)
                handler.send_header("Content-Type", "application/octet-stream")
                handler.send_header("Cache-Control", "no-store")
                handler.send_header("X-Sample-Rate", str(self.samplerate))
                handler.end_headers()
                headers_sent = True
                while not self.stopping.is_set():
                    data, overflowed = stream.read(self.samplerate // 50)
                    samples = np.frombuffer(data, dtype=np.int16)
                    self.peak = max(self.peak, float(np.max(np.abs(samples.astype(np.int32)))) / 32768)
                    if overflowed:
                        self.last_status = "Input overflow: audio frames were dropped."
                    handler.wfile.write(samples.astype("<i2", copy=False).tobytes())
                    handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass
        except Exception as error:
            self.last_status = str(error)
            if not headers_sent:
                handler.send_error(503, "Cannot open microphone; check notebook status.")
        finally:
            self.capture_lock.release()

    def close(self):
        """Stop capture, join HTTP handlers, and release the server port."""
        atexit.unregister(self.close)
        self.stopping.set()
        server, self.server = self.server, None
        thread, self.thread = self.thread, None
        if server is not None:
            try:
                if thread is not None and thread.is_alive():
                    server.shutdown()
                    thread.join()
            finally:
                server.server_close()
                # Daemon request threads are not joined by server_close(). Wait
                # for the capture context to release its microphone explicitly.
                with self.capture_lock:
                    pass
