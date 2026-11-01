"""Publish transcription reports as JSON over MQTT."""

import json
import os
from threading import Event


def transcript_messages(manifest):
    """Build minimal real-time MQTT messages, one per accepted utterance."""
    return [{"transcript": entry["text"],
             "start_time": entry["model_start_utc"],
             "end_time": entry["model_end_utc"]}
            for entry in manifest["segments"]]


class MqttJsonPublisher:
    """Keep one MQTT connection open and confirm each publish before returning."""

    def __init__(self, *, host, port, topic, qos=1, keepalive_seconds=30,
                 connect_timeout_seconds=5.0, publish_timeout_seconds=5.0):
        self.host = host
        self.port = port
        self.topic = topic
        self.qos = qos
        self.keepalive_seconds = keepalive_seconds
        self.connect_timeout_seconds = connect_timeout_seconds
        self.publish_timeout_seconds = publish_timeout_seconds
        self.client = None
        self.connected = Event()
        self.last_error = None
        self.loop_started = False

    def __enter__(self):
        from paho.mqtt import client as mqtt

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"moss-asr-{os.getpid()}",
        )
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        try:
            result = self.client.connect(self.host, self.port, self.keepalive_seconds)
            if result != mqtt.MQTT_ERR_SUCCESS:
                raise ConnectionError(f"MQTT connect failed: {mqtt.error_string(result)}")
            self.client.loop_start()
            self.loop_started = True
            if not self.connected.wait(self.connect_timeout_seconds):
                detail = f": {self.last_error}" if self.last_error else ""
                raise TimeoutError(f"Timed out connecting to MQTT {self.host}:{self.port}{detail}")
            return self
        except BaseException:
            self.close()
            raise

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            self.last_error = str(reason_code)
            self.connected.clear()
        else:
            self.last_error = None
            self.connected.set()

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        self.connected.clear()
        if reason_code.is_failure:
            self.last_error = str(reason_code)

    def publish_json(self, payload):
        """Publish one UTF-8 JSON document and wait for its configured QoS ACK."""
        if self.client is None:
            raise RuntimeError("MQTT publisher is not connected.")
        if not self.connected.wait(self.connect_timeout_seconds):
            detail = f": {self.last_error}" if self.last_error else ""
            raise ConnectionError(f"MQTT broker is disconnected{detail}")
        message = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        result = self.client.publish(self.topic, message, qos=self.qos, retain=False)
        from paho.mqtt import client as mqtt
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            raise ConnectionError(f"MQTT publish failed: {mqtt.error_string(result.rc)}")
        result.wait_for_publish(timeout=self.publish_timeout_seconds)
        if not result.is_published():
            raise TimeoutError(f"MQTT publish to {self.topic} was not acknowledged in time.")
        return result.mid

    def close(self):
        client, self.client = self.client, None
        if client is None:
            return
        try:
            client.disconnect()
        finally:
            if self.loop_started:
                client.loop_stop()
                self.loop_started = False

    def __exit__(self, *exc):
        self.close()
