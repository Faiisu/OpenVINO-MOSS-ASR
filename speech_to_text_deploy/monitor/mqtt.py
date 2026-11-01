"""Receive transient transcript events from the configured MQTT topic."""

from collections import deque
import json
import os
from threading import Condition


class LiveTranscriptHub:
    """Keep a small in-memory event tail for currently connected SSE clients."""

    def __init__(self, max_events=100):
        if max_events < 1:
            raise ValueError("Live transcript buffer must retain at least one event.")
        self._events = deque(maxlen=max_events)
        self._sequence = 0
        self._condition = Condition()

    @property
    def latest_sequence(self):
        with self._condition:
            return self._sequence

    def publish(self, payload):
        """Publish one validated transcript to connected monitor clients."""
        if not isinstance(payload, dict) or not all(
                isinstance(payload.get(key), str) for key in ("transcript", "start_time", "end_time")):
            raise ValueError("MQTT transcript must contain text and start/end timestamps.")
        with self._condition:
            self._sequence += 1
            event = {"sequence": self._sequence, **payload}
            self._events.append(event)
            self._condition.notify_all()
            return self._sequence

    def after(self, sequence, timeout=10.0):
        """Wait for events newer than sequence without replaying earlier events."""
        with self._condition:
            if self._sequence <= sequence:
                self._condition.wait(timeout)
            return tuple(event for event in self._events if event["sequence"] > sequence)


class MqttTranscriptSubscriber:
    """Forward non-retained MOSS reports from MQTT into the in-memory hub."""

    def __init__(self, *, hub, host, port, topic, qos=1, keepalive_seconds=30):
        self.hub = hub
        self.host = host
        self.port = port
        self.topic = topic
        self.qos = qos
        self.keepalive_seconds = keepalive_seconds
        self.client = None
        self.loop_started = False

    def start(self):
        from paho.mqtt import client as mqtt

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"moss-monitor-{os.getpid()}",
        )
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect_async(self.host, self.port, self.keepalive_seconds)
        self.client.loop_start()
        self.loop_started = True

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            print(f"MQTT monitor connection failed: {reason_code}", flush=True)
            return
        result, _ = client.subscribe(self.topic, qos=self.qos)
        if result != 0:
            print(f"MQTT monitor could not subscribe to {self.topic}: {result}", flush=True)
        else:
            print(f"MQTT monitor subscribed to {self.topic}", flush=True)

    def _on_message(self, client, userdata, message):
        try:
            payload = json.loads(message.payload.decode("utf-8"))
            self.hub.publish(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            print(f"Ignoring invalid MQTT transcript on {message.topic}: {error}", flush=True)

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
