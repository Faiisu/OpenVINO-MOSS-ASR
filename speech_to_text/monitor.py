"""Monitor a local microphone through a local playback device."""

import atexit

import numpy as np
import sounddevice as sd


class MicrophoneMonitor:
    """Own one duplex stream and release it on failure, stop, or shutdown."""

    def __init__(self):
        self.stream = None
        self.last_status = ""
        self.peak = 0.0
        self.gain = 0.1

    @property
    def active(self):
        return self.stream is not None and self.stream.active

    def _callback(self, indata, outdata, frames, time_info, status):
        if status:
            self.last_status = str(status)
        self.peak = max(self.peak, float(np.max(np.abs(indata))))
        # Broadcast the first microphone channel to every playback channel.
        np.multiply(indata[:, :1], self.gain, out=outdata)
        np.clip(outdata, -1.0, 1.0, out=outdata)

    def start(self, input_device, output_device=None, samplerate=48000, gain=0.1):
        """Validate device settings and start mono monitoring at a shared rate."""
        self.close()
        if not 0 <= gain <= 1:
            raise ValueError("Gain must be between 0 and 1.")
        output_info = sd.query_devices(output_device, "output")
        output_channels = min(2, int(output_info["max_output_channels"]))
        sd.check_input_settings(device=input_device, channels=1,
                                dtype="float32", samplerate=samplerate)
        sd.check_output_settings(device=output_device, channels=output_channels,
                                 dtype="float32", samplerate=samplerate)
        self.gain = gain
        self.peak = 0.0
        self.last_status = ""
        try:
            self.stream = sd.Stream(
                device=(input_device, output_device), samplerate=samplerate,
                channels=(1, output_channels), dtype="float32", latency="high",
                callback=self._callback,
            )
            self.stream.start()
            atexit.register(self.close)
        except BaseException:
            self.close()
            raise

    def close(self):
        """Stop and close the stream, including when stopping raises."""
        atexit.unregister(self.close)
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
