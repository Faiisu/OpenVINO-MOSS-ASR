"""Native-precision audio I/O and frame copying."""
from contextlib import contextmanager
from pathlib import Path
import shutil
import subprocess
import tempfile
import soundfile as sf


def resolve_ffmpeg():
    """Use FFmpeg on PATH or the binary bundled in the local environment."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "FFmpeg unavailable. Install imageio-ffmpeg into the project .venv using uv."
        ) from error


@contextmanager
def open_audio_source(path):
    """Read supported files directly; decode other containers at native settings."""
    try:
        source = sf.SoundFile(path)
    except sf.LibsndfileError:
        import av

        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            sample_format = stream.codec_context.format.name.rstrip("p")
        codecs = {"u8": "pcm_u8", "s16": "pcm_s16le", "s32": "pcm_s32le",
                  "flt": "pcm_f32le", "dbl": "pcm_f64le"}
        if sample_format not in codecs:
            raise ValueError(f"Cannot preserve decoded audio format: {sample_format}")
        with tempfile.TemporaryDirectory(prefix="moss-decode-") as temporary:
            decoded = Path(temporary) / "decoded.wav"
            subprocess.run([resolve_ffmpeg(), "-nostdin", "-v", "error", "-i", str(path),
                            "-map", "0:a:0", "-vn", "-c:a", codecs[sample_format], str(decoded)],
                           check=True, timeout=300)
            with sf.SoundFile(decoded) as source:
                yield source
    else:
        with source:
            yield source


def open_audio_target(path, source):
    """Write PCM/float WAVs with the source precision, channels and sample rate."""
    subtype = source.subtype
    if subtype not in {"PCM_U8", "PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}:
        # Lossy codecs have no fixed PCM bit depth. Retain floating decoded samples.
        subtype = "FLOAT"
    wav_source = source.format in {"WAV", "WAVEX"}
    return sf.SoundFile(path, mode="w", samplerate=source.samplerate, channels=source.channels,
                        subtype=subtype, format=source.format if wav_source else "WAV",
                        endian=source.endian if wav_source else "FILE")


def copy_audio_frames(source, target, start, count):
    """Stream exact PCM/float samples in bounded blocks, without normalization."""
    dtype = {"PCM_U8": "int16", "PCM_16": "int16", "PCM_24": "int32",
             "PCM_32": "int32", "DOUBLE": "float64"}.get(source.subtype, "float32")
    source.seek(start)
    sample_bytes = {"int16": 2, "int32": 4, "float32": 4, "float64": 8}[dtype]
    while count:
        size = min(count, 65536)
        data = source.buffer_read(size, dtype=dtype)
        if len(data) != size * source.channels * sample_bytes:
            raise ValueError("Audio ended before the requested frame range.")
        target.buffer_write(data, dtype=dtype)
        count -= size
