"""Oscilloscope output: the beam path as X/Y (+Z) audio.

The same path the raster renderer draws is resampled onto the audio clock and
emitted as deflection voltages, so a scope in XY mode reproduces the picture.
Channel 3, when present, is the Z (beam current) axis.
"""

import wave
from dataclasses import dataclass

import numpy as np

from .core import beam_clock, beam_path

BIT_DEPTHS = (16, 24)


@dataclass(frozen=True)
class AudioParams:
    """Oscilloscope signal settings."""

    rate: int = 96000
    channels: int = 2
    bits: int = 16
    z_invert: bool = False

    def __post_init__(self):
        if self.rate < 1:
            raise ValueError("rate must be >= 1")
        if self.channels not in (2, 3):
            raise ValueError("channels must be 2 (XY) or 3 (XYZ)")
        if self.bits not in BIT_DEPTHS:
            raise ValueError(f"bits must be one of {BIT_DEPTHS}")


def samples_per_frame(rate, fps):
    """Audio samples covering one video frame."""
    if fps <= 0:
        raise ValueError("fps must be > 0")
    return max(2, int(round(rate / fps)))


def signals_from_path(path, params, audio, samples):
    """Resample an already-computed beam path to `samples` deflection voltages."""
    clock = beam_clock(path, params.beam)
    grid = np.linspace(0.0, clock[-1], samples, endpoint=False)
    x = np.interp(grid, clock, path.x)
    y = np.interp(grid, clock, path.y)
    z = np.interp(grid, clock, path.z)

    y_min, y_max = params.y_bounds
    out = np.empty((samples, audio.channels), dtype=np.float32)
    out[:, 0] = x / params.x_extent
    out[:, 1] = (2.0 * y - (y_max + y_min)) / (y_max - y_min)
    if audio.channels == 3:
        out[:, 2] = (1.0 - 2.0 * z) if audio.z_invert else (2.0 * z - 1.0)
    return np.clip(out, -1.0, 1.0)


def frame_signals(frame, params, audio, samples):
    """Resample one frame's beam path to `samples` deflection voltages."""
    return signals_from_path(beam_path(frame, params), params, audio, samples)


def quantise(block, bits):
    """Pack float samples in [-1, 1] into little-endian PCM bytes."""
    clipped = np.clip(block, -1.0, 1.0)
    if bits == 16:
        return np.round(clipped * 32767.0).astype("<i2").tobytes()
    ints = np.round(clipped * 8388607.0).astype("<i4")
    return np.ascontiguousarray(ints.view(np.uint8).reshape(-1, 4)[:, :3]).tobytes()


class WavSink:
    """Streaming WAV writer for deflection signals."""

    def __init__(self, path, audio):
        self.audio = audio
        self.handle = wave.open(str(path), "wb")
        self.handle.setnchannels(audio.channels)
        self.handle.setsampwidth(audio.bits // 8)
        self.handle.setframerate(audio.rate)

    def write(self, block):
        """Append one block of (samples, channels) float data."""
        self.handle.writeframes(quantise(block, self.audio.bits))

    def close(self):
        """Finalise the file."""
        self.handle.close()


class SoundCardSink:
    """Live deflection output through a sound card."""

    def __init__(self, audio, device=None):
        # Optional dependency: only needed for live output.
        import sounddevice  # pylint: disable=import-outside-toplevel,import-error

        self.stream = sounddevice.OutputStream(
            samplerate=audio.rate,
            channels=audio.channels,
            device=device,
            dtype="float32",
        )
        self.stream.start()

    def write(self, block):
        """Append one block of (samples, channels) float data."""
        self.stream.write(np.ascontiguousarray(block, dtype=np.float32))

    def close(self):
        """Stop and release the device."""
        self.stream.stop()
        self.stream.close()
