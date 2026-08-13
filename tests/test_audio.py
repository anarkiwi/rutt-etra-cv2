"""Oscilloscope output tests."""

import sys
import types
import wave

import cv2
import numpy as np
import pytest

from ruttetra.audio import (
    AudioParams,
    SoundCardSink,
    WavSink,
    frame_signals,
    quantise,
    samples_per_frame,
    signals_from_path,
)
from ruttetra.core import BeamPath, ScanParams, beam_path
from ruttetra.raster import accumulate, canvas_shape, to_pixels


def test_samples_per_frame():
    """One video frame maps onto its share of the audio clock."""
    assert samples_per_frame(96000, 25) == 3840
    assert samples_per_frame(48000, 30) == 1600
    with pytest.raises(ValueError):
        samples_per_frame(48000, 0)


@pytest.mark.parametrize("channels", [2, 3])
def test_signal_shape_and_range(noise, channels):
    """Deflection voltages fill the available swing without clipping past it."""
    signals = frame_signals(
        noise, ScanParams(lines=8), AudioParams(channels=channels), 512
    )
    assert signals.shape == (512, channels)
    assert signals.min() >= -1.0 and signals.max() <= 1.0
    assert signals[:, 0].max() > 0.9 and signals[:, 0].min() < -0.9


def test_x_sweeps_the_full_width(ramp):
    """X is the horizontal ramp, hitting both ends of the swing."""
    signals = frame_signals(ramp, ScanParams(lines=4), AudioParams(), 4096)
    assert signals[:, 0].max() == pytest.approx(1.0, abs=0.02)
    assert signals[:, 0].min() == pytest.approx(-1.0, abs=0.02)


def test_y_carries_the_displacement(step):
    """Bright picture areas sit higher up the Y axis."""
    params = ScanParams(lines=1, gain=0.4, retrace=0)
    signals = frame_signals(step, params, AudioParams(), 512)
    left = signals[signals[:, 0] < -0.5, 1]
    right = signals[signals[:, 0] > 0.5, 1]
    assert right.mean() > left.mean()


def test_two_channels_omit_z(noise):
    """XY mode carries deflection only."""
    assert frame_signals(noise, ScanParams(lines=4), AudioParams(channels=2), 256)[
        0
    ].shape == (2,)


def test_z_channel_tracks_beam_current(ramp):
    """The third channel is the Z axis, blanked low and lit high."""
    signals = frame_signals(ramp, ScanParams(lines=4), AudioParams(channels=3), 4096)
    assert signals[:, 2].min() == pytest.approx(-1.0, abs=0.01)
    assert signals[:, 2].max() > 0.9


def test_z_invert(ramp):
    """Inverted Z suits scopes whose blanking input is active high."""
    params, samples = ScanParams(lines=4), 1024
    plain = frame_signals(ramp, params, AudioParams(channels=3), samples)
    flipped = frame_signals(
        ramp, params, AudioParams(channels=3, z_invert=True), samples
    )
    assert np.allclose(plain[:, 2], -flipped[:, 2], atol=1e-6)
    assert np.allclose(plain[:, :2], flipped[:, :2])


def test_retrace_is_blanked_in_audio(noise):
    """The Z channel goes fully low somewhere on every interline flyback."""
    signals = frame_signals(
        noise, ScanParams(lines=4, retrace=16), AudioParams(channels=3), 4096
    )
    assert (signals[:, 2] <= -0.999).sum() >= 4


def velocity_spread(noise, beam):
    """Coefficient of variation of beam velocity over one frame."""
    params = ScanParams(lines=8, gain=0.3, beam=beam)
    signals = frame_signals(noise, params, AudioParams(), 1024)
    step = np.hypot(np.diff(signals[:, 0]), np.diff(signals[:, 1]))
    return step.std() / step.mean()


def test_constant_speed_moves_the_beam_evenly(noise):
    """Arc-length resampling gives a near-constant beam velocity."""
    assert velocity_spread(noise, "speed") < 0.3


def test_constant_rate_varies_beam_velocity(noise):
    """Constant-rate deflection races across steep parts of the trace."""
    assert velocity_spread(noise, "rate") > 3 * velocity_spread(noise, "speed")


def test_audio_is_deterministic(noise):
    """Repeated conversions are identical."""
    args = (ScanParams(lines=8), AudioParams(channels=3), 1024)
    first = frame_signals(noise, *args)
    for _ in range(3):
        assert np.array_equal(frame_signals(noise, *args), first)


@pytest.mark.parametrize("bits,width", [(16, 2), (24, 3)])
def test_quantise_round_trip(bits, width):
    """PCM packing preserves the signal to within a quantisation step."""
    block = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(-1, 2)
    raw = quantise(block, bits)
    assert len(raw) == block.size * width
    if bits == 16:
        back = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
        assert np.allclose(back.reshape(block.shape), block, atol=1e-4)


def test_quantise_clips(bits=16):
    """Overdriven deflection saturates instead of wrapping around."""
    raw = np.frombuffer(
        quantise(np.array([[-5.0, 5.0]], dtype=np.float32), bits), "<i2"
    )
    assert raw.tolist() == [-32767, 32767]


@pytest.mark.parametrize("bits", [16, 24])
@pytest.mark.parametrize("channels", [2, 3])
def test_wav_round_trip(tmp_path, noise, bits, channels):
    """A written WAV has the declared format and length."""
    path = tmp_path / "scope.wav"
    audio = AudioParams(rate=48000, channels=channels, bits=bits)
    sink = WavSink(path, audio)
    for _ in range(3):
        sink.write(frame_signals(noise, ScanParams(lines=4), audio, 800))
    sink.close()

    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == channels
        assert handle.getsampwidth() == bits // 8
        assert handle.getframerate() == 48000
        assert handle.getnframes() == 2400


def test_wav_holds_the_deflection(tmp_path, ramp):
    """Samples read back from the file match what was written."""
    path = tmp_path / "scope.wav"
    audio = AudioParams(rate=48000, channels=2, bits=16)
    signals = frame_signals(ramp, ScanParams(lines=4), audio, 512)
    sink = WavSink(path, audio)
    sink.write(signals)
    sink.close()

    with wave.open(str(path), "rb") as handle:
        raw = np.frombuffer(handle.readframes(handle.getnframes()), "<i2")
    assert np.allclose(raw.reshape(-1, 2) / 32767.0, signals, atol=1e-4)


@pytest.mark.parametrize(
    "kwargs", [{"rate": 0}, {"channels": 1}, {"channels": 4}, {"bits": 8}]
)
def test_invalid_audio_params_rejected(kwargs):
    """Unusable formats fail before a file is opened."""
    with pytest.raises(ValueError):
        AudioParams(**kwargs)


class FakeStream:
    """Stand-in for a sounddevice output stream."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.blocks = []
        self.state = "new"

    def start(self):
        """Open the device."""
        self.state = "running"

    def write(self, block):
        """Queue a block of samples."""
        self.blocks.append(block)

    def stop(self):
        """Halt playback."""
        self.state = "stopped"

    def close(self):
        """Release the device."""
        self.state = "closed"


def test_sound_card_sink(monkeypatch, noise):
    """The live sink opens the device with the declared format and feeds it."""
    module = types.SimpleNamespace(OutputStream=FakeStream)
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    audio = AudioParams(rate=48000, channels=3)
    sink = SoundCardSink(audio, device="Scarlett")
    assert sink.stream.kwargs == {
        "samplerate": 48000,
        "channels": 3,
        "device": "Scarlett",
        "dtype": "float32",
    }
    assert sink.stream.state == "running"

    sink.write(frame_signals(noise, ScanParams(lines=4), audio, 256))
    assert sink.stream.blocks[0].shape == (256, 3)
    assert sink.stream.blocks[0].dtype == np.float32

    sink.close()
    assert sink.stream.state == "closed"


def test_audio_and_raster_agree(noise):
    """Both outputs trace one path: every lit audio sample lands on drawn ink.

    Allowing one pixel of slack, since the rasteriser samples segment starts
    while the audio is continuous. This is what makes the two modes one effect.
    """
    height, width = noise.shape[:2]
    params = ScanParams(lines=8, gain=0.3, beam="speed", intensity=0.0, brightness=1.0)
    path = beam_path(noise, params)
    audio = AudioParams(channels=3)
    signals = signals_from_path(path, params, audio, 8192)

    acc = accumulate(path, height, width, params).sum(axis=2)
    out_h, out_w = canvas_shape(height, width, params)
    lit = signals[signals[:, 2] > 0.999]
    y_min, y_max = params.y_bounds
    replayed = BeamPath(
        x=lit[:, 0] * params.x_extent,
        y=(lit[:, 1] * (y_max - y_min) + (y_max + y_min)) / 2.0,
        z=np.ones(len(lit), dtype=np.float32),
        color=np.ones((len(lit), 3), dtype=np.float32),
    )
    px, py = to_pixels(replayed, height, width, params)
    px = np.clip(px.astype(int), 0, out_w - 1)
    py = np.clip(py.astype(int), 0, out_h - 1)
    inked = cv2.dilate((acc > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    assert inked[py, px].mean() > 0.999
    assert (acc[py, px] > 0).mean() > 0.95
