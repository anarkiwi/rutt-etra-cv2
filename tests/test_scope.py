"""Simulated oscilloscope tests."""

import cv2
import numpy as np
import pytest

from ruttetra.audio import AudioParams, WavSink, frame_signals
from ruttetra.core import ScanParams, beam_path
from ruttetra.raster import accumulate, canvas_shape
from ruttetra.scope import (
    PCM_SCALE,
    Screen,
    ScopeParams,
    ScopeSink,
    beam,
    decode,
    render_wav,
    trace,
    wav_blocks,
)

SMALL = ScopeParams(size=64, spot=0.0, bloom=0.0, persistence=0.0)


def circle(count=512, channels=2, radius=0.8):
    """A closed XY circle, the classic scope test figure."""
    angle = np.linspace(0, 2 * np.pi, count, endpoint=False, dtype=np.float32)
    out = np.ones((count, channels), dtype=np.float32)
    out[:, 0] = radius * np.cos(angle)
    out[:, 1] = radius * np.sin(angle)
    return out


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": 4},
        {"aspect": 0.0},
        {"gain": -1.0},
        {"spot": -1.0},
        {"bloom": -1.0},
        {"persistence": 1.0},
        {"persistence": -0.1},
    ],
)
def test_invalid_params_rejected(kwargs):
    """Unusable front panel settings fail loudly."""
    with pytest.raises(ValueError):
        ScopeParams(**kwargs)


def test_shape_follows_aspect():
    """A scope is square unless told otherwise."""
    assert ScopeParams(size=100).shape == (100, 100)
    assert ScopeParams(size=100, aspect=1.5).shape == (100, 150)


@pytest.mark.parametrize("width", [2, 3, 4])
def test_decode_round_trip(width):
    """PCM of every supported width decodes back to the original voltages."""
    from ruttetra.audio import quantise  # pylint: disable=import-outside-toplevel

    block = np.array([[-1.0, 0.0], [0.5, -0.25], [1.0, 0.75]], dtype=np.float64)
    if width == 4:
        raw = np.round(block * 2147483647.0).astype("<i4").tobytes()
    else:
        raw = quantise(block, width * 8)
    assert np.allclose(decode(raw, width, 2), block, atol=2.0 / PCM_SCALE[width])


def test_beam_maps_the_screen_corners():
    """Full scale deflection reaches the edges, +Y at the top."""
    signals = np.array([[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0]], dtype=np.float32)
    params = ScopeParams(size=100, aspect=1.0)
    px, py, level = beam(signals, params)
    assert (px[0], py[0]) == (0.0, 0.0)
    assert (px[1], py[1]) == (99.0, 99.0)
    assert np.allclose(level, 1.0)


def test_z_channel_sets_beam_current():
    """Channel three is intensity, and -1 blanks."""
    signals = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    assert np.allclose(beam(signals, SMALL)[2], [0.0, 1.0])
    flipped = ScopeParams(size=64, z_invert=True)
    assert np.allclose(beam(signals, flipped)[2], [1.0, 0.0])


def test_z_can_be_ignored():
    """Two channel signals, or --no-scope-z, run the beam at full current."""
    assert np.allclose(beam(circle(8, 2), SMALL)[2], 1.0)
    off = ScopeParams(size=64, z=False)
    assert np.allclose(beam(circle(8, 3), off)[2], 1.0)


def test_blanked_beam_draws_nothing():
    """A fully blanked Z channel leaves the screen dark."""
    signals = circle(256, 3)
    signals[:, 2] = -1.0
    assert trace(signals, SMALL).max() == 0.0


def test_circle_is_drawn_as_a_ring():
    """The classic XY test figure comes out as a closed ring."""
    screen = trace(circle(1024), SMALL)
    lit = screen > 0
    assert lit.any()
    centre = SMALL.size // 2
    assert not lit[centre - 4 : centre + 4, centre - 4 : centre + 4].any()
    count, _ = cv2.connectedComponents((~lit).astype(np.uint8), connectivity=4)
    assert count - 1 == 2


def test_fast_travel_writes_fainter():
    """Brightness falls with beam speed, as on a real scope."""
    params = ScopeParams(size=128, spot=0.0, bloom=0.0, persistence=0.0)
    slow = np.zeros((64, 2), dtype=np.float32)
    slow[:, 0] = np.linspace(-0.1, 0.1, 64)
    fast = np.zeros((64, 2), dtype=np.float32)
    fast[:, 0] = np.linspace(-0.9, 0.9, 64)
    assert trace(slow, params).max() > 4 * trace(fast, params).max()


def test_spot_and_bloom_spread_the_trace():
    """Spot size and bloom widen the trace without inventing energy elsewhere."""
    sharp = trace(circle(512), SMALL)
    soft = trace(circle(512), ScopeParams(size=64, spot=1.5, bloom=0.0))
    assert (soft > 0).sum() > (sharp > 0).sum()
    assert soft.max() < sharp.max()


def test_persistence_holds_the_trace():
    """Phosphor fades between frames rather than vanishing."""
    lit, dark = circle(512), np.zeros((512, 2), dtype=np.float32)
    screen = Screen(ScopeParams(size=64, spot=0.0, bloom=0.0, persistence=0.5))
    first = screen.render(lit).astype(int).sum()
    second = screen.render(dark).astype(int).sum()
    assert 0 < second < first


def test_no_persistence_clears_between_frames():
    """With persistence at zero each frame stands alone."""
    screen = Screen(SMALL)
    screen.render(circle(512, 3))
    blanked = np.zeros((512, 3), dtype=np.float32)
    blanked[:, 2] = -1.0
    assert screen.render(blanked).max() == 0


def test_parked_beam_burns_a_spot():
    """A stationary beam dumps every sample into one place, as a real one does."""
    parked = np.zeros((512, 2), dtype=np.float32)
    screen = trace(parked, SMALL)
    assert screen.max() > 100
    assert (screen > 0).sum() <= 4


def test_output_is_monochrome():
    """The simulated screen is black and white."""
    image = Screen(SMALL).render(circle(512))
    assert np.array_equal(image[..., 0], image[..., 1])
    assert np.array_equal(image[..., 1], image[..., 2])


def test_graticule():
    """The optional grid draws divisions on an otherwise dark screen."""
    params = ScopeParams(size=64, spot=0.0, bloom=0.0, persistence=0.0, graticule=True)
    blank = Screen(params).render(np.zeros((8, 2), dtype=np.float32))
    assert blank.max() > 0
    assert (blank > 0).sum() < blank[..., 0].size


def write_wav(path, frames=3, channels=3, rate=48000, bits=16):
    """Write a short deflection WAV for the reader tests."""
    audio = AudioParams(rate=rate, channels=channels, bits=bits)
    noise = np.tile(np.arange(64, dtype=np.uint8), (48, 3, 1)).transpose(0, 2, 1)
    sink = WavSink(path, audio)
    for _ in range(frames):
        sink.write(frame_signals(noise, ScanParams(lines=4), audio, rate // 25))
    sink.close()
    return path


def test_wav_blocks(tmp_path):
    """The reader hands back one video frame of samples at a time."""
    path = write_wav(tmp_path / "d.wav", frames=3)
    blocks = list(wav_blocks(path, 25.0))
    assert len(blocks) == 3
    assert all(b.shape == (1920, 3) for b in blocks)
    assert all(b.dtype == np.float32 for b in blocks)
    assert max(abs(b).max() for b in blocks) <= 1.0


def test_wav_blocks_rejects_unsupported_width(tmp_path):
    """An 8-bit file is refused rather than silently misread."""
    import wave  # pylint: disable=import-outside-toplevel

    path = tmp_path / "eight.wav"
    handle = wave.Wave_write(str(path))
    handle.setnchannels(2)
    handle.setsampwidth(1)
    handle.setframerate(48000)
    handle.writeframes(b"\x00" * 64)
    handle.close()
    with pytest.raises(ValueError):
        list(wav_blocks(path, 25.0))


def test_render_wav(tmp_path):
    """Rendering a WAV yields one monochrome frame per video frame."""
    path = write_wav(tmp_path / "d.wav", frames=4)
    images = list(render_wav(path, 25.0, ScopeParams(size=48, aspect=1.5)))
    assert len(images) == 4
    assert all(im.shape == (48, 72, 3) for im in images)
    assert max(im.max() for im in images) > 0


def test_scope_sink_writes_video(tmp_path):
    """The sink produces a playable file at the screen size."""
    out = tmp_path / "scope.avi"
    params = ScopeParams(size=48, aspect=1.5, persistence=0.0)
    sink = ScopeSink(out, params, 25.0)
    for block in wav_blocks(write_wav(tmp_path / "d.wav", frames=3), 25.0):
        sink.write(block)
    sink.close()
    assert sink.frames == 3

    cap = cv2.VideoCapture(str(out))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape == (48, 72, 3)


def test_scope_sink_without_a_path(tmp_path):
    """The sink can render without writing anything, for preview only."""
    sink = ScopeSink(None, SMALL, 25.0)
    image = sink.write(next(wav_blocks(write_wav(tmp_path / "d.wav"), 25.0)))
    sink.close()
    assert image.shape == (64, 64, 3) and sink.frames == 1


def test_scope_reproduces_the_raster_geometry(noise):
    """The scope view of the WAV lands on the same shape the raster draws.

    Same beam path, two renderers and a round trip through PCM in between.
    """
    height, width = noise.shape[:2]
    params = ScanParams(lines=8, gain=0.3, beam="speed", intensity=0.0, brightness=1.0)
    path = beam_path(noise, params)
    audio = AudioParams(channels=3)
    signals = frame_signals(noise, params, audio, 8192)

    out_h, out_w = canvas_shape(height, width, params)
    ink = accumulate(path, height, width, params).sum(axis=2) > 0

    screen = ScopeParams(size=out_h, aspect=out_w / out_h, spot=0.0, bloom=0.0)
    lit = Screen(screen).render(signals)[..., 0] > 0
    assert lit.shape == ink.shape

    overlap = np.logical_and(lit, cv2.dilate(ink.astype(np.uint8), np.ones((3, 3))) > 0)
    assert overlap.sum() / lit.sum() > 0.99
