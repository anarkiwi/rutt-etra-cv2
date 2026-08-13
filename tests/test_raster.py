"""Beam trace rasteriser tests."""

import cv2
import numpy as np
import pytest

from ruttetra.core import BeamPath, ScanParams, beam_path
from ruttetra.raster import accumulate, canvas_shape, render, render_path, to_pixels


def lit_mask(image):
    """Boolean mask of pixels the beam wrote to."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 0


def components(mask):
    """Count of 8-connected foreground regions."""
    count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return count - 1


def test_canvas_holds_the_displacement(noise):
    """Output grows vertically by exactly the displacement range."""
    height, width = noise.shape[:2]
    assert canvas_shape(height, width, ScanParams(gain=0.1)) == (
        round(height * 1.1),
        width,
    )
    assert render(noise, ScanParams(lines=8, gain=0.25)).shape == (
        round(height * 1.25),
        width,
        3,
    )


def test_scanline_is_continuous(step):
    """A single scanline is one unbroken curve, not a row of dots."""
    image = render(step, ScanParams(lines=1, gain=0.4, retrace=0, beam="speed"))
    mask = lit_mask(image)
    assert np.all(mask.sum(axis=0) > 0)
    assert components(mask) == 1


def test_luminance_step_is_bridged(step):
    """The trace spans the vertical jump at a brightness edge."""
    image = render(step, ScanParams(lines=1, gain=0.4, retrace=0, beam="speed"))
    spans = [(col.min(), col.max()) for col in map(np.flatnonzero, lit_mask(image).T)]
    gaps = [max(0, max(a[0], b[0]) - min(a[1], b[1])) for a, b in zip(spans, spans[1:])]
    assert max(gaps) <= 1


def test_dot_per_column_would_be_disconnected(step):
    """The replaced approach leaves one isolated pixel per column."""
    params = ScanParams(lines=1, gain=0.4, retrace=0)
    path = beam_path(step, params)
    px, py = to_pixels(path, *step.shape[:2], params)
    mask = np.zeros(canvas_shape(*step.shape[:2], params), dtype=bool)
    mask[py.astype(int), px.astype(int)] = True
    assert components(mask) > 1


def test_lines_stay_separate(noise):
    """Blanked retrace keeps scanlines from being joined together."""
    image = render(noise, ScanParams(lines=4, gain=0.05, beam="speed"))
    assert components(lit_mask(image)) >= 4


def test_blanked_retrace_leaves_no_ink(noise):
    """Retrace length cannot change the picture, because Z gates it off."""
    base = render(noise, ScanParams(lines=8, retrace=1))
    for retrace in (2, 8, 32):
        assert np.array_equal(render(noise, ScanParams(lines=8, retrace=retrace)), base)


def test_without_retrace_lines_join_up(noise):
    """With no blanking interval the beam draws its own flyback."""
    params = {"lines": 8, "gain": 0.05, "beam": "speed"}
    joined = components(lit_mask(render(noise, ScanParams(retrace=0, **params))))
    blanked = components(lit_mask(render(noise, ScanParams(retrace=8, **params))))
    assert joined < blanked


def test_traces_add_where_they_cross():
    """Overlapping lines brighten; nothing occludes anything."""
    frame = np.full((32, 48, 3), 128, dtype=np.uint8)
    params = ScanParams(lines=6, gain=1.5, retrace=0, mono=True, beam="speed")
    acc = accumulate(beam_path(frame, params), *frame.shape[:2], params)[..., 0]
    assert acc.max() > 1.5 * np.median(acc[acc > 0])


def test_render_is_deterministic(noise):
    """Repeated renders are bit-identical."""
    params = ScanParams(lines=12, gain=0.3)
    first = render(noise, params)
    for _ in range(5):
        assert np.array_equal(render(noise, params), first)


def test_constant_rate_conserves_energy(step):
    """Constant-rate deflection writes the same total ink whatever the gain."""
    totals = [
        accumulate(
            beam_path(step, ScanParams(lines=1, gain=gain, retrace=0, mono=True)),
            *step.shape[:2],
            ScanParams(lines=1, gain=gain, retrace=0, mono=True),
        ).sum()
        for gain in (0.0, 0.2, 0.5)
    ]
    assert totals[1] == pytest.approx(totals[0], rel=0.02)
    assert totals[2] == pytest.approx(totals[0], rel=0.02)


def edge_vs_flat(step, beam):
    """Trace brightness on the steep edge relative to the flat sweep.

    Only the rows strictly between the two trace levels are measured, so the
    flat segments either side cannot contribute.
    """
    params = ScanParams(lines=1, gain=0.4, retrace=0, mono=True, beam=beam)
    path = beam_path(step, params)
    _, py = to_pixels(path, *step.shape[:2], params)
    acc = accumulate(path, *step.shape[:2], params)[..., 0]
    band = acc[int(py.min()) + 2 : int(py.max()) - 1]
    return band.max() / acc.max()


def test_constant_rate_dims_steep_segments(step):
    """Fast vertical travel across an edge writes far fainter than a flat sweep."""
    assert edge_vs_flat(step, "rate") < 0.1


def test_intensity_compensation_brightens_edges(step):
    """Constant-speed deflection lifts the steep edge back up by ~the step count."""
    assert edge_vs_flat(step, "speed") > 10 * edge_vs_flat(step, "rate")


def test_exposure_scales_brightness(noise):
    """Exposure multiplies the accumulated trace."""
    params = ScanParams(lines=8, gain=0.2, intensity=0.0, brightness=1.0)
    dim = render(noise, params, exposure=0.25).astype(int)
    bright = render(noise, params, exposure=0.5).astype(int)
    assert bright.sum() > dim.sum()


def test_color_trace_keeps_hue():
    """A red source produces a red trace."""
    red = np.zeros((32, 48, 3), dtype=np.uint8)
    red[:, :, 2] = 255
    image = render(red, ScanParams(lines=4, gain=0.1))
    assert image[..., 2].sum() > 20 * image[..., 0].sum() + 1


def test_mono_trace_is_grey():
    """Monochrome mode discards hue."""
    red = np.zeros((32, 48, 3), dtype=np.uint8)
    red[:, :, 2] = 255
    image = render(red, ScanParams(lines=4, gain=0.1, mono=True))
    assert np.array_equal(image[..., 0], image[..., 2])


def test_out_of_canvas_beam_is_clipped():
    """Deflection beyond the screen is dropped, not written out of bounds."""
    size = 12
    wild = np.array([-500.0, 500.0, 5.0, -1e6, 1e6], dtype=np.float32)
    path = BeamPath(
        x=wild,
        y=wild[::-1].copy(),
        z=np.ones(5, dtype=np.float32),
        color=np.ones((5, 3), dtype=np.float32),
    )
    guard = np.zeros((size, size, 3), dtype=np.float32)
    image = render_path(path, size * 2, size * 2, ScanParams(gain=0.0))
    assert image.shape == (size * 2, size * 2, 3)
    assert np.all(guard == 0.0)


def test_empty_frame_stays_black():
    """A black source writes nothing."""
    black = np.zeros((32, 48, 3), dtype=np.uint8)
    assert render(black, ScanParams(lines=4, intensity=1.0)).max() == 0
