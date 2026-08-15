"""Deflection model tests."""

import numpy as np
import pytest

from ruttetra.core import (
    LUMA_COEFFS,
    ScanParams,
    beam_clock,
    beam_path,
    deflection,
    luma,
    scan_lines,
)


def solid(color, shape=(48, 64)):
    """Frame filled with one BGR colour."""
    return np.full(shape + (3,), color, dtype=np.uint8)


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("bt709", (0.0722, 0.7152, 0.2126)),
        ("bt601", (0.1140, 0.5870, 0.2990)),
        ("mean", (1 / 3, 1 / 3, 1 / 3)),
    ],
)
def test_luma_primaries(mode, expected):
    """Each primary weighs in at its standard coefficient."""
    for channel, weight in enumerate(expected):
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        frame[..., channel] = 255
        assert luma(frame, mode)[0, 0] == pytest.approx(weight, abs=1e-4)


def test_luma_extremes():
    """Black and white map to the ends of the range."""
    assert luma(solid((0, 0, 0)))[0, 0] == pytest.approx(0.0)
    assert luma(solid((255, 255, 255)))[0, 0] == pytest.approx(1.0, abs=1e-4)


def test_luma_is_not_channel_mean():
    """Rec.709 luma of pure red differs from the channel average."""
    red = solid((0, 0, 255))
    assert luma(red, "bt709")[0, 0] < luma(red, "mean")[0, 0]


def test_scan_lines_area_averages():
    """Alternating rows average to grey rather than aliasing to one of them."""
    frame = np.zeros((48, 8, 3), dtype=np.uint8)
    frame[::2] = 255
    assert luma(scan_lines(frame, 1)).mean() == pytest.approx(0.5, abs=0.02)


def test_scan_lines_decimation_would_alias():
    """The discarded-rows approach this replaces returns a constant instead."""
    frame = np.zeros((48, 8, 3), dtype=np.uint8)
    frame[::2] = 255
    assert luma(frame[::48]).mean() == pytest.approx(1.0)


@pytest.mark.parametrize("lines", [1, 2, 30, 45, 47, 48, 100, 500])
def test_line_count_is_exact(noise, lines):
    """Every requested line count is honoured, including more lines than rows."""
    params = ScanParams(lines=lines, retrace=0)
    assert deflection(noise, params)[0].shape == (lines, noise.shape[1])
    assert len(beam_path(noise, params)) == lines * noise.shape[1]


def test_samples_override(noise):
    """Samples per line can be set independently of source width."""
    assert deflection(noise, ScanParams(lines=4, samples=17))[0].shape == (4, 17)


def test_bright_displaces_up(ramp):
    """Luminance sums into vertical deflection, brighter meaning higher."""
    _, y, _, _ = deflection(ramp, ScanParams(lines=4, gain=0.25))
    assert np.all(np.diff(y[0]) >= 0)
    assert y[0, -1] > y[0, 0]


def test_gain_scales_displacement(ramp):
    """Displacement is a linear function of gain, in units of frame height."""
    base = deflection(ramp, ScanParams(lines=4, gain=0.0))[1]
    for gain in (0.1, 0.5):
        y = deflection(ramp, ScanParams(lines=4, gain=gain))[1]
        assert (y - base).max() == pytest.approx(2.0 * gain, abs=0.02)


def test_zero_gain_is_a_flat_raster(noise):
    """With no gain the picture leaves the deflection untouched."""
    y = deflection(noise, ScanParams(lines=8, gain=0.0))[1]
    assert np.allclose(y, y[:, :1])


def test_x_is_a_monotonic_ramp(noise):
    """Horizontal deflection is a plain sweep, never picture dependent."""
    x, _, _, _ = deflection(noise, ScanParams(lines=8))
    assert np.all(np.diff(x, axis=1) > 0)
    assert np.allclose(x, x[:1])


def test_skew_leans_the_raster(noise):
    """Skew offsets each line horizontally by its vertical position."""
    plain = deflection(noise, ScanParams(lines=8))[0]
    leaned = deflection(noise, ScanParams(lines=8, skew=0.5))[0]
    base_y = deflection(noise, ScanParams(lines=8, gain=0.0))[1][:, 0]
    assert np.allclose(leaned - plain, 0.5 * base_y[:, None], atol=1e-5)


def test_invert(step):
    """Inversion mirrors the displacement about the mid grey axis."""
    normal = deflection(step, ScanParams(lines=4, gain=0.3))[1]
    flipped = deflection(step, ScanParams(lines=4, gain=0.3, invert=True))[1]
    assert np.allclose(normal + flipped, normal[:, :1] + flipped[:, :1], atol=1e-5)


def test_zero_intensity_is_an_unmodulated_beam(noise):
    """No Z gain plus a pedestal means constant beam current."""
    z = deflection(noise, ScanParams(lines=8, intensity=0.0, brightness=1.0))[2]
    assert np.allclose(z, 1.0)


def test_intensity_tracks_luma(ramp):
    """Z beam current follows luminance through the intensity gain."""
    z = deflection(ramp, ScanParams(lines=4, intensity=1.0))[2]
    assert np.all(np.diff(z[0]) >= 0)
    assert z[0, 0] == pytest.approx(0.0, abs=0.02)


def test_brightness_lifts_the_pedestal(ramp):
    """Z brightness adds a floor under the intensity gain."""
    params = ScanParams(lines=4, intensity=0.5, brightness=0.25)
    signal = luma(scan_lines(ramp, 4))
    assert np.allclose(deflection(ramp, params)[2], 0.25 + 0.5 * signal, atol=1e-6)


def test_displacement_is_unipolar_from_black():
    """Black does not move; only the lit part of the picture deflects."""
    dark = np.zeros((16, 16, 3), dtype=np.uint8)
    flat = deflection(dark, ScanParams(lines=4, gain=0.5))[1]
    base = deflection(dark, ScanParams(lines=4, gain=0.0))[1]
    assert np.allclose(flat, base)


def test_displacement_survives_resizing(noise):
    """Displacement is summed after the size multipliers, as in the RE4 chain."""
    small = ScanParams(lines=8, gain=0.3, v_size=0.5)
    full = ScanParams(lines=8, gain=0.3, v_size=1.0)
    flat_s = ScanParams(lines=8, gain=0.0, v_size=0.5)
    flat_f = ScanParams(lines=8, gain=0.0, v_size=1.0)
    shift_s = deflection(noise, small)[1] - deflection(noise, flat_s)[1]
    shift_f = deflection(noise, full)[1] - deflection(noise, flat_f)[1]
    assert np.allclose(shift_s, shift_f, atol=1e-6)


def test_depth_scales_both_axes(noise):
    """Depth is a uniform size control, not a perspective projection."""
    plain = deflection(noise, ScanParams(lines=8, gain=0.0))
    scaled = deflection(noise, ScanParams(lines=8, gain=0.0, depth=0.5))
    assert np.allclose(scaled[0], plain[0] * 0.5, atol=1e-6)
    assert np.allclose(scaled[1], plain[1] * 0.5, atol=1e-6)


def test_smooth_limits_deflection_bandwidth(step):
    """Bandwidth limiting rounds the deflection step off."""
    sharp = deflection(step, ScanParams(lines=1, gain=0.4))[1][0]
    soft = deflection(step, ScanParams(lines=1, gain=0.4, smooth=3.0))[1][0]
    assert np.abs(np.diff(soft)).max() < 0.5 * np.abs(np.diff(sharp)).max()
    assert soft.mean() == pytest.approx(sharp.mean(), abs=0.02)


def test_smooth_does_not_blur_across_lines():
    """The amplifier rolls off along a line; it cannot smear one into the next."""
    frame = np.zeros((3, 33, 3), dtype=np.uint8)
    frame[1, 16] = 255
    flat = deflection(frame, ScanParams(lines=3, gain=0.0))[1]
    soft = deflection(frame, ScanParams(lines=3, gain=0.4, smooth=2.0))[1]
    assert np.allclose(soft[0], flat[0])
    assert np.allclose(soft[2], flat[2])
    assert np.count_nonzero(soft[1] - flat[1]) > 1


def test_nearest_sampling_aliases(noise):
    """Nearest sampling decimates lines the way alternate-line switching did."""
    rows = scan_lines(noise, 4, sampling="nearest")
    assert np.any(np.all(rows == noise[::12][:4], axis=(1, 2)))


def test_mono_uses_luma_for_all_channels(step):
    """Monochrome renders the trace grey."""
    color = deflection(step, ScanParams(lines=4, mono=True))[3]
    assert np.allclose(color[..., 0], color[..., 2])


def test_color_keeps_chroma():
    """Colour mode keeps the source hue, at the brightness the Z law gives it."""
    color = deflection(solid((0, 0, 255)), ScanParams(lines=4))[3]
    assert color[..., 2].mean() == pytest.approx(LUMA_COEFFS["bt709"][2], abs=1e-3)
    assert color[..., 0].max() == 0.0


def test_serpentine_reverses_alternate_lines(ramp):
    """Odd lines are swept right to left so the beam never flies back."""
    params = ScanParams(lines=4, retrace=0, serpentine=True)
    x = beam_path(ramp, params).x.reshape(4, -1)
    assert np.all(np.diff(x[0]) > 0)
    assert np.all(np.diff(x[1]) < 0)
    assert np.allclose(x[0], x[1][::-1])


def test_non_serpentine_sweeps_one_way(ramp):
    """Without serpentine every line runs left to right."""
    x = beam_path(ramp, ScanParams(lines=4, retrace=0, serpentine=False)).x
    assert np.all(np.diff(x.reshape(4, -1), axis=1) > 0)


def test_retrace_is_blanked(noise):
    """Interline samples carry no beam current and no colour."""
    lines, width, retrace = 6, noise.shape[1], 5
    path = beam_path(noise, ScanParams(lines=lines, retrace=retrace))
    z = path.z.reshape(lines, width + retrace)
    color = path.color.reshape(lines, width + retrace, 3)
    assert np.all(z[:, width:] == 0.0)
    assert np.all(color[:, width:] == 0.0)
    assert np.all(z[:, :width] > 0.0)


def test_serpentine_retrace_is_a_vertical_hop(noise):
    """Alternating sweeps leave only a vertical step between lines."""
    lines, width, retrace = 6, noise.shape[1], 5
    path = beam_path(noise, ScanParams(lines=lines, retrace=retrace))
    x = path.x.reshape(lines, width + retrace)
    assert np.allclose(x[:, width:], x[:, width - 1 : width], atol=1e-6)


def test_retrace_closes_the_frame(noise):
    """The last line flies back to where the next frame starts."""
    lines, width = 6, noise.shape[1]
    path = beam_path(noise, ScanParams(lines=lines, retrace=4))
    y = path.y.reshape(lines, width + 4)
    assert y[-1, -1] == pytest.approx(
        y[-1, width - 1] + 0.8 * (y[0, 0] - y[-1, width - 1]), abs=1e-5
    )


def test_beam_clock_rate_is_uniform(noise):
    """Constant-rate deflection ticks once per sample."""
    path = beam_path(noise, ScanParams(lines=4))
    assert np.array_equal(beam_clock(path, "rate"), np.arange(len(path)))


def test_beam_clock_speed_is_arc_length(noise):
    """Constant-speed deflection ticks by distance travelled."""
    path = beam_path(noise, ScanParams(lines=4))
    clock = beam_clock(path, "speed")
    assert np.all(np.diff(clock) >= 0)
    assert clock[-1] == pytest.approx(
        np.hypot(np.diff(path.x), np.diff(path.y)).sum(), rel=1e-5
    )


def test_beam_path_is_deterministic(noise):
    """Repeated runs produce identical geometry."""
    params = ScanParams(lines=16, gain=0.3)
    first = beam_path(noise, params)
    for _ in range(5):
        assert np.array_equal(beam_path(noise, params).y, first.y)


def test_envelope_bounds_contain_the_path(noise):
    """Declared bounds really do enclose the deflection."""
    params = ScanParams(lines=8, gain=0.4, skew=0.3, v_size=0.8, h_size=1.2)
    path = beam_path(noise, params)
    y_min, y_max = params.y_bounds
    assert y_min - 1e-5 <= path.y.min() and path.y.max() <= y_max + 1e-5
    assert np.abs(path.x).max() <= params.x_extent + 1e-5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lines": 0},
        {"gain": -0.1},
        {"samples": -1},
        {"retrace": -1},
        {"luma": "yuv"},
        {"beam": "warp"},
        {"v_size": 0.0},
        {"h_size": 0.0},
        {"depth": 0.0},
        {"sampling": "cubic"},
        {"intensity": -1.0},
        {"brightness": -1.0},
        {"smooth": -1.0},
    ],
)
def test_invalid_params_rejected(kwargs):
    """Bad settings fail loudly instead of corrupting the render."""
    with pytest.raises(ValueError):
        ScanParams(**kwargs)
