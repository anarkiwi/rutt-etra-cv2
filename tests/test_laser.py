"""Laser projector model tests."""

import numpy as np
import pytest

from ruttetra import colour, ilda, laser
from ruttetra.core import ScanParams, beam_path


def ilda_circle(revolutions=400, points_per_revolution=12):
    """The ILDA speed test figure: a circle of 12 points per revolution."""
    count = revolutions * points_per_revolution
    angle = np.linspace(0.0, 2.0 * np.pi * revolutions, count, endpoint=False)
    return np.stack([np.cos(angle), np.sin(angle)], axis=1)


def settled_radius(projector, rate=None):
    """Radius the scanner actually draws the ILDA circle at."""
    drawn = laser.galvo(ilda_circle(), projector, rate or projector.kpps)
    tail = drawn[drawn.shape[0] // 2 :]
    return float(np.hypot(tail[:, 0], tail[:, 1]).mean())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kpps": 0},
        {"scan_angle": 0.0},
        {"damping": 0.0},
        {"damping": 1.0},
        {"dwell": -1},
        {"blank_delay": -1},
        {"blank_points": -1},
        {"power": (1.0, 2.0)},
        {"wavelengths": (1.0,)},
    ],
)
def test_invalid_projector_rejected(kwargs):
    """Impossible projector settings fail loudly."""
    with pytest.raises(ValueError):
        laser.Projector(**kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"max_points": 1}, {"max_rate": 0}, {"bits": 4}, {"bits": 20}]
)
def test_invalid_dac_rejected(kwargs):
    """Impossible DAC settings fail loudly."""
    with pytest.raises(ValueError):
        laser.Dac(**kwargs)


def test_bandwidth_from_the_ilda_rating():
    """A kpps rating is a bandwidth measurement at one twelfth of it."""
    assert laser.Projector(kpps=30000).bandwidth == 2500
    assert laser.Projector(kpps=12000).bandwidth == 1000
    assert laser.Projector(kpps=60000).bandwidth == 5000
    assert laser.Projector(kpps=30000, galvo_hz=1800).bandwidth == 1800


def test_galvo_filter_is_unity_at_dc():
    """A stationary command must not be attenuated or offset."""
    for bandwidth, damping, rate in ((2500, 0.7, 30000), (1000, 0.6, 12000)):
        b1, b2, a1, a2 = laser.galvo_coefficients(bandwidth, damping, rate)
        assert (b1 + b2) / (1 + a1 + a2) == pytest.approx(1.0, abs=1e-9)


def test_galvo_follows_a_slow_command():
    """Well below the bandwidth the mirrors track the command."""
    projector = laser.Projector(kpps=30000)
    slow = ilda_circle(revolutions=20, points_per_revolution=600)
    drawn = laser.galvo(slow, projector, 30000)
    tail = drawn[drawn.shape[0] // 2 :]
    assert np.hypot(tail[:, 0], tail[:, 1]).mean() == pytest.approx(1.0, abs=0.02)


def test_ilda_circle_shrinks_at_the_rated_speed():
    """At its rating the scanner is at the corner of its response."""
    assert settled_radius(laser.Projector(kpps=30000)) == pytest.approx(0.70, abs=0.06)


def test_rating_is_scale_invariant():
    """Every rating is measured at the same point on its own response."""
    radii = [settled_radius(laser.Projector(kpps=k)) for k in (12000, 30000, 60000)]
    assert max(radii) - min(radii) < 0.01


def test_slow_scanner_collapses_the_figure():
    """Driving a scanner past its bandwidth destroys the picture."""
    slow = laser.Projector(kpps=6000)
    assert settled_radius(slow, rate=30000) < 0.25


def test_wide_scanning_costs_bandwidth():
    """The rating is quoted at 8 degrees; a wider sweep scans worse."""
    narrow = settled_radius(laser.Projector(kpps=30000, scan_angle=8.0))
    wide = settled_radius(laser.Projector(kpps=30000, scan_angle=30.0))
    assert wide < 0.3 * narrow


def test_ilda_pattern_circle_against_square():
    """The pattern's circle is commanded larger and shrinks to fit the square.

    ILDA draws the speed circle 1.5545 times the reference square, so a scanner
    at its rating brings the two together. That is the whole test.
    """
    commanded = 1.5545
    drawn = commanded * settled_radius(laser.Projector(kpps=30000))
    assert 0.95 < drawn < 1.2
    too_slow = commanded * settled_radius(laser.Projector(kpps=10000), rate=30000)
    assert too_slow < 0.6


def test_point_budget_binds_on_the_scanner():
    """At ordinary frame rates the scanner runs out before the DAC does."""
    assert laser.point_budget(laser.Projector(kpps=30000), laser.Dac(), 30) == 1000
    assert laser.point_budget(laser.Projector(kpps=20000), laser.Dac(), 25) == 800


def test_point_budget_binds_on_the_dac():
    """A fast scanner at a low frame rate runs into the DAC's frame limit."""
    fast = laser.Projector(kpps=60000)
    assert laser.point_budget(fast, laser.Dac(max_points=4095), 12) == 4095
    assert laser.point_budget(fast, laser.Dac(max_points=16000), 12) == 5000


def test_faster_scanner_buys_resolution_on_the_same_dac():
    """Upgrading the projector helps until the DAC becomes the limit."""
    dac = laser.Dac()
    budgets = [
        laser.point_budget(laser.Projector(kpps=k), dac, 30)
        for k in (20000, 30000, 40000, 60000)
    ]
    assert budgets == sorted(budgets)
    assert budgets[0] < budgets[-1]


def test_fit_scan_stays_within_budget():
    """The fitted raster never asks for more points than exist."""
    dac = laser.Dac()
    for kpps in (12000, 30000, 60000):
        for fps in (12, 25, 30):
            projector = laser.Projector(kpps=kpps)
            lines, samples = laser.fit_scan(projector, dac, fps, retrace=4)
            overhead = 4 + 2 * projector.dwell
            assert lines >= 1 and samples >= 2
            assert lines * (samples + overhead) <= laser.point_budget(
                projector, dac, fps
            )


def test_dac_quantisation():
    """Deflection lands on the DAC's grid, and 12 bits is coarser than 16."""
    values = np.linspace(-1, 1, 5000)
    coarse = np.unique(laser.Dac(bits=12).quantise(values))
    fine = np.unique(laser.Dac(bits=16).quantise(values))
    assert coarse.size < fine.size
    assert coarse.size <= 4096
    assert np.abs(coarse).max() <= 1.0


def test_shorten_blanked():
    """Blanked runs are cut short; lit points are never dropped."""
    blank = np.array([False] * 3 + [True] * 10 + [False] * 2)
    keep = laser.shorten_blanked(blank, 4)
    assert keep[:3].all() and keep[-2:].all()
    assert keep[3:13].sum() == 4


def test_blanked_runs_cost_less(noise):
    """Shortening blank travel frees budget on a sparse picture."""
    sparse = np.zeros_like(noise)
    sparse[:, 20:24] = 255
    path = beam_path(sparse, ScanParams(lines=8, retrace=8))
    dac = laser.Dac()
    full = laser.laser_points(path, laser.Projector(blank_points=0), dac)[0]
    short = laser.laser_points(path, laser.Projector(blank_points=4), dac)[0]
    assert short.shape[0] < full.shape[0]


def test_dwell_points_are_added_at_lit_edges(noise):
    """Corners and blank jumps get held points so the mirrors can settle."""
    path = beam_path(noise, ScanParams(lines=4, retrace=4))
    dac = laser.Dac()
    plain = laser.laser_points(path, laser.Projector(dwell=0, blank_delay=0), dac)[0]
    held = laser.laser_points(path, laser.Projector(dwell=5, blank_delay=5), dac)[0]
    assert held.shape[0] > plain.shape[0]


def test_frame_points_respect_the_budget(noise):
    """However big the raster, the emitted frame fits the projector."""
    path = beam_path(noise, ScanParams(lines=40, retrace=8))
    projector, dac = laser.Projector(kpps=30000), laser.Dac()
    points, blank, rgb = laser.frame_points(path, projector, dac, 30.0)
    budget = laser.point_budget(projector, dac, 30.0)
    assert points.shape[0] <= budget
    assert blank.shape[0] == points.shape[0] == rgb.shape[0]


def test_blanked_points_carry_no_colour(noise):
    """A blanked point must be dark, whatever the picture said."""
    path = beam_path(noise, ScanParams(lines=6, retrace=6))
    _, blank, rgb = laser.frame_points(path, laser.Projector(), laser.Dac(), 25.0)
    assert rgb[blank].max() == 0.0


def test_white_balance_pins_the_scarcest_channel():
    """Red is the limiter on a typical blue heavy projector."""
    drive = laser.Projector(power=(500, 500, 1000)).drive
    assert drive[0] == pytest.approx(1.0)
    assert drive[1] < 1.0 and drive[2] < drive[1]


def test_uncalibrated_projector_is_not_white():
    """Full drive on a stock unit renders blue violet, not white."""
    projector = laser.Projector(power=(500, 500, 1000), calibrate=False)
    point = colour.chromaticity(
        np.array(projector.power) / 1000.0, projector.wavelengths
    )
    assert point[0] < 0.25 and point[1] < 0.22


def test_calibrated_projector_reaches_white():
    """White balancing lands the mix on the target white point."""
    projector = laser.Projector(power=(500, 500, 1000))
    point = colour.chromaticity(projector.white_power, projector.wavelengths)
    assert point == pytest.approx(colour.D65, abs=0.01)


def test_to_ilda_scales_to_full_deflection():
    """Normalised deflection maps onto the 16 bit ILDA range."""
    points = np.array([[-1.0, -1.0], [1.0, 1.0]])
    frame = laser.to_ilda(points, np.zeros(2, bool), np.ones((2, 3)))
    assert frame.x[0] == pytest.approx(-ilda.COORD_MAX)
    assert frame.x[1] == pytest.approx(ilda.COORD_MAX)


def test_preview_draws_something(noise):
    """The simulated projection lights up where the beam goes."""
    path = beam_path(noise, ScanParams(lines=6, retrace=4))
    projector, dac = laser.Projector(), laser.Dac()
    shot = laser.frame_points(path, projector, dac, 25.0)
    params = laser.PreviewParams(size=64, spot=0.0, bloom=0.0)
    image = laser.preview(*shot, projector, params)
    assert image.shape == (64, 64, 3)
    assert image.max() > 0


def test_preview_shows_galvo_distortion(noise):
    """Modelling the scanner changes the picture; an ideal one does not."""
    path = beam_path(noise, ScanParams(lines=6, retrace=4))
    projector, dac = laser.Projector(kpps=8000), laser.Dac()
    shot = laser.frame_points(path, projector, dac, 25.0)
    ideal = laser.preview(
        *shot, projector, laser.PreviewParams(size=64, show_galvo=False)
    )
    real = laser.preview(
        *shot, projector, laser.PreviewParams(size=64, show_galvo=True)
    )
    assert not np.array_equal(ideal, real)


def test_describe_names_the_binding_limit():
    """The summary says which of the two limits is actually binding."""
    assert "scanner rate" in laser.describe(laser.Projector(), laser.Dac(), 30.0, 900)
    fast = laser.Projector(kpps=60000)
    assert "DAC frame limit" in laser.describe(fast, laser.Dac(), 12.0, 4095)
