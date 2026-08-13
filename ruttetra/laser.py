"""Laser show projector model: point budget, galvo dynamics, and preview.

A galvanometer pair is a mechanical low pass filter, so what a projector draws
is not the point list it was given. This models that, and the point budget the
scanner and the DAC between them impose.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from numba import njit

from . import colour, ilda
from .raster import draw_segments

# ILDA rates scanners on a 12 point circle, so N kpps means bandwidth at N/12 Hz.
ILDA_POINTS_PER_REVOLUTION = 12
ILDA_CIRCLE_ATTENUATION = 0.6433


@dataclass(frozen=True)
class Projector:
    """A laser projector's scanner and light source."""

    kpps: int = 30000
    scan_angle: float = 8.0
    galvo_hz: float = 0.0
    damping: float = 0.65
    power: tuple = (500.0, 500.0, 1000.0)
    wavelengths: tuple = (638.0, 520.0, 450.0)
    dwell: int = 4
    blank_delay: int = 4
    blank_points: int = 6
    calibrate: bool = True

    def __post_init__(self):
        if self.kpps < 1:
            raise ValueError("kpps must be >= 1")
        if self.scan_angle <= 0:
            raise ValueError("scan_angle must be > 0")
        if not 0.0 < self.damping < 1.0:
            raise ValueError("damping must be in (0, 1)")
        if self.dwell < 0 or self.blank_delay < 0 or self.blank_points < 0:
            raise ValueError("dwell, blank_delay and blank_points must be >= 0")
        if len(self.power) != 3 or len(self.wavelengths) != 3:
            raise ValueError("power and wavelengths are per RGB channel")

    @property
    def bandwidth(self):
        """Small signal bandwidth in Hz, from the ILDA rating if not given."""
        return self.galvo_hz or self.kpps / ILDA_POINTS_PER_REVOLUTION

    @property
    def drive(self):
        """Per channel drive in [0, 1] after white balancing."""
        if not self.calibrate:
            return np.ones(3)
        return colour.drive_limits(self.wavelengths, self.power)

    @property
    def white_power(self):
        """Optical milliwatts per channel at full white."""
        return self.drive * np.asarray(self.power, dtype=np.float64)


@dataclass(frozen=True)
class Dac:
    """The DAC between the software and the projector.

    Defaults describe the Helios: 4095 points per frame and 12 bit deflection.
    """

    max_points: int = 4095
    max_rate: int = 65535
    bits: int = 12

    def __post_init__(self):
        if self.max_points < 2:
            raise ValueError("max_points must be >= 2")
        if self.max_rate < 1:
            raise ValueError("max_rate must be >= 1")
        if not 8 <= self.bits <= 16:
            raise ValueError("bits must be between 8 and 16")

    def quantise(self, unit):
        """Snap normalised [-1, 1] deflection to the DAC's grid."""
        levels = (1 << self.bits) - 1
        return np.round((unit + 1.0) * 0.5 * levels) / levels * 2.0 - 1.0


def point_rate(projector, dac):
    """Points per second the pair can actually sustain."""
    return min(projector.kpps, dac.max_rate)


def point_budget(projector, dac, fps):
    """Points available for one frame.

    The scanner caps points per second and the DAC caps points per frame;
    whichever binds first is the real limit.
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")
    return max(2, min(dac.max_points, int(point_rate(projector, dac) / fps)))


def fit_scan(projector, dac, fps, aspect=1.33, retrace=4):
    """Largest (lines, samples) raster that fits the budget.

    aspect is samples per line divided by lines, so the raster keeps roughly
    the picture's shape.
    """
    budget = point_budget(projector, dac, fps)
    overhead = retrace + 2 * projector.dwell
    lines = (-overhead + np.sqrt(overhead**2 + 4.0 * aspect * budget)) / (2.0 * aspect)
    lines = max(1, int(lines))
    samples = max(2, int(budget / lines) - overhead)
    while lines > 1 and lines * (samples + overhead) > budget:
        lines -= 1
        samples = max(2, int(budget / lines) - overhead)
    return lines, samples


@njit(cache=True, fastmath=True)
def _second_order(signal, b1, b2, a1, a2):
    """Zero order hold discretised second order response."""
    out = np.empty_like(signal)
    y1 = y2 = signal[0]
    u1 = u2 = signal[0]
    for i in range(signal.size):
        value = b1 * u1 + b2 * u2 - a1 * y1 - a2 * y2
        out[i] = value
        y2, y1 = y1, value
        u2, u1 = u1, signal[i]
    return out


def natural_frequency(bandwidth, damping):
    """Convert a -3 dB bandwidth, as datasheets quote it, to a resonant one."""
    squared = damping**2
    return bandwidth / np.sqrt(
        1.0 - 2.0 * squared + np.sqrt(2.0 - 4.0 * squared + 4.0 * squared**2)
    )


def galvo_coefficients(bandwidth, damping, rate):
    """ZOH discretised second order coefficients (b1, b2, a1, a2)."""
    omega = 2.0 * np.pi * natural_frequency(bandwidth, damping)
    period = 1.0 / rate
    root = np.sqrt(1.0 - damping**2)
    decay = np.exp(-damping * omega * period)
    angle = omega * root * period
    cosine, sine = np.cos(angle), np.sin(angle)
    ratio = damping / root
    return (
        1.0 - decay * (cosine + ratio * sine),
        decay**2 - decay * (cosine - ratio * sine),
        -2.0 * decay * cosine,
        decay**2,
    )


def galvo(positions, projector, rate):
    """Where the mirrors actually point, given where they were told to point.

    The kpps rating is measured at 8 degrees optical, so asking for a wider
    sweep costs bandwidth in proportion.
    """
    bandwidth = projector.bandwidth * min(1.0, 8.0 / projector.scan_angle)
    coeffs = galvo_coefficients(bandwidth, projector.damping, rate)
    out = np.empty_like(positions)
    for axis in range(positions.shape[1]):
        out[:, axis] = _second_order(
            np.ascontiguousarray(positions[:, axis], dtype=np.float64), *coeffs
        )
    return out


def shorten_blanked(blank, keep):
    """Mask keeping only the first `keep` points of each blanked run.

    Travelling through darkness costs the same points as drawing, so on sparse
    pictures most of the budget is otherwise spent on nothing.
    """
    if keep <= 0 or not blank.any():
        return np.ones(blank.size, dtype=bool)
    change = np.flatnonzero(np.diff(blank.astype(np.int8)) != 0) + 1
    starts = np.concatenate([[0], change])
    lengths = np.diff(np.concatenate([starts, [blank.size]]))
    within = np.arange(blank.size) - np.repeat(starts, lengths)
    return ~blank | (within < keep)


def _repeat_counts(blank, dwell, blank_delay):
    """How many times to hold each point, to settle corners and blank jumps."""
    counts = np.ones(blank.size, dtype=np.int64)
    if dwell == 0 and blank_delay == 0:
        return counts
    lit = ~blank
    starts = np.flatnonzero(lit & ~np.roll(lit, 1))
    ends = np.flatnonzero(lit & ~np.roll(lit, -1))
    counts[starts] += dwell + blank_delay
    counts[ends] += dwell
    return counts


def laser_points(path, projector, dac):
    """Turn a beam path into projector points: positions, blanking, colour."""
    blank = path.z <= 0.0
    keep = shorten_blanked(blank, projector.blank_points)
    blank = blank[keep]
    counts = _repeat_counts(blank, projector.dwell, projector.blank_delay)
    index = np.repeat(np.flatnonzero(keep), counts)
    unit = np.stack([dac.quantise(path.x[index]), dac.quantise(path.y[index])], axis=1)
    rgb = path.color[index][:, ::-1] * projector.drive
    expanded = np.repeat(blank, counts)
    rgb[expanded] = 0.0
    return unit, expanded, rgb


def fit_points(points, blank, rgb, budget):
    """Decimate a point stream that overruns the budget."""
    if points.shape[0] <= budget:
        return points, blank, rgb
    keep = np.linspace(0, points.shape[0] - 1, budget).astype(np.int64)
    return points[keep], blank[keep], rgb[keep]


def frame_points(path, projector, dac, fps):
    """Budget-fitted projector points for one frame."""
    points, blank, rgb = laser_points(path, projector, dac)
    return fit_points(points, blank, rgb, point_budget(projector, dac, fps))


def to_ilda(points, blank, rgb, name="RUTTETRA"):
    """Pack a point stream into an ILDA frame at full 16 bit scale."""
    return ilda.Frame(
        x=points[:, 0] * ilda.COORD_MAX,
        y=points[:, 1] * ilda.COORD_MAX,
        blank=blank,
        color=np.clip(rgb, 0.0, 1.0) * 255.0,
        name=name,
    )


def display_colour(rgb, projector):
    """Convert per channel drive to what the beam would look like."""
    watts = np.asarray(rgb, dtype=np.float64) * np.asarray(projector.power)
    scale = max(projector.white_power.max(), 1e-9)
    return colour.to_display(watts / scale, projector.wavelengths)


@dataclass(frozen=True)
class PreviewParams:
    """Simulated projection screen."""

    size: int = 480
    aspect: float = 1.0
    gain: float = 1.0
    spot: float = 1.2
    bloom: float = 0.4
    show_galvo: bool = True

    @property
    def shape(self):
        """Canvas (height, width)."""
        return self.size, max(8, int(round(self.size * self.aspect)))


def preview(points, blank, rgb, projector, params, rate=None):
    """Render what the projector would put on a screen.

    Brightness follows dwell time, so corners and held points burn bright and
    fast strokes go dim, exactly as a real beam does.
    """
    height, width = params.shape
    drawn = (
        galvo(points, projector, rate or projector.kpps)
        if params.show_galvo
        else points
    )
    px = np.ascontiguousarray((drawn[:, 0] + 1.0) * 0.5 * (width - 1), np.float32)
    py = np.ascontiguousarray((1.0 - drawn[:, 1]) * 0.5 * (height - 1), np.float32)
    lit = np.ascontiguousarray((~blank).astype(np.float32))
    shown = display_colour(rgb, projector).astype(np.float32) * params.gain
    acc = np.zeros((height, width, 3), dtype=np.float32)
    draw_segments(acc, px, py, lit, np.ascontiguousarray(shown[:, ::-1]), False)
    if params.spot > 0:
        acc = cv2.GaussianBlur(acc, (0, 0), sigmaX=params.spot)
    if params.bloom > 0:
        acc = acc + params.bloom * cv2.GaussianBlur(
            acc, (0, 0), sigmaX=max(1.0, params.spot * 6.0)
        )
    return (np.clip(acc, 0.0, 1.0) * 255.0).astype(np.uint8)


def add_arguments(group):
    """Attach the projector front panel to an argument parser group."""
    flag = __import__("argparse").BooleanOptionalAction
    group.add_argument("--kpps", default=30000, type=int, help="ILDA scanner rating")
    group.add_argument(
        "--scan-angle", default=8.0, type=float, help="optical degrees, 8 = ILDA rating"
    )
    group.add_argument(
        "--galvo-hz", default=0.0, type=float, help="measured -3 dB bandwidth"
    )
    group.add_argument("--damping", default=0.65, type=float)
    group.add_argument(
        "--laser-power",
        default="500,500,1000",
        type=str,
        help="optical mW per R,G,B channel",
    )
    group.add_argument(
        "--wavelengths", default="638,520,450", type=str, help="nm per R,G,B channel"
    )
    group.add_argument("--dwell", default=4, type=int, help="points held at corners")
    group.add_argument("--blank-delay", default=4, type=int)
    group.add_argument(
        "--blank-points", default=6, type=int, help="points kept per blanked run"
    )
    group.add_argument("--calibrate", action=flag, default=True, help="white balance")
    group.add_argument("--dac-points", default=4095, type=int, help="DAC frame limit")
    group.add_argument("--dac-rate", default=65535, type=int)
    group.add_argument("--dac-bits", default=12, type=int)
    return group


def _triple(text):
    """Parse a comma separated triple of floats."""
    parts = tuple(float(v) for v in str(text).split(","))
    if len(parts) != 3:
        raise ValueError(f"expected three comma separated values, got {text!r}")
    return parts


def params_from(args):
    """Build (projector, dac) from parsed arguments."""
    projector = Projector(
        kpps=args.kpps,
        scan_angle=args.scan_angle,
        galvo_hz=args.galvo_hz,
        damping=args.damping,
        power=_triple(args.laser_power),
        wavelengths=_triple(args.wavelengths),
        dwell=args.dwell,
        blank_delay=args.blank_delay,
        blank_points=args.blank_points,
        calibrate=args.calibrate,
    )
    return projector, Dac(
        max_points=args.dac_points, max_rate=args.dac_rate, bits=args.dac_bits
    )


def describe(projector, dac, fps, points):
    """One line summary of how the budget was spent."""
    budget = point_budget(projector, dac, fps)
    binds = "DAC frame limit" if budget == dac.max_points else "scanner rate"
    return (
        f"laser: {points} points/frame of {budget} available at {fps:.2f} fps "
        f"(limited by the {binds}); scanner {projector.kpps} pps, "
        f"{projector.bandwidth:.0f} Hz at {projector.scan_angle:g} degrees"
    )


class IldSink:
    """Write projector points to an ILDA file."""

    def __init__(self, path, fmt=5, name="RUTTETRA"):
        self.writer = ilda.IldWriter(path, fmt=fmt)
        self.name = name

    def write(self, points, blank, rgb):
        """Append one frame."""
        self.writer.write(to_ilda(points, blank, rgb, self.name))

    def close(self):
        """Finalise the file."""
        self.writer.close()


class PreviewSink:
    """Render projector points as video of a simulated projection."""

    def __init__(self, path, projector, params, fps, monitor=False):
        height, width = params.shape
        self.projector = projector
        self.params = params
        self.monitor = monitor
        self.frames = 0
        self.writer = None
        if path:
            self.writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height)
            )

    def write(self, points, blank, rgb):
        """Render and append one frame."""
        image = preview(points, blank, rgb, self.projector, self.params)
        if self.monitor:
            cv2.imshow("laser", image)
            cv2.waitKey(1)
        if self.writer is not None:
            self.writer.write(image)
        self.frames += 1
        return image

    def close(self):
        """Finalise the file."""
        if self.writer is not None:
            self.writer.release()
