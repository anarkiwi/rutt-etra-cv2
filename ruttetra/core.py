"""Core Rutt/Etra scan-processor model.

The device is an XY vector display: a horizontal ramp drives X, video luminance
is summed into Y deflection, and Z (beam current) carries intensity. This module
emits that geometry as one beam path, consumed by every renderer.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Luminance weights in OpenCV BGR channel order.
LUMA_COEFFS = {
    "bt709": (0.0722, 0.7152, 0.2126),
    "bt601": (0.1140, 0.5870, 0.2990),
    "mean": (1 / 3, 1 / 3, 1 / 3),
}

BEAM_MODES = ("rate", "speed")
SAMPLING = {"area": cv2.INTER_AREA, "nearest": cv2.INTER_NEAREST}


@dataclass(frozen=True)
class ScanParams:
    """Deflection settings, shared by every output mode.

    Named after the RE4 display control unit: v_size/h_size are HEIGHT/WIDTH,
    depth scales both, and intensity/brightness are the Z-axis gain and pedestal.
    """

    lines: int = 60
    gain: float = 0.1
    samples: int = 0
    luma: str = "bt709"
    sampling: str = "area"
    mono: bool = False
    invert: bool = False
    intensity: float = 1.0
    brightness: float = 0.0
    v_size: float = 1.0
    h_size: float = 1.0
    depth: float = 1.0
    skew: float = 0.0
    smooth: float = 0.0
    serpentine: bool = True
    retrace: int = 8
    beam: str = "rate"

    def __post_init__(self):
        for name in ("gain", "samples", "retrace", "intensity", "brightness", "smooth"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.lines < 1:
            raise ValueError("lines must be >= 1")
        if self.luma not in LUMA_COEFFS:
            raise ValueError(f"luma must be one of {sorted(LUMA_COEFFS)}")
        if self.sampling not in SAMPLING:
            raise ValueError(f"sampling must be one of {sorted(SAMPLING)}")
        if self.beam not in BEAM_MODES:
            raise ValueError(f"beam must be one of {BEAM_MODES}")
        if self.v_size <= 0 or self.h_size <= 0 or self.depth <= 0:
            raise ValueError("v_size, h_size and depth must be > 0")

    @property
    def y_scale(self):
        """Vertical raster size after the HEIGHT and DEPTH multipliers."""
        return self.v_size * self.depth

    @property
    def x_extent(self):
        """Half-width of the deflection envelope in normalised units."""
        return self.h_size * self.depth + abs(self.skew) * self.y_scale

    @property
    def y_bounds(self):
        """(min, max) of the deflection envelope in normalised units."""
        return -self.y_scale, self.y_scale + 2.0 * self.gain


@dataclass(frozen=True)
class BeamPath:
    """One frame of beam motion in beam order.

    x, y are normalised deflection with y positive up, z is beam current in
    [0, 1] where 0 is blanked, and color is per-sample BGR in [0, 1].
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    color: np.ndarray

    def __len__(self):
        return int(self.x.size)


def luma(frame, mode="bt709"):
    """Luminance of a BGR uint8 frame as float32 in [0, 1]."""
    coeffs = np.asarray(LUMA_COEFFS[mode], dtype=np.float32)
    return (frame.astype(np.float32) @ coeffs) / 255.0


def scan_lines(frame, lines, samples=0, sampling="area"):
    """Resample a frame to the scan raster.

    Area averaging is the default because the beam integrates the video signal
    over each line period; "nearest" reproduces the alternate-line decimation
    that is the only line-dropping feature the original hardware had.
    """
    width = frame.shape[1]
    return cv2.resize(
        frame, (samples or width, lines), interpolation=SAMPLING[sampling]
    ).reshape(lines, samples or width, -1)


def _bandwidth_limit(signal, smooth):
    """Roll off the deflection signal along each line.

    The kernel height is pinned to 1: lines are swept independently, so the
    amplifier cannot smear one into the next. OpenCV reads sigmaY=0 as "same
    as sigmaX", which would blur across lines as well as along them.
    """
    if smooth <= 0:
        return signal
    return cv2.GaussianBlur(signal, (0, 1), sigmaX=smooth).reshape(signal.shape)


def deflection(frame, params):
    """Per-line deflection grids (x, y, z, color), each (lines, samples)."""
    small = scan_lines(frame, params.lines, params.samples, params.sampling)
    signal = luma(small, params.luma)
    color = small.astype(np.float32) / 255.0
    if params.invert:
        signal, color = 1.0 - signal, 1.0 - color
    if params.mono:
        color = np.repeat(signal[..., None], 3, axis=-1)

    rows = (np.arange(params.lines, dtype=np.float32) + 0.5) / params.lines
    base_y = (1.0 - 2.0 * rows) * params.y_scale
    cols = (np.arange(small.shape[1], dtype=np.float32) + 0.5) / small.shape[1]
    base_x = (2.0 * cols - 1.0) * params.h_size * params.depth

    # Displacement is summed after the size multipliers, as in the RE4 chain.
    y = base_y[:, None] + 2.0 * params.gain * _bandwidth_limit(signal, params.smooth)
    x = base_x[None, :] + params.skew * base_y[:, None]
    z = np.clip(params.brightness + params.intensity * signal, 0.0, 1.0)
    peak = np.maximum(color.max(axis=-1, keepdims=True), 1e-6)
    return (
        np.broadcast_to(x, signal.shape).astype(np.float32),
        y.astype(np.float32),
        z.astype(np.float32),
        (color / peak * z[..., None]).astype(np.float32),
    )


def _retrace_block(values, count, blank=False):
    """Interpolate from the end of each line to the start of the next."""
    if count == 0:
        return values[:, :0]
    starts = values[:, -1]
    ends = np.roll(values, -1, axis=0)[:, 0]
    if blank:
        starts, ends = np.zeros_like(starts), np.zeros_like(ends)
    ramp = (np.arange(count, dtype=np.float32) + 1.0) / (count + 1)
    ramp = ramp.reshape((count,) + (1,) * (values.ndim - 2))
    return starts[:, None] + (ends - starts)[:, None] * ramp


def beam_path(frame, params):
    """Flatten a frame's deflection into one ordered, blanked beam path."""
    x, y, z, color = deflection(frame, params)

    if params.serpentine:
        # Alternating line direction removes the flyback streak on a scope.
        x, y, z, color = (a.copy() for a in (x, y, z, color))
        for arr in (x, y, z, color):
            arr[1::2] = arr[1::2, ::-1]

    parts = [
        np.concatenate([a, _retrace_block(a, params.retrace, blank=b)], axis=1)
        for a, b in ((x, False), (y, False), (z, True), (color, True))
    ]
    return BeamPath(
        x=np.ascontiguousarray(parts[0].reshape(-1), dtype=np.float32),
        y=np.ascontiguousarray(parts[1].reshape(-1), dtype=np.float32),
        z=np.ascontiguousarray(parts[2].reshape(-1), dtype=np.float32),
        color=np.ascontiguousarray(parts[3].reshape(-1, 3), dtype=np.float32),
    )


def beam_clock(path, mode):
    """Cumulative beam time along a path.

    "rate" ticks once per sample, matching a constant-rate horizontal ramp, so
    steep trace segments are swept faster and written more faintly. "speed"
    ticks by arc length, giving the even brightness oscilloscope art wants.
    """
    if mode == "rate":
        return np.arange(path.x.size, dtype=np.float64)
    step = np.hypot(np.diff(path.x), np.diff(path.y)).astype(np.float64)
    return np.concatenate([[0.0], np.cumsum(step)])
