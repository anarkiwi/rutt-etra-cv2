"""Rasterise a beam path as a continuous phosphor trace."""

import numpy as np
from numba import njit

from .core import beam_path


@njit(cache=True, fastmath=True, nogil=True)
def _clip(lo, hi, p, q):
    """Narrow a parametric range against one canvas edge."""
    if p == 0.0:
        return (lo, hi) if q >= 0.0 else (1.0, 0.0)
    ratio = q / p
    if p < 0.0:
        return max(lo, ratio), hi
    return lo, min(hi, ratio)


@njit(cache=True, fastmath=True, nogil=True)
def _visible(x0, y0, dx, dy, width, height):
    """Parametric range of a segment that lies inside the canvas."""
    lo, hi = _clip(0.0, 1.0, -dx, x0)
    lo, hi = _clip(lo, hi, dx, width - 1.0 - x0)
    lo, hi = _clip(lo, hi, -dy, y0)
    return _clip(lo, hi, dy, height - 1.0 - y0)


@njit(cache=True, fastmath=True, nogil=True)
def draw_segments(acc, px, py, pz, pc, per_length):
    """Accumulate antialiased beam segments into a float32 BGR canvas."""
    height, width = acc.shape[0], acc.shape[1]
    for i in range(px.size - 1):
        # Z is a gate, not a fade: a blanked endpoint means no trace at all.
        if pz[i] <= 0.0 or pz[i + 1] <= 0.0:
            continue
        x0, y0 = np.float64(px[i]), np.float64(py[i])
        dx, dy = np.float64(px[i + 1]) - x0, np.float64(py[i + 1]) - y0
        if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(dx + dy)):
            continue
        # Arc length, as in core.beam_clock, so both renderings agree.
        span = np.hypot(dx, dy)
        steps = max(1, int(np.ceil(min(span, 1e7))))
        # One beam period per segment however far it travels: long segments fainter.
        amp = 1.0 if per_length else 1.0 / steps
        lo, hi = _visible(x0, y0, dx, dy, width, height)
        if lo > hi:
            continue
        for step in range(int(lo * steps), min(steps, int(hi * steps) + 2)):
            frac = step / steps
            fx, fy = x0 + dx * frac, y0 + dy * frac
            ix, iy = int(np.floor(fx)), int(np.floor(fy))
            tx, ty = fx - ix, fy - iy
            for oy in range(2):
                yy = iy + oy
                if yy < 0 or yy >= height:
                    continue
                wy = ty if oy else 1.0 - ty
                for ox in range(2):
                    xx = ix + ox
                    if xx < 0 or xx >= width:
                        continue
                    weight = wy * (tx if ox else 1.0 - tx) * amp
                    if weight <= 0.0:
                        continue
                    for chan in range(3):
                        value = pc[i, chan] + (pc[i + 1, chan] - pc[i, chan]) * frac
                        acc[yy, xx, chan] += weight * value
    return acc


def canvas_shape(height, width, params):
    """Output frame size that holds the whole deflection envelope."""
    y_min, y_max = params.y_bounds
    return (
        max(1, int(round((y_max - y_min) * height / 2.0))),
        max(1, int(round(params.x_extent * width))),
    )


def to_pixels(path, height, width, params):
    """Map normalised deflection onto canvas pixel coordinates."""
    _, y_max = params.y_bounds
    return (
        (path.x + params.x_extent) * (width / 2.0),
        (y_max - path.y) * (height / 2.0),
    )


def accumulate(path, height, width, params):
    """Unclipped float32 phosphor canvas for a beam path."""
    out_h, out_w = canvas_shape(height, width, params)
    px, py = to_pixels(path, height, width, params)
    acc = np.zeros((out_h, out_w, 3), dtype=np.float32)
    return draw_segments(acc, px, py, path.z, path.color, params.beam == "speed")


def render_path(path, height, width, params, exposure=1.0):
    """Draw an already-computed beam path at the source frame's pixel scale."""
    acc = accumulate(path, height, width, params)
    return (np.clip(acc * exposure, 0.0, 1.0) * 255.0).astype(np.uint8)


def render(frame, params, exposure=1.0):
    """Render one BGR frame through the scan processor."""
    height, width = frame.shape[:2]
    return render_path(beam_path(frame, params), height, width, params, exposure)
