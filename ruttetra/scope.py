"""Simulate a monochrome XY oscilloscope fed from the deflection signals.

Reads the X/Y(/Z) audio the scan processor emits and draws what a scope would
show: a phosphor trace whose brightness falls with beam speed, blurred by the
spot size and decaying between frames.
"""

import argparse
import wave
from dataclasses import dataclass

import cv2
import numpy as np
from numba import njit

from .raster import draw_segments

PCM_SCALE = {2: 32767.0, 3: 8388607.0, 4: 2147483647.0}


@dataclass(frozen=True)
class ScopeParams:
    """Front panel of the simulated scope."""

    size: int = 480
    aspect: float = 1.0
    gain: float = 1.0
    spot: float = 1.0
    bloom: float = 0.35
    persistence: float = 0.35
    z: bool = True
    z_invert: bool = False
    graticule: bool = False

    def __post_init__(self):
        for name in ("gain", "spot", "bloom"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.size < 8:
            raise ValueError("size must be >= 8")
        if self.aspect <= 0:
            raise ValueError("aspect must be > 0")
        if not 0.0 <= self.persistence < 1.0:
            raise ValueError("persistence must be in [0, 1)")

    @property
    def shape(self):
        """Canvas (height, width)."""
        return self.size, max(8, int(round(self.size * self.aspect)))


def decode(raw, width, channels):
    """Decode interleaved little-endian PCM to float32 in [-1, 1]."""
    if width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = packed[:, 0] | packed[:, 1] << 8 | packed[:, 2] << 16
        values = np.where(values & 0x800000, values - 0x1000000, values)
    else:
        values = np.frombuffer(raw, dtype=f"<i{width}")
    voltages = np.clip(values / PCM_SCALE[width], -1.0, 1.0)
    return voltages.astype(np.float32).reshape(-1, channels)


def wav_blocks(path, fps):
    """Yield one video frame's worth of samples at a time from a WAV."""
    with wave.open(str(path), "rb") as handle:
        channels, width = handle.getnchannels(), handle.getsampwidth()
        if width not in PCM_SCALE:
            raise ValueError(f"unsupported sample width {width}")
        block = max(2, int(round(handle.getframerate() / fps)))
        while True:
            raw = handle.readframes(block)
            if len(raw) < width * channels * 2:
                return
            yield decode(raw, width, channels)


@njit(cache=True, fastmath=True, nogil=True)
def _decay(state, frame, persistence):
    """Fade the phosphor, then write the new trace over it at full brightness."""
    for y in range(state.shape[0]):
        for x in range(state.shape[1]):
            faded = state[y, x] * persistence
            state[y, x] = frame[y, x] if frame[y, x] > faded else faded
    return state


def beam(signals, params):
    """Map deflection voltages to canvas coordinates and beam current."""
    height, width = params.shape
    px = np.ascontiguousarray((signals[:, 0] + 1.0) * 0.5 * (width - 1), np.float32)
    py = np.ascontiguousarray((1.0 - signals[:, 1]) * 0.5 * (height - 1), np.float32)
    if params.z and signals.shape[1] >= 3:
        level = (signals[:, 2] + 1.0) * 0.5
        if params.z_invert:
            level = 1.0 - level
    else:
        level = np.ones(len(signals), dtype=np.float32)
    level = np.clip(level, 0.0, 1.0).astype(np.float32)
    return px, py, level


def trace(signals, params):
    """Draw one frame of beam travel, before persistence."""
    height, width = params.shape
    px, py, level = beam(signals, params)
    acc = np.zeros((height, width, 3), dtype=np.float32)
    color = np.repeat((level * params.gain)[:, None], 3, axis=1).astype(np.float32)
    # per_length is False: equal time per sample, so fast travel writes fainter.
    draw_segments(acc, px, py, level, color, False)
    spot = acc[:, :, 0]
    if params.spot > 0:
        spot = cv2.GaussianBlur(spot, (0, 0), sigmaX=params.spot)
    if params.bloom > 0:
        spot = spot + params.bloom * cv2.GaussianBlur(
            spot, (0, 0), sigmaX=max(1.0, params.spot * 6.0)
        )
    return spot


def draw_graticule(image):
    """Overlay the usual 8 x 10 division grid."""
    height, width = image.shape[:2]
    for i in range(1, 10):
        cv2.line(
            image, (width * i // 10, 0), (width * i // 10, height), (28, 28, 28), 1
        )
    for i in range(1, 8):
        cv2.line(image, (0, height * i // 8), (width, height * i // 8), (28, 28, 28), 1)
    return image


class Screen:
    """Stateful phosphor screen, one instance per render."""

    def __init__(self, params):
        self.params = params
        self.state = np.zeros(params.shape, dtype=np.float32)

    def render(self, signals):
        """Render one frame of signal and return an 8-bit BGR image."""
        frame = trace(signals, self.params)
        _decay(self.state, frame, self.params.persistence)
        grey = (np.clip(self.state, 0.0, 1.0) * 255.0).astype(np.uint8)
        image = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
        return draw_graticule(image) if self.params.graticule else image


def render_wav(path, fps, params):
    """Yield scope frames for every video frame's worth of a WAV."""
    screen = Screen(params)
    for block in wav_blocks(path, fps):
        yield screen.render(block)


class ScopeSink:
    """Sink that renders deflection blocks as oscilloscope video."""

    def __init__(self, path, params, fps, monitor=False):
        height, width = params.shape
        self.screen = Screen(params)
        self.monitor = monitor
        self.frames = 0
        self.writer = None
        if path:
            self.writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height)
            )

    def write(self, block):
        """Render one block of (samples, channels) deflection voltages."""
        image = self.screen.render(block)
        if self.monitor:
            cv2.imshow("scope", image)
            cv2.waitKey(1)
        if self.writer is not None:
            self.writer.write(image)
        self.frames += 1
        return image

    def close(self):
        """Finalise the file."""
        if self.writer is not None:
            self.writer.release()


def add_arguments(group):
    """Attach the scope front panel to an argument parser group."""
    flag = argparse.BooleanOptionalAction
    group.add_argument("--scope-size", default=480, type=int, help="screen height")
    group.add_argument("--scope-aspect", default=1.0, type=float, help="width / height")
    group.add_argument("--scope-gain", default=1.0, type=float, help="trace brightness")
    group.add_argument("--scope-spot", default=1.0, type=float, help="beam spot sigma")
    group.add_argument("--scope-bloom", default=0.35, type=float, help="halo around it")
    group.add_argument(
        "--scope-persistence", default=0.35, type=float, help="phosphor decay, 0 to 1"
    )
    group.add_argument("--scope-z", action=flag, default=True, help="use the Z channel")
    group.add_argument("--scope-graticule", action=flag, default=False)
    return group


def params_from(args):
    """Build scope settings from parsed arguments."""
    return ScopeParams(
        size=args.scope_size,
        aspect=args.scope_aspect,
        gain=args.scope_gain,
        spot=args.scope_spot,
        bloom=args.scope_bloom,
        persistence=args.scope_persistence,
        z=args.scope_z,
        z_invert=getattr(args, "z_invert", False),
        graticule=args.scope_graticule,
    )
