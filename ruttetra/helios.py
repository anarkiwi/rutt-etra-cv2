"""Helios laser DAC output, via ctypes onto the vendor shared library.

Build libHeliosDacAPI from github.com/Grix/helios_dac and put it on the loader
path. Coordinates are unsigned 12 bit, colour 8 bit, 4095 points per frame.
"""

import ctypes
import ctypes.util

import numpy as np

MAX_POINTS = 0xFFF
COORD_MAX = 0xFFF
FLAGS_START_IMMEDIATELY = 1 << 0
FLAGS_SINGLE_MODE = 1 << 1
FLAGS_DONT_BLOCK = 1 << 2
FLAGS_DEFAULT = FLAGS_SINGLE_MODE

LIBRARY_NAMES = ("HeliosDacAPI", "libHeliosDacAPI.so", "HeliosLaserDAC")


class HeliosPoint(ctypes.Structure):
    """The SDK's 8 byte point record."""

    _pack_ = 1
    _fields_ = [
        ("x", ctypes.c_uint16),
        ("y", ctypes.c_uint16),
        ("r", ctypes.c_uint8),
        ("g", ctypes.c_uint8),
        ("b", ctypes.c_uint8),
        ("i", ctypes.c_uint8),
    ]


def load_library(path=None):
    """Load the Helios shared library, raising a useful error if absent."""
    candidates = [path] if path else []
    candidates += list(LIBRARY_NAMES)
    found = next((ctypes.util.find_library(n) for n in LIBRARY_NAMES if n), None)
    if found:
        candidates.append(found)
    for name in candidates:
        try:
            return ctypes.cdll.LoadLibrary(name)
        except OSError:
            continue
    raise OSError(
        "libHeliosDacAPI not found. Build it from github.com/Grix/helios_dac "
        "and put it on LD_LIBRARY_PATH, or pass --helios-library."
    )


def encode(points, blank, rgb):
    """Pack normalised deflection and colour into a HeliosPoint array."""
    count = points.shape[0]
    if count > MAX_POINTS:
        raise ValueError(f"{count} points exceeds the Helios limit of {MAX_POINTS}")
    coords = np.clip(np.round((points + 1.0) * 0.5 * COORD_MAX), 0, COORD_MAX)
    levels = np.clip(np.round(np.asarray(rgb) * 255.0), 0, 255).astype(np.uint8)
    levels = levels.reshape(count, 3).copy()
    levels[np.asarray(blank, dtype=bool)] = 0

    array = (HeliosPoint * count)()
    buffer = np.frombuffer(array, dtype=np.uint8).reshape(count, 8)
    buffer[:, 0:2] = coords[:, 0].astype("<u2").view(np.uint8).reshape(count, 2)
    buffer[:, 2:4] = coords[:, 1].astype("<u2").view(np.uint8).reshape(count, 2)
    buffer[:, 4:7] = levels
    buffer[:, 7] = levels.max(axis=1)
    return array


class HeliosSink:
    """Stream frames to a Helios DAC."""

    def __init__(self, projector, dac_number=0, library=None, flags=FLAGS_DEFAULT):
        self.lib = load_library(library)
        self.dac_number = dac_number
        self.flags = flags
        self.pps = min(projector.kpps, 0xFFFF)
        self.frames = 0
        count = self.lib.OpenDevices()
        if count <= dac_number:
            raise OSError(f"no Helios DAC at index {dac_number}; found {count}")
        self.lib.SetShutter(dac_number, True)

    def write(self, points, blank, rgb):
        """Send one frame, waiting for the DAC to be ready for it."""
        array = encode(points, blank, rgb)
        while self.lib.GetStatus(self.dac_number) != 1:
            pass
        result = self.lib.WriteFrame(
            self.dac_number, self.pps, self.flags, array, len(array)
        )
        if result < 0:
            raise OSError(f"Helios WriteFrame failed with {result}")
        self.frames += 1

    def close(self):
        """Stop output, close the shutter and release the device."""
        self.lib.SetShutter(self.dac_number, False)
        self.lib.Stop(self.dac_number)
        self.lib.CloseDevices()
