"""Helios DAC binding tests, against a stand-in library."""

import ctypes

import numpy as np
import pytest

from ruttetra import helios, laser


class FakeLib:
    """Stand-in for libHeliosDacAPI."""

    def __init__(self, devices=1):
        self.devices = devices
        self.frames = []
        self.shutter = []
        self.stopped = False
        self.closed = False
        self.status_calls = 0

    def OpenDevices(self):  # pylint: disable=invalid-name
        """Report how many DACs are attached."""
        return self.devices

    def GetStatus(self, _):  # pylint: disable=invalid-name
        """Report ready on the second poll, to exercise the wait loop."""
        self.status_calls += 1
        return 1 if self.status_calls % 2 == 0 else 0

    def WriteFrame(
        self, dac, pps, flags, points, count
    ):  # pylint: disable=invalid-name
        """Record a frame."""
        raw = bytes(bytearray(points)[: count * 8])
        self.frames.append((dac, pps, flags, count, raw))
        return 0

    def SetShutter(self, _, value):  # pylint: disable=invalid-name
        """Record shutter changes."""
        self.shutter.append(bool(value))
        return 0

    def Stop(self, _):  # pylint: disable=invalid-name
        """Record the stop."""
        self.stopped = True
        return 0

    def CloseDevices(self):  # pylint: disable=invalid-name
        """Record the close."""
        self.closed = True
        return 0


@pytest.fixture(name="fake")
def fake_fixture(monkeypatch):
    """Patch the library loader to return the stand-in."""
    lib = FakeLib()
    monkeypatch.setattr(helios, "load_library", lambda path=None: lib)
    return lib


def test_point_struct_layout():
    """The SDK's record is 8 packed bytes in x, y, r, g, b, i order."""
    assert ctypes.sizeof(helios.HeliosPoint) == 8
    offsets = [getattr(helios.HeliosPoint, f).offset for f in "xyrgbi"]
    assert offsets == [0, 2, 4, 5, 6, 7]


def test_encode_maps_full_scale():
    """Normalised deflection spans the DAC's unsigned 12 bit range."""
    points = np.array([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]])
    array = helios.encode(points, np.zeros(3, bool), np.ones((3, 3)))
    assert (array[0].x, array[0].y) == (0, 0)
    assert (array[1].x, array[1].y) == (helios.COORD_MAX, helios.COORD_MAX)
    assert array[2].x == pytest.approx(helios.COORD_MAX // 2, abs=1)


def test_encode_colour_and_intensity():
    """Colour is 8 bit per channel and intensity follows the brightest."""
    array = helios.encode(
        np.zeros((1, 2)), np.zeros(1, bool), np.array([[1.0, 0.5, 0.0]])
    )
    assert (array[0].r, array[0].g, array[0].b) == (255, 128, 0)
    assert array[0].i == 255


def test_encode_blanks_are_dark():
    """A blanked point carries no colour, whatever was passed in."""
    array = helios.encode(np.zeros((1, 2)), np.ones(1, bool), np.ones((1, 3)))
    assert (array[0].r, array[0].g, array[0].b, array[0].i) == (0, 0, 0, 0)


def test_encode_rejects_oversized_frames():
    """The Helios takes at most 4095 points per frame."""
    count = helios.MAX_POINTS + 1
    with pytest.raises(ValueError):
        helios.encode(np.zeros((count, 2)), np.zeros(count, bool), np.zeros((count, 3)))


def test_sink_opens_and_writes(fake):
    """The sink opens the shutter, writes frames and closes cleanly."""
    sink = helios.HeliosSink(laser.Projector(kpps=30000))
    assert fake.shutter == [True]

    points = np.linspace(-1, 1, 20).reshape(10, 2)
    sink.write(points, np.zeros(10, bool), np.ones((10, 3)))
    sink.write(points, np.zeros(10, bool), np.ones((10, 3)))
    assert sink.frames == 2

    dac, pps, flags, count, _ = fake.frames[0]
    assert (dac, pps, count) == (0, 30000, 10)
    assert flags == helios.FLAGS_DEFAULT

    sink.close()
    assert fake.shutter == [True, False]
    assert fake.stopped and fake.closed


def test_sink_clamps_the_point_rate(fake):
    """The wire protocol carries the rate in 16 bits."""
    sink = helios.HeliosSink(laser.Projector(kpps=100000))
    assert sink.pps == 0xFFFF
    sink.close()
    assert fake.closed


def test_sink_reports_a_missing_device(monkeypatch):
    """Asking for a DAC that is not there fails clearly."""
    monkeypatch.setattr(helios, "load_library", lambda path=None: FakeLib(devices=0))
    with pytest.raises(OSError, match="no Helios DAC"):
        helios.HeliosSink(laser.Projector())


def test_write_failure_is_raised(fake):
    """A negative return from the SDK becomes an exception."""
    sink = helios.HeliosSink(laser.Projector())
    fake.WriteFrame = lambda *args: -3
    with pytest.raises(OSError, match="WriteFrame failed"):
        sink.write(np.zeros((4, 2)), np.zeros(4, bool), np.ones((4, 3)))


def test_missing_library_explains_itself(monkeypatch):
    """Without the shared library the error says where to get it."""
    monkeypatch.setattr(ctypes.util, "find_library", lambda _: None)
    monkeypatch.setattr(
        ctypes.cdll, "LoadLibrary", lambda _: (_ for _ in ()).throw(OSError("nope"))
    )
    with pytest.raises(OSError, match="helios_dac"):
        helios.load_library()
