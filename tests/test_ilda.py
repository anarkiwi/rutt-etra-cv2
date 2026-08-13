"""ILDA file format tests."""

import struct

import numpy as np
import pytest

from ruttetra import ilda


def frame(count=8, blank=None, color=None):
    """A small frame spanning the coordinate range."""
    return ilda.Frame(
        x=np.linspace(-32767, 32767, count),
        y=np.linspace(32767, -32767, count),
        blank=np.zeros(count, bool) if blank is None else blank,
        color=np.full((count, 3), 128.0) if color is None else color,
    )


def test_header_layout():
    """The header is 32 bytes with the fields at their specified offsets."""
    raw = ilda.header(5, 1191, 3, 40, "NAME", "COMPANY")
    assert len(raw) == ilda.HEADER_SIZE
    assert raw[0:4] == b"ILDA"
    assert raw[4:7] == b"\0\0\0"
    assert raw[7] == 5
    assert raw[8:16] == b"NAME\0\0\0\0"
    assert raw[16:24] == b"COMPANY\0"
    assert struct.unpack(">H", raw[24:26])[0] == 1191
    assert struct.unpack(">H", raw[26:28])[0] == 3
    assert struct.unpack(">H", raw[28:30])[0] == 40
    assert raw[30] == 0 and raw[31] == 0


def test_coordinates_are_big_endian():
    """ILDA is big endian, which is the easiest thing to get wrong."""
    one = ilda.Frame(
        x=np.array([256.0]),
        y=np.array([1.0]),
        blank=np.array([False]),
        color=np.zeros((1, 3)),
    )
    body = ilda.encode_frame(one)[ilda.HEADER_SIZE :]
    assert body[0:2] == b"\x01\x00"
    assert body[2:4] == b"\x00\x01"


def test_colour_byte_order_is_bgr():
    """The spec puts blue first, then green, then red."""
    one = ilda.Frame(
        x=np.zeros(1),
        y=np.zeros(1),
        blank=np.array([False]),
        color=np.array([[10.0, 20.0, 30.0]]),
    )
    body = ilda.encode_frame(one, fmt=5)[ilda.HEADER_SIZE :]
    assert (body[5], body[6], body[7]) == (30, 20, 10)


def test_status_bits():
    """Blanking is bit 6, and only the final point carries the last point bit."""
    blank = np.array([False, True, False, True])
    body = ilda.encode_frame(frame(4, blank=blank))[ilda.HEADER_SIZE :]
    status = [body[i * 8 + 4] for i in range(4)]
    assert [bool(s & ilda.BLANKING) for s in status] == list(blank)
    assert [bool(s & ilda.LAST_POINT) for s in status] == [False, False, False, True]


@pytest.mark.parametrize("fmt,size", [(4, 10), (5, 8)])
def test_record_sizes(fmt, size):
    """True colour records are 10 bytes in 3D and 8 bytes in 2D."""
    blob = ilda.encode_frame(frame(6), fmt=fmt)
    assert len(blob) == ilda.HEADER_SIZE + 6 * size


def test_indexed_formats_are_not_written():
    """Only the true colour formats are produced."""
    for fmt in (0, 1, 2):
        with pytest.raises(ValueError):
            ilda.encode_frame(frame(), fmt=fmt)


def test_end_of_file_declares_no_records():
    """The terminator is a header with a zero record count."""
    raw = ilda.end_of_file()
    assert raw[0:4] == b"ILDA"
    assert struct.unpack(">H", raw[24:26])[0] == 0


@pytest.mark.parametrize("fmt", [4, 5])
def test_round_trip(tmp_path, fmt):
    """What is written reads back unchanged."""
    original = frame(64, blank=np.arange(64) % 3 == 0)
    path = tmp_path / "out.ild"
    writer = ilda.IldWriter(path, fmt=fmt)
    for _ in range(3):
        writer.write(original)
    writer.close()

    frames = ilda.read(path)
    assert len(frames) == 3
    back = frames[0]
    assert np.allclose(back.x, np.round(original.x))
    assert np.allclose(back.y, np.round(original.y))
    assert np.array_equal(back.blank, original.blank)
    assert np.allclose(back.color, original.color)
    assert back.name == "RUTTETRA"


def test_coordinates_are_clamped():
    """Out of range deflection saturates rather than wrapping."""
    wild = ilda.Frame(
        x=np.array([-90000.0, 90000.0]),
        y=np.array([0.0, 0.0]),
        blank=np.zeros(2, bool),
        color=np.zeros((2, 3)),
    )
    path = ilda.encode_frame(wild)[ilda.HEADER_SIZE :]
    values = [struct.unpack(">h", path[i * 8 : i * 8 + 2])[0] for i in range(2)]
    assert values == [ilda.COORD_MIN, ilda.COORD_MAX]


def test_too_many_records_rejected():
    """A frame beyond the 16 bit record count is refused."""
    with pytest.raises(ValueError):
        ilda.encode_frame(frame(70000))


def test_bad_magic_rejected(tmp_path):
    """A file that is not ILDA fails loudly."""
    path = tmp_path / "junk.ild"
    path.write_bytes(b"NOPE" + b"\0" * 40)
    with pytest.raises(ValueError):
        ilda.read(path)


def test_reads_indexed_formats(tmp_path):
    """Format 0 and 1 frames are readable, with the index in the red channel."""
    body = struct.pack(">hhhBB", 100, -200, 0, 0x40, 24)
    path = tmp_path / "indexed.ild"
    path.write_bytes(
        ilda.header(0, 1, 0, 1, "IDX", "test") + body + ilda.end_of_file(0)
    )
    got = ilda.read(path)[0]
    assert (got.x[0], got.y[0]) == (100.0, -200.0)
    assert bool(got.blank[0])
    assert got.color[0, 0] == 24
