"""ILDA Image Data Transfer Format, revision 011.

Big-endian throughout. A file is a run of 32 byte headers, each followed by its
records, terminated by a header declaring zero records.
"""

import struct
from dataclasses import dataclass

import numpy as np

HEADER = struct.Struct(">4s3sB8s8sHHHBB")
HEADER_SIZE = 32
RECORD_SIZE = {0: 8, 1: 6, 2: 3, 4: 10, 5: 8}
POINT_FORMATS = (0, 1, 4, 5)

BLANKING = 0x40
LAST_POINT = 0x80
COORD_MIN, COORD_MAX = -32768, 32767


@dataclass(frozen=True)
class Frame:
    """One ILDA frame: pixel coordinates and colour, both already scaled."""

    x: np.ndarray
    y: np.ndarray
    blank: np.ndarray
    color: np.ndarray
    z: np.ndarray = None
    name: str = "RUTTETRA"
    company: str = "ruttetra"

    def __len__(self):
        return int(self.x.size)


def _text(value, size=8):
    """ASCII field padded with NULs, as the header fields are defined."""
    return value.encode("ascii", "replace")[:size].ljust(size, b"\0")


def header(fmt, records, frame_number, total_frames, name, company, projector=0):
    """Pack a 32 byte section header."""
    return HEADER.pack(
        b"ILDA",
        b"\0\0\0",
        fmt,
        _text(name),
        _text(company),
        records,
        frame_number,
        total_frames,
        projector,
        0,
    )


def _status(blank):
    """Status bytes for a frame, with the final point flagged as the last."""
    status = np.where(blank, BLANKING, 0).astype(np.uint8)
    if status.size:
        status[-1] |= LAST_POINT
    return status


def encode_frame(frame, number=0, total=1, fmt=5, projector=0):
    """Encode one frame as a header plus records."""
    if fmt not in (4, 5):
        raise ValueError("only true colour formats 4 and 5 are written")
    count = len(frame)
    if count > 65535:
        raise ValueError(f"{count} points exceeds the 65535 record limit")
    out = bytearray(
        header(fmt, count, number, total, frame.name, frame.company, projector)
    )
    coords = [
        np.clip(np.round(frame.x), COORD_MIN, COORD_MAX).astype(">i2"),
        np.clip(np.round(frame.y), COORD_MIN, COORD_MAX).astype(">i2"),
    ]
    if fmt == 4:
        depth = np.zeros(count) if frame.z is None else frame.z
        coords.append(np.clip(np.round(depth), COORD_MIN, COORD_MAX).astype(">i2"))
    rgb = np.clip(np.round(frame.color), 0, 255).astype(np.uint8).reshape(count, 3)
    status = _status(np.asarray(frame.blank, dtype=bool))

    record = np.zeros((count, RECORD_SIZE[fmt]), dtype=np.uint8)
    offset = 0
    for axis in coords:
        record[:, offset : offset + 2] = axis.view(np.uint8).reshape(count, 2)
        offset += 2
    record[:, offset] = status
    # The spec orders the colour bytes blue, green, red.
    record[:, offset + 1] = rgb[:, 2]
    record[:, offset + 2] = rgb[:, 1]
    record[:, offset + 3] = rgb[:, 0]
    out += record.tobytes()
    return bytes(out)


def end_of_file(fmt=5):
    """The terminating header, declaring no records."""
    return header(fmt, 0, 0, 0, "", "ruttetra")


class IldWriter:
    """Streaming ILDA file writer, one frame per call."""

    def __init__(self, path, fmt=5, total=0, projector=0):
        if fmt not in (4, 5):
            raise ValueError("only true colour formats 4 and 5 are written")
        self.handle = open(path, "wb")  # pylint: disable=consider-using-with
        self.fmt = fmt
        self.total = total
        self.projector = projector
        self.frames = 0

    def write(self, frame):
        """Append one frame."""
        self.handle.write(
            encode_frame(frame, self.frames, self.total or 1, self.fmt, self.projector)
        )
        self.frames += 1

    def close(self):
        """Write the end of file header and close."""
        self.handle.write(end_of_file(self.fmt))
        self.handle.close()


def read(path):
    """Read every point frame in a file.

    Indexed colour formats are returned with the raw index in the red channel,
    since resolving a palette is out of scope here.
    """
    data = pathlib_read(path)
    if len(data) < HEADER_SIZE or data[:4] != b"ILDA":
        raise ValueError("not an ILDA file: no ILDA header at the start")
    offset, frames = 0, []
    while offset + HEADER_SIZE <= len(data):
        magic, _, fmt, name, company, count, _, _, _, _ = HEADER.unpack(
            data[offset : offset + HEADER_SIZE]
        )
        if magic != b"ILDA":
            raise ValueError(f"bad section header at byte {offset}")
        offset += HEADER_SIZE
        if count == 0:
            break
        if fmt not in RECORD_SIZE:
            raise ValueError(f"unknown format code {fmt}")
        size = RECORD_SIZE[fmt]
        body = np.frombuffer(data, np.uint8, count * size, offset).reshape(count, size)
        offset += count * size
        if fmt in POINT_FORMATS:
            frames.append(_decode(body, fmt, name, company))
    return frames


def _decode(body, fmt, name, company):
    """Turn a record block into a Frame."""
    count = body.shape[0]
    axes = 3 if fmt in (0, 4) else 2
    coords = [
        body[:, i * 2 : i * 2 + 2].copy().view(">i2").reshape(count).astype(np.float64)
        for i in range(axes)
    ]
    at = axes * 2
    status = body[:, at]
    color = np.zeros((count, 3))
    if fmt in (4, 5):
        color[:, 0] = body[:, at + 3]
        color[:, 1] = body[:, at + 2]
        color[:, 2] = body[:, at + 1]
    else:
        color[:, 0] = body[:, at + 1]
    return Frame(
        x=coords[0],
        y=coords[1],
        z=coords[2] if axes == 3 else None,
        blank=(status & BLANKING) != 0,
        color=color,
        name=name.split(b"\0")[0].decode("ascii", "replace").strip(),
        company=company.split(b"\0")[0].decode("ascii", "replace").strip(),
    )


def pathlib_read(path):
    """Read a file whole."""
    with open(path, "rb") as handle:
        return handle.read()
