"""Laser renderer command line tests."""

import io

import cv2
import numpy as np
import pytest

from ruttetra import ilda, lasercli


@pytest.fixture(name="ildfile")
def ildfile_fixture(tmp_path):
    """A three frame ILDA file spanning the deflection range."""
    path = tmp_path / "shape.ild"
    writer = ilda.IldWriter(path)
    for shift in range(3):
        angle = np.linspace(0, 2 * np.pi, 64, endpoint=False) + shift
        writer.write(
            ilda.Frame(
                x=20000 * np.cos(angle),
                y=20000 * np.sin(angle),
                blank=np.arange(64) % 16 == 0,
                color=np.full((64, 3), 200.0),
            )
        )
    writer.close()
    return path


def parse(*argv):
    """Parse arguments as the entry point would."""
    return lasercli.build_parser().parse_args(["in.ild", *argv])


def test_defaults():
    """A default projector is a 30K scanner on a Helios."""
    args = parse()
    assert args.kpps == 30000 and args.dac_points == 4095 and args.dac_bits == 12
    assert args.galvo and args.fps == 25.0


def test_renders_a_video(tmp_path, ildfile):
    """Every ILDA frame becomes a video frame."""
    out = tmp_path / "laser.avi"
    assert (
        lasercli.main(
            [str(ildfile), "--outfile", str(out), "--laser-size", "64", "--fps", "25"]
        )
        == 0
    )
    cap = cv2.VideoCapture(str(out))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    assert len(frames) == 3
    assert frames[0].shape == (64, 64, 3)
    assert max(f.max() for f in frames) > 0


def test_repeat_loops_the_file(ildfile):
    """Repeating plays the file more than once."""
    args = parse()
    args.ildfile, args.video, args.laser_size, args.repeat = str(ildfile), False, 48, 4
    assert lasercli.process(args, out_stream=io.StringIO()) == 12


def test_reports_the_budget(ildfile):
    """The summary names the point budget and what binds it."""
    args = parse()
    args.ildfile, args.video, args.laser_size = str(ildfile), False, 48
    stream = io.StringIO()
    lasercli.process(args, out_stream=stream)
    text = stream.getvalue()
    assert "64 points in the first" in text and "points/frame" in text


def test_missing_file_reports_failure(capsys, tmp_path):
    """An unreadable file exits non-zero rather than raising."""
    assert lasercli.main([str(tmp_path / "nope.ild")]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_not_an_ild_reports_failure(capsys, tmp_path):
    """A file that is not ILDA is refused."""
    junk = tmp_path / "junk.ild"
    junk.write_bytes(b"definitely not an ilda header")
    assert lasercli.main([str(junk)]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_empty_file_renders_nothing(tmp_path):
    """A file with only a terminator produces no frames."""
    path = tmp_path / "empty.ild"
    path.write_bytes(ilda.end_of_file())
    assert lasercli.main([str(path)]) == 0
