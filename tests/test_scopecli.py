"""Oscilloscope renderer command line tests."""

import io

import cv2
import numpy as np
import pytest

from ruttetra import scopecli
from ruttetra.audio import AudioParams, WavSink, frame_signals
from ruttetra.core import ScanParams


@pytest.fixture(name="wavfile")
def wavfile_fixture(tmp_path, noise):
    """Three video frames of deflection signal at 48 kHz."""
    path = tmp_path / "deflection.wav"
    audio = AudioParams(rate=48000, channels=3, bits=16)
    sink = WavSink(path, audio)
    for shift in range(3):
        rolled = np.roll(noise, shift * 4, axis=1)
        sink.write(frame_signals(rolled, ScanParams(lines=6), audio, 1920))
    sink.close()
    return path


def parse(*argv):
    """Parse arguments as the entry point would."""
    return scopecli.build_parser().parse_args(["in.wav", *argv])


def test_defaults():
    """A scope is square, monochrome and mildly persistent by default."""
    args = parse()
    params = scopecli.scope.params_from(args)
    assert (params.size, params.aspect) == (480, 1.0)
    assert params.z and not params.graticule
    assert args.fps == 25.0 and args.video


def test_flags_map_to_params():
    """Front panel flags reach the renderer."""
    params = scopecli.scope.params_from(
        parse(
            "--scope-size",
            "256",
            "--scope-aspect",
            "1.6",
            "--scope-gain",
            "2",
            "--scope-spot",
            "0.5",
            "--scope-bloom",
            "0",
            "--scope-persistence",
            "0.8",
            "--no-scope-z",
            "--scope-graticule",
        )
    )
    assert (params.size, params.aspect, params.gain) == (256, 1.6, 2.0)
    assert (params.spot, params.bloom, params.persistence) == (0.5, 0.0, 0.8)
    assert not params.z and params.graticule


def test_renders_a_video(tmp_path, wavfile):
    """Every frame's worth of the WAV becomes a video frame."""
    out = tmp_path / "scope.avi"
    assert (
        scopecli.main(
            [str(wavfile), "--outfile", str(out), "--scope-size", "64", "--fps", "25"]
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


def test_no_video_still_counts_frames(wavfile):
    """Rendering without writing is allowed, for preview only."""
    args = parse()
    args.wavfile, args.video, args.scope_size = str(wavfile), False, 48
    assert scopecli.process(args, out_stream=io.StringIO()) == 3


def test_missing_file_reports_failure(capsys, tmp_path):
    """An unreadable WAV exits non-zero rather than raising."""
    assert scopecli.main([str(tmp_path / "nope.wav")]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_not_a_wav_reports_failure(capsys, tmp_path):
    """A file that is not a WAV is refused."""
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not a riff header at all")
    assert scopecli.main([str(junk)]) == 1
    assert "cannot read" in capsys.readouterr().err
