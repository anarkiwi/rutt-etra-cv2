"""Command line tests."""

import io
import wave

import cv2
import numpy as np
import pytest

from ruttetra import cli


@pytest.fixture(name="clip")
def clip_fixture(tmp_path, noise):
    """Short AVI written to disk for the CLI to read back."""
    path = tmp_path / "in.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        25.0,
        (noise.shape[1], noise.shape[0]),
    )
    for shift in range(4):
        writer.write(np.roll(noise, shift * 4, axis=1))
    writer.release()
    return path


def parse(*argv):
    """Parse arguments as the entry point would."""
    return cli.build_parser().parse_args(["in.avi", *argv])


def test_defaults_match_the_documented_effect():
    """Defaults give a colour, constant-rate, serpentine scan."""
    args = parse()
    params = cli.scan_params(args)
    assert (params.lines, params.gain, params.beam) == (60, 0.1, "rate")
    assert params.luma == "bt709" and params.sampling == "area"
    assert not params.mono and params.serpentine


def test_legacy_flags_still_work():
    """The original --lines, --scale and --no-color flags keep their meaning."""
    params = cli.scan_params(parse("--lines", "24", "--scale", "0.4", "--no-color"))
    assert (params.lines, params.gain, params.mono) == (24, 0.4, True)


def test_control_flags_map_to_params():
    """Deflection flags reach the model."""
    params = cli.scan_params(
        parse(
            "--intensity",
            "0.5",
            "--brightness",
            "0.25",
            "--v-size",
            "0.8",
            "--h-size",
            "1.2",
            "--depth",
            "0.9",
            "--skew",
            "0.3",
            "--smooth",
            "2",
            "--beam",
            "speed",
            "--no-serpentine",
            "--retrace",
            "3",
            "--invert",
            "--sampling",
            "nearest",
            "--luma",
            "bt601",
            "--samples",
            "40",
        )
    )
    assert params.intensity == 0.5 and params.brightness == 0.25
    assert (params.v_size, params.h_size, params.depth) == (0.8, 1.2, 0.9)
    assert params.skew == 0.3 and params.smooth == 2.0
    assert params.beam == "speed" and not params.serpentine
    assert params.retrace == 3 and params.invert
    assert params.sampling == "nearest" and params.luma == "bt601"
    assert params.samples == 40


def test_source_fps_prefers_the_override():
    """An explicit rate wins over whatever the container claims."""

    class Cap:
        """Capture reporting a fixed frame rate."""

        def __init__(self, fps):
            self.fps = fps

        def get(self, _):
            """Report the frame rate."""
            return self.fps

    assert cli.source_fps(Cap(30.0), 0.0) == 30.0
    assert cli.source_fps(Cap(30.0), 50.0) == 50.0
    assert cli.source_fps(Cap(0.0), 0.0) == cli.DEFAULT_FPS
    assert cli.source_fps(Cap(-1.0), 0.0) == cli.DEFAULT_FPS


def test_device_accepts_index_or_name():
    """Sound cards can be chosen either way."""
    assert cli.device_id("3") == 3
    assert cli.device_id("Scarlett 2i2") == "Scarlett 2i2"


def test_open_capture_handles_camera_index():
    """A bare number is treated as a camera."""
    cap = cli.open_capture("99")
    assert not cap.isOpened()
    cap.release()


def test_missing_input_reports_failure(capsys):
    """A file that cannot be opened exits non-zero."""
    assert cli.main(["/nonexistent/clip.avi", "--no-monitor", "--no-video"]) == 1
    assert "cannot open" in capsys.readouterr().err


def test_video_output(tmp_path, clip):
    """Video mode writes a playable file, taller than the source."""
    out = tmp_path / "out.avi"
    assert (
        cli.main(
            [
                str(clip),
                "--outfile",
                str(out),
                "--no-monitor",
                "--lines",
                "8",
                "--scale",
                "0.25",
            ]
        )
        == 0
    )
    cap = cv2.VideoCapture(str(out))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[0] == round(48 * 1.25) and frame.shape[1] == 64


@pytest.mark.parametrize("channels", [2, 3])
def test_wav_output(tmp_path, clip, channels):
    """Audio mode writes one frame's worth of samples per video frame."""
    out = tmp_path / "scope.wav"
    assert (
        cli.main(
            [
                str(clip),
                "--no-video",
                "--no-monitor",
                "--wav",
                str(out),
                "--audio-rate",
                "48000",
                "--audio-channels",
                str(channels),
                "--lines",
                "8",
                "--fps",
                "25",
            ]
        )
        == 0
    )
    with wave.open(str(out), "rb") as handle:
        assert handle.getnchannels() == channels
        assert handle.getnframes() == 4 * 1920


def test_video_and_audio_together(tmp_path, clip):
    """Both sinks can run off the one pass over the source."""
    video, audio = tmp_path / "out.avi", tmp_path / "scope.wav"
    assert (
        cli.main(
            [
                str(clip),
                "--outfile",
                str(video),
                "--wav",
                str(audio),
                "--no-monitor",
                "--lines",
                "8",
            ]
        )
        == 0
    )
    assert video.stat().st_size > 0 and audio.stat().st_size > 0


def test_process_reports_frame_count(clip):
    """Every source frame is accounted for."""
    args = parse()
    args.infile, args.video, args.monitor, args.lines = str(clip), False, False, 8
    cap = cli.open_capture(str(clip))
    try:
        assert cli.process(cap, args, out_stream=io.StringIO()) == 4
    finally:
        cap.release()


def test_missing_sounddevice_is_reported(clip, monkeypatch):
    """Asking for a sound card without the optional dependency fails clearly."""
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", None)
    with pytest.raises((ImportError, AttributeError)):
        cli.main(
            [
                str(clip),
                "--no-video",
                "--no-monitor",
                "--audio-device",
                "0",
                "--lines",
                "4",
            ]
        )


def test_undersampled_audio_is_reported(clip):
    """Too few audio samples for the beam path draws a warning."""
    args = parse()
    args.infile, args.video, args.monitor = str(clip), False, False
    args.lines, args.wav, args.audio_rate, args.fps = 60, None, 8000, 25.0
    args.audio_device = None
    stream = io.StringIO()
    cap = cli.open_capture(str(clip))
    try:
        cli.process(cap, args, out_stream=stream)
    finally:
        cap.release()
    assert stream.getvalue() == ""

    args.wav = str(clip.parent / "warn.wav")
    stream = io.StringIO()
    cap = cli.open_capture(str(clip))
    try:
        cli.process(cap, args, out_stream=stream)
    finally:
        cap.release()
    assert "warning:" in stream.getvalue() and "beam path" in stream.getvalue()


def test_no_warning_when_the_clock_is_fast_enough(clip):
    """A sufficient sample budget stays quiet."""
    assert cli.warn_undersampled(20000, 4000) is not None
    args = parse()
    args.infile, args.video, args.monitor = str(clip), False, False
    args.lines, args.samples, args.retrace = 4, 8, 2
    args.wav = str(clip.parent / "quiet.wav")
    args.audio_device, args.audio_rate, args.fps = None, 48000, 25.0
    stream = io.StringIO()
    cap = cli.open_capture(str(clip))
    try:
        cli.process(cap, args, out_stream=stream)
    finally:
        cap.release()
    assert "warning:" not in stream.getvalue()


def test_scope_out_renders_alongside(tmp_path, clip):
    """The simulated scope is just another sink on the same pass."""
    video, scope_video = tmp_path / "out.avi", tmp_path / "scope.avi"
    assert (
        cli.main(
            [
                str(clip),
                "--outfile",
                str(video),
                "--scope-out",
                str(scope_video),
                "--scope-size",
                "48",
                "--no-monitor",
                "--lines",
                "4",
                "--samples",
                "8",
                "--audio-rate",
                "48000",
                "--fps",
                "25",
            ]
        )
        == 0
    )
    cap = cv2.VideoCapture(str(scope_video))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape == (48, 48, 3)
    assert video.stat().st_size > 0


def test_laser_sinks_render_alongside(tmp_path, clip):
    """ILDA and the simulated projection are sinks on the same pass."""
    ild, laser_video = tmp_path / "out.ild", tmp_path / "laser.avi"
    assert (
        cli.main(
            [
                str(clip),
                "--no-video",
                "--no-monitor",
                "--ild-out",
                str(ild),
                "--laser-out",
                str(laser_video),
                "--laser-size",
                "48",
                "--fit-scan",
                "--fps",
                "25",
            ]
        )
        == 0
    )
    from ruttetra import ilda  # pylint: disable=import-outside-toplevel

    frames = ilda.read(ild)
    assert len(frames) == 4
    assert laser_video.stat().st_size > 0


def test_fit_scan_sizes_the_raster(clip):
    """Fitting picks a raster the projector can actually draw."""
    args = parse()
    args.infile, args.video, args.monitor = str(clip), False, False
    args.fit_scan, args.kpps, args.fps = True, 30000, 25.0
    args.ild_out = None
    cap = cli.open_capture(str(clip))
    try:
        cli.process(cap, args, out_stream=io.StringIO())
    finally:
        cap.release()
    assert args.lines * (args.samples + args.retrace) <= 1200
