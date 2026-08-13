"""Command line front end."""

import argparse
import sys

import cv2

from .audio import (
    AudioParams,
    SoundCardSink,
    WavSink,
    samples_per_frame,
    signals_from_path,
)
from . import laser, scope
from .core import BEAM_MODES, LUMA_COEFFS, SAMPLING, ScanParams, beam_path
from .raster import render_path

DEFAULT_FPS = 25.0


def build_parser():
    """Argument parser for the scan processor."""
    parser = argparse.ArgumentParser(
        description="Rutt/Etra scan processor: video and oscilloscope output."
    )
    flag = argparse.BooleanOptionalAction
    parser.add_argument("infile", type=str, help="video file or camera index")

    scan = parser.add_argument_group("deflection")
    scan.add_argument("--lines", default=60, type=int, help="scan lines per frame")
    scan.add_argument(
        "--scale", default=0.1, type=float, help="displacement, fraction of height"
    )
    scan.add_argument(
        "--samples", default=0, type=int, help="samples per line (0 = source width)"
    )
    scan.add_argument("--luma", default="bt709", choices=sorted(LUMA_COEFFS))
    scan.add_argument(
        "--sampling", default="area", choices=sorted(SAMPLING), help="line resampling"
    )
    scan.add_argument("--color", action=flag, default=True, help="colour the trace")
    scan.add_argument("--invert", action=flag, default=False)
    scan.add_argument("--intensity", default=1.0, type=float, help="Z-axis gain")
    scan.add_argument("--brightness", default=0.0, type=float, help="Z-axis pedestal")
    scan.add_argument("--v-size", default=1.0, type=float, help="vertical raster size")
    scan.add_argument("--h-size", default=1.0, type=float, help="horizontal size")
    scan.add_argument("--depth", default=1.0, type=float, help="uniform size")
    scan.add_argument("--skew", default=0.0, type=float, help="raster lean, extension")
    scan.add_argument(
        "--smooth", default=0.0, type=float, help="deflection bandwidth limit"
    )
    scan.add_argument(
        "--beam", default="rate", choices=BEAM_MODES, help="constant rate or speed"
    )
    scan.add_argument("--serpentine", action=flag, default=True)
    scan.add_argument(
        "--retrace", default=8, type=int, help="blanked interline samples"
    )

    video = parser.add_argument_group("video output")
    video.add_argument("--outfile", default="output.avi", type=str)
    video.add_argument("--video", action=flag, default=True)
    video.add_argument("--monitor", action=flag, default=True)
    video.add_argument("--exposure", default=1.0, type=float, help="trace brightness")
    video.add_argument("--fps", default=0.0, type=float, help="override source rate")

    audio = parser.add_argument_group("oscilloscope output")
    audio.add_argument("--wav", default=None, type=str, help="write deflection WAV")
    audio.add_argument(
        "--audio-device", default=None, type=str, help="sound card name or index"
    )
    audio.add_argument("--audio-rate", default=96000, type=int)
    audio.add_argument(
        "--audio-channels", default=2, type=int, choices=(2, 3), help="XY or XYZ"
    )
    audio.add_argument("--audio-bits", default=16, type=int, choices=(16, 24))
    audio.add_argument("--z-invert", action=flag, default=False)

    projector = parser.add_argument_group("laser projector")
    projector.add_argument("--ild-out", default=None, type=str, help="write ILDA file")
    projector.add_argument(
        "--laser-out", default=None, type=str, help="render a simulated projection"
    )
    projector.add_argument("--helios", action=flag, default=False, help="drive a DAC")
    projector.add_argument("--helios-device", default=0, type=int)
    projector.add_argument("--helios-library", default=None, type=str)
    projector.add_argument("--laser-size", default=480, type=int)
    projector.add_argument("--laser-gain", default=1.0, type=float)
    projector.add_argument(
        "--galvo", action=flag, default=True, help="model scanner dynamics"
    )
    projector.add_argument(
        "--fit-scan", action=flag, default=False, help="size the raster to the budget"
    )
    laser.add_arguments(projector)

    simulated = parser.add_argument_group("simulated oscilloscope")
    simulated.add_argument(
        "--scope-out", default=None, type=str, help="render a scope view to this file"
    )
    scope.add_arguments(simulated)
    return parser


def scan_params(args):
    """Build deflection settings from parsed arguments."""
    return ScanParams(
        lines=args.lines,
        gain=args.scale,
        samples=args.samples,
        luma=args.luma,
        sampling=args.sampling,
        mono=not args.color,
        invert=args.invert,
        intensity=args.intensity,
        brightness=args.brightness,
        v_size=args.v_size,
        h_size=args.h_size,
        depth=args.depth,
        skew=args.skew,
        smooth=args.smooth,
        serpentine=args.serpentine,
        retrace=args.retrace,
        beam=args.beam,
    )


def open_capture(infile):
    """Open a file path or numeric camera index."""
    try:
        return cv2.VideoCapture(int(infile))
    except (TypeError, ValueError):
        return cv2.VideoCapture(str(infile))


def source_fps(cap, override):
    """Frame rate to drive both the video writer and the audio clock."""
    if override > 0:
        return override
    reported = cap.get(cv2.CAP_PROP_FPS)
    return reported if reported and reported > 0 else DEFAULT_FPS


def device_id(name):
    """Sound card index if numeric, otherwise the name itself."""
    try:
        return int(name)
    except (TypeError, ValueError):
        return name


def warn_undersampled(block, points):
    """Warn that the audio clock cannot resolve every point of the beam path."""
    return (
        f"warning: {block} audio samples per frame for a {points}-point beam path; "
        "the scope will skip detail. Raise --audio-rate, or lower --lines/--samples "
        f"so that lines x (samples + retrace) <= {block}."
    )


def open_laser_sinks(args, fps, aspect):
    """Projector sinks requested on the command line."""
    projector, dac = laser.params_from(args)
    sinks = []
    if args.ild_out:
        sinks.append(laser.IldSink(args.ild_out))
    if args.laser_out:
        sinks.append(
            laser.PreviewSink(
                args.laser_out,
                projector,
                laser.PreviewParams(
                    size=args.laser_size,
                    aspect=aspect,
                    gain=args.laser_gain,
                    show_galvo=args.galvo,
                ),
                fps,
            )
        )
    if args.helios:
        from . import helios  # pylint: disable=import-outside-toplevel

        sinks.append(
            helios.HeliosSink(projector, args.helios_device, args.helios_library)
        )
    return projector, dac, sinks


def open_sinks(args, audio, fps):
    """Deflection sinks requested on the command line."""
    sinks = []
    if args.scope_out:
        sinks.append(scope.ScopeSink(args.scope_out, scope.params_from(args), fps))
    if args.wav:
        sinks.append(WavSink(args.wav, audio))
    if args.audio_device is not None:
        sinks.append(SoundCardSink(audio, device_id(args.audio_device)))
    return sinks


def preview(args, params, path, shape):
    """Rasterise a frame if any video output needs it, showing it if asked."""
    if not (args.video or args.monitor):
        return None
    image = render_path(path, shape[0], shape[1], params, args.exposure)
    if args.monitor:
        cv2.imshow("rutt etra", image)
        cv2.waitKey(1)
    return image


def open_writer(args, image, fps, out_stream):
    """Open the video writer once the output size is known."""
    print(f"opened {args.outfile} for writing", file=out_stream)
    return cv2.VideoWriter(
        args.outfile,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (image.shape[1], image.shape[0]),
    )


def feed_audio(sinks, path, params, audio, block, first, out_stream):
    """Resample a beam path and hand it to every deflection sink."""
    if not sinks:
        return
    if first and block < len(path):
        print(warn_undersampled(block, len(path)), file=out_stream)
    signals = signals_from_path(path, params, audio, block)
    for sink in sinks:
        sink.write(signals)


def feed_laser(sinks, path, projector_dac, fps, first, out_stream):
    """Fit a beam path to the projector budget and hand it to every sink."""
    if not sinks:
        return
    projector, dac = projector_dac
    shot = laser.frame_points(path, projector, dac, fps)
    if first:
        print(laser.describe(projector, dac, fps, shot[0].shape[0]), file=out_stream)
    for sink in sinks:
        sink.write(*shot)


def process(cap, args, out_stream=sys.stdout):
    """Run the capture through the scan processor into every requested sink."""
    fps = source_fps(cap, args.fps)
    if args.fit_scan:
        args.lines, args.samples = laser.fit_scan(
            *laser.params_from(args), fps, retrace=args.retrace
        )
    params = scan_params(args)
    audio = AudioParams(
        rate=args.audio_rate,
        channels=args.audio_channels,
        bits=args.audio_bits,
        z_invert=args.z_invert,
    )
    block = samples_per_frame(audio.rate, fps)
    sinks = open_sinks(args, audio, fps)
    projector, dac, beam_sinks = open_laser_sinks(args, fps, 1.0)
    writer, frames = None, 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            height, width = frame.shape[:2]
            path = beam_path(frame, params)
            feed_audio(sinks, path, params, audio, block, frames == 0, out_stream)
            feed_laser(beam_sinks, path, (projector, dac), fps, frames == 0, out_stream)
            image = preview(args, params, path, (height, width))
            if args.video and image is not None:
                writer = writer or open_writer(args, image, fps, out_stream)
                writer.write(image)
            frames += 1
    except KeyboardInterrupt:
        pass
    finally:
        for sink in sinks + beam_sinks:
            sink.close()
        if writer is not None:
            writer.release()
    return frames


def main(argv=None):
    """Entry point."""
    args = build_parser().parse_args(argv)
    cap = open_capture(args.infile)
    if not cap.isOpened():
        print(f"cannot open {args.infile}", file=sys.stderr)
        return 1
    try:
        frames = process(cap, args)
    finally:
        cap.release()
    print(f"processed {frames} frames")
    return 0
