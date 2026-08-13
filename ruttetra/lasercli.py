"""Render an ILDA file as a simulated laser projection."""

import argparse
import struct
import sys

import numpy as np

from . import ilda, laser


def build_parser():
    """Argument parser for the laser renderer."""
    parser = argparse.ArgumentParser(
        description="Render an ILDA file as a simulated laser projection."
    )
    flag = argparse.BooleanOptionalAction
    parser.add_argument("ildfile", type=str, help="ILDA .ild file")
    parser.add_argument("--outfile", default="laser.avi", type=str)
    parser.add_argument("--fps", default=25.0, type=float)
    parser.add_argument("--video", action=flag, default=True)
    parser.add_argument("--monitor", action=flag, default=False)
    parser.add_argument("--repeat", default=1, type=int, help="times to loop the file")
    parser.add_argument("--laser-size", default=480, type=int)
    parser.add_argument("--laser-gain", default=1.0, type=float)
    parser.add_argument("--galvo", action=flag, default=True)
    laser.add_arguments(parser.add_argument_group("laser projector"))
    return parser


def frame_stream(frames, projector, dac, repeat):
    """Yield projector points for every frame, looped."""
    scale = float(ilda.COORD_MAX)
    for _ in range(max(1, repeat)):
        for frame in frames:
            unit = np.stack(
                [dac.quantise(frame.x / scale), dac.quantise(frame.y / scale)], axis=1
            )
            yield unit, frame.blank, frame.color / 255.0 * projector.drive


def process(args, out_stream=sys.stdout):
    """Render every frame of the file, returning the frame count."""
    frames = ilda.read(args.ildfile)
    if not frames:
        return 0
    projector, dac = laser.params_from(args)
    params = laser.PreviewParams(
        size=args.laser_size, gain=args.laser_gain, show_galvo=args.galvo
    )
    sink = laser.PreviewSink(
        args.outfile if args.video else None, projector, params, args.fps, args.monitor
    )
    print(
        f"{args.ildfile}: {len(frames)} frames, "
        f"{len(frames[0])} points in the first; "
        f"{laser.describe(projector, dac, args.fps, len(frames[0]))}",
        file=out_stream,
    )
    try:
        for shot in frame_stream(frames, projector, dac, args.repeat):
            sink.write(*shot)
    except KeyboardInterrupt:
        pass
    finally:
        sink.close()
    return sink.frames


def main(argv=None):
    """Entry point."""
    args = build_parser().parse_args(argv)
    try:
        frames = process(args)
    except (OSError, ValueError, struct.error) as error:
        print(f"cannot read {args.ildfile}: {error}", file=sys.stderr)
        return 1
    print(f"rendered {frames} frames")
    return 0
