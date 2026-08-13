"""Render a deflection WAV as oscilloscope video."""

import argparse
import sys
import wave

from . import scope


def build_parser():
    """Argument parser for the scope renderer."""
    parser = argparse.ArgumentParser(
        description="Render a Rutt/Etra deflection WAV as an XY oscilloscope."
    )
    flag = argparse.BooleanOptionalAction
    parser.add_argument("wavfile", type=str, help="X/Y(/Z) deflection WAV")
    parser.add_argument("--outfile", default="scope.avi", type=str)
    parser.add_argument(
        "--fps", default=25.0, type=float, help="frames the WAV was written for"
    )
    parser.add_argument("--video", action=flag, default=True)
    parser.add_argument("--monitor", action=flag, default=False)
    parser.add_argument("--z-invert", action=flag, default=False)
    scope.add_arguments(parser.add_argument_group("scope"))
    return parser


def process(args, out_stream=sys.stdout):
    """Render every frame's worth of the WAV, returning the frame count."""
    path = args.outfile if args.video else None
    sink = scope.ScopeSink(path, scope.params_from(args), args.fps, args.monitor)
    if path:
        print(f"opened {path} for writing", file=out_stream)
    try:
        for block in scope.wav_blocks(args.wavfile, args.fps):
            sink.write(block)
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
    except (OSError, ValueError, EOFError, wave.Error) as error:
        print(f"cannot read {args.wavfile}: {error}", file=sys.stderr)
        return 1
    print(f"rendered {frames} frames")
    return 0
