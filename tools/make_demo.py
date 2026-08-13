#!/usr/bin/env python3
"""Generate the demo clip and the animated PNGs the README embeds.

Run from the repository root: python3 tools/make_demo.py
"""

import argparse
import dataclasses
import pathlib
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# pylint: disable=wrong-import-position
from ruttetra.audio import AudioParams, WavSink, frame_signals  # noqa: E402
from ruttetra.core import ScanParams, beam_path  # noqa: E402
from ruttetra.raster import render  # noqa: E402
from ruttetra.scope import ScopeParams, render_wav  # noqa: E402
from ruttetra import laser  # noqa: E402

WIDTH, HEIGHT, FRAMES, FPS = 256, 192, 36, 24.0
SCAN = {"lines": 48, "samples": 152, "gain": 0.35}
RATE = 192000
SCOPE_GAIN = 3.0


def shape_frame(step):
    """One frame of a vector drawing that rotates and translates on a loop."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    phase = 2.0 * np.pi * step / FRAMES
    centre = np.array(
        [WIDTH / 2 + 36 * np.cos(phase), HEIGHT / 2 + 24 * np.sin(2 * phase)]
    )
    for sides, radius, spin, color in (
        (4, 50.0, 1.0, (255, 210, 90)),
        (3, 32.0, -1.6, (120, 245, 255)),
        (6, 66.0, 0.5, (200, 130, 255)),
    ):
        angle = phase * spin + np.linspace(0, 2 * np.pi, sides, endpoint=False)
        points = centre + radius * np.stack([np.cos(angle), np.sin(angle)], axis=1)
        cv2.polylines(
            frame, [np.round(points).astype(np.int32)], True, color, 3, cv2.LINE_AA
        )
    cv2.circle(frame, tuple(np.round(centre).astype(int)), 3, (255, 255, 255), -1)
    return frame


def write_clip(path):
    """Write the demo clip and return its frames."""
    frames = [shape_frame(i) for i in range(FRAMES)]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    for frame in frames:
        writer.write(frame)
    writer.release()
    return frames


def write_apng(path, frames, colors=48):
    """Save BGR frames as a looping animated PNG."""
    images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    if colors:
        images = [
            im.quantize(
                colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
            )
            for im in images
        ]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(round(1000.0 / FPS)),
        loop=0,
        optimize=True,
    )
    return path.stat().st_size


def scope_frames(frames, channels, params, scope_params, workdir):
    """Route frames through a real WAV and render the simulated scope."""
    audio = AudioParams(rate=RATE, channels=channels, bits=16)
    block = int(round(RATE / FPS))
    wav = workdir / f"deflection{channels}.wav"
    sink = WavSink(wav, audio)
    for frame in frames:
        sink.write(frame_signals(frame, params, audio, block))
    sink.close()
    return list(render_wav(wav, FPS, scope_params))


def laser_frames(frames):
    """Route frames through a projector budget and simulate the projection."""
    projector, dac = laser.Projector(kpps=30000), laser.Dac()
    lines, samples = laser.fit_scan(projector, dac, FPS, retrace=4)
    params = ScanParams(lines=lines, samples=samples, gain=SCAN["gain"], retrace=4)
    screen = laser.PreviewParams(size=259, aspect=256 / 259, gain=5.0, spot=1.0)
    out = []
    for frame in frames:
        shot = laser.frame_points(beam_path(frame, params), projector, dac, FPS)
        out.append(laser.preview(*shot, projector, screen))
    return out


def main():
    """Build every asset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="docs/media", type=str)
    parser.add_argument("--scope-gain", default=SCOPE_GAIN, type=float)
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    source = write_clip(outdir / "vector-test.mp4")

    rate = ScanParams(**SCAN)
    speed = ScanParams(**SCAN, beam="speed")
    mono = ScanParams(**SCAN, beam="speed", mono=True)
    out_h, out_w = render(source[0], rate).shape[:2]
    screen = ScopeParams(
        size=out_h,
        aspect=out_w / out_h,
        gain=args.scope_gain,
        spot=0.7,
        bloom=0.25,
        persistence=0.45,
    )

    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp)
        assets = {
            "source": source,
            "raster-color": [render(f, rate, exposure=3.0) for f in source],
            "raster-speed": [render(f, speed) for f in source],
            "raster-mono": [render(f, mono) for f in source],
            "scope-xy": scope_frames(
                source, 2, speed, dataclasses.replace(screen, z=False), work
            ),
            "scope-xyz": scope_frames(source, 3, speed, screen, work),
            "laser": laser_frames(source),
        }

    for name, frames in assets.items():
        colors = 24 if "mono" in name or "scope" in name else 48
        size = write_apng(outdir / f"{name}.png", frames, colors)
        print(f"{name:14s} {len(frames):3d} frames  {size / 1024:7.1f} kB")


if __name__ == "__main__":
    main()
