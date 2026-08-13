# Oscilloscope output

The scan processor is an XY vector display, so its deflection signals can drive a
scope directly. `--wav` writes them to a file; `--audio-device` streams them live.

## Channel map

| Channel | Signal | Scope input |
|---|---|---|
| 1 (left) | X, horizontal ramp | CH1 / X |
| 2 (right) | Y, ramp + luminance displacement | CH2 / Y |
| 3 | Z, beam current, `-1` blanked | Z / intensity input |

Set the scope to XY mode. Use `--audio-channels 2` for a plain XY scope and
`--audio-channels 3` if it has a Z input. `--z-invert` suits scopes whose blanking
input is active high.

```sh
# two channel XY, to a file
./rutt-etra.py clip.mp4 --no-video --wav scope.wav --audio-channels 2

# three channel XYZ, live, no preview window
./rutt-etra.py clip.mp4 --no-video --no-monitor \
  --audio-device "Scarlett 2i2" --audio-channels 3 --audio-rate 192000
```

## Sample budget

This is the setting that matters most. One video frame becomes `audio-rate / fps`
samples, and the beam path is `lines x (samples + retrace)` points. If the budget is
smaller than the path, the scope skips detail and the picture breaks into dashes. The
CLI warns when this happens.

At 96 kHz and 25 fps you get 3840 samples per frame. A 60-line, 320-sample raster
needs 19680 points — five times too many. Either raise the rate or cut the raster:

```sh
# 192 kHz / 25 fps = 7680 samples; 48 x (152 + 8) = 7680. Exact fit.
./rutt-etra.py clip.mp4 --no-video --wav scope.wav \
  --audio-rate 192000 --lines 48 --samples 152 --beam speed
```

## Settings that matter on a scope

- `--beam speed` spaces samples evenly along the trace, giving even brightness. The
  default `rate` is faithful to the hardware but leaves fast segments dim.
- `--serpentine` (on by default) sweeps alternate lines in opposite directions, so the
  beam never flies back across the screen. Turn it off only if you want the flyback.
- `--retrace` sets how many blanked samples bridge each line. Raise it if the vertical
  jumps between lines are visibly streaking.
- `--audio-bits 24` gives finer deflection steps than 16-bit if your interface takes it.

## Simulated scope

`rutt-scope.py` renders a deflection WAV as a monochrome XY scope, so you can see the
result without hardware. It reuses the raster's beam kernel, so brightness falls with
beam speed exactly as it does on a real tube, then adds spot size, bloom and phosphor
decay.

```sh
./rutt-scope.py scope.wav --outfile scope-view.avi --fps 23.976 \
  --scope-size 494 --scope-aspect 1.449 --scope-gain 8 \
  --scope-persistence 0.6 --scope-spot 0.7 --scope-bloom 0.25
```

The same view can be produced in one pass, straight from the deflection signals:

```sh
./rutt-etra.py clip.mp4 --scope-out scope-view.avi --no-video --no-monitor \
  --audio-rate 192000 --lines 48 --samples 152 --beam speed
```

| Control | |
|---|---|
| `--scope-size` / `--scope-aspect` | screen height, and width divided by height (1.0 = square, as a scope is) |
| `--scope-gain` | trace brightness |
| `--scope-spot` / `--scope-bloom` | beam spot sigma, and the halo around it |
| `--scope-persistence` | phosphor decay per frame, 0 to just under 1 |
| `--scope-z` / `--no-scope-z` | use channel 3 as beam current, or run flat out |
| `--scope-graticule` | overlay the 8 x 10 division grid |

**`--scope-gain` needs setting per configuration, and that is not a bug.** Each sample
deposits one beam period's worth of energy however far it travels, so brightness
depends on how many samples land per screen pixel. Doubling `--scope-size` at a fixed
sample rate spreads the same energy over four times the area. Turning `--no-scope-z`
on also brightens everything, because the picture is no longer modulating the beam —
and it reveals the flat undisplaced raster behind the trace.

## Sound card caveats

Audio outputs are **AC coupled**. Constant deflection decays toward centre, so large
flat areas drift and the picture leans. Steve Rutt made the same point about the
original hardware: "we DC coupled everything which had been AC coupled. That was the
main thing. Without that you couldn't get positional movement, you could only get
waveform distortion."

Options, in order of effort: use a DC-coupled interface; use a scope with AC coupling
off and accept the droop; or keep `--v-size` and `--h-size` modest so the signal stays
near centre. Cheap interfaces also low-pass well below Nyquist, which rounds the
raster corners.

Start the volume low. These are full-scale signals into an amplifier input.
