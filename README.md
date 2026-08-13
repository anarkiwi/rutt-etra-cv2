# rutt-etra-cv2

Rutt/Etra scan processor. Displaces each scan line vertically by video luminance and
renders the result as video, as oscilloscope XY(Z) audio, or both from one pass.

The device is an XY vector display, so the core model produces deflection signals
rather than pixels. Video and audio are two renderings of the same beam path.

![raster and oscilloscope output of the same frame](docs/scope-vs-raster.png)

## Install

```sh
pip install -r requirements.txt
pip install sounddevice   # optional, only for live sound card output
```

## Use

```sh
./rutt-etra.py input.mp4                                    # video, with preview
./rutt-etra.py 0 --lines 90 --scale 0.4                     # webcam
./rutt-etra.py input.mp4 --no-video --wav scope.wav \
  --audio-rate 192000 --lines 48 --samples 152 --beam speed # oscilloscope
./rutt-scope.py scope.wav --outfile scope-view.avi          # simulate a scope
```

`--outfile` sets the video path, `--wav` the audio path, `--audio-device` streams to a
sound card, `--scope-out` renders the simulated scope. Any combination can run at once.

`rutt-scope.py` renders a deflection WAV as a monochrome XY oscilloscope, reusing the
same beam kernel as the raster so brightness falls with beam speed as it does on a
real tube, then adding spot size, bloom and phosphor decay.

## Options

| Deflection | |
|---|---|
| `--lines` | scan lines per frame (60) |
| `--scale` | displacement as a fraction of frame height (0.1) |
| `--samples` | samples per line, 0 = source width |
| `--luma` | `bt709`, `bt601`, `mean` |
| `--sampling` | `area` averages lines, `nearest` decimates |
| `--intensity` / `--brightness` | Z-axis gain and pedestal |
| `--v-size` / `--h-size` / `--depth` | raster size, vertical, horizontal, both |
| `--skew` | lean the raster (not an original control) |
| `--smooth` | limit deflection bandwidth along each line |
| `--beam` | `rate` dims fast segments, `speed` compensates |
| `--serpentine` | sweep alternate lines in reverse (on) |
| `--retrace` | blanked samples between lines (8) |
| `--color` / `--invert` | colour the trace, invert the picture |

| Output | |
|---|---|
| `--outfile` / `--video` / `--monitor` | video file, enable, preview window |
| `--exposure` / `--fps` | trace brightness, override source rate |
| `--wav` / `--audio-device` | file or sound card |
| `--audio-rate` / `--audio-channels` / `--audio-bits` | 96000, 2 (XY) or 3 (XYZ), 16 or 24 |
| `--z-invert` | for scopes whose blanking is active high |
| `--scope-out` | render a simulated scope view to this file |
| `--scope-size` / `--scope-aspect` | screen height, width / height (1.0 = square) |
| `--scope-gain` / `--scope-spot` / `--scope-bloom` | brightness, beam spot, halo |
| `--scope-persistence` / `--scope-z` / `--scope-graticule` | phosphor decay, Z channel, grid |

## Docs

- [docs/algorithm.md](docs/algorithm.md) — the signal model, its sources, and how
  faithful it is to the RE4 hardware
- [docs/oscilloscope.md](docs/oscilloscope.md) — channel map, sample budget, sound
  card caveats

## Develop

```sh
pip install -r requirements-dev.txt
pytest                        # 149 tests, coverage gate at 85%
black --check . && pylint ruttetra tests
docker build -t ruttetra-test . && docker run --rm ruttetra-test
```
