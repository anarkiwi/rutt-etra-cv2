# Laser projector output

The scan processor already produces deflection signals, which is what a laser
projector wants. `--ild-out` writes an ILDA file, `--helios` drives a DAC, and
`--laser-out` renders what the projector would actually put on a wall.

```sh
# ILDA file plus a simulated projection, raster sized to what the scanner can draw
./rutt-etra.py clip.mp4 --no-video --no-monitor --fit-scan \
  --ild-out show.ild --laser-out preview.avi --kpps 30000

# drive a Helios DAC directly
./rutt-etra.py clip.mp4 --no-video --no-monitor --fit-scan --helios

# preview any ILDA file, including ILDA's own test patterns
./rutt-laser.py TEST_30K.ILD --kpps 30000 --fps 21 --outfile scanned.avi
```

## The point budget, which is the whole problem

A projector draws one point at a time. Points per frame is the point rate
divided by the frame rate, and the DAC caps how many it will accept per frame,
so:

```
budget = min(dac_points_per_frame, point_rate / fps)
```

That is a far harder limit than anything else in this project. The audio path
manages 7680 points per frame at 192 kHz; a 30K scanner at 30 fps gives **1000**.

| projector | 30 fps | 25 fps | 20 fps | 12 fps | binds at 12 fps |
|---|---|---|---|---|---|
| cheap 20K | 666 pts, 18x25 | 800 pts, 20x28 | 1000 pts, 23x31 | 1666 pts, 31x41 | scanner |
| cheap 30K | 1000 pts, 23x31 | 1200 pts, 25x36 | 1500 pts, 29x39 | 2500 pts, 39x52 | scanner |
| good 40K | 1333 pts, 27x37 | 1600 pts, 30x41 | 2000 pts, 34x46 | 3333 pts, 45x62 | scanner |
| pro 60K | 2000 pts, 34x46 | 2400 pts, 38x51 | 3000 pts, 43x57 | 4095 pts, 51x68 | **DAC** |
| pro 100K | 2184 pts, 36x48 | 2621 pts, 40x53 | 3276 pts, 45x60 | 4095 pts, 51x68 | **DAC** |

`--fit-scan` picks the raster for you. The rasters above are the worst case, all
points lit; `--blank-points` shortens travel through dark areas, which on sparse
pictures frees more than half the budget.

**Upgrading the projector stops helping once the DAC binds.** On a Helios
(4095 points per frame) a 60K scanner is already at the frame limit by 12 fps,
and a 100K scanner buys nothing below about 16 fps. Above that the scanner binds
again and the faster unit is worth having. Set `--dac-points` and `--dac-rate`
for a different DAC.

The Helios is also **12 bit**, so deflection lands on one of 4096 steps per axis
where an ILDA file carries 16 bit coordinates. `--dac-bits 16` models a DAC that
does not throw those bits away.

## What the scanner does to the picture

Galvanometers are mechanical, so they low pass the point stream. The model is a
second order response, zero order hold discretised at the point rate, per axis.

The bandwidth comes free with the kpps rating, because of how ILDA rates
scanners. The speed test circle in the ILDA pattern is **12 points per
revolution**, so a scanner run at N points per second is being tested at N/12 Hz:

```
scanner bandwidth (Hz) = kpps / 12       30K -> 2500 Hz, 60K -> 5000 Hz
```

The pattern's circle is commanded 1.5545 times the radius of its reference
square and relies on the scanner's own attenuation to shrink it. A projector at
its rating draws the two the same size; a slow one collapses the circle inside
the square. `rutt-laser.py` on a real ILDA test file shows this directly.

```
    commanded          at the rating         far too slow
   O   over the        O  circle meets       o  circle collapsed,
   +---+---+           +---+---+                +--/---+ square skewed
   | square|           | square|                |  \_  |
   +---+---+           +---+---+                +------+
```

Two caveats worth knowing. `--scan-angle` above the 8 degree rating angle costs
bandwidth in proportion, which is why you cannot scan both fast and wide. And
the model puts the rated circle at 0.70 where the pattern's own geometry implies
0.643 — about 9% optimistic, because datasheets quote a −3 dB bandwidth while
the pattern implies scanners run slightly past it. Measure your own hardware and
set `--galvo-hz` and `--damping` if it matters.

`--laser-out` renders brightness from dwell time, so held points at corners burn
bright and fast strokes go dim, as a real beam does. `--no-galvo` renders the
ideal point list instead, which is what most preview software shows.

## Colour

Cheap RGB projectors are blue heavy and red scarce, because blue diodes are the
cheapest. Red is what pulls the mix back toward white, so red is the channel
that limits achievable white and the one the vendor skimped on.

`--calibrate` (on by default) solves for the optical watt ratios that put the
three primaries on D65 and pins the scarcest channel at full drive. For 638, 520
and 450 nm that is roughly **R 1.37 : G 1.00 : B 0.56 optical watts**.

A "2W" unit split 500/500/1000 mW therefore delivers about 500/365/278 mW of
balanced white, so **around half the sticker wattage is unusable**. Blue runs at
about 20% duty and red is pinned at 100%.

With `--no-calibrate` the simulation renders what a stock unit really does at
full drive: chromaticity around (0.21, 0.16), a blue violet, not white. If a
laser simulator shows neutral white at full RGB, its colour model is wrong.

Set `--laser-power` and `--wavelengths` to your unit. The colorimetry uses the
Wyman, Sloan & Shirley analytic fit to the CIE 1931 observer, which agrees with
tabulated values to a few percent; the balance ratios shift by about that much
depending on which table you compare against.

## Driving a Helios

Build `libHeliosDacAPI` from [github.com/Grix/helios_dac](https://github.com/Grix/helios_dac)
and put it on `LD_LIBRARY_PATH`, or pass `--helios-library /path/to/libHeliosDacAPI.so`.
On Linux you also need the udev rule from that repository, or run as root.

The binding sends 8 byte points, unsigned 12 bit X and Y with 8 bit colour and
intensity, at most 4095 per frame, and waits for the DAC to report ready before
each frame. It opens the shutter on connect and closes it on exit.

**Point a laser somewhere safe before running any of this.** A stationary beam
puts the whole frame's energy in one spot; the simulator shows that as a burnt
white dot, and a real projector means it literally.
