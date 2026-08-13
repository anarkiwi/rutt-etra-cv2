# The algorithm, and how faithful it is

## What the hardware did

The Rutt/Etra Video Synthesizer (Rutt Electrophysics, models RE4-A and RE4-B, ©1974)
is an XY vector display driven to paint a raster. The manufacturer's own price list
calls the display unit "a high resolution x/y kinescope display with deflection
circuitry translating synthesizer commands into a standard TV image for re-scanning
with a TV camera". Its output was re-photographed by a camera.

Signal chain, from the block diagram in *Eigenwelt der Apparatewelt* (Ars Electronica,
1992), p.139:

```
VIDEO IN ─┬─→ [x INTENSITY] ─→ [+ BRIGHTNESS] ────────────────→ CRT BEAM   (Z)
          └─→ [SYNC SEP] ─┬→ [H RAMP GEN] → H RAMP
                          └→ [V RAMP GEN] → V RAMP
H RAMP → [x H SIZE] → [x DEPTH] → [+ H POSITION] → H DEFLECTION            (X)
V RAMP → [x V SIZE] → [x DEPTH] → [+ V POSITION] → V DEFLECTION            (Y)
```

The displacement effect is a patch, not a stock signal path: video is patched into the
VERTICAL position input, whose LEVEL attenuator sets the depth. *Eigenwelt* names it
the "Vasulka Effect": "the input video brightness connects to the vertical position
control. This causes the brighter parts of the video to 'pull' the raster lines
upward." The same video simultaneously drives Z, with independent gain.

Three consequences that most emulations miss:

- **Displacement is unipolar from black.** Spielmann, *Video and Computer* (Daniel
  Langlois Foundation, 2004): "the black areas of the image will not be affected
  (these areas are neutral because they lack voltage)". Black does not move.
- **POSITION is summed after the SIZE and DEPTH multipliers**, so displacement
  magnitude does not change when you resize the raster.
- **Brightness varies with beam velocity.** *Eigenwelt* records "the need for
  intensity compensation, to correct for brightness changes due to the speed of the
  beam". Where the trace is steep it is swept faster, so it is written fainter.
  Automatic Intensity Compensation was a purchasable option.

There is no occlusion anywhere in the chain: overlapping traces excite the same
phosphor twice and are integrated by the rescan camera. Crossing lines get brighter
and nothing hides anything. A depth-buffered mesh render is categorically wrong.

Sources: the RE4 operator's manual is scanned at
[archive.org/details/ETC2732](https://archive.org/details/ETC2732) and
[ETC2532](https://archive.org/details/ETC2532); the *Eigenwelt* chapter is at
[vasulka.org](https://www.vasulka.org/archive/eigenwelt/pdf/136-139.pdf).

## The model here

`ruttetra.core` emits deflection geometry, not pixels:

```
x = (2(col + 0.5)/samples - 1) * h_size * depth  + skew * base_y
y = base_y + 2 * gain * luma                       base_y = (1 - 2(row+0.5)/lines) * v_size * depth
z = brightness + intensity * luma
```

`beam_path` flattens that into one ordered sample stream with blanked retrace between
lines. Both outputs render that single path, so they cannot drift apart:
`ruttetra.raster` walks it with an additive antialiased line kernel, and
`ruttetra.audio` resamples it onto the audio clock. `tests/test_audio.py::
test_audio_and_raster_agree` asserts every lit audio sample lands on drawn ink.

`beam` selects the tonal law, and it applies identically to both outputs:

- `rate` — one beam period per sample, matching the constant-rate horizontal ramp.
  Steep segments are swept faster and written fainter. This is the hardware default.
- `speed` — uniform arc length per sample. This is Automatic Intensity Compensation,
  and it is what oscilloscope output usually wants.

## Corrections to the previous implementation

| # | Was | Is | Why |
|---|-----|----|-----|
| 1 | One pixel written per (line, column); nothing between | Continuous antialiased polyline along the beam path | The beam sweeps continuously. The old output was a dot field: on a 320-wide frame a single scanline lit exactly 320 isolated pixels, jumping 62px across a luminance edge with nothing drawn between |
| 2 | `line_spacing = h // num_lines`, then `img[::spacing]` | `cv2.resize` to exactly `lines` rows | `--lines 45` produced 48 lines and `--lines 100` produced 120. `--lines 500` on a 240px frame raised `slice step cannot be zero` |
| 3 | Row decimation | Area averaging (`--sampling nearest` restores decimation) | Decimation aliases and discards most of the picture. The hardware's only line-dropping feature, ALTERNATE LINE, does decimate — hence the option |
| 4 | `np.mean` of B, G, R | Rec.709 luma, `--luma` selectable | The device deflects on the luminance signal. Pure red displaced by 85/255 instead of 76/255 |
| 5 | `prange` over source rows writing a shared output array | Serial deterministic accumulation | Two source rows can map to the same output row. The result depended on thread scheduling |
| 6 | Brightness independent of trace slope | `--beam rate` dims fast segments; `--beam speed` compensates | Documented hardware behaviour, and the reason Intensity Compensation was sold as an option |
| 7 | No Z path | `--intensity` and `--brightness` | `Z = luma x INTENSITY + BRIGHTNESS` is the documented chain |
| 8 | Displacement downward (increasing row index) | Upward | "brighter parts of the video to 'pull' the raster lines upward" |
| 9 | No bounds checking in `njit` kernels | Segments clipped parametrically before stepping | numba does not bounds-check. Clipping also bounded the work: a wild deflection previously drove a million-iteration inner loop |
| 10 | `main()` called at import | `if __name__ == "__main__"` guard, package layout | The module could not be imported or tested |

Two things the previous code already had right, which many published emulations get
wrong: displacement is unipolar from black rather than `luma - 0.5`, and it is applied
in screen space rather than as +Z depth on a perspective-projected mesh.

## Deliberate departures

- `--skew` leans the raster to give the familiar oblique look. The stock RE4 had only
  a 0°/90° ROTATION switch; continuous rotation needed the optional Q18 module, which
  *Eigenwelt* says "remains empty in most units". Default 0.0.
- `--lines` has no hardware counterpart. Line count was fixed by the scan rate
  (525/625/945/1050). The sparse-line look came from raising HEIGHT/DEPTH until most
  lines fell off-screen.
- `--smooth` low-passes the deflection signal along each line, modelling a deflection
  amplifier that rolls off below video bandwidth. Plausible but unverified — no
  bandwidth figure for the RE4 was found. Default 0.0, off.
- Colour is an emulation extra; the display unit was monochrome. Trace brightness
  follows the Z law, so saturated colours read dim, as they would on a luma-driven
  display.

## Not modelled

- **Interlace.** The V ramp ran at field rate, two interleaved fields per frame. This
  renders whole frames progressively.
- **Genlock.** The hardware derived its ramps from the incoming video's sync via a
  sync separator, so sweep and picture drifted together. Here the sweep is exact.
- **Dual trace / alternate line**, which routed odd and even lines through two
  independent control groups.
- **Phosphor persistence** between frames.
