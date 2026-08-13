# The algorithm, and how faithful it is

## What the hardware did

The Rutt/Etra Video Synthesizer (Rutt Electrophysics, models RE4-A and RE4-B, ©1974)
is an XY vector display driven to paint a raster. The manufacturer's own price list
calls the display unit "a high resolution x/y kinescope display with deflection
circuitry translating synthesizer commands into a standard TV image for re-scanning
with a TV camera". Its output was re-photographed by a camera.

### The RE4 signal chain

Redrawn from the block diagram in *Eigenwelt der Apparatewelt* (Ars Electronica,
1992), p.139, with control names as they appear on the Display Control Unit panel.
The dashed line is the patch cord that makes the effect.

```
                              ┌────────────────────────────────────────────┐
                              │            DISPLAY CONTROL UNIT            │
                              └────────────────────────────────────────────┘

                 ┌───────────┐   ┌────────────┐
 VIDEO IN ──┬───►│ x INTENSITY├──►│+ BRIGHTNESS├───────────────────────► Z  (beam
            │    └───────────┘   └────────────┘                              current)
            │
            ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
            │        the "Vasulka Effect" patch: video into    ╎
            │        the VERTICAL jack, LEVEL sets the depth   ╎
            │                                                  ▼
            │    ┌─────────┐   ┌────────┐   ┌───────┐   ┌────────────┐
            │ ┌─►│ H RAMP  ├──►│ x WIDTH├──►│ x DEPTH├─►│+ HORIZONTAL├──► X  (H
            │ │  │  GEN    │   └────────┘   └───────┘   └────────────┘       yoke)
            │ │  └─────────┘
            └►│ SYNC SEP                                              genlocked to
              │  ┌─────────┐   ┌─────────┐  ┌───────┐   ┌────────────┐  the source
              └─►│ V RAMP  ├──►│ x HEIGHT├─►│ x DEPTH├─►│+ VERTICAL  ├──► Y  (V
                 │  GEN    │   └─────────┘  └───────┘   └────────────┘       yoke)
                 └─────────┘

        X, Y, Z ──► high resolution B&W x/y kinescope ──► rescan TV camera ──► out
```

Note the order: POSITION is summed **after** the SIZE and DEPTH multipliers, so
displacement magnitude does not change when the raster is resized. VERTICAL CENTER,
by contrast, is summed before the HEIGHT multiplier — the manual describes it as
adjusting "the Height control's 0 voltage point".

The displacement effect is a patch, not a stock signal path: video is patched into the
VERTICAL position input, whose LEVEL attenuator sets the depth. Every one of the eight
DCU parameters has a BIAS knob, a LEVEL attenuator and an external patch jack — "Level
controls act as potentiometers, allowing more or less voltage from outside sources to
pass to the deflection circuits". *Eigenwelt* names the result the "Vasulka Effect":
"the input video brightness connects to the vertical position control. This causes the
brighter parts of the video to 'pull' the raster lines upward." The same video
simultaneously drives Z, with independent gain.

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

### RE4 front panel

Eight parameters, each with a BIAS knob, a LEVEL attenuator and a patch jack, in two
independent control groups. All BIAS pots are unmarked multi-turn; the manual's
"Synthesizer Zen (You Become the Feedback Circuit)" section apologises for this.

| Control | Function, in the manual's words | Range |
|---|---|---|
| HEIGHT | "varies the amplitude of the vertical sweep"; through zero inverts | -10 to +10 V |
| WIDTH | "varies the amplitude of the horizontal sweep"; through zero inverts | -10 to +10 V |
| DEPTH | "varies the height and width simultaneously, causing the image to appear to advance or recede" | 0 to +10 V |
| INTENSITY | Z-axis gain; "should always be used together" with Depth to avoid phosphor burn | 0 to +10 V |
| HORIZONTAL | moves the whole raster left and right | ±10 V |
| H. CENTER | "controls the horizontal phase of the synthesizer in relation to the incoming video" | ±10 V |
| VERTICAL | pans the raster up and down; this is where video is patched | ±10 V |
| V. CENTER | sets HEIGHT's zero crossing; BIAS only, no jack | ±10 V |

Switches: BLACK LEVEL, DUAL TRACE (position / alternate line), ROTATION 0°/90°, H and
V SYNC INT/EXT, VIDEO SYNC H/V, SCAN RATE A/B. Plug-in modules included the Q7
multiplier, Q8 summing amplifier, Q10 ramp generator, Q12 joystick, Q17 waveform
generator and the optional Q18 angular rotation module.

There is **no line count, line skip, perspective or negative control**. Scan rate was
factory-fitted from 525 / 625 / 945 / 1050 lines, switchable A/B.

### References

- Rutt Electrophysics Corp., *Video Synthesizer Systems, Models RE4-A and RE4-B*,
  operator's manual, ©1974 — scanned at
  [archive.org/details/ETC2732](https://archive.org/details/ETC2732) and
  [ETC2532](https://archive.org/details/ETC2532). Includes the front panel drawing and
  the "VARIABLE FUNCTIONS of the R/E-4 DCU" spec sheet.
- *Eigenwelt der Apparatewelt: Pioneers of Electronic Art* (Ars Electronica, 1992),
  "Bill Etra & Steve Rutt: Rutt/Etra Scan Processor (Analog), 1973", pp.136-139 —
  [PDF](https://www.vasulka.org/archive/eigenwelt/pdf/136-139.pdf), OCR at
  [ETC2207](https://archive.org/details/ETC2207). Source of the block diagram, the
  "Vasulka Effect" passage and the intensity-compensation note.
- Yvonne Spielmann, *Video and Computer: The Aesthetics of Steina and Woody Vasulka*
  (Daniel Langlois Foundation, 2004) —
  [tool page](https://www.fondation-langlois.org/html/e/page.php?NumPage=456). Source
  of the black-is-neutral point and the raster / line-deflection distinction.
- Experimental Television Center, *Rutt/Etra: Notes on Development* —
  [videohistoryproject.org](https://www.videohistoryproject.org/ruttetra-notes-development).
- Interviews with Bill Etra ([ETC1786](https://archive.org/details/ETC1786)) and Steve
  Rutt ([ETC1790](https://archive.org/details/ETC1790)). Etra: "The first machine we
  built was really deflection on a regular oscilloscope... all you need is a locked
  vertical and horizontal ramp and multipliers and summing amplifiers into it to
  create the Rutt/Etra type effects." Rutt: "we DC coupled everything which had been
  AC coupled."
- RE4 schematics, if anyone wants to settle the deflection bandwidth question:
  [ETC2428](https://archive.org/details/ETC2428),
  [ETC1742](https://archive.org/details/ETC1742),
  [ETC3224](https://archive.org/details/ETC3224). These are page images; no OCR will
  do it.
- Woody Vasulka & Scott Nygren, "Didactic Video: Organizational Models of the
  Electronic Image", *Afterimage* 3, no. 4 (October 1975). Not open access; cited but
  not consulted here.

## The model here

`ruttetra.core` emits deflection geometry, not pixels:

```
x = (2(col + 0.5)/samples - 1) * h_size * depth  + skew * base_y
y = base_y + 2 * gain * luma                       base_y = (1 - 2(row+0.5)/lines) * v_size * depth
z = brightness + intensity * luma
```

`beam_path` flattens that into one ordered sample stream with blanked retrace between
lines. Every output renders that single path, so they cannot drift apart:

```
                      ┌──────────────────── core.py ────────────────────┐
                      │                                                 │
  frame ──► scan_lines ──► luma ──┬──► + into Y  ──┐                    │
  (BGR)     (area avg)            │                ├──► deflection ──►  │
                                  └──► x INTENSITY ┘    x, y, z, colour │
                                       + BRIGHTNESS ──► Z               │
                      │                                    │            │
                      │            beam_path: serpentine order,         │
                      │            blanked retrace between lines        │
                      └────────────────────┬────────────────────────────┘
                                           │
                                      BeamPath
                        ┌──────────────────┼──────────────────┐
                        ▼                  ▼                  ▼
                 raster.py           audio.py           (audio.py)
                 draw_segments       beam_clock +        WavSink /
                 additive AA         np.interp           SoundCardSink
                 line kernel         resample                │
                        │                  │                 ▼
                        ▼                  ▼             X/Y/Z WAV
                   video frame        X/Y/Z block            │
                                           │                 │
                                           └──────┬──────────┘
                                                  ▼
                                             scope.py
                                        draw_segments again,
                                        + spot blur, bloom,
                                          phosphor decay
                                                  │
                                                  ▼
                                        B&W oscilloscope frame
```

`ruttetra.scope` reuses the same `draw_segments` kernel as the raster, with
`per_length=False`, so a simulated scope obeys exactly the tonal law the hardware
does. Two tests pin the agreement down: `test_audio_and_raster_agree` asserts every
lit audio sample lands on drawn ink, and `test_scope_reproduces_the_raster_geometry`
asserts the scope view of the signal — after a round trip through PCM — lands on the
shape the raster drew.

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
