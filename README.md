This repo is vibe coded!!!
# MBTransient

A multiband transient shaper for [Rust](https://www.rust-lang.org/) built with
[NIH-plug](https://github.com/robbert-vdh/nih-plug), inspired by
[LHI Audio ST4b](https://lhiaudio.com/st4b/). Exports **VST3** and **CLAP**.

## What it does

The signal is split into four bands with a phase-coherent
Linkwitz-Riley 24 dB/oct crossover. Each band has a single shaping value that
sweeps from "cut the attack" (soft) through neutral to "cut the sustain"
(punchy). The bands are recombined flat (the crossover is all-pass phase
compensated), so there is no comb filtering.

The transient detector tracks a fast peak envelope and a slower body envelope;
the difference between the two decides, per sample, how much of the current
level is an attack transient versus a sustained/decaying body. `speed` controls
how fast the envelopes recover.

## Parameters

| Parameter | Range | Description |
| --- | --- | --- |
| shape (per band) | soft ↔ neutral ↔ punchy | One value per band; sweeps `att cut` → `att/sus unity` → `sus cut` |
| bands | 2 … 6 | Number of bands (managed from the GUI: right-click deletes, double-click adds) |
| speed | 2 … 200 ms | Envelope recovery time (lower = snappier) |
| level | -30 … +6 dB | Output trim |

The crossovers are adjusted by dragging the vertical lines on the spectrum.
Dragging (or the mouse wheel) on a band's region changes that band's shape; a
horizontal bar inside each band shows the current shape.

## GUI

A compact, flat, muted light-gray editor with a dot-grid background and Ubuntu
Mono throughout: one horizontal spectrum (a connected output curve with a
gradient fill fading toward the bottom), horizontal per-band shape bars, and
`speed`/`level` as plain text readouts below. The interface scales
proportionally with the window. Spectrum curve colour encodes shaping per band
(light gray = soft, mid gray = neutral, near black = punchy).

`nih_plug_egui` is vendored under `vendor/` with a one-line fix so the in-window
resize corner uses the correct physical pixel size on HiDPI (Retina) displays.

## Building

```sh
cargo xtask bundle mbtransient --release
```

This produces:

- `target/bundled/MBTransient.vst3`
- `target/bundled/MBTransient.clap`

Copy those into your DAW's plugin directory.

## Layout

- `src/lib.rs` — plugin, parameters, audio processing, GUI.
- `src/dsp.rs` — biquad, crossover, transient-shaper and FFT spectrum-analyzer DSP.
- `assets/fonts/` — bundled Ubuntu Mono font.
- `vendor/nih_plug_egui` — vendored egui adapter with the Retina resize fix.
