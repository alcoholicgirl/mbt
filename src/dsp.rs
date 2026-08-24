//! DSP core for MBTransient: a Linkwitz-Riley 24 dB/oct multiband crossover
//! (up to six bands), a per-band transient shaper, and an FFT-based spectrum
//! analyzer.

use std::f32::consts;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use nih_plug::prelude::AtomicF32;
use realfft::num_complex::Complex32;
use realfft::{RealFftPlanner, RealToComplex};

/// Maximum number of frequency bands.
pub const MAX_BANDS: usize = 6;
/// Maximum number of crossovers (one fewer than the number of bands).
pub const MAX_CROSSOVERS: usize = MAX_BANDS - 1;

/// Butterworth Q used to build the Linkwitz-Riley (cascaded) crossover sections.
const Q: f32 = consts::FRAC_1_SQRT_2;

/// The fixed time (in seconds) used to smooth the per-band transient/sustain gain.
const GAIN_SMOOTH_TIME: f32 = 0.003;

// ---------------------------------------------------------------------------
// Biquad
// ---------------------------------------------------------------------------

/// Pre-normalized (divided by `a0`) transposed direct form II biquad coefficients.
#[derive(Clone, Copy, Debug, Default)]
pub struct BiquadCoeffs {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
}

/// Per-channel biquad state.
#[derive(Clone, Copy, Debug, Default)]
pub struct BiquadState {
    pub z1: f32,
    pub z2: f32,
}

impl BiquadCoeffs {
    pub fn identity() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }
    }

    pub fn lowpass(sample_rate: f32, frequency: f32) -> Self {
        let omega0 = consts::TAU * (frequency / sample_rate);
        let cos = omega0.cos();
        let alpha = omega0.sin() / (2.0 * Q);
        let a0 = 1.0 + alpha;

        Self {
            b0: ((1.0 - cos) / 2.0) / a0,
            b1: (1.0 - cos) / a0,
            b2: ((1.0 - cos) / 2.0) / a0,
            a1: (-2.0 * cos) / a0,
            a2: (1.0 - alpha) / a0,
        }
    }

    pub fn highpass(sample_rate: f32, frequency: f32) -> Self {
        let omega0 = consts::TAU * (frequency / sample_rate);
        let cos = omega0.cos();
        let alpha = omega0.sin() / (2.0 * Q);
        let a0 = 1.0 + alpha;

        Self {
            b0: ((1.0 + cos) / 2.0) / a0,
            b1: -(1.0 + cos) / a0,
            b2: ((1.0 + cos) / 2.0) / a0,
            a1: (-2.0 * cos) / a0,
            a2: (1.0 - alpha) / a0,
        }
    }

    pub fn allpass(sample_rate: f32, frequency: f32) -> Self {
        let omega0 = consts::TAU * (frequency / sample_rate);
        let cos = omega0.cos();
        let alpha = omega0.sin() / (2.0 * Q);
        let a0 = 1.0 + alpha;

        Self {
            b0: (1.0 - alpha) / a0,
            b1: (-2.0 * cos) / a0,
            b2: (1.0 + alpha) / a0,
            a1: (-2.0 * cos) / a0,
            a2: (1.0 - alpha) / a0,
        }
    }
}

#[inline(always)]
fn process_biquad(c: &BiquadCoeffs, s: &mut BiquadState, x: f32) -> f32 {
    let y = c.b0 * x + s.z1;
    s.z1 = c.b1 * x - c.a1 * y + s.z2;
    s.z2 = c.b2 * x - c.a2 * y;
    y
}

#[inline(always)]
fn process_biquad_pair(c: &[BiquadCoeffs; 2], s: &mut [BiquadState; 2], x: f32) -> f32 {
    let y = process_biquad(&c[0], &mut s[0], x);
    process_biquad(&c[1], &mut s[1], y)
}

// ---------------------------------------------------------------------------
// Crossover
// ---------------------------------------------------------------------------

/// Linkwitz-Riley 24 dB/oct crossover coefficients. Shared between channels.
///
/// The bands are ordered so that the signal can be reconstructed flat by simply
/// summing all band outputs. This requires the lower bands to be phase
/// compensated with second-order all-pass filters matching the higher crossover
/// points. For a band `k` out of `N` bands, the crossover at index `i`
/// (`0..N-2`) contributes:
///
/// * `i < k`  — a high-pass section
/// * `i == k` — a low-pass section
/// * `i > k`  — an all-pass section
pub struct Crossover {
    lp: [[BiquadCoeffs; 2]; MAX_CROSSOVERS],
    hp: [[BiquadCoeffs; 2]; MAX_CROSSOVERS],
    ap: [BiquadCoeffs; MAX_CROSSOVERS],
}

/// Per-channel crossover state: each band owns the state of the `N-1` sections
/// it passes through.
#[derive(Clone)]
pub struct CrossoverState {
    sections: [[[BiquadState; 2]; MAX_CROSSOVERS]; MAX_BANDS],
}

impl Default for CrossoverState {
    fn default() -> Self {
        Self {
            sections: [[[BiquadState::default(); 2]; MAX_CROSSOVERS]; MAX_BANDS],
        }
    }
}

impl Crossover {
    pub fn new(sample_rate: f32, frequencies: &[f32]) -> Self {
        let mut crossover = Self {
            lp: [[BiquadCoeffs::default(); 2]; MAX_CROSSOVERS],
            hp: [[BiquadCoeffs::default(); 2]; MAX_CROSSOVERS],
            ap: [BiquadCoeffs::default(); MAX_CROSSOVERS],
        };
        crossover.update(sample_rate, frequencies);
        crossover
    }

    pub fn update(&mut self, sample_rate: f32, frequencies: &[f32]) {
        let nyquist = sample_rate * 0.5;
        let clamp = |f: f32| f.clamp(20.0, nyquist * 0.95);
        let num_crossovers = frequencies.len().min(MAX_CROSSOVERS);

        for i in 0..num_crossovers {
            let f = clamp(frequencies[i]);
            let lp = BiquadCoeffs::lowpass(sample_rate, f);
            let hp = BiquadCoeffs::highpass(sample_rate, f);
            let ap = BiquadCoeffs::allpass(sample_rate, f);
            self.lp[i] = [lp, lp];
            self.hp[i] = [hp, hp];
            self.ap[i] = ap;
        }

        // Unused sections pass audio through unchanged.
        for i in num_crossovers..MAX_CROSSOVERS {
            self.lp[i] = [BiquadCoeffs::identity(); 2];
            self.hp[i] = [BiquadCoeffs::identity(); 2];
            self.ap[i] = BiquadCoeffs::identity();
        }
    }

    /// Split `x` into `num_bands` bands that sum back to a flat (all-pass)
    /// version of the input. Unused entries in the result are zero.
    #[inline]
    pub fn process(
        &self,
        st: &mut CrossoverState,
        x: f32,
        num_bands: usize,
    ) -> [f32; MAX_BANDS] {
        let mut out = [0.0f32; MAX_BANDS];
        let num_crossovers = num_bands.saturating_sub(1).min(MAX_CROSSOVERS);

        for band in 0..num_bands.min(MAX_BANDS) {
            let mut y = x;
            for i in 0..num_crossovers {
                let section = &mut st.sections[band][i];
                if i < band {
                    y = process_biquad_pair(&self.hp[i], section, y);
                } else if i == band {
                    y = process_biquad_pair(&self.lp[i], section, y);
                } else {
                    y = process_biquad(&self.ap[i], &mut section[0], y);
                }
            }
            out[band] = y;
        }

        out
    }
}

impl CrossoverState {
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ---------------------------------------------------------------------------
// Transient shaper
// ---------------------------------------------------------------------------

/// A per-band transient shaper.
///
/// It tracks a fast peak envelope and a slower "body" envelope. The difference
/// between the two indicates how much of the signal is currently an attack
/// transient versus a sustained/decaying body, and is used to crossfade between
/// an attack gain and a sustain gain. The two gains are derived from a single
/// "shaping" value so the band only ever cuts the attack or the sustain.
pub struct TransientShaper {
    env: f32,
    body: f32,
    gain: f32,
    release_coef: f32,
    body_coef: f32,
    gain_coef: f32,
}

impl TransientShaper {
    pub fn new() -> Self {
        Self {
            env: 0.0,
            body: 0.0,
            gain: 1.0,
            release_coef: 0.0,
            body_coef: 0.0,
            gain_coef: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.env = 0.0;
        self.body = 0.0;
        self.gain = 1.0;
    }

    /// Recompute the envelope smoothing coefficients. `speed_ms` is the fast
    /// envelope's release/recovery time in milliseconds.
    pub fn update(&mut self, sample_rate: f32, speed_ms: f32) {
        let sr = sample_rate.max(1.0);
        let release_time = speed_ms.max(2.0) / 1000.0;
        // Keep the body envelope on the same timescale as the fast envelope so
        // the `speed` control really reads as the transient window length.
        let body_time = release_time.max(0.003);

        self.release_coef = (-1.0 / (sr * release_time)).exp();
        self.body_coef = 1.0 - (-1.0 / (sr * body_time)).exp();
        self.gain_coef = 1.0 - (-1.0 / (sr * GAIN_SMOOTH_TIME)).exp();
    }

    #[inline]
    pub fn process(&mut self, x: f32, attack_gain: f32, sustain_gain: f32) -> f32 {
        let ax = x.abs();

        // Fast peak envelope: instant attack, exponential release.
        if ax > self.env {
            self.env = ax;
        } else {
            self.env = ax + self.release_coef * (self.env - ax);
        }

        // Body envelope: a slow low-pass of the fast envelope.
        self.body += self.body_coef * (self.env - self.body);

        // Fraction of the current level that is an attack transient, in [0, 1].
        let frac = if self.env > 1e-6 {
            ((self.env - self.body) / self.env).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let target = frac * attack_gain + (1.0 - frac) * sustain_gain;
        self.gain += self.gain_coef * (target - self.gain);

        x * self.gain
    }
}

impl Default for TransientShaper {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Spectrum analyzer
// ---------------------------------------------------------------------------

/// FFT size used for the spectrum analyzer.
pub const SPECTRUM_FFT_SIZE: usize = 4096;
/// Number of real FFT bins (`FFT_SIZE / 2 + 1`).
pub const SPECTRUM_BINS: usize = SPECTRUM_FFT_SIZE / 2 + 1;

/// Number of samples between successive FFT frames.
const SPECTRUM_HOP: usize = 256;
/// Value shown for silence, in dBFS.
const SPECTRUM_MIN_DB: f32 = -120.0;
/// How fast held spectrum peaks fall, in dB per second.
const SPECTRUM_FALL_DB_PER_SECOND: f32 = 40.0;

/// Spectrum data shared between the audio thread (writer) and the GUI (reader).
/// Magnitudes are stored in dBFS, with `SPECTRUM_MIN_DB` representing silence.
pub struct SpectrumData {
    pub sample_rate: AtomicF32,
    pub input: Box<[AtomicF32]>,
    pub output: Box<[AtomicF32]>,
}

impl SpectrumData {
    pub fn new() -> Self {
        Self {
            sample_rate: AtomicF32::new(44_100.0),
            input: (0..SPECTRUM_BINS)
                .map(|_| AtomicF32::new(SPECTRUM_MIN_DB))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            output: (0..SPECTRUM_BINS)
                .map(|_| AtomicF32::new(SPECTRUM_MIN_DB))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }
}

impl Default for SpectrumData {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes and publishes a smoothed input/output spectrum using a Hann-windowed
/// FFT. Runs on the audio thread; all allocations happen in [`SpectrumAnalyzer::new()`],
/// which is called during initialization rather than in the hot path.
pub struct SpectrumAnalyzer {
    fft_size: usize,
    hop: usize,
    input_ring: Vec<f32>,
    output_ring: Vec<f32>,
    ring_pos: usize,
    hop_counter: usize,
    window: Vec<f32>,
    coherent_gain: f32,
    fft_in: Vec<f32>,
    fft_out: Vec<Complex32>,
    scratch: Vec<Complex32>,
    r2c: Arc<dyn RealToComplex<f32>>,
    fall_db: f32,
    smoothed_input: Vec<f32>,
    smoothed_output: Vec<f32>,
}

impl SpectrumAnalyzer {
    pub fn new(sample_rate: f32) -> Self {
        let fft_size = SPECTRUM_FFT_SIZE;
        let hop = SPECTRUM_HOP;

        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(fft_size);
        let fft_in = r2c.make_input_vec();
        let fft_out = r2c.make_output_vec();
        let scratch = r2c.make_scratch_vec();

        let window: Vec<f32> = (0..fft_size)
            .map(|i| {
                0.5 * (1.0 - (consts::TAU * i as f32 / (fft_size - 1) as f32).cos())
            })
            .collect();
        let coherent_gain = window.iter().sum::<f32>() / fft_size as f32;

        Self {
            fft_size,
            hop,
            input_ring: vec![0.0; fft_size],
            output_ring: vec![0.0; fft_size],
            ring_pos: 0,
            hop_counter: 0,
            window,
            coherent_gain,
            fft_in,
            fft_out,
            scratch,
            r2c,
            fall_db: SPECTRUM_FALL_DB_PER_SECOND * hop as f32 / sample_rate.max(1.0),
            smoothed_input: vec![SPECTRUM_MIN_DB; SPECTRUM_BINS],
            smoothed_output: vec![SPECTRUM_MIN_DB; SPECTRUM_BINS],
        }
    }

    pub fn reset(&mut self) {
        self.input_ring.fill(0.0);
        self.output_ring.fill(0.0);
        self.ring_pos = 0;
        self.hop_counter = 0;
        self.smoothed_input.fill(SPECTRUM_MIN_DB);
        self.smoothed_output.fill(SPECTRUM_MIN_DB);
    }

    /// Feed one mono sample each of the input and output. Every `hop` samples an
    /// FFT is computed and published to `data`.
    #[inline]
    pub fn process(&mut self, input: f32, output: f32, data: &SpectrumData) {
        self.input_ring[self.ring_pos] = input;
        self.output_ring[self.ring_pos] = output;

        self.ring_pos += 1;
        if self.ring_pos == self.fft_size {
            self.ring_pos = 0;
        }

        self.hop_counter += 1;
        if self.hop_counter >= self.hop {
            self.hop_counter = 0;
            self.analyze(data);
        }
    }

    fn analyze(&mut self, data: &SpectrumData) {
        let norm = 2.0 / (self.fft_size as f32 * self.coherent_gain);
        let fall_db = self.fall_db;

        // Input spectrum.
        for i in 0..self.fft_size {
            let idx = (self.ring_pos + i) % self.fft_size;
            self.fft_in[i] = self.input_ring[idx] * self.window[i];
        }
        self.r2c
            .process_with_scratch(&mut self.fft_in, &mut self.fft_out, &mut self.scratch)
            .expect("real FFT failed");
        publish_spectrum(
            &self.fft_out,
            &mut self.smoothed_input,
            &data.input,
            norm,
            fall_db,
        );

        // Output spectrum.
        for i in 0..self.fft_size {
            let idx = (self.ring_pos + i) % self.fft_size;
            self.fft_in[i] = self.output_ring[idx] * self.window[i];
        }
        self.r2c
            .process_with_scratch(&mut self.fft_in, &mut self.fft_out, &mut self.scratch)
            .expect("real FFT failed");
        publish_spectrum(
            &self.fft_out,
            &mut self.smoothed_output,
            &data.output,
            norm,
            fall_db,
        );
    }
}

fn publish_spectrum(
    fft_out: &[Complex32],
    smoothed: &mut [f32],
    bins: &[AtomicF32],
    norm: f32,
    fall_db: f32,
) {
    for i in 0..SPECTRUM_BINS {
        let mag = fft_out[i].norm() * norm;
        let db = if mag > 1e-9 {
            20.0 * mag.log10()
        } else {
            SPECTRUM_MIN_DB
        };

        let prev = smoothed[i];
        let next = if db > prev {
            db
        } else {
            (prev - fall_db).max(db)
        };
        smoothed[i] = next;
        bins[i].store(next, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crossover_reconstructs_flat() {
        let sr = 48_000.0;
        let crossover = Crossover::new(sr, &[150.0, 1500.0, 8000.0]);
        let mut state = CrossoverState::default();
        let num_bands = 4;

        // Feed a steady signal so the filters settle, then check the RMS.
        for n in 0..48_000 {
            let x = (consts::TAU * 440.0 * n as f32 / sr).sin() * 0.5;
            let bands = crossover.process(&mut state, x, num_bands);
            let sum: f32 = bands.iter().sum();
            assert!(sum.is_finite());
        }

        // A steady sine should sum back to (approximately) the same sine, just
        // with an all-pass phase shift. Check that the RMS of the sum matches
        // the input RMS.
        let mut state = CrossoverState::default();
        let mut input_rms = 0.0;
        let mut output_rms = 0.0;
        for n in 0..48_000 {
            let x = (consts::TAU * 440.0 * n as f32 / sr).sin() * 0.5;
            let bands = crossover.process(&mut state, x, num_bands);
            let sum: f32 = bands.iter().sum();
            if n > 4096 {
                input_rms += x * x;
                output_rms += sum * sum;
            }
        }
        input_rms = (input_rms / 48_000.0).sqrt();
        output_rms = (output_rms / 48_000.0).sqrt();

        // The all-pass reconstruction keeps the magnitude flat, so the RMS
        // should be essentially identical.
        assert!((input_rms - output_rms).abs() < 1e-3, "in={input_rms} out={output_rms}");
    }

    #[test]
    fn transient_shaper_passthrough() {
        let mut shaper = TransientShaper::new();
        shaper.update(48_000.0, 50.0);

        let mut out = 0.0;
        for n in 0..48_000 {
            let x = (consts::TAU * 440.0 * n as f32 / 48_000.0).sin() * 0.5;
            out = shaper.process(x, 1.0, 1.0);
        }
        assert!(out.is_finite());
    }
}
