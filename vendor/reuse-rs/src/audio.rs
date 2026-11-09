use anyhow::{anyhow, Result};
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::f32::consts::PI;
use std::sync::Arc;

/// Sample rates the model was trained on.
pub const SAMPLE_RATES: [u32; 7] = [8000, 16000, 22050, 24000, 32000, 44100, 48000];

/// Periodic Hann window of length `n`, like `torch.hann_window(n)`.
pub(crate) fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos())
        .collect()
}

/// Reflect-pad a signal by `pad` samples on each side.
/// Matches `torch.stft(..., pad_mode='reflect', center=True)`.
fn reflect_pad(signal: &[f32], pad: usize) -> Vec<f32> {
    let n = signal.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    // left reflect
    for i in (1..=pad).rev() {
        let idx = if i < n { i } else { n - 1 };
        out.push(signal[idx]);
    }
    out.extend_from_slice(signal);
    // right reflect
    for i in 1..=pad {
        let idx = if n >= 2 && i < n { n - 1 - i } else { 0 };
        out.push(signal[idx]);
    }
    out
}

/// Per-bin rows of per-frame values, `[freq][time]`.
pub type Spectrogram = Vec<Vec<f32>>;

/// Reusable FFT plans and window for analysis and overlap-add synthesis.
pub(crate) struct Stft {
    window: Vec<f32>,
    forward: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f32>>,
}

impl Stft {
    pub(crate) fn new(n_fft: usize, win_size: usize) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        Self {
            window: hann_window(win_size),
            forward: planner.plan_fft_forward(n_fft),
            inverse: planner.plan_fft_inverse(n_fft),
        }
    }

    /// Complex STFT matching `torch.stft(y, n_fft, hop_length, win_length, hann_window, center=True, pad_mode='reflect', normalized=False, return_complex=True)`.
    ///
    /// Returns `(mag, phase)` each shaped `[F, T]` where `F = n_fft/2 + 1`.
    pub(crate) fn stft(
        &self,
        signal: &[f32],
        hop_size: usize,
    ) -> Result<(Spectrogram, Spectrogram)> {
        let n_fft = self.forward.len();
        let window = &self.window;
        let win_size = window.len();
        let pad = n_fft / 2;
        let padded = reflect_pad(signal, pad);
        let freq_bins = n_fft / 2 + 1;
        let num_frames = (padded.len() - n_fft) / hop_size + 1;

        let r2c = &self.forward;
        let mut input_buf = vec![0.0f32; n_fft];
        let mut output_buf = r2c.make_output_vec();
        let mut scratch = r2c.make_scratch_vec();

        let mut mag = vec![vec![0.0f32; num_frames]; freq_bins];
        let mut phase = vec![vec![0.0f32; num_frames]; freq_bins];

        for t in 0..num_frames {
            let start = t * hop_size;
            input_buf.fill(0.0);
            for i in 0..win_size.min(padded.len() - start) {
                input_buf[i] = padded[start + i] * window[i];
            }

            r2c.process_with_scratch(&mut input_buf, &mut output_buf, &mut scratch)
                .map_err(|e| anyhow!("FFT failed: {e}"))?;

            for k in 0..freq_bins {
                let re = output_buf[k].re;
                let im = output_buf[k].im;
                mag[k][t] = (re * re + im * im).sqrt();
                phase[k][t] = im.atan2(re);
            }
        }

        Ok((mag, phase))
    }

    /// Inverse STFT matching `torch.istft(complex, n_fft, hop_length, win_length, hann_window, center=True)`.
    ///
    /// `mag` and `phase` are `[F, T]` where `F = n_fft/2 + 1`.
    pub(crate) fn istft(
        &self,
        mag: &[Vec<f32>],
        phase: &[Vec<f32>],
        hop_size: usize,
    ) -> Result<Vec<f32>> {
        let n_fft = self.inverse.len();
        let window = &self.window;
        let win_size = window.len();
        let freq_bins = n_fft / 2 + 1;
        let num_frames = mag[0].len();
        let c2r = &self.inverse;
        let mut spec_buf = c2r.make_input_vec();
        let mut time_buf = vec![0.0f32; n_fft];
        let mut scratch = c2r.make_scratch_vec();

        // Output length includes center padding on both sides
        let out_len = n_fft / 2 + (num_frames - 1) * hop_size + n_fft / 2;
        let mut output = vec![0.0f32; out_len];
        let mut window_sum = vec![0.0f32; out_len];

        for t in 0..num_frames {
            // Build complex spectrum
            for k in 0..freq_bins {
                let m = mag[k][t];
                let p = phase[k][t];
                let (sin, cos) = p.sin_cos();
                spec_buf[k] = realfft::num_complex::Complex::new(m * cos, m * sin);
            }
            // DC and Nyquist bins must be real for realfft
            spec_buf[0].im = 0.0;
            if freq_bins > 1 {
                spec_buf[freq_bins - 1].im = 0.0;
            }

            c2r.process_with_scratch(&mut spec_buf, &mut time_buf, &mut scratch)
                .map_err(|e| anyhow!("IFFT failed: {e}"))?;

            let offset = t * hop_size;
            // realfft returns unnormalized output, divide by n_fft
            let norm = 1.0 / n_fft as f32;
            for i in 0..win_size {
                let pos = offset + i;
                if pos < out_len {
                    output[pos] += time_buf[i] * norm * window[i];
                    window_sum[pos] += window[i] * window[i];
                }
            }
        }

        // Normalize by window sum (overlap-add)
        let tiny = 1e-11;
        for i in 0..out_len {
            if window_sum[i] > tiny {
                output[i] /= window_sum[i];
            }
        }

        // Remove center padding (n_fft/2 on each side)
        let pad = n_fft / 2;
        if out_len > 2 * pad {
            output.copy_within(pad..out_len - pad, 0);
            output.truncate(out_len - 2 * pad);
        }
        Ok(output)
    }
}

/// Load a WAV file as mono f32 samples at its native sample rate.
pub fn load_wav(path: &str) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    let mut samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1u32 << (bits - 1)) as f32;
            if bits <= 16 {
                reader
                    .samples::<i16>()
                    .map(|s| s.map(|s| s as f32 / max_val))
                    .collect::<std::result::Result<Vec<_>, _>>()?
            } else {
                reader
                    .samples::<i32>()
                    .map(|s| s.map(|s| s as f32 / max_val))
                    .collect::<std::result::Result<Vec<_>, _>>()?
            }
        }
    };

    // Mix to mono
    if spec.channels > 1 {
        let ch = spec.channels as usize;
        samples = samples
            .chunks(ch)
            .map(|c| c.iter().sum::<f32>() / ch as f32)
            .collect();
    }

    Ok((samples, spec.sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stft_istft_roundtrip() {
        // A simple sine wave should survive STFT -> ISTFT with low error
        let sr = 16000;
        let duration = 0.1; // 100ms
        let n = (sr as f32 * duration) as usize;
        let freq = 440.0;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sr as f32).sin())
            .collect();

        let n_fft = 320;
        let hop = 40;
        let win = 320;

        let plan = Stft::new(n_fft, win);
        let (mag, phase) = plan.stft(&signal, hop).unwrap();
        let reconstructed = plan.istft(&mag, &phase, hop).unwrap();

        // Compare overlapping region
        let len = signal.len().min(reconstructed.len());
        let max_err: f32 = signal[..len]
            .iter()
            .zip(&reconstructed[..len])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_err < 0.01,
            "STFT/ISTFT roundtrip error too large: {max_err}"
        );
    }
}
