use anyhow::{Context, Result};
use ndarray::Array2;
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;
use serde::Deserialize;
use std::f32::consts::PI;
use std::sync::Arc;

#[derive(Debug, Deserialize)]
pub(crate) struct CoherePreprocessorConfig {
    pub(crate) dither: f32,
    pub(crate) feature_size: usize,
    pub(crate) n_fft: usize,
    pub(crate) n_window_size: usize,
    pub(crate) n_window_stride: usize,
    pub(crate) normalize: String,
    pub(crate) padding_value: f32,
    pub(crate) sampling_rate: u32,
    pub(crate) window: String,
}

pub(crate) struct CohereFrontend {
    n_fft: usize,
    hop_size: usize,
    feature_size: usize,
    padding_value: f32,
    dither: f32,
    mel_filters: Vec<f32>,
    fft_bins: usize,
    window: Vec<f32>,
    fft: Arc<dyn rustfft::Fft<f32>>,
}

impl CohereFrontend {
    pub(crate) fn new(config: CoherePreprocessorConfig) -> Result<Self> {
        anyhow::ensure!(
            config.sampling_rate == 16_000,
            "unsupported Cohere sampling rate {}; expected 16000",
            config.sampling_rate
        );
        anyhow::ensure!(
            config.window == "hann",
            "unsupported Cohere window {}; expected hann",
            config.window
        );
        anyhow::ensure!(
            config.normalize == "per_feature",
            "unsupported Cohere normalization {}; expected per_feature",
            config.normalize
        );

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(config.n_fft);
        let fft_bins = (config.n_fft / 2) + 1;
        let window = build_centered_hann_window(config.n_fft, config.n_window_size);
        let mel_filters = build_slaney_mel_filters(
            config.sampling_rate,
            config.n_fft,
            config.feature_size,
            0.0,
            (config.sampling_rate as f32) / 2.0,
        );
        Ok(Self {
            n_fft: config.n_fft,
            hop_size: config.n_window_stride,
            feature_size: config.feature_size,
            padding_value: config.padding_value,
            dither: config.dither,
            mel_filters,
            fft_bins,
            window,
            fft,
        })
    }

    pub(crate) fn compute(&self, samples: &[f32]) -> Result<Array2<f32>> {
        if samples.is_empty() {
            return Ok(Array2::zeros((self.feature_size, 0)));
        }

        let seq_len = samples.len() / self.hop_size;
        if seq_len == 0 {
            return Ok(Array2::zeros((self.feature_size, 0)));
        }

        let mut waveform = samples.to_vec();
        if self.dither > 0.0 {
            apply_dither(&mut waveform, self.dither);
        }
        apply_preemphasis(&mut waveform, 0.97);

        let pad = self.n_fft / 2;
        let mut padded = vec![0.0f32; waveform.len() + (pad * 2)];
        padded[pad..pad + waveform.len()].copy_from_slice(&waveform);

        let mut features = vec![0.0f32; self.feature_size * seq_len];
        let mut fft_input = vec![Complex32::new(0.0, 0.0); self.n_fft];
        for frame_idx in 0..seq_len {
            let start = frame_idx * self.hop_size;
            let frame = &padded[start..start + self.n_fft];
            for i in 0..self.n_fft {
                fft_input[i] = Complex32::new(frame[i] * self.window[i], 0.0);
            }
            self.fft.process(&mut fft_input);

            let mut power = vec![0.0f32; self.fft_bins];
            for (bin_idx, value) in fft_input.iter().take(self.fft_bins).enumerate() {
                power[bin_idx] = value.norm_sqr();
            }

            for mel_idx in 0..self.feature_size {
                let filter =
                    &self.mel_filters[(mel_idx * self.fft_bins)..((mel_idx + 1) * self.fft_bins)];
                let mut energy = 0.0f32;
                for (weight, bin_power) in filter.iter().zip(power.iter()) {
                    energy += *weight * *bin_power;
                }
                let logged = (energy + 2f32.powi(-24)).ln();
                features[(mel_idx * seq_len) + frame_idx] = logged;
            }
        }

        normalize_per_feature(
            &mut features,
            self.feature_size,
            seq_len,
            self.padding_value,
        );
        Array2::from_shape_vec((self.feature_size, seq_len), features)
            .context("failed to shape Cohere mel features")
    }
}

fn apply_dither(waveform: &mut [f32], scale: f32) {
    let mut state = waveform.len() as u64 + 1;
    for sample in waveform {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let uniform = ((state as f64 / u64::MAX as f64) as f32) - 0.5;
        *sample += uniform * 2.0 * scale;
    }
}

fn apply_preemphasis(waveform: &mut [f32], coeff: f32) {
    if waveform.is_empty() {
        return;
    }
    let mut prev = waveform[0];
    for sample in waveform.iter_mut().skip(1) {
        let current = *sample;
        *sample = current - (coeff * prev);
        prev = current;
    }
}

fn build_centered_hann_window(n_fft: usize, win_length: usize) -> Vec<f32> {
    let mut window = vec![0.0f32; n_fft];
    let offset = (n_fft.saturating_sub(win_length)) / 2;
    if win_length <= 1 {
        return window;
    }
    for i in 0..win_length {
        let phase = (2.0 * PI * i as f32) / (win_length as f32 - 1.0);
        window[offset + i] = 0.5 - (0.5 * phase.cos());
    }
    window
}

fn build_slaney_mel_filters(
    sample_rate: u32,
    n_fft: usize,
    n_mels: usize,
    f_min: f32,
    f_max: f32,
) -> Vec<f32> {
    let fft_bins = (n_fft / 2) + 1;
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for idx in 0..(n_mels + 2) {
        let ratio = idx as f32 / (n_mels + 1) as f32;
        mel_points.push(mel_to_hz(mel_min + ((mel_max - mel_min) * ratio)));
    }

    let mut filters = vec![0.0f32; n_mels * fft_bins];
    for mel_idx in 0..n_mels {
        let lower = mel_points[mel_idx];
        let center = mel_points[mel_idx + 1];
        let upper = mel_points[mel_idx + 2];
        let enorm = 2.0 / (upper - lower).max(f32::EPSILON);
        for bin_idx in 0..fft_bins {
            let freq = (sample_rate as f32 / n_fft as f32) * bin_idx as f32;
            let lower_slope = (freq - lower) / (center - lower).max(f32::EPSILON);
            let upper_slope = (upper - freq) / (upper - center).max(f32::EPSILON);
            let weight = lower_slope.min(upper_slope).max(0.0) * enorm;
            filters[(mel_idx * fft_bins) + bin_idx] = weight;
        }
    }
    filters
}

fn hz_to_mel(freq_hz: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if freq_hz < min_log_hz {
        freq_hz / f_sp
    } else {
        min_log_mel + (freq_hz / min_log_hz).ln() / logstep
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if mel < min_log_mel {
        mel * f_sp
    } else {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    }
}

fn normalize_per_feature(features: &mut [f32], n_mels: usize, seq_len: usize, pad_value: f32) {
    if seq_len == 0 {
        return;
    }
    for mel_idx in 0..n_mels {
        let row = &mut features[(mel_idx * seq_len)..((mel_idx + 1) * seq_len)];
        let mean = row.iter().sum::<f32>() / seq_len as f32;
        let denom = (seq_len as f32 - 1.0).max(1.0);
        let variance = row
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f32>()
            / denom;
        let std = variance.sqrt() + 1e-5;
        for value in row.iter_mut() {
            *value = (*value - mean) / std;
            if !value.is_finite() {
                *value = pad_value;
            }
        }
    }
}
