use crate::asr::WindowTranscription;
use crate::chunking::TimedWord;
use anyhow::{Context, Result};
use asr_onnx::{Config as TdtConfig, JobMeta, TranscriberPool, TranscriptionResult};
use mel_spec::mel::{BatchLogMelConfig, BatchLogMelScratch, BatchLogMelSpectrogram};
use ndarray::Array2;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use tracing::info;

pub struct ParakeetBackend {
    frontend: ParakeetFrontend,
    pool: TranscriberPool,
    state: Arc<Mutex<ParakeetState>>,
}

struct ParakeetState {
    pending: HashMap<u64, oneshot::Sender<std::result::Result<WindowTranscription, String>>>,
    completed: HashMap<u64, std::result::Result<WindowTranscription, String>>,
}

impl ParakeetBackend {
    pub fn new(model_dir: &Path, device_ids: &[usize], onnx_sessions: usize) -> Result<Self> {
        validate_model_dir(model_dir)?;

        let config = TdtConfig::default().with_num_sessions(onnx_sessions.max(1));
        let frontend = ParakeetFrontend::from_tdt_config(&config)?;
        let model_path = model_dir.to_path_buf();
        let vocab_path = model_dir.join("vocab.txt");
        let pool = TranscriberPool::new(model_path, vocab_path, device_ids, config.clone())
            .context("failed to initialize Parakeet ONNX/TDT transcriber pool")?;

        let ready_count = device_ids.len().max(1) * config.num_sessions.max(1);
        for _ in 0..ready_count {
            pool.ready()
                .recv_timeout(Duration::from_secs(240))
                .context("timed out waiting for Parakeet ONNX/TDT worker initialization")?;
        }

        let state = Arc::new(Mutex::new(ParakeetState {
            pending: HashMap::new(),
            completed: HashMap::new(),
        }));
        let dispatch_state = Arc::clone(&state);
        let result_rx = pool.result_rx().clone();
        thread::spawn(move || {
            for result in result_rx {
                dispatch_result(&dispatch_state, result);
            }

            let mut guard = dispatch_state
                .lock()
                .expect("parakeet state mutex poisoned");
            for (_, sender) in guard.pending.drain() {
                let _ = sender.send(Err("Parakeet worker pool closed".into()));
            }
            guard.completed.clear();
        });

        info!(
            model_dir = %model_dir.display(),
            onnx_sessions = onnx_sessions.max(1),
            "initialized Parakeet ONNX/TDT backend"
        );

        Ok(Self {
            frontend,
            pool,
            state,
        })
    }

    pub async fn transcribe_window(
        &self,
        samples: Vec<f32>,
        seq: u32,
    ) -> Result<WindowTranscription> {
        let started = Instant::now();
        let features = self.frontend.compute(&samples)?;
        let meta = JobMeta {
            seq,
            chunk_id: u64::from(seq),
        };
        let job_id = self
            .pool
            .submit(format!("parakeet-{seq}"), features, meta)
            .context("failed to submit Parakeet TDT job")?;

        let (sender, receiver) = oneshot::channel();
        {
            let mut guard = self.state.lock().expect("parakeet state mutex poisoned");
            if let Some(result) = guard.completed.remove(&job_id) {
                let _ = sender.send(result);
            } else {
                guard.pending.insert(job_id, sender);
            }
        }

        let result = receiver
            .await
            .context("Parakeet worker dropped response")?
            .map_err(|error| anyhow::anyhow!(error))?;

        if env_var_truthy("ASR_PARAKEET_TIMINGS") {
            let audio_seconds = samples.len() as f64 / 16_000.0;
            let elapsed_s = started.elapsed().as_secs_f64();
            eprintln!(
                "parakeet_timing total_ms={:.2} audio_seconds={:.3} rtfx={:.2}",
                elapsed_s * 1000.0,
                audio_seconds,
                audio_seconds / elapsed_s
            );
        }

        Ok(result)
    }
}

fn dispatch_result(state: &Arc<Mutex<ParakeetState>>, result: TranscriptionResult) {
    let job_id = result.job_id;
    let value = Ok(WindowTranscription {
        text: result.text,
        words: result
            .words
            .into_iter()
            .map(|(word, start_ms, end_ms)| TimedWord {
                word,
                start_ms,
                end_ms,
            })
            .collect(),
        stitch_words: None,
    });

    let mut guard = state.lock().expect("parakeet state mutex poisoned");
    if let Some(sender) = guard.pending.remove(&job_id) {
        let _ = sender.send(value);
    } else {
        guard.completed.insert(job_id, value);
    }
}

struct ParakeetFrontend {
    frontend: BatchLogMelSpectrogram,
    scratch: Mutex<BatchLogMelScratch>,
}

impl ParakeetFrontend {
    fn from_tdt_config(config: &TdtConfig) -> Result<Self> {
        let sample_rate = env_usize("ASR_PARAKEET_SAMPLE_RATE").unwrap_or(config.sample_rate);
        let n_fft = env_usize("ASR_PARAKEET_N_FFT").unwrap_or(512);
        let win_length = env_usize("ASR_PARAKEET_WIN_LENGTH").unwrap_or(config.window);
        let hop_length = env_usize("ASR_PARAKEET_HOP_LENGTH").unwrap_or(config.hop);
        let n_mels = env_usize("ASR_PARAKEET_N_MELS").unwrap_or(config.features_size);
        let preemph = env_f32("ASR_PARAKEET_PREEMPH").unwrap_or(0.97);
        let log_zero_guard = env_f32("ASR_PARAKEET_LOG_ZERO_GUARD").unwrap_or(2.0_f32.powi(-24));
        let pad_to = env_usize("ASR_PARAKEET_PAD_TO").unwrap_or(0);

        anyhow::ensure!(
            sample_rate == 16_000,
            "unsupported Parakeet sample rate {}; expected 16000",
            sample_rate
        );
        anyhow::ensure!(
            win_length <= n_fft,
            "ASR_PARAKEET_WIN_LENGTH must be <= ASR_PARAKEET_N_FFT"
        );
        anyhow::ensure!(hop_length > 0, "ASR_PARAKEET_HOP_LENGTH must be > 0");
        anyhow::ensure!(n_mels > 0, "ASR_PARAKEET_N_MELS must be > 0");

        let frontend = BatchLogMelSpectrogram::new(BatchLogMelConfig {
            sample_rate,
            n_fft,
            win_length,
            hop_length,
            n_mels,
            f_min: 0.0,
            f_max: Some(sample_rate as f64 / 2.0),
            htk: false,
            norm: true,
            preemphasis: preemph,
            center: true,
            log_zero_guard,
            pad_to,
            normalize_per_feature: true,
        })
        .context("failed to initialize mel-spec Parakeet frontend")?;
        let scratch = Mutex::new(frontend.scratch());

        Ok(Self { frontend, scratch })
    }

    fn compute(&self, samples: &[f32]) -> Result<Array2<f32>> {
        let mut scratch = self
            .scratch
            .lock()
            .expect("Parakeet frontend scratch mutex poisoned");
        let output = self
            .frontend
            .compute_flat_with_scratch(samples, &mut scratch)
            .context("failed to compute Parakeet mel features")?;
        Array2::from_shape_vec((output.rows, output.cols), output.data)
            .context("failed to shape Parakeet mel features")
    }
}

fn validate_model_dir(model_dir: &Path) -> Result<()> {
    for file in [
        "encoder.onnx",
        "decoder.onnx",
        "joint.enc.onnx",
        "joint.pred.onnx",
        "joint.joint_net.onnx",
        "tokens.txt",
    ] {
        anyhow::ensure!(
            model_dir.join(file).is_file(),
            "Parakeet backend requires {}",
            model_dir.join(file).display()
        );
    }
    Ok(())
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.trim().parse().ok()
}

fn env_f32(name: &str) -> Option<f32> {
    std::env::var(name).ok()?.trim().parse().ok()
}

fn env_var_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "True"))
        .unwrap_or(false)
}
