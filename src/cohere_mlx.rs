use crate::asr::WindowTranscription;
use crate::config::DEFAULT_LANGUAGE;
use anyhow::{Context, Result};
use cohere_transcribe_rs::audio::{compute_mel_features, mel_to_tensor_data, MelConfig};
use cohere_transcribe_rs::config::ModelConfig;
use cohere_transcribe_rs::mlx;
use cohere_transcribe_rs::tokenizer::Tokenizer;
use crossbeam_channel::{bounded, Sender};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use tracing::info;

type MlxJob = (
    u64,
    u32,
    Vec<f32>,
    oneshot::Sender<std::result::Result<WindowTranscription, String>>,
);

pub struct CohereMlxBackend {
    next_id: AtomicU64,
    sender: Sender<MlxJob>,
}

impl CohereMlxBackend {
    pub fn new(model_dir: &Path, max_new_tokens: usize) -> Result<Self> {
        validate_model_dir(model_dir)?;

        let model_dir = model_dir.to_path_buf();
        let (sender, receiver) = bounded::<MlxJob>(2);
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();

        thread::spawn(move || {
            let mut worker = match CohereMlxWorker::new(&model_dir, max_new_tokens) {
                Ok(worker) => {
                    let _ = ready_tx.send(Ok(()));
                    worker
                }
                Err(error) => {
                    let _ = ready_tx.send(Err(error.to_string()));
                    return;
                }
            };

            for (_id, seq, samples, sender) in receiver {
                let result = worker
                    .transcribe(samples, seq)
                    .map_err(|error| error.to_string());
                let _ = sender.send(result);
            }
        });

        let ready = ready_rx
            .recv_timeout(Duration::from_secs(240))
            .context("timed out waiting for Cohere MLX worker initialization")?;
        if let Err(error) = ready {
            anyhow::bail!(error);
        }

        Ok(Self {
            next_id: AtomicU64::new(1),
            sender,
        })
    }

    pub async fn transcribe_window(
        &self,
        samples: Vec<f32>,
        seq: u32,
    ) -> Result<WindowTranscription> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = oneshot::channel();
        self.sender
            .send((id, seq, samples, sender))
            .map_err(|_| anyhow::anyhow!("Cohere MLX worker pool closed"))?;
        receiver
            .await
            .context("Cohere MLX worker dropped response")?
            .map_err(|error| anyhow::anyhow!(error))
    }
}

struct CohereMlxWorker {
    encoder: mlx::encoder::ConformerEncoder,
    decoder: mlx::decoder::TransformerDecoder,
    tokenizer: Tokenizer,
    mel_config: MelConfig,
    max_new_tokens: usize,
}

impl CohereMlxWorker {
    fn new(model_dir: &Path, max_new_tokens: usize) -> Result<Self> {
        let started = Instant::now();
        mlx::stream::init_mlx(true);

        let config = ModelConfig::load(model_dir).context("failed to load Cohere MLX config")?;
        let tokenizer =
            Tokenizer::load(model_dir).context("failed to load Cohere MLX tokenizer")?;
        let mel_config = MelConfig::from_model_config(&config);
        let weights = mlx::weights::MlxWeights::load(model_dir.join("model.safetensors"))
            .context("failed to load Cohere MLX safetensors")?;
        let encoder = mlx::encoder::ConformerEncoder::load(&weights, &config)
            .context("failed to load Cohere MLX encoder")?;
        let decoder = mlx::decoder::TransformerDecoder::load(&weights, &config)
            .context("failed to load Cohere MLX decoder")?;
        mlx::stream::synchronize();

        info!(
            elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
            "initialized Cohere MLX backend"
        );

        Ok(Self {
            encoder,
            decoder,
            tokenizer,
            mel_config,
            max_new_tokens,
        })
    }

    fn transcribe(&mut self, samples: Vec<f32>, seq: u32) -> Result<WindowTranscription> {
        let started = Instant::now();
        let dithered = add_dither(&samples, self.mel_config.dither as f32, u64::from(seq));
        let mel = compute_mel_features(&dithered, &self.mel_config);
        let (flat, shape) = mel_to_tensor_data(&mel);
        let shape = shape
            .into_iter()
            .map(|dim| i32::try_from(dim).context("Cohere MLX mel shape overflow"))
            .collect::<Result<Vec<_>>>()?;
        let mel = mlx::array::Array::from_data_f32(&flat, &shape);
        let text = mlx::inference::transcribe(
            &mel,
            &self.encoder,
            &self.decoder,
            &self.tokenizer,
            DEFAULT_LANGUAGE,
            true,
            self.max_new_tokens,
        )
        .context("Cohere MLX inference failed")?;
        mlx::stream::synchronize();

        if env_var_truthy("ASR_COHERE_TIMINGS") {
            let audio_seconds = samples.len() as f64 / 16_000.0;
            let elapsed_s = started.elapsed().as_secs_f64();
            eprintln!(
                "cohere_mlx_timing total_ms={:.2} audio_seconds={:.3} rtfx={:.2}",
                elapsed_s * 1000.0,
                audio_seconds,
                audio_seconds / elapsed_s
            );
        }

        Ok(WindowTranscription {
            text,
            words: Vec::new(),
        })
    }
}

fn validate_model_dir(model_dir: &Path) -> Result<()> {
    for file in ["config.json", "model.safetensors", "vocab.json"] {
        anyhow::ensure!(
            model_dir.join(file).is_file(),
            "Cohere MLX backend requires {}; sync the Hugging Face safetensors bundle and generate vocab.json",
            model_dir.join(file).display()
        );
    }
    Ok(())
}

fn add_dither(samples: &[f32], dither: f32, seed: u64) -> Vec<f32> {
    if dither == 0.0 {
        return samples.to_vec();
    }

    let mut rng = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let mut out = samples.to_vec();
    for sample in out.iter_mut() {
        rng = rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = (rng >> 33) as f32 / (u32::MAX as f32);
        rng = rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let v = (rng >> 33) as f32 / (u32::MAX as f32);
        let noise = (-2.0 * u.max(1e-38).ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos();
        *sample += dither * noise;
    }
    out
}

fn env_var_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "True"))
        .unwrap_or(false)
}
