use anyhow::{Context, Result};
use clap::Parser;
use std::path::{Path, PathBuf};
use web_service::{load_default_tls_base64, load_tls_base64_from_paths};

pub const ASR_SAMPLE_RATE: u32 = 16_000;
pub const DEFAULT_MODEL_NAME: &str = "wavey-parakeet-tdt-onnx";
pub const DEFAULT_LANGUAGE: &str = "en";

#[derive(Debug, Clone, Parser)]
#[command(
    name = "transcriber",
    about = "Deepgram-compatible ASR service over Wavey's web-service stack"
)]
pub struct AppConfig {
    #[arg(long, env = "RUST_LOG", default_value = "info")]
    pub rust_log: String,

    #[arg(long, env = "PORT", default_value_t = 8443)]
    pub port: u16,

    #[arg(long, env = "ENABLE_H3", default_value_t = false)]
    pub enable_h3: bool,

    #[arg(long, env = "TLS_CERT_PATH")]
    pub tls_cert_path: Option<PathBuf>,

    #[arg(long, env = "TLS_KEY_PATH")]
    pub tls_key_path: Option<PathBuf>,

    #[arg(long = "model-dir", env = "ASR_MODEL_DIR")]
    pub model_dir: PathBuf,

    #[arg(long = "vocab-path", env = "ASR_VOCAB_PATH")]
    pub vocab_path: Option<PathBuf>,

    #[arg(
        long,
        env = "ASR_DEVICE_IDS",
        value_delimiter = ',',
        default_value = "0"
    )]
    pub device_ids: Vec<usize>,

    #[arg(long, env = "ASR_TORCH_SESSIONS", default_value_t = 1)]
    pub torch_sessions: usize,

    #[arg(long, env = "ASR_ONNX_SESSIONS", default_value_t = 1)]
    pub onnx_sessions: usize,

    #[arg(long, env = "CHUNK_SECONDS", default_value_t = 30.0)]
    pub chunk_seconds: f32,

    #[arg(long, env = "OVERLAP_SECONDS", default_value_t = 2.0)]
    pub overlap_seconds: f32,

    #[arg(long, env = "FINAL_MIN_SECONDS", default_value_t = 0.5)]
    pub final_min_seconds: f32,

    #[arg(long, env = "UTT_SPLIT_SECONDS", default_value_t = 0.8)]
    pub utt_split_seconds: f64,
}

impl AppConfig {
    pub fn validate(&self) -> Result<()> {
        let model_dir = &self.model_dir;
        anyhow::ensure!(
            model_dir.is_dir(),
            "ASR_MODEL_DIR must point to a directory"
        );
        anyhow::ensure!(
            !self.device_ids.is_empty(),
            "ASR_DEVICE_IDS must include at least one device id"
        );
        anyhow::ensure!(
            self.torch_sessions > 0,
            "ASR_TORCH_SESSIONS must be greater than 0"
        );
        anyhow::ensure!(
            self.onnx_sessions > 0,
            "ASR_ONNX_SESSIONS must be greater than 0"
        );
        anyhow::ensure!(self.chunk_seconds > 0.0, "CHUNK_SECONDS must be > 0");
        anyhow::ensure!(self.overlap_seconds >= 0.0, "OVERLAP_SECONDS must be >= 0");
        anyhow::ensure!(
            self.chunk_seconds > self.overlap_seconds,
            "CHUNK_SECONDS must be larger than OVERLAP_SECONDS"
        );
        anyhow::ensure!(
            self.final_min_seconds >= 0.0,
            "FINAL_MIN_SECONDS must be >= 0"
        );
        anyhow::ensure!(
            self.utt_split_seconds >= 0.0,
            "UTT_SPLIT_SECONDS must be >= 0"
        );

        ensure_any_exists(
            model_dir,
            &["encoder.fp16.onnx", "encoder.onnx", "encoder.int8.onnx"],
            "encoder",
        )?;
        ensure_any_exists(
            model_dir,
            &["decoder.fp16.onnx", "decoder.onnx", "decoder.int8.onnx"],
            "decoder",
        )?;
        ensure_any_exists(
            model_dir,
            &[
                "joint.enc.fp16.onnx",
                "joint.enc.onnx",
                "joint.enc.int8.onnx",
            ],
            "joint.enc",
        )?;
        ensure_any_exists(
            model_dir,
            &[
                "joint.pred.fp16.onnx",
                "joint.pred.onnx",
                "joint.pred.int8.onnx",
            ],
            "joint.pred",
        )?;
        ensure_any_exists(
            model_dir,
            &[
                "joint.joint_net.fp16.onnx",
                "joint.joint_net.onnx",
                "joint.joint_net.int8.onnx",
            ],
            "joint.joint_net",
        )?;

        let tokens_path = self.model_dir.join("tokens.txt");
        if !tokens_path.is_file() {
            let vocab_path = self.resolve_vocab_path()?;
            anyhow::ensure!(
                vocab_path.is_file(),
                "ASR_VOCAB_PATH must point to a readable file when tokens.txt is absent"
            );
        }

        Ok(())
    }

    pub fn resolve_vocab_path(&self) -> Result<PathBuf> {
        let tokens_path = self.model_dir.join("tokens.txt");
        if tokens_path.is_file() {
            return Ok(tokens_path);
        }

        if let Some(path) = &self.vocab_path {
            return Ok(path.clone());
        }

        let default_vocab = self.model_dir.join("vocab.txt");
        anyhow::ensure!(
            default_vocab.is_file(),
            "model directory must contain tokens.txt or vocab.txt, or set ASR_VOCAB_PATH"
        );
        Ok(default_vocab)
    }

    pub fn tls_base64(&self) -> Result<(String, String)> {
        match (&self.tls_cert_path, &self.tls_key_path) {
            (Some(cert), Some(key)) => load_tls_base64_from_paths(cert, key).with_context(|| {
                format!(
                    "failed to load TLS PEMs from {} and {}",
                    cert.display(),
                    key.display()
                )
            }),
            (None, None) => load_default_tls_base64()
                .context("failed to load default local Wavey TLS certificate"),
            _ => anyhow::bail!("set both TLS_CERT_PATH and TLS_KEY_PATH, or neither"),
        }
    }

    pub fn chunk_samples(&self) -> usize {
        seconds_to_samples(self.chunk_seconds)
    }

    pub fn overlap_samples(&self) -> usize {
        seconds_to_samples(self.overlap_seconds)
    }

    pub fn min_final_samples(&self) -> usize {
        seconds_to_samples(self.final_min_seconds)
    }
}

fn ensure_any_exists(model_dir: &Path, candidates: &[&str], label: &str) -> Result<()> {
    if candidates.iter().any(|name| model_dir.join(name).is_file()) {
        return Ok(());
    }

    anyhow::bail!(
        "missing {label} model component in {} (checked: {})",
        model_dir.display(),
        candidates.join(", ")
    )
}

fn seconds_to_samples(seconds: f32) -> usize {
    (seconds.max(0.0) * ASR_SAMPLE_RATE as f32).round() as usize
}
