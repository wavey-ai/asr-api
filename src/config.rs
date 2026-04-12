use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use std::path::{Path, PathBuf};
use upload_response::UploadResponseConfig;
use web_service::{load_default_tls_base64, load_tls_base64_from_paths};

pub const ASR_SAMPLE_RATE: u32 = 16_000;
pub const DEFAULT_MODEL_NAME: &str = "wavey-parakeet-tdt-onnx";
pub const DEFAULT_LANGUAGE: &str = "en";

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum AppRole {
    Monolith,
    Ingress,
    Worker,
}

impl AppRole {
    pub fn uses_asr_backend(self) -> bool {
        matches!(self, Self::Monolith | Self::Worker)
    }

    pub fn serves_listen(self) -> bool {
        matches!(self, Self::Monolith | Self::Ingress)
    }

    pub fn exposes_upload_cache(self) -> bool {
        matches!(self, Self::Monolith | Self::Ingress)
    }

    pub fn runs_local_worker(self) -> bool {
        matches!(self, Self::Monolith)
    }

    pub fn runs_remote_worker(self) -> bool {
        matches!(self, Self::Worker)
    }
}

#[derive(Debug, Clone, Parser)]
#[command(
    name = "asr-api",
    about = "Deepgram-compatible ASR service over Wavey's web-service stack"
)]
pub struct AppConfig {
    #[arg(long, env = "ASR_API_ROLE", value_enum, default_value_t = AppRole::Monolith)]
    pub role: AppRole,

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
    pub model_dir: Option<PathBuf>,

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

    #[arg(long, env = "UPLOAD_RESPONSE_NUM_STREAMS", default_value_t = 16)]
    pub upload_response_num_streams: usize,

    #[arg(long, env = "UPLOAD_RESPONSE_SLOT_SIZE_KB", default_value_t = 32)]
    pub upload_response_slot_size_kb: usize,

    #[arg(long, env = "UPLOAD_RESPONSE_SLOTS_PER_STREAM", default_value_t = 1_024)]
    pub upload_response_slots_per_stream: usize,

    #[arg(
        long,
        env = "UPLOAD_RESPONSE_TIMEOUT_MS",
        default_value_t = 30_000
    )]
    pub upload_response_timeout_ms: u64,

    #[arg(
        long,
        env = "UPLOAD_RESPONSE_WATCH_POLL_MS",
        default_value_t = 1
    )]
    pub upload_response_watch_poll_ms: u64,

    #[arg(
        long,
        env = "UPLOAD_RESPONSE_WORKER_POLL_MS",
        default_value_t = 2
    )]
    pub upload_response_worker_poll_ms: u64,

    #[arg(
        long,
        env = "UPLOAD_RESPONSE_MAX_INFLIGHT",
        default_value_t = 2
    )]
    pub upload_response_max_inflight: usize,

    #[arg(
        long,
        env = "UPLOAD_RESPONSE_WORKER_ID",
        default_value = "asr-api-monolith"
    )]
    pub upload_response_worker_id: String,

    #[arg(long, env = "UPLOAD_RESPONSE_INGRESS_URLS", value_delimiter = ',')]
    pub upload_response_ingress_urls: Vec<String>,

    #[arg(long, env = "UPLOAD_RESPONSE_DISCOVERY_DNS")]
    pub upload_response_discovery_dns: Option<String>,

    #[arg(
        long,
        env = "UPLOAD_RESPONSE_DISCOVERY_INTERVAL_MS",
        default_value_t = 2_000
    )]
    pub upload_response_discovery_interval_ms: u64,

    #[arg(long, env = "UPLOAD_RESPONSE_INSECURE_TLS", default_value_t = false)]
    pub upload_response_insecure_tls: bool,
}

impl AppConfig {
    pub fn validate(&self) -> Result<()> {
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
        anyhow::ensure!(
            self.upload_response_num_streams > 0,
            "UPLOAD_RESPONSE_NUM_STREAMS must be > 0"
        );
        anyhow::ensure!(
            self.upload_response_slot_size_kb > 0,
            "UPLOAD_RESPONSE_SLOT_SIZE_KB must be > 0"
        );
        anyhow::ensure!(
            self.upload_response_slots_per_stream > 0,
            "UPLOAD_RESPONSE_SLOTS_PER_STREAM must be > 0"
        );
        anyhow::ensure!(
            self.upload_response_timeout_ms > 0,
            "UPLOAD_RESPONSE_TIMEOUT_MS must be > 0"
        );
        anyhow::ensure!(
            self.upload_response_max_inflight > 0,
            "UPLOAD_RESPONSE_MAX_INFLIGHT must be > 0"
        );
        anyhow::ensure!(
            !self.upload_response_worker_id.trim().is_empty(),
            "UPLOAD_RESPONSE_WORKER_ID must not be empty"
        );
        anyhow::ensure!(
            self.upload_response_discovery_interval_ms > 0,
            "UPLOAD_RESPONSE_DISCOVERY_INTERVAL_MS must be > 0"
        );

        if self.role.uses_asr_backend() {
            let model_dir = self.model_dir()?;
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

            let tokens_path = model_dir.join("tokens.txt");
            if !tokens_path.is_file() {
                let vocab_path = self.resolve_vocab_path()?;
                anyhow::ensure!(
                    vocab_path.is_file(),
                    "ASR_VOCAB_PATH must point to a readable file when tokens.txt is absent"
                );
            }
        }

        if self.role.runs_remote_worker() {
            anyhow::ensure!(
                !self.upload_response_ingress_urls.is_empty()
                    || self
                        .upload_response_discovery_dns
                        .as_ref()
                        .map(|value| !value.trim().is_empty())
                        .unwrap_or(false),
                "worker role requires UPLOAD_RESPONSE_INGRESS_URLS or UPLOAD_RESPONSE_DISCOVERY_DNS"
            );
        }

        Ok(())
    }

    pub fn resolve_vocab_path(&self) -> Result<PathBuf> {
        let model_dir = self.model_dir()?;
        let tokens_path = model_dir.join("tokens.txt");
        if tokens_path.is_file() {
            return Ok(tokens_path);
        }

        if let Some(path) = &self.vocab_path {
            return Ok(path.clone());
        }

        let default_vocab = model_dir.join("vocab.txt");
        anyhow::ensure!(
            default_vocab.is_file(),
            "model directory must contain tokens.txt or vocab.txt, or set ASR_VOCAB_PATH"
        );
        Ok(default_vocab)
    }

    pub fn model_dir(&self) -> Result<&Path> {
        let model_dir = self
            .model_dir
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("ASR_MODEL_DIR is required for this asr-api role"))?;
        anyhow::ensure!(model_dir.is_dir(), "ASR_MODEL_DIR must point to a directory");
        Ok(model_dir)
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

    pub fn upload_response_config(&self) -> UploadResponseConfig {
        UploadResponseConfig {
            num_streams: self.upload_response_num_streams,
            slot_size_kb: self.upload_response_slot_size_kb,
            slots_per_stream: self.upload_response_slots_per_stream,
            response_timeout_ms: self.upload_response_timeout_ms,
        }
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
