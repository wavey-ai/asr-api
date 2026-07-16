use crate::asr::WindowTranscription;
use crate::chunking::TimedWord;
use crate::cohere_frontend::{CohereFrontend, CoherePreprocessorConfig};
use crate::config::ASR_SAMPLE_RATE;
use crate::timestamps::{duration_ms_for_samples, estimate_word_timestamps_from_token_count};
use anyhow::{Context, Result};
use crossbeam_channel::{bounded, Sender};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
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
        let runtime = resolve_runtime_binary()?;

        let model_dir = model_dir.to_path_buf();
        let (sender, receiver) = bounded::<MlxJob>(2);
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();

        thread::spawn(move || {
            let mut worker = match CohereMlxWorker::new(runtime, model_dir, max_new_tokens) {
                Ok(worker) => {
                    let _ = ready_tx.send(Ok(()));
                    worker
                }
                Err(error) => {
                    let _ = ready_tx.send(Err(error.to_string()));
                    return;
                }
            };

            for (id, seq, samples, sender) in receiver {
                let result = worker
                    .transcribe(id, seq, samples)
                    .map_err(|error| error.to_string());
                let _ = sender.send(result);
            }
        });

        let ready = ready_rx
            .recv_timeout(Duration::from_secs(120))
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
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    frontend: CohereFrontend,
}

impl CohereMlxWorker {
    fn new(runtime: PathBuf, model_dir: PathBuf, max_new_tokens: usize) -> Result<Self> {
        anyhow::ensure!(
            runtime.is_file(),
            "Cohere MLX runtime binary not found at {}; run `swift build -c release` in asr-api/apple or set ASR_MLX_TRANSCRIBE_BIN",
            runtime.display()
        );
        let preprocessor =
            load_json::<CoherePreprocessorConfig>(&model_dir.join("preprocessor_config.json"))
                .context("failed to load Cohere preprocessor_config.json")?;
        let frontend = CohereFrontend::new(preprocessor)?;
        let mut child = Command::new(&runtime)
            .arg("--model-dir")
            .arg(&model_dir)
            .arg("--server")
            .arg("--max-new-tokens")
            .arg(max_new_tokens.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("failed to start {}", runtime.display()))?;
        let stdin = child
            .stdin
            .take()
            .context("Cohere Swift MLX runtime did not open stdin")?;
        let stdout = child
            .stdout
            .take()
            .context("Cohere Swift MLX runtime did not open stdout")?;
        let mut stdout = BufReader::new(stdout);
        let mut ready_line = String::new();
        anyhow::ensure!(
            stdout.read_line(&mut ready_line)? > 0,
            "Cohere Swift MLX runtime exited before initialization"
        );
        let ready: SwiftServerReady = serde_json::from_str(&ready_line)
            .context("failed to parse Cohere Swift MLX runtime ready response")?;
        anyhow::ensure!(ready.ready, "Cohere Swift MLX runtime was not ready");
        info!(runtime = %runtime.display(), "initialized persistent Cohere Swift MLX backend");
        Ok(Self {
            child,
            stdin,
            stdout,
            frontend,
        })
    }

    fn transcribe(&mut self, id: u64, seq: u32, samples: Vec<f32>) -> Result<WindowTranscription> {
        let started = Instant::now();
        let features = self.frontend.compute(&samples)?;
        let feature_shape = features.dim();
        let feature_path = write_temp_features(id, seq, &features)?;
        let response = (|| -> Result<SwiftServerResponse> {
            serde_json::to_writer(
                &mut self.stdin,
                &SwiftServerRequest {
                    features_f32le: feature_path.to_string_lossy().as_ref(),
                    feature_count: feature_shape.0,
                    feature_steps: feature_shape.1,
                },
            )?;
            self.stdin.write_all(b"\n")?;
            self.stdin.flush()?;

            let mut line = String::new();
            anyhow::ensure!(
                self.stdout.read_line(&mut line)? > 0,
                "Cohere Swift MLX runtime exited while transcribing"
            );
            serde_json::from_str(&line)
                .context("failed to parse Cohere Swift MLX runtime server response")
        })();
        let _ = fs::remove_file(&feature_path);
        let response = response?;
        if !response.ok {
            anyhow::bail!(
                "Cohere Swift MLX runtime failed: {}",
                response.error.as_deref().unwrap_or("unknown server error")
            );
        }
        let response = response
            .result
            .context("Cohere Swift MLX runtime omitted its transcription result")?;
        let mut words = response
            .words
            .unwrap_or_default()
            .into_iter()
            .map(|word| TimedWord {
                word: word.word,
                start_ms: word.start_ms,
                end_ms: word.end_ms,
            })
            .collect::<Vec<_>>();
        if words.is_empty() {
            words = estimate_word_timestamps_from_token_count(
                &response.text,
                response
                    .token_ids
                    .as_ref()
                    .map(|tokens| tokens.len())
                    .unwrap_or_default(),
                duration_ms_for_samples(samples.len(), ASR_SAMPLE_RATE),
            );
        }

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
            text: response.text,
            words,
        })
    }
}

impl Drop for CohereMlxWorker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[derive(Debug, Deserialize)]
struct SwiftServerReady {
    ready: bool,
}

#[derive(Debug, Serialize)]
struct SwiftServerRequest<'a> {
    features_f32le: &'a str,
    feature_count: usize,
    feature_steps: usize,
}

#[derive(Debug, Deserialize)]
struct SwiftServerResponse {
    ok: bool,
    result: Option<SwiftTranscription>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SwiftTranscription {
    text: String,
    words: Option<Vec<SwiftTimedWord>>,
    token_ids: Option<Vec<u32>>,
}

#[derive(Debug, Deserialize)]
struct SwiftTimedWord {
    word: String,
    start_ms: u32,
    end_ms: u32,
}

fn validate_model_dir(model_dir: &Path) -> Result<()> {
    for file in [
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "vocab.json",
    ] {
        anyhow::ensure!(
            model_dir.join(file).is_file(),
            "Cohere MLX backend requires {}; sync the Hugging Face safetensors bundle and generate vocab.json",
            model_dir.join(file).display()
        );
    }
    Ok(())
}

fn resolve_runtime_binary() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("ASR_MLX_TRANSCRIBE_BIN") {
        return Ok(PathBuf::from(path));
    }
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("apple")
        .join(".build")
        .join("release")
        .join("asr-mlx-transcribe"))
}

fn write_temp_features(id: u64, seq: u32, features: &Array2<f32>) -> Result<PathBuf> {
    let path = std::env::temp_dir().join(format!(
        "asr-api-cohere-mlx-features-{}-{id}-{seq}.f32le",
        std::process::id()
    ));
    let mut file =
        fs::File::create(&path).with_context(|| format!("failed to create {}", path.display()))?;
    let slice = features
        .as_slice_memory_order()
        .context("Cohere MLX feature tensor is not contiguous")?;
    for value in slice {
        file.write_all(&value.to_le_bytes())
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    Ok(path)
}

fn load_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("failed to parse {}", path.display()))
}

fn env_var_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "True"))
        .unwrap_or(false)
}
