use crate::chunking::TimedWord;
use crate::config::ASR_SAMPLE_RATE;
use anyhow::{Context, Result};
use crossbeam_channel::{bounded, Receiver, Sender};
use mel_spec::mel::{BatchLogMelConfig, BatchLogMelScratch, BatchLogMelSpectrogram};
use ndarray::{Array2, ArrayD, IxDyn};
use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, ExecutionProvider, ExecutionProviderDispatch,
    TensorRTExecutionProvider,
};
use ort::logging::LogLevel;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor as OrtTensor;
use serde::Deserialize;
use std::env;
use std::fs;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;
use tracing::{info, warn};

static ORT_INIT: OnceLock<Result<(), String>> = OnceLock::new();

const DEFAULT_CTC_MODEL_DIR: &str = "models/parakeet-ctc-0.6b-onnx";
const DEFAULT_CTC_ONNX_FILE: &str = "onnx/model.onnx";
const DEFAULT_CTC_ALIGN_SESSIONS: usize = 1;
const DIRECT_ANCHOR_MAX_GROUP_WORDS: usize = 3;
const DEFAULT_DIRECT_MIN_MATCH_RATIO: f32 = 0.65;
const DEFAULT_DIRECT_MIN_MATCHED_WORDS: usize = 2;
const DEFAULT_DIRECT_MAX_UNMATCHED_WORDS: usize = 8;

#[derive(Debug, Deserialize)]
struct ParakeetCtcModelConfig {
    encoder_config: Option<ParakeetCtcEncoderConfig>,
    pad_token_id: Option<usize>,
    vocab_size: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ParakeetCtcEncoderConfig {
    num_mel_bins: Option<usize>,
    subsampling_factor: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ParakeetCtcPreprocessorConfig {
    feature_size: Option<usize>,
    hop_length: Option<usize>,
    n_fft: Option<usize>,
    preemphasis: Option<f32>,
    sampling_rate: Option<usize>,
    win_length: Option<usize>,
}

#[derive(Debug, Clone)]
struct ParakeetCtcRuntimeConfig {
    model_dir: PathBuf,
    onnx_file: PathBuf,
    execution_provider: CtcExecutionProvider,
    cache_dir: PathBuf,
    min_duration_s: usize,
    opt_duration_s: usize,
    max_duration_s: usize,
    workspace_bytes: usize,
    builder_optimization_level: u8,
    fp16: bool,
    detailed_build_log: bool,
    sample_rate: usize,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    n_mels: usize,
    preemphasis: f32,
    log_zero_guard: f32,
    pad_to: usize,
    normalize_per_feature: bool,
    subsampling_factor: usize,
    blank_id: usize,
    vocab_size: usize,
    timestamp_offset_ms: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CtcTimestampMode {
    ForcedAlignment,
    DirectAnchors,
}

#[derive(Debug, Clone)]
struct DirectAnchorConfig {
    min_match_ratio: f32,
    min_matched_words: usize,
    max_unmatched_words: usize,
    start_offset_ms: i32,
    end_offset_ms: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CtcExecutionProvider {
    Auto,
    TensorRt,
    Cuda,
    Cpu,
}

#[derive(Debug)]
struct AlignmentTarget {
    token_ids: Vec<usize>,
    words: Vec<AlignmentWord>,
}

#[derive(Debug)]
struct AlignmentWord {
    original: String,
    token_start: usize,
    token_end: usize,
}

pub struct ParakeetCtcAligner {
    frontend: BatchLogMelSpectrogram,
    scratch: Mutex<BatchLogMelScratch>,
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    config: ParakeetCtcRuntimeConfig,
    input_names: Vec<String>,
    output_name: String,
}

pub(crate) struct ParakeetCtcTimestampEngine {
    aligners: LeasePool<ParakeetCtcAligner>,
    mode: CtcTimestampMode,
    direct_config: Option<DirectAnchorConfig>,
}

struct LeasePool<T> {
    available_tx: Sender<T>,
    available_rx: Receiver<T>,
    capacity: usize,
}

struct PoolLease<'a, T> {
    return_tx: &'a Sender<T>,
    value: Option<T>,
}

#[derive(Debug, Clone)]
pub struct ParakeetCtcTranscription {
    pub text: String,
    pub words: Vec<TimedWord>,
}

#[derive(Debug, Clone)]
struct CtcTokenFrame {
    id: usize,
    start: usize,
    end: usize,
}

impl ParakeetCtcTimestampEngine {
    pub(crate) fn from_env(device_ids: &[usize]) -> Result<Option<Self>> {
        let Some(mode) = ctc_timestamp_mode()? else {
            return Ok(None);
        };

        let direct_config = match mode {
            CtcTimestampMode::ForcedAlignment => None,
            CtcTimestampMode::DirectAnchors => Some(DirectAnchorConfig::from_env()?),
        };
        let session_count = ctc_align_session_count()?;
        let aligners =
            LeasePool::try_initialize(session_count, "Parakeet CTC aligner session", |_| {
                ParakeetCtcAligner::new(device_ids)
            })?;
        info!(
            sessions = aligners.capacity,
            "initialized Parakeet CTC timestamp aligner pool"
        );
        Ok(Some(Self {
            aligners,
            mode,
            direct_config,
        }))
    }

    pub(crate) fn timestamp_words(
        &self,
        samples: &[f32],
        text: &str,
        reference_words: &[TimedWord],
    ) -> Result<Vec<TimedWord>> {
        let total_started = Instant::now();
        let lease_started = Instant::now();
        let aligner = self.aligners.lease();
        let lease_wait = lease_started.elapsed();
        let ctc_started = Instant::now();
        let (result, ctc_elapsed, post_elapsed) = match self.mode {
            CtcTimestampMode::ForcedAlignment => {
                let result = aligner.align(samples, text);
                (result, ctc_started.elapsed(), Duration::ZERO)
            }
            CtcTimestampMode::DirectAnchors => {
                let duration_ms =
                    duration_ms_for_samples(samples.len(), aligner.config.sample_rate);
                match aligner.transcribe(samples) {
                    Ok(transcription) => {
                        let ctc_elapsed = ctc_started.elapsed();
                        let post_started = Instant::now();
                        let result = direct_anchor_reference_words(
                            reference_words,
                            &transcription.words,
                            duration_ms,
                            self.direct_config
                                .as_ref()
                                .expect("direct CTC mode must have direct anchor configuration"),
                        );
                        (result, ctc_elapsed, post_started.elapsed())
                    }
                    Err(error) => (Err(error), ctc_started.elapsed(), Duration::ZERO),
                }
            }
        };
        if env_var_truthy("ASR_CTC_ALIGN_TIMINGS") {
            eprintln!(
                "ctc_timestamp_timing mode={} status={} lease_wait_ms={:.2} ctc_ms={:.2} post_ms={:.2} total_ms={:.2} words={}",
                self.mode_name(),
                if result.is_ok() { "ok" } else { "error" },
                lease_wait.as_secs_f64() * 1000.0,
                ctc_elapsed.as_secs_f64() * 1000.0,
                post_elapsed.as_secs_f64() * 1000.0,
                total_started.elapsed().as_secs_f64() * 1000.0,
                result.as_ref().map(|words| words.len()).unwrap_or(0),
            );
        }
        result
    }

    pub(crate) fn mode_name(&self) -> &'static str {
        match self.mode {
            CtcTimestampMode::ForcedAlignment => "forced",
            CtcTimestampMode::DirectAnchors => "direct",
        }
    }
}

impl<T> LeasePool<T> {
    fn try_initialize(
        capacity: usize,
        resource_name: &str,
        mut initialize: impl FnMut(usize) -> Result<T>,
    ) -> Result<Self> {
        anyhow::ensure!(capacity > 0, "{resource_name} count must be positive");

        let mut resources = Vec::new();
        for index in 0..capacity {
            let resource = initialize(index).with_context(|| {
                format!(
                    "failed to initialize {resource_name} {} of {capacity}; initialized {index} before failure",
                    index + 1
                )
            })?;
            resources.push(resource);
        }

        let (available_tx, available_rx) = bounded(capacity);
        for resource in resources {
            available_tx
                .send(resource)
                .expect("new lease pool receiver must remain connected");
        }
        Ok(Self {
            available_tx,
            available_rx,
            capacity,
        })
    }

    fn lease(&self) -> PoolLease<'_, T> {
        let value = self
            .available_rx
            .recv()
            .expect("lease pool sender must remain connected");
        PoolLease {
            return_tx: &self.available_tx,
            value: Some(value),
        }
    }

    #[cfg(test)]
    fn available(&self) -> usize {
        self.available_rx.len()
    }
}

impl<T> Deref for PoolLease<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value
            .as_ref()
            .expect("leased resource must exist until the lease is dropped")
    }
}

impl<T> Drop for PoolLease<'_, T> {
    fn drop(&mut self) {
        let Some(value) = self.value.take() else {
            return;
        };
        assert!(
            self.return_tx.send(value).is_ok(),
            "lease pool receiver must remain connected"
        );
    }
}

impl ParakeetCtcAligner {
    pub fn new(device_ids: &[usize]) -> Result<Self> {
        ensure_ort_initialized()?;
        let config = ParakeetCtcRuntimeConfig::from_env()?;
        fs::create_dir_all(&config.cache_dir).with_context(|| {
            format!(
                "failed to create Parakeet CTC TensorRT cache dir {}",
                config.cache_dir.display()
            )
        })?;

        let frontend = BatchLogMelSpectrogram::new(BatchLogMelConfig {
            sample_rate: config.sample_rate,
            n_fft: config.n_fft,
            win_length: config.win_length,
            hop_length: config.hop_length,
            n_mels: config.n_mels,
            f_min: 0.0,
            f_max: Some(config.sample_rate as f64 / 2.0),
            htk: false,
            norm: true,
            preemphasis: config.preemphasis,
            center: true,
            log_zero_guard: config.log_zero_guard,
            pad_to: config.pad_to,
            normalize_per_feature: config.normalize_per_feature,
        })
        .context("failed to initialize Parakeet CTC mel frontend")?;
        let scratch = Mutex::new(frontend.scratch());
        let tokenizer = Tokenizer::from_file(config.model_dir.join("tokenizer.json"))
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .context("failed to load Parakeet CTC tokenizer.json")?;

        let device_id = ctc_device_id(device_ids);
        validate_provider(&config, device_id)?;
        let providers = ctc_provider_chain(&config, device_id);
        let session = session_from_providers(&config.onnx_file, &providers)
            .with_context(|| format!("failed to initialize {}", config.onnx_file.display()))?;
        let input_names = session
            .inputs()
            .iter()
            .map(|input| input.name().to_string())
            .collect::<Vec<_>>();
        anyhow::ensure!(
            input_names.len() >= 2,
            "Parakeet CTC ONNX model must expose input_features and attention_mask inputs"
        );
        let output_name = session
            .outputs()
            .first()
            .map(|output| output.name().to_string())
            .context("Parakeet CTC ONNX model exposes no outputs")?;

        info!(
            model = %config.onnx_file.display(),
            provider = ?config.execution_provider,
            device_id = ?device_id,
            blank_id = config.blank_id,
            vocab_size = config.vocab_size,
            input_names = ?input_names,
            output_name = %output_name,
            "initialized Parakeet CTC timestamp aligner"
        );

        Ok(Self {
            frontend,
            scratch,
            session: Mutex::new(session),
            tokenizer,
            config,
            input_names,
            output_name,
        })
    }

    pub fn align(&self, samples: &[f32], text: &str) -> Result<Vec<TimedWord>> {
        let duration_ms = duration_ms_for_samples(samples.len(), self.config.sample_rate);
        if duration_ms == 0 || text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let target = self.prepare_target(text)?;
        if target.token_ids.is_empty() || target.words.is_empty() {
            return Ok(Vec::new());
        }

        let features = self.compute_features(samples)?;
        let logits = self.run_ctc(&features)?;
        let (logit_values, frame_count, vocab_size) =
            logits_to_time_major(&logits, self.config.vocab_size)?;
        anyhow::ensure!(
            vocab_size > self.config.blank_id,
            "Parakeet CTC blank id {} is outside logits vocab size {}",
            self.config.blank_id,
            vocab_size
        );
        anyhow::ensure!(
            target.token_ids.len() <= frame_count,
            "Parakeet CTC target has {} tokens but only {} acoustic frames",
            target.token_ids.len(),
            frame_count
        );
        let max_token_id = target.token_ids.iter().copied().max().unwrap_or(0);
        anyhow::ensure!(
            max_token_id < vocab_size,
            "Parakeet CTC target token id {} is outside logits vocab size {}",
            max_token_id,
            vocab_size
        );
        if env_var_truthy("ASR_CTC_ALIGN_TIMINGS") {
            print_logit_stats(
                &logit_values,
                frame_count,
                vocab_size,
                &target.token_ids,
                self.config.blank_id,
            );
        }

        let token_frames = forced_align_tokens(
            &logit_values,
            frame_count,
            vocab_size,
            &target.token_ids,
            self.config.blank_id,
        )?;
        Ok(words_from_token_frames(
            &target,
            &token_frames,
            frame_count,
            duration_ms,
            self.config.hop_length,
            self.config.sample_rate,
            self.config.subsampling_factor,
            self.config.timestamp_offset_ms,
        ))
    }

    pub fn transcribe(&self, samples: &[f32]) -> Result<ParakeetCtcTranscription> {
        let duration_ms = duration_ms_for_samples(samples.len(), self.config.sample_rate);
        if duration_ms == 0 {
            return Ok(ParakeetCtcTranscription {
                text: String::new(),
                words: Vec::new(),
            });
        }

        let features = self.compute_features(samples)?;
        let logits = self.run_ctc(&features)?;
        let (logit_values, frame_count, vocab_size) =
            logits_to_time_major(&logits, self.config.vocab_size)?;
        anyhow::ensure!(
            vocab_size > self.config.blank_id,
            "Parakeet CTC blank id {} is outside logits vocab size {}",
            self.config.blank_id,
            vocab_size
        );

        let token_frames =
            greedy_ctc_token_frames(&logit_values, frame_count, vocab_size, self.config.blank_id)?;
        let words = greedy_words_from_token_frames(
            &self.tokenizer,
            &token_frames,
            duration_ms,
            self.config.hop_length,
            self.config.sample_rate,
            self.config.subsampling_factor,
            self.config.timestamp_offset_ms,
        );
        let text = words
            .iter()
            .map(|word| word.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        Ok(ParakeetCtcTranscription { text, words })
    }

    fn prepare_target(&self, text: &str) -> Result<AlignmentTarget> {
        let mut token_ids = Vec::new();
        let mut words = Vec::new();

        for raw_word in text.split_whitespace() {
            let normalized = normalize_ctc_word(raw_word);
            if normalized.is_empty() {
                continue;
            }

            let encoding = self
                .tokenizer
                .encode(normalized.as_str(), false)
                .map_err(|error| anyhow::anyhow!(error.to_string()))
                .with_context(|| format!("failed to tokenize Parakeet CTC word {normalized:?}"))?;
            let ids = encoding
                .get_ids()
                .iter()
                .filter_map(|&id| {
                    let id = id as usize;
                    (id != self.config.blank_id).then_some(id)
                })
                .collect::<Vec<_>>();
            if ids.is_empty() {
                continue;
            }

            let token_start = token_ids.len();
            token_ids.extend(ids);
            let token_end = token_ids.len();
            words.push(AlignmentWord {
                original: raw_word.to_string(),
                token_start,
                token_end,
            });
        }

        Ok(AlignmentTarget { token_ids, words })
    }

    fn compute_features(&self, samples: &[f32]) -> Result<Array2<f32>> {
        let mut scratch = self
            .scratch
            .lock()
            .expect("Parakeet CTC frontend scratch mutex poisoned");
        let output = self
            .frontend
            .compute_flat_with_scratch(samples, &mut scratch)
            .context("failed to compute Parakeet CTC mel features")?;
        Array2::from_shape_vec((output.rows, output.cols), output.data)
            .context("failed to shape Parakeet CTC mel features")
    }

    fn run_ctc(&self, features: &Array2<f32>) -> Result<ArrayD<f32>> {
        let (rows, cols) = features.dim();
        let mut flat = Vec::with_capacity(rows * cols);
        for frame in 0..cols {
            for mel in 0..rows {
                flat.push(features[(mel, frame)]);
            }
        }
        let feature_tensor = OrtTensor::from_array(([1, cols as i64, rows as i64], flat))?;
        let attention_mask = OrtTensor::from_array(([1, cols as i64], vec![1i64; cols]))?;
        let inputs = ort::inputs! {
            self.input_names[0].as_str() => feature_tensor,
            self.input_names[1].as_str() => attention_mask,
        };
        let mut session = self
            .session
            .lock()
            .expect("Parakeet CTC session mutex poisoned");
        let outputs = session
            .run(inputs)
            .context("Parakeet CTC ONNX run failed")?;
        anyhow::ensure!(
            outputs.len() > 0,
            "Parakeet CTC ONNX model returned no logits"
        );
        let output = outputs
            .get(self.output_name.as_str())
            .unwrap_or_else(|| &outputs[0]);
        output
            .try_extract_array::<f32>()
            .map(|array| array.to_owned())
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .context("failed to extract Parakeet CTC logits")
    }
}

impl ParakeetCtcRuntimeConfig {
    fn from_env() -> Result<Self> {
        let model_dir = env_var_nonempty("ASR_CTC_ALIGN_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_CTC_MODEL_DIR));
        let config_path = model_dir.join("config.json");
        let preprocessor_path = model_dir.join("preprocessor_config.json");
        let model_config = load_json::<ParakeetCtcModelConfig>(&config_path)
            .with_context(|| format!("failed to load {}", config_path.display()))?;
        let preprocessor = load_json::<ParakeetCtcPreprocessorConfig>(&preprocessor_path)
            .with_context(|| format!("failed to load {}", preprocessor_path.display()))?;
        let onnx_file = env_var_nonempty("ASR_CTC_ALIGN_ONNX_FILE")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_CTC_ONNX_FILE));
        let onnx_file = if onnx_file.is_absolute() {
            onnx_file
        } else {
            model_dir.join(onnx_file)
        };
        anyhow::ensure!(
            onnx_file.is_file(),
            "Parakeet CTC ONNX file does not exist: {}",
            onnx_file.display()
        );

        let sample_rate = preprocessor
            .sampling_rate
            .unwrap_or(ASR_SAMPLE_RATE as usize);
        anyhow::ensure!(
            sample_rate == ASR_SAMPLE_RATE as usize,
            "unsupported Parakeet CTC sample rate {}; expected {}",
            sample_rate,
            ASR_SAMPLE_RATE
        );
        let n_fft = preprocessor.n_fft.unwrap_or(512);
        let win_length = preprocessor.win_length.unwrap_or(400);
        let hop_length = preprocessor.hop_length.unwrap_or(160);
        let n_mels = preprocessor
            .feature_size
            .or_else(|| {
                model_config
                    .encoder_config
                    .as_ref()
                    .and_then(|config| config.num_mel_bins)
            })
            .unwrap_or(80);
        anyhow::ensure!(
            win_length <= n_fft,
            "Parakeet CTC win_length must be <= n_fft"
        );
        anyhow::ensure!(hop_length > 0, "Parakeet CTC hop_length must be > 0");
        anyhow::ensure!(n_mels > 0, "Parakeet CTC feature_size must be > 0");

        let cache_dir = env_var_nonempty("ASR_CTC_ALIGN_TRT_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| model_dir.join(".trt-cache").join("ctc-aligner"));

        Ok(Self {
            model_dir,
            onnx_file,
            execution_provider: ctc_execution_provider()?,
            cache_dir,
            min_duration_s: env_var_usize("ASR_CTC_ALIGN_TRT_MIN_DURATION_S").unwrap_or(1),
            opt_duration_s: env_var_usize("ASR_CTC_ALIGN_TRT_OPT_DURATION_S").unwrap_or(15),
            max_duration_s: env_var_usize("ASR_CTC_ALIGN_TRT_MAX_DURATION_S").unwrap_or(35),
            workspace_bytes: env_var_usize("ASR_CTC_ALIGN_TRT_WORKSPACE_BYTES")
                .unwrap_or(8 * 1024 * 1024 * 1024),
            builder_optimization_level: env_var_u8("ASR_CTC_ALIGN_TRT_BUILDER_OPT_LEVEL")
                .unwrap_or(5),
            fp16: env_var_truthy("ASR_CTC_ALIGN_TRT_FP16"),
            detailed_build_log: env_var_truthy("ASR_CTC_ALIGN_TRT_DETAILED_BUILD_LOG"),
            sample_rate,
            n_fft,
            win_length,
            hop_length,
            n_mels,
            preemphasis: preprocessor.preemphasis.unwrap_or(0.97),
            log_zero_guard: env_var_f32("ASR_CTC_ALIGN_LOG_ZERO_GUARD")
                .unwrap_or(2.0_f32.powi(-24)),
            pad_to: env_var_usize("ASR_CTC_ALIGN_PAD_TO").unwrap_or(0),
            normalize_per_feature: !env_var_falsey("ASR_CTC_ALIGN_NORMALIZE_PER_FEATURE"),
            subsampling_factor: model_config
                .encoder_config
                .as_ref()
                .and_then(|config| config.subsampling_factor)
                .unwrap_or(8),
            blank_id: model_config.pad_token_id.unwrap_or(1024),
            vocab_size: model_config.vocab_size.unwrap_or(1025),
            timestamp_offset_ms: env_var_i32("ASR_CTC_ALIGN_OFFSET_MS").unwrap_or(0),
        })
    }
}

impl DirectAnchorConfig {
    fn from_env() -> Result<Self> {
        let min_match_ratio =
            env_var_f32("ASR_CTC_DIRECT_MIN_MATCH_RATIO").unwrap_or(DEFAULT_DIRECT_MIN_MATCH_RATIO);
        anyhow::ensure!(
            min_match_ratio.is_finite() && (0.0..=1.0).contains(&min_match_ratio),
            "ASR_CTC_DIRECT_MIN_MATCH_RATIO must be between 0 and 1"
        );

        Ok(Self {
            min_match_ratio,
            min_matched_words: env_var_usize("ASR_CTC_DIRECT_MIN_MATCHED_WORDS")
                .unwrap_or(DEFAULT_DIRECT_MIN_MATCHED_WORDS)
                .max(1),
            max_unmatched_words: env_var_usize("ASR_CTC_DIRECT_MAX_UNMATCHED_WORDS")
                .unwrap_or(DEFAULT_DIRECT_MAX_UNMATCHED_WORDS),
            start_offset_ms: env_var_i32("ASR_CTC_DIRECT_START_OFFSET_MS").unwrap_or(0),
            end_offset_ms: env_var_i32("ASR_CTC_DIRECT_END_OFFSET_MS").unwrap_or(0),
        })
    }
}

fn validate_provider(config: &ParakeetCtcRuntimeConfig, device_id: Option<usize>) -> Result<()> {
    match config.execution_provider {
        CtcExecutionProvider::Cpu => Ok(()),
        CtcExecutionProvider::Cuda => {
            anyhow::ensure!(
                device_id.is_some(),
                "ASR_CTC_ALIGN_EXECUTION_PROVIDER=cuda requires ASR_DEVICE_IDS or ASR_CTC_ALIGN_DEVICE_ID"
            );
            anyhow::ensure!(
                CUDAExecutionProvider::default().is_available().unwrap_or(false),
                "Parakeet CTC CUDA requested, but the CUDA execution provider is unavailable in the linked ONNX Runtime build"
            );
            Ok(())
        }
        CtcExecutionProvider::TensorRt => {
            anyhow::ensure!(
                device_id.is_some(),
                "ASR_CTC_ALIGN_EXECUTION_PROVIDER=tensorrt requires ASR_DEVICE_IDS or ASR_CTC_ALIGN_DEVICE_ID"
            );
            anyhow::ensure!(
                TensorRTExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false),
                "Parakeet CTC TensorRT requested, but the TensorRT execution provider is unavailable; TensorRT requires a CUDA/NVIDIA ONNX Runtime build and does not run on macOS"
            );
            Ok(())
        }
        CtcExecutionProvider::Auto => Ok(()),
    }
}

fn ctc_provider_chain(
    config: &ParakeetCtcRuntimeConfig,
    device_id: Option<usize>,
) -> Vec<ExecutionProviderDispatch> {
    if matches!(config.execution_provider, CtcExecutionProvider::Cpu) {
        return vec![CPUExecutionProvider::default().build()];
    }

    let Some(device_id) = device_id else {
        return vec![CPUExecutionProvider::default().build()];
    };

    let mut providers = Vec::new();
    let trt_available = TensorRTExecutionProvider::default()
        .is_available()
        .unwrap_or(false);
    let cuda_available = CUDAExecutionProvider::default()
        .is_available()
        .unwrap_or(false);

    let wants_trt = matches!(
        config.execution_provider,
        CtcExecutionProvider::Auto | CtcExecutionProvider::TensorRt
    );
    if wants_trt && trt_available {
        let cache_dir = config.cache_dir.to_string_lossy().into_owned();
        let mut tensorrt = TensorRTExecutionProvider::default()
            .with_device_id(device_id as i32)
            .with_engine_cache(true)
            .with_engine_cache_path(&cache_dir)
            .with_engine_cache_prefix("parakeet_ctc")
            .with_timing_cache(true)
            .with_timing_cache_path(&cache_dir)
            .with_max_workspace_size(config.workspace_bytes)
            .with_builder_optimization_level(config.builder_optimization_level)
            .with_force_sequential_engine_build(true)
            .with_layer_norm_fp32_fallback(true)
            .with_detailed_build_log(config.detailed_build_log)
            .with_profile_min_shapes(ctc_profile_shapes(config, config.min_duration_s))
            .with_profile_opt_shapes(ctc_profile_shapes(config, config.opt_duration_s))
            .with_profile_max_shapes(ctc_profile_shapes(config, config.max_duration_s));
        if config.fp16 {
            tensorrt = tensorrt.with_fp16(true);
        }
        let provider = tensorrt.build();
        providers.push(
            if matches!(config.execution_provider, CtcExecutionProvider::TensorRt) {
                provider.error_on_failure()
            } else {
                provider.fail_silently()
            },
        );
    } else if wants_trt && matches!(config.execution_provider, CtcExecutionProvider::Auto) {
        warn!("Parakeet CTC TensorRT unavailable; falling back to CUDA/CPU");
    }

    let wants_cuda = matches!(
        config.execution_provider,
        CtcExecutionProvider::Auto | CtcExecutionProvider::Cuda | CtcExecutionProvider::TensorRt
    );
    if wants_cuda && cuda_available {
        let provider = CUDAExecutionProvider::default()
            .with_device_id(device_id as i32)
            .build();
        providers.push(
            if matches!(config.execution_provider, CtcExecutionProvider::Cuda) {
                provider.error_on_failure()
            } else {
                provider.fail_silently()
            },
        );
    } else if wants_cuda && matches!(config.execution_provider, CtcExecutionProvider::Auto) {
        warn!("Parakeet CTC CUDA unavailable; falling back to CPU");
    }

    providers.push(CPUExecutionProvider::default().build());
    providers
}

fn ctc_profile_shapes(config: &ParakeetCtcRuntimeConfig, seconds: usize) -> String {
    let feature_steps = ((config.sample_rate * seconds.max(1)) / config.hop_length).max(1);
    format!(
        "input_features:1x{}x{},attention_mask:1x{}",
        feature_steps, config.n_mels, feature_steps
    )
}

fn session_from_providers(path: &Path, providers: &[ExecutionProviderDispatch]) -> Result<Session> {
    let mut builder = Session::builder()
        .map_err(ort_error)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_error)?
        .with_log_level(LogLevel::Info)
        .map_err(ort_error)?
        .with_execution_providers(providers)
        .map_err(ort_error)?;

    if let Some(threads) =
        env_var_usize("ASR_CTC_ALIGN_INTRA_THREADS").filter(|threads| *threads > 0)
    {
        builder = builder.with_intra_threads(threads).map_err(ort_error)?;
    }
    if let Some(threads) =
        env_var_usize("ASR_CTC_ALIGN_INTER_THREADS").filter(|threads| *threads > 0)
    {
        builder = builder.with_inter_threads(threads).map_err(ort_error)?;
    }

    builder.commit_from_file(path).map_err(ort_error)
}

fn logits_to_time_major(
    logits: &ArrayD<f32>,
    expected_vocab: usize,
) -> Result<(Vec<f32>, usize, usize)> {
    let shape = logits.shape();
    anyhow::ensure!(
        shape.len() == 2 || shape.len() == 3,
        "expected Parakeet CTC logits rank 2 or 3, got shape {:?}",
        shape
    );

    if shape.len() == 3 {
        let batch = shape[0];
        anyhow::ensure!(
            batch == 1,
            "expected Parakeet CTC batch size 1, got {batch}"
        );
        if shape[2] == expected_vocab {
            let time = shape[1];
            let vocab = shape[2];
            let mut values = Vec::with_capacity(time * vocab);
            for t in 0..time {
                for v in 0..vocab {
                    values.push(*logits.get(IxDyn(&[0, t, v])).unwrap_or(&f32::NEG_INFINITY));
                }
            }
            return Ok((values, time, vocab));
        }
        if shape[1] == expected_vocab {
            let vocab = shape[1];
            let time = shape[2];
            let mut values = Vec::with_capacity(time * vocab);
            for t in 0..time {
                for v in 0..vocab {
                    values.push(*logits.get(IxDyn(&[0, v, t])).unwrap_or(&f32::NEG_INFINITY));
                }
            }
            return Ok((values, time, vocab));
        }
    } else if shape[1] == expected_vocab {
        let time = shape[0];
        let vocab = shape[1];
        let mut values = Vec::with_capacity(time * vocab);
        for t in 0..time {
            for v in 0..vocab {
                values.push(*logits.get(IxDyn(&[t, v])).unwrap_or(&f32::NEG_INFINITY));
            }
        }
        return Ok((values, time, vocab));
    } else if shape[0] == expected_vocab {
        let vocab = shape[0];
        let time = shape[1];
        let mut values = Vec::with_capacity(time * vocab);
        for t in 0..time {
            for v in 0..vocab {
                values.push(*logits.get(IxDyn(&[v, t])).unwrap_or(&f32::NEG_INFINITY));
            }
        }
        return Ok((values, time, vocab));
    }

    anyhow::bail!(
        "could not identify Parakeet CTC vocab axis in logits shape {:?} with expected vocab {}",
        shape,
        expected_vocab
    )
}

fn forced_align_tokens(
    logits: &[f32],
    frame_count: usize,
    vocab_size: usize,
    token_ids: &[usize],
    blank_id: usize,
) -> Result<Vec<Option<(usize, usize)>>> {
    anyhow::ensure!(!token_ids.is_empty(), "cannot align an empty CTC target");
    anyhow::ensure!(frame_count > 0, "cannot align with zero CTC frames");
    anyhow::ensure!(
        logits.len() == frame_count * vocab_size,
        "CTC logits length {} did not match frames*vocab {}",
        logits.len(),
        frame_count * vocab_size
    );

    let state_count = (token_ids.len() * 2) + 1;
    let mut labels = Vec::with_capacity(state_count);
    for &token in token_ids {
        labels.push(blank_id);
        labels.push(token);
    }
    labels.push(blank_id);

    let neg_inf = f32::NEG_INFINITY;
    let mut backpointers = vec![0usize; frame_count * state_count];
    let mut prev = vec![neg_inf; state_count];
    prev[0] = logit_score(logits, vocab_size, 0, labels[0]);
    if state_count > 1 {
        prev[1] = logit_score(logits, vocab_size, 0, labels[1]);
    }

    for frame in 1..frame_count {
        let mut next = vec![neg_inf; state_count];
        for state in 0..state_count {
            let mut best_state = state;
            let mut best_score = prev[state];

            if state > 0 && prev[state - 1] > best_score {
                best_state = state - 1;
                best_score = prev[state - 1];
            }
            if state > 1
                && labels[state] != blank_id
                && labels[state] != labels[state - 2]
                && prev[state - 2] > best_score
            {
                best_state = state - 2;
                best_score = prev[state - 2];
            }

            if best_score.is_finite() {
                next[state] = best_score + logit_score(logits, vocab_size, frame, labels[state]);
                backpointers[frame * state_count + state] = best_state;
            }
        }
        prev = next;
    }

    let final_blank = state_count - 1;
    let final_label = state_count - 2;
    let mut state = if prev[final_blank] >= prev[final_label] {
        final_blank
    } else {
        final_label
    };
    anyhow::ensure!(
        prev[state].is_finite(),
        "Parakeet CTC forced alignment did not find a finite path: frames={} target_tokens={} states={} max_token_id={} blank_id={} final_blank_score={} final_label_score={}",
        frame_count,
        token_ids.len(),
        state_count,
        token_ids.iter().copied().max().unwrap_or(0),
        blank_id,
        prev[final_blank],
        prev[final_label],
    );

    let mut states = vec![0usize; frame_count];
    for frame in (0..frame_count).rev() {
        states[frame] = state;
        if frame > 0 {
            state = backpointers[frame * state_count + state];
        }
    }

    let mut token_frames = vec![None; token_ids.len()];
    for (frame, state) in states.into_iter().enumerate() {
        if state % 2 == 1 {
            let token_index = state / 2;
            let entry = &mut token_frames[token_index];
            match entry {
                Some((start, end)) => *entry = Some((*start, frame.max(*end))),
                None => *entry = Some((frame, frame)),
            }
        }
    }

    Ok(token_frames)
}

fn greedy_ctc_token_frames(
    logits: &[f32],
    frame_count: usize,
    vocab_size: usize,
    blank_id: usize,
) -> Result<Vec<CtcTokenFrame>> {
    anyhow::ensure!(frame_count > 0, "cannot decode with zero CTC frames");
    anyhow::ensure!(
        logits.len() == frame_count * vocab_size,
        "CTC logits length {} did not match frames*vocab {}",
        logits.len(),
        frame_count * vocab_size
    );

    let mut token_frames: Vec<CtcTokenFrame> = Vec::new();
    let mut active_token: Option<usize> = None;
    for frame in 0..frame_count {
        let token = argmax_token(logits, vocab_size, frame);
        if token == blank_id {
            active_token = None;
            continue;
        }

        if active_token == Some(token) {
            if let Some(last) = token_frames.last_mut() {
                last.end = frame;
            }
        } else {
            token_frames.push(CtcTokenFrame {
                id: token,
                start: frame,
                end: frame,
            });
            active_token = Some(token);
        }
    }

    Ok(token_frames)
}

fn words_from_token_frames(
    target: &AlignmentTarget,
    token_frames: &[Option<(usize, usize)>],
    frame_count: usize,
    duration_ms: u32,
    hop_length: usize,
    sample_rate: usize,
    subsampling_factor: usize,
    offset_ms: i32,
) -> Vec<TimedWord> {
    let word_frames = target
        .words
        .iter()
        .map(|word| word_token_frame_range(word, target.token_ids.len(), token_frames, frame_count))
        .collect::<Vec<_>>();
    let mut words = Vec::with_capacity(target.words.len());
    let mut last_end_ms = 0u32;

    for (index, word) in target.words.iter().enumerate() {
        let (start_frame, last_token_frame) = word_frames[index];
        let next_start_frame = word_frames
            .get(index + 1)
            .map(|(start_frame, _)| *start_frame);
        let end_frame = next_start_frame
            .filter(|next| *next > start_frame)
            .unwrap_or_else(|| last_token_frame.saturating_add(1));
        let mut start_ms = frame_to_ms(
            start_frame,
            hop_length,
            sample_rate,
            subsampling_factor,
            offset_ms,
            duration_ms,
        );
        let mut end_ms = frame_to_ms(
            end_frame,
            hop_length,
            sample_rate,
            subsampling_factor,
            offset_ms,
            duration_ms,
        );

        start_ms = start_ms.max(last_end_ms);
        if end_ms <= start_ms {
            end_ms = start_ms.saturating_add(1).min(duration_ms.max(start_ms));
        }
        last_end_ms = end_ms;
        words.push(TimedWord {
            word: word.original.clone(),
            start_ms,
            end_ms,
        });
    }

    words
}

fn greedy_words_from_token_frames(
    tokenizer: &Tokenizer,
    token_frames: &[CtcTokenFrame],
    duration_ms: u32,
    hop_length: usize,
    sample_rate: usize,
    subsampling_factor: usize,
    offset_ms: i32,
) -> Vec<TimedWord> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut current_start = 0usize;
    let mut current_end = 0usize;

    for token_frame in token_frames {
        let Some(token) = tokenizer.id_to_token(token_frame.id as u32) else {
            continue;
        };
        let Some((piece, starts_word)) = ctc_token_piece(&token) else {
            continue;
        };
        if piece.is_empty() {
            continue;
        }

        if starts_word && !current.is_empty() {
            push_greedy_word(
                &mut words,
                std::mem::take(&mut current),
                current_start,
                current_end,
                duration_ms,
                hop_length,
                sample_rate,
                subsampling_factor,
                offset_ms,
            );
        }

        if current.is_empty() {
            current_start = token_frame.start;
        }
        current.push_str(&piece);
        current_end = token_frame.end;
    }

    if !current.is_empty() {
        push_greedy_word(
            &mut words,
            current,
            current_start,
            current_end,
            duration_ms,
            hop_length,
            sample_rate,
            subsampling_factor,
            offset_ms,
        );
    }

    words
}

#[derive(Debug, Clone, Copy)]
struct DirectWordMatch {
    hypothesis_start: usize,
    hypothesis_end: usize,
    reference_start: usize,
    reference_end: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct AnchorAlignmentScore {
    matched_reference_words: usize,
    matched_characters: usize,
    matched_hypothesis_words: usize,
    timing_error_ms: u64,
    skipped_words: usize,
}

#[derive(Debug, Clone, Copy)]
enum AnchorAlignmentStep {
    SkipHypothesis,
    SkipReference,
    Match {
        hypothesis_words: usize,
        reference_words: usize,
    },
}

#[derive(Debug, Clone, Copy)]
struct AnchorAlignmentPrevious {
    hypothesis_index: usize,
    reference_index: usize,
    step: AnchorAlignmentStep,
}

#[derive(Debug, Clone, Copy)]
struct AnchorAlignmentCell {
    score: AnchorAlignmentScore,
    previous: Option<AnchorAlignmentPrevious>,
}

#[derive(Debug, Clone, Copy)]
enum CharacterBoundary {
    Start,
    End,
}

fn direct_anchor_reference_words(
    reference: &[TimedWord],
    hypothesis: &[TimedWord],
    duration_ms: u32,
    config: &DirectAnchorConfig,
) -> Result<Vec<TimedWord>> {
    if reference.is_empty() {
        return Ok(Vec::new());
    }
    anyhow::ensure!(
        duration_ms > 0,
        "direct CTC anchors require non-empty audio"
    );
    validate_timestamp_sequence(hypothesis, duration_ms, "Parakeet CTC hypothesis")?;

    let reference_normalized = reference
        .iter()
        .map(|word| normalize_ctc_word(&word.word))
        .collect::<Vec<_>>();
    let hypothesis_normalized = hypothesis
        .iter()
        .map(|word| normalize_ctc_word(&word.word))
        .collect::<Vec<_>>();
    let matches = align_direct_word_groups(
        hypothesis,
        &hypothesis_normalized,
        reference,
        &reference_normalized,
    );

    let mut anchored = vec![false; reference.len()];
    for matched in &matches {
        anchored[matched.reference_start..matched.reference_end].fill(true);
    }
    let eligible_words = reference_normalized
        .iter()
        .filter(|word| !word.is_empty())
        .count();
    anyhow::ensure!(
        eligible_words > 0,
        "direct CTC anchors found no reference speech words"
    );
    let matched_words = reference_normalized
        .iter()
        .enumerate()
        .filter(|(index, word)| !word.is_empty() && anchored[*index])
        .count();
    let required_by_ratio = (eligible_words as f32 * config.min_match_ratio).ceil() as usize;
    let required_words = config
        .min_matched_words
        .max(required_by_ratio)
        .min(eligible_words);
    let longest_unmatched = longest_unmatched_word_run(&reference_normalized, &anchored);
    anyhow::ensure!(
        matched_words >= required_words,
        "direct CTC anchors rejected: matched {matched_words}/{eligible_words} reference words; required {required_words}"
    );
    anyhow::ensure!(
        longest_unmatched <= config.max_unmatched_words,
        "direct CTC anchors rejected: longest unmatched run was {longest_unmatched} words; maximum is {}",
        config.max_unmatched_words
    );
    if env_var_truthy("ASR_CTC_ALIGN_TIMINGS") {
        eprintln!(
            "ctc_direct_anchor matched_words={} eligible_words={} match_ratio={:.4} groups={} longest_unmatched_words={}",
            matched_words,
            eligible_words,
            matched_words as f64 / eligible_words as f64,
            matches.len(),
            longest_unmatched,
        );
    }

    let mut output = vec![None; reference.len()];
    for matched in &matches {
        map_direct_word_group(
            &mut output,
            reference,
            hypothesis,
            &reference_normalized,
            &hypothesis_normalized,
            *matched,
            duration_ms,
            config,
        )?;
    }
    reconcile_anchor_order(&mut output)?;
    fill_unanchored_reference_words(&mut output, reference, duration_ms)?;

    let words = output
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .context("direct CTC anchors did not timestamp every reference word")?;
    validate_timestamp_sequence(&words, duration_ms, "direct CTC result")?;
    anyhow::ensure!(
        words
            .iter()
            .zip(reference)
            .all(|(word, reference_word)| word.word == reference_word.word),
        "direct CTC anchors changed the Cohere word sequence"
    );
    Ok(words)
}

fn align_direct_word_groups(
    hypothesis: &[TimedWord],
    hypothesis_normalized: &[String],
    reference: &[TimedWord],
    reference_normalized: &[String],
) -> Vec<DirectWordMatch> {
    let hypothesis_count = hypothesis.len();
    let reference_count = reference.len();
    let width = reference_count + 1;
    let mut cells = vec![None; (hypothesis_count + 1) * width];
    cells[0] = Some(AnchorAlignmentCell {
        score: AnchorAlignmentScore::default(),
        previous: None,
    });

    for hypothesis_index in 0..=hypothesis_count {
        for reference_index in 0..=reference_count {
            let Some(cell) = cells[hypothesis_index * width + reference_index] else {
                continue;
            };

            for hypothesis_words in
                1..=DIRECT_ANCHOR_MAX_GROUP_WORDS.min(hypothesis_count - hypothesis_index)
            {
                for reference_words in
                    1..=DIRECT_ANCHOR_MAX_GROUP_WORDS.min(reference_count - reference_index)
                {
                    let hypothesis_end = hypothesis_index + hypothesis_words;
                    let reference_end = reference_index + reference_words;
                    let Some(matched_characters) = matching_group_character_count(
                        &hypothesis_normalized[hypothesis_index..hypothesis_end],
                        &reference_normalized[reference_index..reference_end],
                    ) else {
                        continue;
                    };
                    let timing_error_ms =
                        group_midpoint_ms(&hypothesis[hypothesis_index..hypothesis_end]).abs_diff(
                            group_midpoint_ms(&reference[reference_index..reference_end]),
                        );
                    let candidate = AnchorAlignmentCell {
                        score: AnchorAlignmentScore {
                            matched_reference_words: cell
                                .score
                                .matched_reference_words
                                .saturating_add(reference_words),
                            matched_characters: cell
                                .score
                                .matched_characters
                                .saturating_add(matched_characters),
                            matched_hypothesis_words: cell
                                .score
                                .matched_hypothesis_words
                                .saturating_add(hypothesis_words),
                            timing_error_ms: cell
                                .score
                                .timing_error_ms
                                .saturating_add(u64::from(timing_error_ms)),
                            skipped_words: cell.score.skipped_words,
                        },
                        previous: Some(AnchorAlignmentPrevious {
                            hypothesis_index,
                            reference_index,
                            step: AnchorAlignmentStep::Match {
                                hypothesis_words,
                                reference_words,
                            },
                        }),
                    };
                    update_anchor_alignment_cell(
                        &mut cells[hypothesis_end * width + reference_end],
                        candidate,
                    );
                }
            }

            if hypothesis_index < hypothesis_count {
                let candidate = AnchorAlignmentCell {
                    score: AnchorAlignmentScore {
                        skipped_words: cell.score.skipped_words.saturating_add(1),
                        ..cell.score
                    },
                    previous: Some(AnchorAlignmentPrevious {
                        hypothesis_index,
                        reference_index,
                        step: AnchorAlignmentStep::SkipHypothesis,
                    }),
                };
                update_anchor_alignment_cell(
                    &mut cells[(hypothesis_index + 1) * width + reference_index],
                    candidate,
                );
            }
            if reference_index < reference_count {
                let candidate = AnchorAlignmentCell {
                    score: AnchorAlignmentScore {
                        skipped_words: cell.score.skipped_words.saturating_add(1),
                        ..cell.score
                    },
                    previous: Some(AnchorAlignmentPrevious {
                        hypothesis_index,
                        reference_index,
                        step: AnchorAlignmentStep::SkipReference,
                    }),
                };
                update_anchor_alignment_cell(
                    &mut cells[hypothesis_index * width + reference_index + 1],
                    candidate,
                );
            }
        }
    }

    let mut matches = Vec::new();
    let mut hypothesis_index = hypothesis_count;
    let mut reference_index = reference_count;
    while hypothesis_index > 0 || reference_index > 0 {
        let Some(previous) =
            cells[hypothesis_index * width + reference_index].and_then(|cell| cell.previous)
        else {
            break;
        };
        if let AnchorAlignmentStep::Match {
            hypothesis_words,
            reference_words,
        } = previous.step
        {
            matches.push(DirectWordMatch {
                hypothesis_start: previous.hypothesis_index,
                hypothesis_end: previous.hypothesis_index + hypothesis_words,
                reference_start: previous.reference_index,
                reference_end: previous.reference_index + reference_words,
            });
        }
        hypothesis_index = previous.hypothesis_index;
        reference_index = previous.reference_index;
    }
    matches.reverse();
    matches
}

fn update_anchor_alignment_cell(
    destination: &mut Option<AnchorAlignmentCell>,
    candidate: AnchorAlignmentCell,
) {
    let replace = destination
        .as_ref()
        .map(|current| anchor_score_is_better(candidate.score, current.score))
        .unwrap_or(true);
    if replace {
        *destination = Some(candidate);
    }
}

fn anchor_score_is_better(candidate: AnchorAlignmentScore, current: AnchorAlignmentScore) -> bool {
    candidate.matched_reference_words > current.matched_reference_words
        || (candidate.matched_reference_words == current.matched_reference_words
            && (candidate.matched_characters > current.matched_characters
                || (candidate.matched_characters == current.matched_characters
                    && (candidate.matched_hypothesis_words > current.matched_hypothesis_words
                        || (candidate.matched_hypothesis_words
                            == current.matched_hypothesis_words
                            && (candidate.timing_error_ms < current.timing_error_ms
                                || (candidate.timing_error_ms == current.timing_error_ms
                                    && candidate.skipped_words < current.skipped_words)))))))
}

fn matching_group_character_count(hypothesis: &[String], reference: &[String]) -> Option<usize> {
    if hypothesis.iter().any(String::is_empty) || reference.iter().any(String::is_empty) {
        return None;
    }
    let hypothesis = hypothesis.concat();
    let reference = reference.concat();
    (hypothesis == reference).then(|| reference.chars().count())
}

fn group_midpoint_ms(words: &[TimedWord]) -> u32 {
    let start_ms = words.first().map(|word| word.start_ms).unwrap_or(0);
    let end_ms = words.last().map(|word| word.end_ms).unwrap_or(start_ms);
    start_ms.saturating_add(end_ms).saturating_div(2)
}

fn longest_unmatched_word_run(normalized: &[String], anchored: &[bool]) -> usize {
    let mut longest = 0usize;
    let mut current = 0usize;
    for (word, anchored) in normalized.iter().zip(anchored) {
        if word.is_empty() {
            continue;
        }
        if *anchored {
            current = 0;
        } else {
            current += 1;
            longest = longest.max(current);
        }
    }
    longest
}

#[allow(clippy::too_many_arguments)]
fn map_direct_word_group(
    output: &mut [Option<TimedWord>],
    reference: &[TimedWord],
    hypothesis: &[TimedWord],
    reference_normalized: &[String],
    hypothesis_normalized: &[String],
    matched: DirectWordMatch,
    duration_ms: u32,
    config: &DirectAnchorConfig,
) -> Result<()> {
    let hypothesis_words = &hypothesis[matched.hypothesis_start..matched.hypothesis_end];
    let hypothesis_lengths = hypothesis_normalized
        [matched.hypothesis_start..matched.hypothesis_end]
        .iter()
        .map(|word| word.chars().count())
        .collect::<Vec<_>>();
    let reference_lengths = reference_normalized[matched.reference_start..matched.reference_end]
        .iter()
        .map(|word| word.chars().count())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        hypothesis_lengths.iter().sum::<usize>() == reference_lengths.iter().sum::<usize>(),
        "direct CTC match groups had different character counts"
    );

    let mut reference_character_start = 0usize;
    for (offset, reference_length) in reference_lengths.into_iter().enumerate() {
        let reference_index = matched.reference_start + offset;
        let reference_character_end = reference_character_start + reference_length;
        let start_ms = character_boundary_time(
            hypothesis_words,
            &hypothesis_lengths,
            reference_character_start,
            CharacterBoundary::Start,
        );
        let end_ms = character_boundary_time(
            hypothesis_words,
            &hypothesis_lengths,
            reference_character_end,
            CharacterBoundary::End,
        );
        let mut start_ms = offset_timestamp_ms(start_ms, config.start_offset_ms, duration_ms);
        let mut end_ms = offset_timestamp_ms(end_ms, config.end_offset_ms, duration_ms);
        if end_ms <= start_ms {
            if start_ms < duration_ms {
                end_ms = start_ms + 1;
            } else if start_ms > 0 {
                start_ms = start_ms.saturating_sub(1);
            }
        }
        anyhow::ensure!(
            end_ms > start_ms,
            "direct CTC anchor produced an empty word interval"
        );
        output[reference_index] = Some(TimedWord {
            word: reference[reference_index].word.clone(),
            start_ms,
            end_ms,
        });
        reference_character_start = reference_character_end;
    }
    Ok(())
}

fn character_boundary_time(
    words: &[TimedWord],
    character_lengths: &[usize],
    position: usize,
    boundary: CharacterBoundary,
) -> u32 {
    let total_characters = character_lengths.iter().sum::<usize>();
    if position == 0 {
        return words.first().map(|word| word.start_ms).unwrap_or(0);
    }
    if position >= total_characters {
        return words.last().map(|word| word.end_ms).unwrap_or(0);
    }

    let mut character_start = 0usize;
    for (index, character_length) in character_lengths.iter().copied().enumerate() {
        let character_end = character_start + character_length;
        if position < character_end {
            return interpolate_ms(
                words[index].start_ms,
                words[index].end_ms,
                position - character_start,
                character_length,
            );
        }
        if position == character_end {
            return match boundary {
                CharacterBoundary::End => words[index].end_ms,
                CharacterBoundary::Start => words
                    .get(index + 1)
                    .map(|word| word.start_ms)
                    .unwrap_or(words[index].end_ms),
            };
        }
        character_start = character_end;
    }
    words.last().map(|word| word.end_ms).unwrap_or(0)
}

fn interpolate_ms(start_ms: u32, end_ms: u32, numerator: usize, denominator: usize) -> u32 {
    if denominator == 0 || end_ms <= start_ms {
        return start_ms;
    }
    start_ms.saturating_add(
        ((u64::from(end_ms - start_ms) * numerator as u64) / denominator as u64) as u32,
    )
}

fn offset_timestamp_ms(timestamp_ms: u32, offset_ms: i32, duration_ms: u32) -> u32 {
    (i64::from(timestamp_ms) + i64::from(offset_ms)).clamp(0, i64::from(duration_ms)) as u32
}

fn reconcile_anchor_order(words: &mut [Option<TimedWord>]) -> Result<()> {
    let anchor_indices = words
        .iter()
        .enumerate()
        .filter_map(|(index, word)| word.as_ref().map(|_| index))
        .collect::<Vec<_>>();
    for anchors in anchor_indices.windows(2) {
        let left_index = anchors[0];
        let right_index = anchors[1];
        let gap_words = u32::try_from(right_index - left_index - 1)
            .context("direct CTC anchor gap exceeded the timestamp range")?;
        let (left_words, right_words) = words.split_at_mut(right_index);
        let left = left_words[left_index]
            .as_mut()
            .context("direct CTC left anchor was unavailable")?;
        let right = right_words[0]
            .as_mut()
            .context("direct CTC right anchor was unavailable")?;
        if u64::from(right.start_ms) >= u64::from(left.end_ms) + u64::from(gap_words) {
            continue;
        }

        let minimum_boundary = left.start_ms.saturating_add(1);
        let maximum_boundary = right
            .end_ms
            .checked_sub(gap_words.saturating_add(1))
            .context("direct CTC calibration left no time between anchors")?;
        anyhow::ensure!(
            minimum_boundary <= maximum_boundary,
            "direct CTC calibration left no time between anchors"
        );
        let preferred_boundary =
            ((u64::from(left.end_ms) + u64::from(right.start_ms))
                .saturating_sub(u64::from(gap_words))
                / 2) as u32;
        let boundary = preferred_boundary.clamp(minimum_boundary, maximum_boundary);
        left.end_ms = boundary;
        right.start_ms = boundary + gap_words;
    }
    validate_anchor_order(words)
}

fn validate_anchor_order(words: &[Option<TimedWord>]) -> Result<()> {
    let mut previous_end_ms = 0u32;
    for word in words.iter().flatten() {
        anyhow::ensure!(
            word.start_ms >= previous_end_ms,
            "direct CTC anchors overlapped after boundary calibration"
        );
        previous_end_ms = word.end_ms;
    }
    Ok(())
}

fn fill_unanchored_reference_words(
    output: &mut [Option<TimedWord>],
    reference: &[TimedWord],
    duration_ms: u32,
) -> Result<()> {
    let anchor_indices = output
        .iter()
        .enumerate()
        .filter_map(|(index, word)| word.as_ref().map(|_| index))
        .collect::<Vec<_>>();
    anyhow::ensure!(
        !anchor_indices.is_empty(),
        "direct CTC anchors did not match a reference word"
    );

    let mut previous_anchor: Option<usize> = None;
    for next_anchor in anchor_indices
        .into_iter()
        .map(Some)
        .chain(std::iter::once(None))
    {
        let gap_start = previous_anchor.map_or(0, |index| index + 1);
        let gap_end = next_anchor.unwrap_or(reference.len());
        if gap_start < gap_end {
            let target_start_ms = previous_anchor
                .and_then(|index| output[index].as_ref().map(|word| word.end_ms))
                .unwrap_or(0);
            let target_end_ms = next_anchor
                .and_then(|index| output[index].as_ref().map(|word| word.start_ms))
                .unwrap_or(duration_ms);
            let source_start_ms = previous_anchor
                .map(|index| reference[index].end_ms.min(duration_ms))
                .unwrap_or(0);
            let source_end_ms = next_anchor
                .map(|index| reference[index].start_ms.min(duration_ms))
                .unwrap_or(duration_ms);
            fill_unanchored_gap(
                output,
                reference,
                gap_start,
                gap_end,
                source_start_ms,
                source_end_ms,
                target_start_ms,
                target_end_ms,
            )?;
        }
        previous_anchor = next_anchor;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn fill_unanchored_gap(
    output: &mut [Option<TimedWord>],
    reference: &[TimedWord],
    start_index: usize,
    end_index: usize,
    source_start_ms: u32,
    source_end_ms: u32,
    target_start_ms: u32,
    target_end_ms: u32,
) -> Result<()> {
    let word_count = end_index - start_index;
    anyhow::ensure!(
        target_end_ms >= target_start_ms.saturating_add(word_count as u32),
        "direct CTC anchors left too little time for {word_count} unmatched words"
    );

    let fallback_boundaries = if source_end_ms > source_start_ms {
        reference[start_index..end_index]
            .iter()
            .map(|word| {
                (
                    remap_interval_ms(
                        word.start_ms,
                        source_start_ms,
                        source_end_ms,
                        target_start_ms,
                        target_end_ms,
                    ),
                    remap_interval_ms(
                        word.end_ms,
                        source_start_ms,
                        source_end_ms,
                        target_start_ms,
                        target_end_ms,
                    ),
                )
            })
            .collect::<Vec<_>>()
    } else {
        let weights = reference[start_index..end_index]
            .iter()
            .map(|word| normalize_ctc_word(&word.word).chars().count().max(1))
            .collect::<Vec<_>>();
        let total_weight = weights.iter().sum::<usize>().max(1);
        let mut consumed_weight = 0usize;
        weights
            .into_iter()
            .map(|weight| {
                let start_ms = interpolate_ms(
                    target_start_ms,
                    target_end_ms,
                    consumed_weight,
                    total_weight,
                );
                consumed_weight += weight;
                let end_ms = interpolate_ms(
                    target_start_ms,
                    target_end_ms,
                    consumed_weight,
                    total_weight,
                );
                (start_ms, end_ms)
            })
            .collect::<Vec<_>>()
    };

    let mut previous_end_ms = target_start_ms;
    for (offset, (fallback_start_ms, fallback_end_ms)) in
        fallback_boundaries.into_iter().enumerate()
    {
        let remaining_words = word_count - offset - 1;
        let latest_end_ms = target_end_ms - remaining_words as u32;
        let latest_start_ms = latest_end_ms - 1;
        let start_ms = fallback_start_ms.max(previous_end_ms).min(latest_start_ms);
        let end_ms = fallback_end_ms.max(start_ms + 1).min(latest_end_ms);
        let word_index = start_index + offset;
        output[word_index] = Some(TimedWord {
            word: reference[word_index].word.clone(),
            start_ms,
            end_ms,
        });
        previous_end_ms = end_ms;
    }
    Ok(())
}

fn remap_interval_ms(
    value_ms: u32,
    source_start_ms: u32,
    source_end_ms: u32,
    target_start_ms: u32,
    target_end_ms: u32,
) -> u32 {
    let source_width = source_end_ms - source_start_ms;
    let target_width = target_end_ms - target_start_ms;
    let source_offset = value_ms
        .clamp(source_start_ms, source_end_ms)
        .saturating_sub(source_start_ms);
    target_start_ms.saturating_add(
        ((u64::from(target_width) * u64::from(source_offset)) / u64::from(source_width)) as u32,
    )
}

fn validate_timestamp_sequence(words: &[TimedWord], duration_ms: u32, label: &str) -> Result<()> {
    let mut previous_end_ms = 0u32;
    for (index, word) in words.iter().enumerate() {
        anyhow::ensure!(
            word.start_ms < word.end_ms && word.end_ms <= duration_ms,
            "{label} word {index} has invalid interval {}..{} for duration {duration_ms}",
            word.start_ms,
            word.end_ms
        );
        anyhow::ensure!(
            word.start_ms >= previous_end_ms,
            "{label} word {index} overlaps the previous word"
        );
        previous_end_ms = word.end_ms;
    }
    Ok(())
}

fn push_greedy_word(
    words: &mut Vec<TimedWord>,
    word: String,
    start_frame: usize,
    end_frame: usize,
    duration_ms: u32,
    hop_length: usize,
    sample_rate: usize,
    subsampling_factor: usize,
    offset_ms: i32,
) {
    let normalized = normalize_ctc_word(&word);
    if normalized.is_empty() {
        return;
    }

    let mut start_ms = frame_to_ms(
        start_frame,
        hop_length,
        sample_rate,
        subsampling_factor,
        offset_ms,
        duration_ms,
    );
    let mut end_ms = frame_to_ms(
        end_frame.saturating_add(1),
        hop_length,
        sample_rate,
        subsampling_factor,
        offset_ms,
        duration_ms,
    );
    if let Some(previous) = words.last() {
        start_ms = start_ms.max(previous.end_ms);
    }
    if end_ms <= start_ms {
        end_ms = start_ms.saturating_add(1).min(duration_ms.max(start_ms));
    }
    words.push(TimedWord {
        word: normalized,
        start_ms,
        end_ms,
    });
}

fn ctc_token_piece(token: &str) -> Option<(String, bool)> {
    if token == "<unk>" {
        return None;
    }

    const METASPACE: char = '\u{2581}';
    let starts_word = token.starts_with(METASPACE);
    let piece = token.trim_start_matches(METASPACE).to_string();
    Some((piece, starts_word))
}

fn word_token_frame_range(
    word: &AlignmentWord,
    target_token_count: usize,
    token_frames: &[Option<(usize, usize)>],
    frame_count: usize,
) -> (usize, usize) {
    let mut start_frame = None;
    let mut end_frame = None;
    for frame in token_frames
        .get(word.token_start..word.token_end)
        .unwrap_or_default()
        .iter()
        .flatten()
    {
        start_frame = Some(start_frame.map_or(frame.0, |value: usize| value.min(frame.0)));
        end_frame = Some(end_frame.map_or(frame.1, |value: usize| value.max(frame.1)));
    }

    let token_mid = (word.token_start + word.token_end).saturating_div(2);
    let fallback_frame = ((token_mid as f64 / target_token_count.max(1) as f64)
        * frame_count as f64)
        .round()
        .clamp(0.0, frame_count.saturating_sub(1) as f64) as usize;
    let start_frame = start_frame.unwrap_or(fallback_frame);
    let end_frame = end_frame.unwrap_or(start_frame);
    (start_frame, end_frame)
}

fn frame_to_ms(
    frame: usize,
    hop_length: usize,
    sample_rate: usize,
    subsampling_factor: usize,
    offset_ms: i32,
    duration_ms: u32,
) -> u32 {
    let frame_ms = (frame as f64 * hop_length as f64 * subsampling_factor.max(1) as f64 * 1000.0)
        / sample_rate.max(1) as f64;
    (frame_ms + f64::from(offset_ms))
        .round()
        .clamp(0.0, f64::from(duration_ms)) as u32
}

fn logit_score(logits: &[f32], vocab_size: usize, frame: usize, token: usize) -> f32 {
    logits
        .get(frame * vocab_size + token)
        .copied()
        .unwrap_or(f32::NEG_INFINITY)
}

fn argmax_token(logits: &[f32], vocab_size: usize, frame: usize) -> usize {
    let offset = frame * vocab_size;
    let mut best_token = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for token in 0..vocab_size {
        let score = logits[offset + token];
        if score > best_score {
            best_token = token;
            best_score = score;
        }
    }
    best_token
}

fn print_logit_stats(
    logits: &[f32],
    frame_count: usize,
    vocab_size: usize,
    token_ids: &[usize],
    blank_id: usize,
) {
    let finite_logits = logits.iter().filter(|value| value.is_finite()).count();
    let nan_logits = logits.iter().filter(|value| value.is_nan()).count();
    let neg_inf_logits = logits
        .iter()
        .filter(|value| **value == f32::NEG_INFINITY)
        .count();
    let blank_finite_frames = (0..frame_count)
        .filter(|frame| logit_score(logits, vocab_size, *frame, blank_id).is_finite())
        .count();
    let mut min_target_finite_frames = usize::MAX;
    let mut max_target_finite_frames = 0usize;
    for &token in token_ids {
        let count = (0..frame_count)
            .filter(|frame| logit_score(logits, vocab_size, *frame, token).is_finite())
            .count();
        min_target_finite_frames = min_target_finite_frames.min(count);
        max_target_finite_frames = max_target_finite_frames.max(count);
    }
    if min_target_finite_frames == usize::MAX {
        min_target_finite_frames = 0;
    }
    eprintln!(
        "ctc_align_debug frames={} vocab={} target_tokens={} finite_logits={} nan_logits={} neg_inf_logits={} blank_finite_frames={} min_target_finite_frames={} max_target_finite_frames={}",
        frame_count,
        vocab_size,
        token_ids.len(),
        finite_logits,
        nan_logits,
        neg_inf_logits,
        blank_finite_frames,
        min_target_finite_frames,
        max_target_finite_frames,
    );
}

fn normalize_ctc_word(word: &str) -> String {
    let mut normalized = String::new();
    for character in word.chars() {
        if character.is_alphanumeric() {
            normalized.extend(character.to_lowercase());
        } else if matches!(character, '\'' | '\u{2018}' | '\u{2019}') {
            normalized.push('\'');
        }
    }
    normalized
}

fn duration_ms_for_samples(sample_count: usize, sample_rate: usize) -> u32 {
    if sample_rate == 0 {
        return 0;
    }
    ((sample_count as f64 / sample_rate as f64) * 1000.0)
        .round()
        .clamp(0.0, f64::from(u32::MAX)) as u32
}

fn ctc_timestamp_mode() -> Result<Option<CtcTimestampMode>> {
    let backend =
        env_var_nonempty("ASR_COHERE_TIMESTAMP_BACKEND").map(|value| normalize_env_token(&value));
    let legacy_requested = env_var_nonempty("ASR_CTC_ALIGN_MODEL_DIR").is_some()
        || env_var_truthy("ASR_COHERE_CTC_TIMESTAMPS");
    parse_ctc_timestamp_mode(backend.as_deref(), legacy_requested)
}

fn parse_ctc_timestamp_mode(
    backend: Option<&str>,
    legacy_requested: bool,
) -> Result<Option<CtcTimestampMode>> {
    match backend {
        Some("token" | "tokens" | "token-frequency" | "token-frequency-estimate" | "none") => {
            Ok(None)
        }
        Some("parakeet-ctc" | "ctc" | "onnx-ctc") => Ok(Some(CtcTimestampMode::ForcedAlignment)),
        Some(
            "parakeet-ctc-direct" | "parakeet-direct" | "ctc-direct" | "direct-ctc" | "direct",
        ) => Ok(Some(CtcTimestampMode::DirectAnchors)),
        Some(value) => {
            anyhow::bail!(
                "unsupported ASR_COHERE_TIMESTAMP_BACKEND={value}; expected token-frequency, parakeet-ctc, or parakeet-ctc-direct"
            )
        }
        None if legacy_requested => Ok(Some(CtcTimestampMode::ForcedAlignment)),
        None => Ok(None),
    }
}

fn ctc_execution_provider() -> Result<CtcExecutionProvider> {
    if env_var_truthy("ASR_CTC_ALIGN_FORCE_CPU") {
        return Ok(CtcExecutionProvider::Cpu);
    }

    match env_var_nonempty("ASR_CTC_ALIGN_EXECUTION_PROVIDER")
        .map(|value| normalize_env_token(&value))
        .as_deref()
    {
        Some("auto") | None => Ok(CtcExecutionProvider::Auto),
        Some("tensorrt" | "trt") => Ok(CtcExecutionProvider::TensorRt),
        Some("cuda" | "gpu") => Ok(CtcExecutionProvider::Cuda),
        Some("cpu") => Ok(CtcExecutionProvider::Cpu),
        Some(value) => anyhow::bail!(
            "unsupported ASR_CTC_ALIGN_EXECUTION_PROVIDER={value}; expected auto, tensorrt, cuda, or cpu"
        ),
    }
}

fn ctc_device_id(device_ids: &[usize]) -> Option<usize> {
    env_var_usize("ASR_CTC_ALIGN_DEVICE_ID").or_else(|| device_ids.first().copied())
}

fn ctc_align_session_count() -> Result<usize> {
    let value = env::var("ASR_CTC_ALIGN_SESSIONS").ok();
    parse_ctc_align_session_count(value.as_deref())
}

fn parse_ctc_align_session_count(value: Option<&str>) -> Result<usize> {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(DEFAULT_CTC_ALIGN_SESSIONS);
    };
    let sessions = value.parse::<usize>().with_context(|| {
        format!("ASR_CTC_ALIGN_SESSIONS must be a positive integer; got {value:?}")
    })?;
    anyhow::ensure!(
        sessions > 0,
        "ASR_CTC_ALIGN_SESSIONS must be a positive integer; got {value:?}"
    );
    Ok(sessions)
}

fn ensure_ort_initialized() -> Result<()> {
    let result = ORT_INIT.get_or_init(|| {
        if let Some(path) = configured_onnxruntime_lib_path() {
            let created = ort::init_from(path.as_str())
                .map_err(|error| error.to_string())?
                .commit();
            info!(
                created,
                path, "initialized dynamic ONNX Runtime for CTC aligner"
            );
            return Ok(());
        }
        let created = ort::init().commit();
        info!(created, "initialized ONNX Runtime for CTC aligner");
        Ok(())
    });
    result
        .as_ref()
        .map_err(|error| anyhow::anyhow!(error.clone()))?;
    Ok(())
}

fn configured_onnxruntime_lib_path() -> Option<String> {
    env_var_nonempty("ASR_ONNX_RUNTIME_LIB")
        .or_else(|| env_var_nonempty("ORT_DYLIB_PATH"))
        .or_else(default_macos_onnxruntime_lib_path)
}

#[cfg(target_os = "macos")]
fn default_macos_onnxruntime_lib_path() -> Option<String> {
    [
        "/opt/homebrew/lib/libonnxruntime.dylib",
        "/usr/local/lib/libonnxruntime.dylib",
    ]
    .into_iter()
    .find(|path| Path::new(path).is_file())
    .map(str::to_string)
}

#[cfg(not(target_os = "macos"))]
fn default_macos_onnxruntime_lib_path() -> Option<String> {
    None
}

fn load_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("failed to parse {}", path.display()))
}

fn normalize_env_token(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(['_', ' '], "-")
}

fn ort_error<E: std::fmt::Display>(error: E) -> anyhow::Error {
    anyhow::anyhow!(error.to_string())
}

fn env_var_nonempty(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_var_truthy(name: &str) -> bool {
    env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "True"))
        .unwrap_or(false)
}

fn env_var_falsey(name: &str) -> bool {
    env::var(name)
        .map(|value| matches!(value.as_str(), "0" | "false" | "FALSE" | "False"))
        .unwrap_or(false)
}

fn env_var_usize(name: &str) -> Option<usize> {
    env::var(name).ok()?.trim().parse::<usize>().ok()
}

fn env_var_i32(name: &str) -> Option<i32> {
    env::var(name).ok()?.trim().parse::<i32>().ok()
}

fn env_var_u8(name: &str) -> Option<u8> {
    env::var(name).ok()?.trim().parse::<u8>().ok()
}

fn env_var_f32(name: &str) -> Option<f32> {
    env::var(name).ok()?.trim().parse::<f32>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    fn timed_word(word: &str, start_ms: u32, end_ms: u32) -> TimedWord {
        TimedWord {
            word: word.to_string(),
            start_ms,
            end_ms,
        }
    }

    fn permissive_direct_config() -> DirectAnchorConfig {
        DirectAnchorConfig {
            min_match_ratio: 0.5,
            min_matched_words: 1,
            max_unmatched_words: 8,
            start_offset_ms: 0,
            end_offset_ms: 0,
        }
    }

    #[test]
    fn ctc_session_count_defaults_and_requires_a_positive_integer() {
        assert_eq!(
            parse_ctc_align_session_count(None).unwrap(),
            DEFAULT_CTC_ALIGN_SESSIONS
        );
        assert_eq!(
            parse_ctc_align_session_count(Some("")).unwrap(),
            DEFAULT_CTC_ALIGN_SESSIONS
        );
        assert_eq!(parse_ctc_align_session_count(Some(" 2 ")).unwrap(), 2);

        for value in ["0", "-1", "two"] {
            let error = parse_ctc_align_session_count(Some(value)).unwrap_err();
            assert!(error
                .to_string()
                .contains("ASR_CTC_ALIGN_SESSIONS must be a positive integer"));
        }
    }

    #[test]
    fn lease_pool_reports_partial_initialization() {
        let error = LeasePool::<usize>::try_initialize(3, "test session", |index| {
            if index == 1 {
                anyhow::bail!("deliberate initialization failure");
            }
            Ok(index)
        })
        .err()
        .expect("pool initialization must fail");
        let error = format!("{error:#}");

        assert!(error.contains("failed to initialize test session 2 of 3"));
        assert!(error.contains("initialized 1 before failure"));
        assert!(error.contains("deliberate initialization failure"));
    }

    #[test]
    fn lease_pool_returns_resources_after_errors_and_panics() {
        let pool = LeasePool::try_initialize(1, "test session", |index| Ok(index)).unwrap();

        let error = (|| -> Result<()> {
            let lease = pool.lease();
            assert_eq!(*lease, 0);
            anyhow::bail!("deliberate request failure");
        })();
        assert!(error.is_err());
        assert_eq!(pool.available(), 1);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let lease = pool.lease();
            assert_eq!(*lease, 0);
            panic!("deliberate request panic");
        }));
        assert!(panic.is_err());
        assert_eq!(pool.available(), 1);
        assert_eq!(*pool.lease(), 0);
    }

    #[test]
    fn lease_pool_bounds_parallel_use_without_resource_collisions() {
        struct TestResource {
            active: AtomicBool,
        }

        let pool = Arc::new(
            LeasePool::try_initialize(2, "test session", |_| {
                Ok(TestResource {
                    active: AtomicBool::new(false),
                })
            })
            .unwrap(),
        );
        let active = Arc::new(AtomicUsize::new(0));
        let maximum_active = Arc::new(AtomicUsize::new(0));
        let start = Arc::new(Barrier::new(9));
        let mut threads = Vec::new();

        for _ in 0..8 {
            let pool = Arc::clone(&pool);
            let active = Arc::clone(&active);
            let maximum_active = Arc::clone(&maximum_active);
            let start = Arc::clone(&start);
            threads.push(thread::spawn(move || {
                start.wait();
                for _ in 0..32 {
                    let lease = pool.lease();
                    assert!(!lease.active.swap(true, Ordering::SeqCst));
                    let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                    maximum_active.fetch_max(current, Ordering::SeqCst);
                    thread::yield_now();
                    active.fetch_sub(1, Ordering::SeqCst);
                    assert!(lease.active.swap(false, Ordering::SeqCst));
                }
            }));
        }
        start.wait();
        for thread in threads {
            thread.join().unwrap();
        }

        assert!(maximum_active.load(Ordering::SeqCst) <= 2);
        assert_eq!(active.load(Ordering::SeqCst), 0);
        assert_eq!(pool.available(), 2);
    }

    #[test]
    fn forced_alignment_recovers_token_frames() {
        let vocab_size = 3;
        let blank = 2;
        let tokens = vec![0, 1];
        let frame_count = 5;
        let mut logits = vec![-10.0; frame_count * vocab_size];
        for (frame, token) in [blank, 0, 0, 1, blank].into_iter().enumerate() {
            logits[frame * vocab_size + token] = 10.0;
        }

        let frames = forced_align_tokens(&logits, frame_count, vocab_size, &tokens, blank).unwrap();

        assert_eq!(frames, vec![Some((1, 2)), Some((3, 3))]);
    }

    #[test]
    fn ctc_word_normalization_keeps_speech_text() {
        assert_eq!(normalize_ctc_word("Americans,"), "americans");
        assert_eq!(normalize_ctc_word("don't"), "don't");
        assert_eq!(normalize_ctc_word("Cohere’s"), "cohere's");
        assert_eq!(normalize_ctc_word("CAFÉ"), "café");
    }

    #[test]
    fn greedy_ctc_collapses_repeats_and_blanks() {
        let vocab_size = 4;
        let blank = 3;
        let frame_count = 7;
        let mut logits = vec![-10.0; frame_count * vocab_size];
        for (frame, token) in [blank, 0, 0, blank, 0, 1, blank].into_iter().enumerate() {
            logits[frame * vocab_size + token] = 10.0;
        }

        let frames = greedy_ctc_token_frames(&logits, frame_count, vocab_size, blank).unwrap();

        assert_eq!(frames.len(), 3);
        assert_eq!((frames[0].id, frames[0].start, frames[0].end), (0, 1, 2));
        assert_eq!((frames[1].id, frames[1].start, frames[1].end), (0, 4, 4));
        assert_eq!((frames[2].id, frames[2].start, frames[2].end), (1, 5, 5));
    }

    #[test]
    fn direct_alignment_uses_time_to_disambiguate_repeated_words() {
        let reference = vec![
            timed_word("go", 0, 100),
            timed_word("go", 100, 200),
            timed_word("now", 200, 300),
        ];
        let hypothesis = vec![timed_word("go", 110, 180), timed_word("now", 220, 280)];

        let words = direct_anchor_reference_words(
            &reference,
            &hypothesis,
            300,
            &permissive_direct_config(),
        )
        .unwrap();

        assert_eq!((words[1].start_ms, words[1].end_ms), (110, 180));
        assert_eq!((words[2].start_ms, words[2].end_ms), (220, 280));
        assert_eq!(
            words
                .iter()
                .map(|word| word.word.as_str())
                .collect::<Vec<_>>(),
            vec!["go", "go", "now"]
        );
    }

    #[test]
    fn direct_alignment_maps_split_and_merged_word_tokenization() {
        let reference = vec![timed_word("New", 0, 500), timed_word("York", 500, 1_000)];
        let hypothesis = vec![timed_word("newyork", 100, 800)];

        let words = direct_anchor_reference_words(
            &reference,
            &hypothesis,
            1_000,
            &permissive_direct_config(),
        )
        .unwrap();

        assert_eq!((words[0].start_ms, words[0].end_ms), (100, 400));
        assert_eq!((words[1].start_ms, words[1].end_ms), (400, 800));
        assert_eq!(words[0].word, "New");
        assert_eq!(words[1].word, "York");
    }

    #[test]
    fn direct_alignment_preserves_ctc_blank_gaps() {
        let reference = vec![timed_word("hello", 0, 500), timed_word("world", 500, 1_000)];
        let hypothesis = vec![timed_word("hello", 100, 200), timed_word("world", 400, 500)];

        let words = direct_anchor_reference_words(
            &reference,
            &hypothesis,
            1_000,
            &permissive_direct_config(),
        )
        .unwrap();

        assert_eq!(words[0].end_ms, 200);
        assert_eq!(words[1].start_ms, 400);
        assert_eq!(words[1].start_ms - words[0].end_ms, 200);
    }

    #[test]
    fn direct_alignment_reconciles_calibrated_anchor_overlaps() {
        let reference = vec![timed_word("one", 100, 200), timed_word("two", 200, 300)];
        let hypothesis = reference.clone();
        let config = DirectAnchorConfig {
            start_offset_ms: -69,
            end_offset_ms: 18,
            ..permissive_direct_config()
        };

        let words =
            direct_anchor_reference_words(&reference, &hypothesis, 400, &config).unwrap();

        assert_eq!((words[0].start_ms, words[0].end_ms), (31, 174));
        assert_eq!((words[1].start_ms, words[1].end_ms), (174, 318));
    }

    #[test]
    fn direct_alignment_reserves_time_for_unmatched_words_after_calibration() {
        let reference = vec![
            timed_word("one", 0, 100),
            timed_word("missing", 100, 200),
            timed_word("three", 200, 300),
        ];
        let hypothesis = vec![timed_word("one", 10, 100), timed_word("three", 110, 200)];
        let config = DirectAnchorConfig {
            start_offset_ms: -50,
            end_offset_ms: 50,
            ..permissive_direct_config()
        };

        let words =
            direct_anchor_reference_words(&reference, &hypothesis, 300, &config).unwrap();

        assert!(words[0].end_ms <= words[1].start_ms);
        assert!(words[1].end_ms <= words[2].start_ms);
        assert_eq!(
            words
                .iter()
                .map(|word| word.word.as_str())
                .collect::<Vec<_>>(),
            vec!["one", "missing", "three"]
        );
    }

    #[test]
    fn direct_alignment_rejects_calibration_without_room_for_words() {
        let reference = vec![
            timed_word("one", 0, 1),
            timed_word("missing", 1, 2),
            timed_word("three", 1, 2),
        ];
        let hypothesis = vec![timed_word("one", 0, 1), timed_word("three", 1, 2)];
        let config = DirectAnchorConfig {
            start_offset_ms: -69,
            end_offset_ms: 18,
            ..permissive_direct_config()
        };

        let error = direct_anchor_reference_words(&reference, &hypothesis, 2, &config).unwrap_err();

        assert!(error
            .to_string()
            .contains("calibration left no time between anchors"));
    }

    #[test]
    fn direct_alignment_rejects_low_match_coverage() {
        let reference = vec![
            timed_word("one", 0, 100),
            timed_word("two", 100, 200),
            timed_word("three", 200, 300),
            timed_word("four", 300, 400),
        ];
        let hypothesis = vec![timed_word("one", 0, 100)];
        let config = DirectAnchorConfig {
            min_match_ratio: DEFAULT_DIRECT_MIN_MATCH_RATIO,
            min_matched_words: DEFAULT_DIRECT_MIN_MATCHED_WORDS,
            max_unmatched_words: DEFAULT_DIRECT_MAX_UNMATCHED_WORDS,
            start_offset_ms: 0,
            end_offset_ms: 0,
        };

        let error =
            direct_anchor_reference_words(&reference, &hypothesis, 400, &config).unwrap_err();

        assert!(error.to_string().contains("matched 1/4"));
    }

    #[test]
    fn direct_timestamp_mode_requires_explicit_direct_selection() {
        assert_eq!(
            parse_ctc_timestamp_mode(Some("parakeet-ctc"), false).unwrap(),
            Some(CtcTimestampMode::ForcedAlignment)
        );
        assert_eq!(
            parse_ctc_timestamp_mode(Some("parakeet-ctc-direct"), false).unwrap(),
            Some(CtcTimestampMode::DirectAnchors)
        );
        assert_eq!(
            parse_ctc_timestamp_mode(None, true).unwrap(),
            Some(CtcTimestampMode::ForcedAlignment)
        );
        assert_eq!(parse_ctc_timestamp_mode(None, false).unwrap(), None);
    }
}
