use crate::chunking::TimedWord;
use crate::config::ASR_SAMPLE_RATE;
use anyhow::{Context, Result};
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
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use tokenizers::Tokenizer;
use tracing::{info, warn};

static ORT_INIT: OnceLock<Result<(), String>> = OnceLock::new();

const DEFAULT_CTC_MODEL_DIR: &str = "models/parakeet-ctc-0.6b-onnx";
const DEFAULT_CTC_ONNX_FILE: &str = "onnx/model.onnx";

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

impl ParakeetCtcAligner {
    pub(crate) fn from_env(device_ids: &[usize]) -> Result<Option<Self>> {
        if !ctc_alignment_requested()? {
            return Ok(None);
        }

        Self::new(device_ids).map(Some)
    }

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
    word.chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '\'')
        .flat_map(char::to_lowercase)
        .collect()
}

fn duration_ms_for_samples(sample_count: usize, sample_rate: usize) -> u32 {
    if sample_rate == 0 {
        return 0;
    }
    ((sample_count as f64 / sample_rate as f64) * 1000.0)
        .round()
        .clamp(0.0, f64::from(u32::MAX)) as u32
}

fn ctc_alignment_requested() -> Result<bool> {
    let backend =
        env_var_nonempty("ASR_COHERE_TIMESTAMP_BACKEND").map(|value| normalize_env_token(&value));
    match backend.as_deref() {
        Some("token" | "tokens" | "token-frequency" | "token-frequency-estimate" | "none") => {
            Ok(false)
        }
        Some("parakeet-ctc" | "ctc" | "onnx-ctc") => Ok(true),
        Some(value) => {
            anyhow::bail!(
                "unsupported ASR_COHERE_TIMESTAMP_BACKEND={value}; expected token-frequency or parakeet-ctc"
            )
        }
        None => Ok(env_var_nonempty("ASR_CTC_ALIGN_MODEL_DIR").is_some()
            || env_var_truthy("ASR_COHERE_CTC_TIMESTAMPS")),
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
}
