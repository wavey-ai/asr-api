use crate::asr::WindowTranscription;
use crate::chunking::TimedWord;
use crate::cohere_frontend::{CohereFrontend, CoherePreprocessorConfig};
use crate::config::ASR_SAMPLE_RATE;
use crate::config::DEFAULT_LANGUAGE;
use crate::ctc_align::ParakeetCtcTimestampEngine;
use crate::timestamps::{
    duration_ms_for_samples, estimate_word_timestamps_from_tokens, TokenTextSpan,
};
use anyhow::{Context, Result};
use crossbeam_channel::{bounded, Receiver, Sender};
use ndarray::{Array2, Array3, ArrayD, Axis, Ix1};
use ort::execution_providers::{
    coreml, CPUExecutionProvider, CUDAExecutionProvider, CoreML, ExecutionProvider,
    ExecutionProviderDispatch, TensorRTExecutionProvider,
};
use ort::logging::LogLevel;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor as OrtTensor;
use ort::value::ValueType;
use serde::Deserialize;
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Instant;
use tokenizers::Tokenizer;
use tokio::sync::oneshot;
use tracing::{info, warn};
static ORT_INIT: OnceLock<Result<(), String>> = OnceLock::new();

#[derive(Debug, Deserialize)]
struct GenerationConfig {
    bos_token_id: i64,
    decoder_start_token_id: Option<i64>,
    eos_token_id: i64,
    pad_token_id: i64,
}

#[derive(Debug, Deserialize, Default)]
struct ExportMetadata {
    prompt_text: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct CohereModelConfig {
    encoder: Option<CohereEncoderConfig>,
    max_audio_clip_s: Option<f32>,
}

#[derive(Debug, Deserialize, Default)]
struct CohereEncoderConfig {
    subsampling_factor: Option<usize>,
}

#[derive(Clone)]
struct DecodeConfig {
    prompt_text: String,
    prompt_ids: Vec<i64>,
    bos_token_id: i64,
    decoder_start_token_id: Option<i64>,
    eos_token_id: i64,
    pad_token_id: i64,
    max_new_tokens: usize,
}

#[derive(Clone, Debug)]
struct CohereRuntimeConfig {
    force_cpu: bool,
    coreml: CohereCoreMlConfig,
    trt: CohereTensorRtConfig,
}

#[derive(Clone, Debug)]
struct CohereCoreMlConfig {
    enabled: bool,
    compute_units: CohereCoreMlComputeUnits,
    cache_dir: Option<PathBuf>,
    low_precision_accumulation_on_gpu: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CohereCoreMlComputeUnits {
    All,
    CpuAndNeuralEngine,
    CpuAndGpu,
    CpuOnly,
}

impl CohereCoreMlComputeUnits {
    fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::CpuAndNeuralEngine => "cpu-and-neural-engine",
            Self::CpuAndGpu => "cpu-and-gpu",
            Self::CpuOnly => "cpu-only",
        }
    }
}

impl From<CohereCoreMlComputeUnits> for coreml::ComputeUnits {
    fn from(value: CohereCoreMlComputeUnits) -> Self {
        match value {
            CohereCoreMlComputeUnits::All => Self::All,
            CohereCoreMlComputeUnits::CpuAndNeuralEngine => Self::CPUAndNeuralEngine,
            CohereCoreMlComputeUnits::CpuAndGpu => Self::CPUAndGPU,
            CohereCoreMlComputeUnits::CpuOnly => Self::CPUOnly,
        }
    }
}

#[derive(Clone, Debug)]
struct CohereTensorRtConfig {
    components_raw: String,
    enabled_components: HashSet<String>,
    cache_dir: PathBuf,
    min_duration_s: usize,
    opt_duration_s: usize,
    max_duration_s: usize,
    min_feature_steps: usize,
    opt_feature_steps: usize,
    max_feature_steps: usize,
    workspace_bytes: usize,
    builder_optimization_level: u8,
    fp16: bool,
    detailed_build_log: bool,
    feature_size: usize,
    subsampling_factor: usize,
    prompt_len: usize,
    max_new_tokens: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelComponent {
    Encoder,
    DecoderPrefill,
    DecoderCachedStep,
}

impl ModelComponent {
    fn config_token(self) -> &'static str {
        match self {
            Self::Encoder => "encoder",
            Self::DecoderPrefill => "decoder_prefill",
            Self::DecoderCachedStep => "decoder_cached_step",
        }
    }

    fn cache_prefix(self) -> &'static str {
        self.config_token()
    }
}

#[derive(Clone, Copy, Debug)]
enum CohereProfileTarget {
    Min,
    Opt,
    Max,
}

pub struct CohereBackend {
    frontend: CohereFrontend,
    decoder: CohereDecoderClient,
    ctc_aligner: Option<Arc<ParakeetCtcTimestampEngine>>,
}

impl CohereBackend {
    pub fn new(
        model_dir: &Path,
        device_ids: &[usize],
        onnx_sessions: usize,
        max_new_tokens: usize,
    ) -> Result<Self> {
        let preprocessor =
            load_json::<CoherePreprocessorConfig>(&model_dir.join("preprocessor_config.json"))
                .context("failed to load Cohere preprocessor_config.json")?;
        let generation = load_json::<GenerationConfig>(&model_dir.join("generation_config.json"))
            .context("failed to load Cohere generation_config.json")?;
        let export =
            load_json::<ExportMetadata>(&model_dir.join("export.json")).unwrap_or_default();
        let model_config =
            load_json::<CohereModelConfig>(&model_dir.join("config.json")).unwrap_or_default();
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .context("failed to load Cohere tokenizer.json")?;

        let prompt_text = export
            .prompt_text
            .unwrap_or_else(|| build_prompt(DEFAULT_LANGUAGE, true));
        let prompt_ids = tokenizer
            .encode(prompt_text.as_str(), false)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .context("failed to encode Cohere prompt")?
            .get_ids()
            .iter()
            .map(|&id| i64::from(id))
            .collect::<Vec<_>>();
        anyhow::ensure!(
            !prompt_ids.is_empty(),
            "Cohere tokenizer produced an empty decoder prompt"
        );

        let runtime = CohereRuntimeConfig::from_env(
            model_dir,
            &preprocessor,
            &model_config,
            prompt_ids.len(),
            max_new_tokens,
        );
        let decode = DecodeConfig {
            prompt_text,
            prompt_ids,
            bos_token_id: generation.bos_token_id,
            decoder_start_token_id: generation.decoder_start_token_id,
            eos_token_id: generation.eos_token_id,
            pad_token_id: generation.pad_token_id,
            max_new_tokens,
        };

        let frontend = CohereFrontend::new(preprocessor)?;
        let decoder = CohereDecoderClient::new(
            model_dir,
            device_ids,
            onnx_sessions,
            tokenizer,
            decode,
            runtime,
        )?;
        let ctc_aligner = ParakeetCtcTimestampEngine::from_env(device_ids)
            .context("failed to initialize Parakeet CTC timestamp aligner")?
            .map(Arc::new);
        Ok(Self {
            frontend,
            decoder,
            ctc_aligner,
        })
    }

    pub async fn transcribe_window(
        &self,
        samples: Vec<f32>,
        _seq: u32,
    ) -> Result<WindowTranscription> {
        let duration_ms = duration_ms_for_samples(samples.len(), ASR_SAMPLE_RATE);
        let features = self.frontend.compute(&samples)?;
        let mut result = self.decoder.decode(features, duration_ms).await?;

        if let Some(aligner) = self.ctc_aligner.clone() {
            let text = result.text.clone();
            let reference_words = result.words.clone();
            let mode = aligner.mode_name();
            match tokio::task::spawn_blocking(move || {
                aligner.timestamp_words(&samples, &text, &reference_words)
            })
            .await
            {
                Ok(Ok(words)) if !words.is_empty() => {
                    match apply_side_model_timestamps(&mut result, &words) {
                        Ok(()) => {}
                        Err(error) => {
                            if env_var_truthy("ASR_CTC_ALIGN_TIMINGS") {
                                eprintln!(
                                    "ctc_align_timing mode={} status=rejected error={error:?}",
                                    mode
                                );
                            }
                            warn!(
                                error = %error,
                                mode,
                                "Parakeet CTC timestamps did not match Cohere words; keeping token-frequency Cohere timestamps"
                            );
                        }
                    }
                }
                Ok(Ok(_)) => {
                    if env_var_truthy("ASR_CTC_ALIGN_TIMINGS") {
                        eprintln!("ctc_align_timing mode={mode} status=empty_words");
                    }
                    warn!(
                        mode,
                        "Parakeet CTC timestamp aligner returned no words; keeping token-frequency Cohere timestamps"
                    );
                }
                Ok(Err(error)) => {
                    if env_var_truthy("ASR_CTC_ALIGN_TIMINGS") {
                        eprintln!(
                            "ctc_align_timing mode={} status=error error={error:?}",
                            mode
                        );
                    }
                    warn!(
                        error = %error,
                        mode,
                        "Parakeet CTC timestamp alignment failed; keeping token-frequency Cohere timestamps"
                    );
                }
                Err(error) => {
                    if env_var_truthy("ASR_CTC_ALIGN_TIMINGS") {
                        eprintln!(
                            "ctc_align_timing mode={} status=task_error error={error:?}",
                            mode
                        );
                    }
                    warn!(
                        error = %error,
                        mode,
                        "Parakeet CTC timestamp alignment task failed; keeping token-frequency Cohere timestamps"
                    );
                }
            }
        }

        Ok(result)
    }
}

struct CohereDecoderClient {
    next_id: AtomicU64,
    job_tx: Option<Sender<CohereJob>>,
    state: Arc<Mutex<CohereDecoderState>>,
    worker_handles: Vec<thread::JoinHandle<()>>,
    result_handle: Option<thread::JoinHandle<()>>,
}

struct CohereDecoderState {
    pending: HashMap<u64, oneshot::Sender<std::result::Result<WindowTranscription, String>>>,
    completed: HashMap<u64, std::result::Result<WindowTranscription, String>>,
}

struct CohereJob {
    job_id: u64,
    features: Array2<f32>,
    duration_ms: u32,
}

struct CohereJobResult {
    job_id: u64,
    result: std::result::Result<WindowTranscription, String>,
}

impl CohereDecoderClient {
    fn new(
        model_dir: &Path,
        device_ids: &[usize],
        onnx_sessions: usize,
        tokenizer: Tokenizer,
        decode: DecodeConfig,
        runtime: CohereRuntimeConfig,
    ) -> Result<Self> {
        if runtime.requires_device_ids() {
            anyhow::ensure!(
                !device_ids.is_empty(),
                "Cohere backend requires at least one GPU device id; set ASR_DEVICE_IDS, ASR_COHERE_COREML=true for Apple GPU/CoreML, or ASR_COHERE_FORCE_CPU=true for explicit CPU compare mode"
            );
        }

        let worker_count = if runtime.uses_single_execution_target() {
            onnx_sessions.max(1)
        } else {
            device_ids.len().max(1) * onnx_sessions.max(1)
        };
        let (job_tx, job_rx) = bounded::<CohereJob>(worker_count * 2);
        let (result_tx, result_rx) = bounded::<CohereJobResult>(worker_count * 2);
        let state = Arc::new(Mutex::new(CohereDecoderState {
            pending: HashMap::new(),
            completed: HashMap::new(),
        }));

        let effective_device_ids = if runtime.uses_single_execution_target() {
            vec![None]
        } else {
            device_ids.iter().copied().map(Some).collect()
        };
        let mut workers = Vec::with_capacity(worker_count);
        for device_id in effective_device_ids {
            for _ in 0..onnx_sessions.max(1) {
                workers.push(CohereWorker::new(
                    model_dir,
                    device_id,
                    tokenizer.clone(),
                    decode.clone(),
                    runtime.clone(),
                )?);
            }
        }
        let worker_handles = workers
            .into_iter()
            .map(|worker| {
                let worker_job_rx = job_rx.clone();
                let worker_result_tx = result_tx.clone();
                thread::spawn(move || worker_loop(worker, worker_job_rx, worker_result_tx))
            })
            .collect();
        drop(result_tx);

        let dispatch_state = Arc::clone(&state);
        let result_handle = thread::spawn(move || {
            for result in result_rx {
                dispatch_result(&dispatch_state, result);
            }

            let mut guard = dispatch_state.lock().expect("cohere state mutex poisoned");
            for (_, sender) in guard.pending.drain() {
                let _ = sender.send(Err("cohere worker pool closed".into()));
            }
            guard.completed.clear();
        });

        Ok(Self {
            next_id: AtomicU64::new(0),
            job_tx: Some(job_tx),
            state,
            worker_handles,
            result_handle: Some(result_handle),
        })
    }

    async fn decode(&self, features: Array2<f32>, duration_ms: u32) -> Result<WindowTranscription> {
        let job_id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        {
            let mut guard = self.state.lock().expect("cohere state mutex poisoned");
            if let Some(result) = guard.completed.remove(&job_id) {
                return result.map_err(anyhow::Error::msg);
            }
            guard.pending.insert(job_id, tx);
        }

        let Some(job_tx) = self.job_tx.as_ref() else {
            self.state
                .lock()
                .expect("cohere state mutex poisoned")
                .pending
                .remove(&job_id);
            anyhow::bail!("Cohere worker pool is closed");
        };
        if let Err(error) = job_tx.send(CohereJob {
            job_id,
            features,
            duration_ms,
        }) {
            self.state
                .lock()
                .expect("cohere state mutex poisoned")
                .pending
                .remove(&job_id);
            anyhow::bail!("failed to enqueue Cohere job: {error}");
        }

        match rx.await {
            Ok(Ok(text)) => Ok(text),
            Ok(Err(error)) => anyhow::bail!(error),
            Err(_) => anyhow::bail!("Cohere request was canceled"),
        }
    }
}

impl Drop for CohereDecoderClient {
    fn drop(&mut self) {
        drop(self.job_tx.take());

        for handle in self.worker_handles.drain(..) {
            if handle.join().is_err() {
                warn!("Cohere decoder worker panicked during shutdown");
            }
        }

        if self
            .result_handle
            .take()
            .is_some_and(|handle| handle.join().is_err())
        {
            warn!("Cohere result dispatcher panicked during shutdown");
        }
    }
}

fn worker_loop(
    mut worker: CohereWorker,
    job_rx: Receiver<CohereJob>,
    result_tx: Sender<CohereJobResult>,
) {
    for job in job_rx {
        let result = worker
            .decode(job.features, job.duration_ms)
            .map_err(|error| error.to_string());
        let _ = result_tx.send(CohereJobResult {
            job_id: job.job_id,
            result,
        });
    }
}

fn dispatch_result(state: &Arc<Mutex<CohereDecoderState>>, result: CohereJobResult) {
    let mut guard = state.lock().expect("cohere state mutex poisoned");
    if let Some(sender) = guard.pending.remove(&result.job_id) {
        let _ = sender.send(result.result);
    } else {
        guard.completed.insert(result.job_id, result.result);
    }
}

struct CohereWorker {
    encoder: Session,
    decoder_prefill: Session,
    decoder_cached_step: Session,
    tokenizer: Tokenizer,
    decode: DecodeConfig,
    decoder_num_layers: usize,
}

impl CohereWorker {
    fn new(
        model_dir: &Path,
        device_id: Option<usize>,
        tokenizer: Tokenizer,
        decode: DecodeConfig,
        runtime: CohereRuntimeConfig,
    ) -> Result<Self> {
        ensure_ort_initialized()?;
        runtime.validate(device_id)?;

        let encoder_path = model_dir.join("encoder.onnx");
        let decoder_prefill_path = model_dir.join("decoder_prefill.onnx");
        let decoder_cached_step_path = model_dir.join("decoder_cached_step.onnx");
        let encoder_shapes = inspect_shapes(
            &encoder_path,
            runtime.trt.enabled_for(ModelComponent::Encoder),
            "Cohere encoder",
        );
        let decoder_prefill_shapes = inspect_shapes(
            &decoder_prefill_path,
            runtime.trt.enabled_for(ModelComponent::DecoderPrefill),
            "Cohere decoder_prefill",
        );
        let decoder_cached_step_shapes = inspect_shapes(
            &decoder_cached_step_path,
            runtime.trt.enabled_for(ModelComponent::DecoderCachedStep),
            "Cohere decoder_cached_step",
        );

        if let Some(device_id) = device_id {
            info!(
                device = device_id,
                trt_available = TensorRTExecutionProvider::default().is_available().unwrap_or(false),
                trt_components = %runtime.trt.components_raw,
                trt_cache_dir = %runtime.trt.cache_dir.display(),
                trt_profile_min_s = runtime.trt.min_duration_s,
                trt_profile_opt_s = runtime.trt.opt_duration_s,
                trt_profile_max_s = runtime.trt.max_duration_s,
                trt_profile_min_frames = runtime.trt.min_feature_steps,
                trt_profile_opt_frames = runtime.trt.opt_feature_steps,
                trt_profile_max_frames = runtime.trt.max_feature_steps,
                "Cohere execution providers ready"
            );
        } else if runtime.coreml.enabled {
            info!(
                compute_units = runtime.coreml.compute_units.as_str(),
                cache_dir = runtime
                    .coreml
                    .cache_dir
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "disabled".to_string()),
                low_precision_accumulation_on_gpu =
                    runtime.coreml.low_precision_accumulation_on_gpu,
                "Cohere CoreML execution provider ready"
            );
        } else {
            info!("Cohere CPU execution provider ready");
        }

        let encoder = session_from_providers(
            &encoder_path,
            &provider_chain(
                ModelComponent::Encoder,
                device_id,
                &runtime,
                encoder_shapes.as_deref(),
            ),
        )
        .context("failed to initialize Cohere encoder session")?;
        let decoder_prefill = session_from_providers(
            &decoder_prefill_path,
            &provider_chain(
                ModelComponent::DecoderPrefill,
                device_id,
                &runtime,
                decoder_prefill_shapes.as_deref(),
            ),
        )
        .context("failed to initialize Cohere decoder_prefill session")?;
        let decoder_cached_step = session_from_providers(
            &decoder_cached_step_path,
            &provider_chain(
                ModelComponent::DecoderCachedStep,
                device_id,
                &runtime,
                decoder_cached_step_shapes.as_deref(),
            ),
        )
        .context("failed to initialize Cohere decoder_cached_step session")?;
        let decoder_num_layers = decoder_prefill.outputs().len().saturating_sub(1) / 4;
        anyhow::ensure!(
            decoder_num_layers > 0,
            "Cohere decoder_prefill session did not expose cached layer outputs"
        );
        Ok(Self {
            encoder,
            decoder_prefill,
            decoder_cached_step,
            tokenizer,
            decode,
            decoder_num_layers,
        })
    }

    fn decode(&mut self, features: Array2<f32>, duration_ms: u32) -> Result<WindowTranscription> {
        let timings_enabled = env_var_truthy("ASR_COHERE_TIMINGS");
        let decode_started = Instant::now();
        let feature_shape = features.dim();
        let raw_feature_length = feature_shape.1 as i64;
        let (feature_data, feature_offset) = features.into_raw_vec_and_offset();
        anyhow::ensure!(
            feature_offset.unwrap_or(0) == 0,
            "Cohere feature tensor had a non-zero storage offset"
        );
        let feature_tensor =
            Array3::from_shape_vec((1, feature_shape.0, feature_shape.1), feature_data)?;
        let feature_length = OrtTensor::from_array(([1], vec![raw_feature_length]))?;
        let feature_tensor = OrtTensor::from_array(feature_tensor)?;
        let encoder_input_names = self
            .encoder
            .inputs()
            .iter()
            .map(|input| input.name().to_string())
            .collect::<Vec<_>>();
        let encoder_inputs = ort::inputs! {
            encoder_input_names[0].as_str() => feature_tensor,
            encoder_input_names[1].as_str() => feature_length,
        };
        let encoder_started = Instant::now();
        let encoder_outputs = self
            .encoder
            .run(encoder_inputs)
            .context("Cohere encoder session failed")?;
        let encoder_run_s = encoder_started.elapsed().as_secs_f64();
        anyhow::ensure!(
            encoder_outputs.len() >= 2,
            "Cohere encoder did not return encoder_hidden_states and encoded_length"
        );
        let encoder_extract_started = Instant::now();
        let encoder_hidden_states = extract_array_f32(&encoder_outputs[0])
            .context("failed to extract Cohere encoder_hidden_states")?;
        let encoded_length_arr = encoder_outputs[1]
            .try_extract_array::<i64>()
            .context("failed to extract Cohere encoded_length")?
            .into_owned()
            .into_dimensionality::<Ix1>()?;
        let encoded_length = *encoded_length_arr
            .first()
            .context("Cohere encoded_length output was empty")?;
        if let Some(path) = env_var_nonempty("ASR_COHERE_DUMP_ENCODER") {
            dump_f32_array(Path::new(&path), &encoder_hidden_states)
                .with_context(|| format!("failed to dump Cohere encoder output to {path}"))?;
            eprintln!(
                "cohere_encoder_dump={} shape={:?}",
                path,
                encoder_hidden_states.shape()
            );
        }
        let encoder_extract_s = encoder_extract_started.elapsed().as_secs_f64();

        let prompt_len = self.decode.prompt_ids.len();
        let prompt_ids =
            OrtTensor::from_array(([1i64, prompt_len as i64], self.decode.prompt_ids.clone()))?;
        let prompt_mask =
            OrtTensor::from_array(([1i64, prompt_len as i64], vec![1i64; prompt_len]))?;
        let encoder_hidden_tensor = OrtTensor::from_array(encoder_hidden_states.clone())?;
        let raw_length_tensor = OrtTensor::from_array(([1], vec![raw_feature_length]))?;
        let prefill_input_names = self
            .decoder_prefill
            .inputs()
            .iter()
            .map(|input| input.name().to_string())
            .collect::<Vec<_>>();
        let prefill_inputs = ort::inputs! {
            prefill_input_names[0].as_str() => encoder_hidden_tensor,
            prefill_input_names[1].as_str() => raw_length_tensor,
            prefill_input_names[2].as_str() => prompt_ids,
            prefill_input_names[3].as_str() => prompt_mask,
        };
        let prefill_started = Instant::now();
        let prefill_outputs = self
            .decoder_prefill
            .run(prefill_inputs)
            .context("Cohere decoder_prefill session failed")?;
        let prefill_run_s = prefill_started.elapsed().as_secs_f64();
        anyhow::ensure!(
            prefill_outputs.len() == 1 + (self.decoder_num_layers * 4),
            "unexpected Cohere decoder_prefill output count {}",
            prefill_outputs.len()
        );

        let prefill_extract_started = Instant::now();
        let mut generated_ids = Vec::new();
        let mut current_token = argmax_last_token(&extract_array_f32(&prefill_outputs[0])?)?;
        let mut self_keys = Vec::with_capacity(self.decoder_num_layers);
        let mut self_values = Vec::with_capacity(self.decoder_num_layers);
        let mut cross_keys = Vec::with_capacity(self.decoder_num_layers);
        let mut cross_values = Vec::with_capacity(self.decoder_num_layers);
        for layer_idx in 0..self.decoder_num_layers {
            let base = 1 + (layer_idx * 4);
            self_keys.push(extract_array_f32(&prefill_outputs[base])?);
            self_values.push(extract_array_f32(&prefill_outputs[base + 1])?);
            cross_keys.push(extract_array_f32(&prefill_outputs[base + 2])?);
            cross_values.push(extract_array_f32(&prefill_outputs[base + 3])?);
        }
        if let Some(path) = env_var_nonempty("ASR_COHERE_DUMP_SELF_KEY0") {
            dump_f32_array(Path::new(&path), &self_keys[0])
                .with_context(|| format!("failed to dump Cohere self key to {path}"))?;
            eprintln!(
                "cohere_self_key0_dump={} shape={:?}",
                path,
                self_keys[0].shape()
            );
        }
        if let Some(path) = env_var_nonempty("ASR_COHERE_DUMP_CROSS_KEY0") {
            dump_f32_array(Path::new(&path), &cross_keys[0])
                .with_context(|| format!("failed to dump Cohere cross key to {path}"))?;
            eprintln!(
                "cohere_cross_key0_dump={} shape={:?}",
                path,
                cross_keys[0].shape()
            );
        }
        let prefill_extract_s = prefill_extract_started.elapsed().as_secs_f64();

        let mut cached_input_s = 0.0;
        let mut cached_run_s = 0.0;
        let mut cached_extract_s = 0.0;
        let mut cached_steps = 0usize;
        for _ in 0..self.decode.max_new_tokens {
            if current_token == self.decode.eos_token_id {
                break;
            }
            generated_ids.push(current_token as u32);

            let cached_input_started = Instant::now();
            let decoder_input_ids = OrtTensor::from_array(([1i64, 1i64], vec![current_token]))?;
            let encoded_length_tensor = OrtTensor::from_array(([1], vec![encoded_length]))?;
            let cached_input_names = self
                .decoder_cached_step
                .inputs()
                .iter()
                .map(|input| input.name().to_string())
                .collect::<Vec<_>>();
            let mut inputs = ort::inputs! {
                cached_input_names[0].as_str() => encoded_length_tensor,
                cached_input_names[1].as_str() => decoder_input_ids,
            };
            for layer_idx in 0..self.decoder_num_layers {
                let base = 2 + (layer_idx * 4);
                inputs.push((
                    cached_input_names[base].as_str().into(),
                    OrtTensor::from_array(self_keys[layer_idx].clone())?.into(),
                ));
                inputs.push((
                    cached_input_names[base + 1].as_str().into(),
                    OrtTensor::from_array(self_values[layer_idx].clone())?.into(),
                ));
                inputs.push((
                    cached_input_names[base + 2].as_str().into(),
                    OrtTensor::from_array(cross_keys[layer_idx].clone())?.into(),
                ));
                inputs.push((
                    cached_input_names[base + 3].as_str().into(),
                    OrtTensor::from_array(cross_values[layer_idx].clone())?.into(),
                ));
            }
            cached_input_s += cached_input_started.elapsed().as_secs_f64();

            let cached_started = Instant::now();
            let cached_outputs = self
                .decoder_cached_step
                .run(inputs)
                .context("Cohere decoder_cached_step session failed")?;
            cached_run_s += cached_started.elapsed().as_secs_f64();
            anyhow::ensure!(
                cached_outputs.len() == 1 + (self.decoder_num_layers * 2),
                "unexpected Cohere decoder_cached_step output count {}",
                cached_outputs.len()
            );

            let cached_extract_started = Instant::now();
            current_token = argmax_last_token(&extract_array_f32(&cached_outputs[0])?)?;
            for layer_idx in 0..self.decoder_num_layers {
                let base = 1 + (layer_idx * 2);
                self_keys[layer_idx] = extract_array_f32(&cached_outputs[base])?;
                self_values[layer_idx] = extract_array_f32(&cached_outputs[base + 1])?;
            }
            cached_extract_s += cached_extract_started.elapsed().as_secs_f64();
            cached_steps += 1;
        }

        let token_decode_started = Instant::now();
        if env_var_truthy("ASR_COHERE_DEBUG_TOKENS") {
            eprintln!("cohere_tokens={generated_ids:?}");
        }
        let text = self
            .tokenizer
            .decode(&generated_ids, true)
            .map(|text| text.trim().to_string())
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .context("failed to decode Cohere token ids")?;
        let token_spans = decoded_token_spans(&self.tokenizer, &generated_ids, &text);
        let words = estimate_word_timestamps_from_tokens(
            &text,
            &token_spans,
            generated_ids.len(),
            duration_ms,
        );
        let token_decode_s = token_decode_started.elapsed().as_secs_f64();

        if timings_enabled {
            eprintln!(
                "cohere_timing total_ms={:.2} feature_steps={} encoded_length={} tokens={} encoder_run_ms={:.2} encoder_extract_ms={:.2} prefill_run_ms={:.2} prefill_extract_ms={:.2} cached_input_ms={:.2} cached_run_ms={:.2} cached_extract_ms={:.2} token_decode_ms={:.2}",
                decode_started.elapsed().as_secs_f64() * 1000.0,
                raw_feature_length,
                encoded_length,
                cached_steps,
                encoder_run_s * 1000.0,
                encoder_extract_s * 1000.0,
                prefill_run_s * 1000.0,
                prefill_extract_s * 1000.0,
                cached_input_s * 1000.0,
                cached_run_s * 1000.0,
                cached_extract_s * 1000.0,
                token_decode_s * 1000.0,
            );
        }

        Ok(WindowTranscription {
            text,
            words,
            stitch_words: None,
        })
    }
}

fn decoded_token_spans(
    tokenizer: &Tokenizer,
    generated_ids: &[u32],
    text: &str,
) -> Vec<TokenTextSpan> {
    let mut spans = Vec::new();
    let mut previous_end = 0usize;

    for end in 1..=generated_ids.len() {
        let Ok(prefix) = tokenizer.decode(&generated_ids[..end], true) else {
            continue;
        };
        let prefix_end = prefix.trim_start().len().min(text.len());
        if prefix_end > previous_end {
            spans.push(TokenTextSpan {
                token_index: end - 1,
                start: previous_end,
                end: prefix_end,
            });
            previous_end = prefix_end;
        }
    }

    spans
}

fn extract_array_f32(value: &ort::value::Value) -> Result<ArrayD<f32>> {
    value
        .try_extract_array::<f32>()
        .map(|array| array.to_owned())
        .map_err(|error| anyhow::anyhow!(error.to_string()))
}

fn dump_f32_array(path: &Path, array: &ArrayD<f32>) -> Result<()> {
    let mut bytes = Vec::with_capacity(array.len() * std::mem::size_of::<f32>());
    for value in array.iter() {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn argmax_last_token(logits: &ArrayD<f32>) -> Result<i64> {
    let mut view = logits.view();
    while view.ndim() > 2 {
        view = view.index_axis_move(Axis(0), 0);
    }

    let last_token = if view.ndim() == 1 {
        view.to_owned()
            .into_dimensionality::<Ix1>()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?
    } else {
        anyhow::ensure!(
            !view.is_empty() && view.shape()[0] > 0,
            "Cohere logits did not contain any token steps"
        );
        view.index_axis(Axis(0), view.shape()[0] - 1)
            .to_owned()
            .into_dimensionality::<Ix1>()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?
    };

    let (index, _) = last_token
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .context("Cohere logits were empty")?;
    Ok(index as i64)
}

fn squeeze_to_1d(array: &ArrayD<f32>) -> Result<ndarray::Array1<f32>> {
    let mut view = array.view();
    while view.ndim() > 1 {
        view = view.index_axis_move(Axis(0), 0);
    }
    view.to_owned()
        .into_dimensionality::<Ix1>()
        .map_err(|error| anyhow::anyhow!(error.to_string()))
}

fn build_prompt(language: &str, punctuation: bool) -> String {
    let pnc_token = if punctuation { "<|pnc|>" } else { "<|nopnc|>" };
    format!(
        "<|startofcontext|><|startoftranscript|><|emo:undefined|><|{language}|><|{language}|>{pnc_token}<|noitn|><|notimestamp|><|nodiarize|>"
    )
}

fn load_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("failed to parse {}", path.display()))
}

fn ort_error<E: std::fmt::Display>(error: E) -> anyhow::Error {
    anyhow::anyhow!(error.to_string())
}

fn env_var_truthy(name: &str) -> bool {
    env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "True"))
        .unwrap_or(false)
}

fn env_var_nonempty(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_var_usize(name: &str) -> Option<usize> {
    env::var(name).ok()?.trim().parse::<usize>().ok()
}

fn cohere_intra_threads() -> Option<usize> {
    match env_var_usize("ASR_COHERE_INTRA_THREADS") {
        Some(0) => None,
        Some(threads) => Some(threads),
        None => default_cohere_intra_threads(),
    }
}

#[cfg(target_os = "macos")]
fn default_cohere_intra_threads() -> Option<usize> {
    None
}

#[cfg(not(target_os = "macos"))]
fn default_cohere_intra_threads() -> Option<usize> {
    Some(1)
}

fn cohere_inter_threads() -> Option<usize> {
    env_var_usize("ASR_COHERE_INTER_THREADS").filter(|threads| *threads > 0)
}

fn env_var_u8(name: &str) -> Option<u8> {
    env::var(name).ok()?.trim().parse::<u8>().ok()
}

fn ensure_ort_initialized() -> Result<()> {
    let result = ORT_INIT.get_or_init(|| {
        if let Some(path) = configured_onnxruntime_lib_path() {
            let created = ort::init_from(path.as_str())
                .map_err(|error| error.to_string())?
                .commit();
            info!(created, path, "initialized dynamic ONNX Runtime");
            return Ok(());
        }
        let created = ort::init().commit();
        info!(created, "initialized ONNX Runtime");
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
        .or_else(default_macos_coreml_onnxruntime_lib_path)
}

#[cfg(target_os = "macos")]
fn default_macos_coreml_onnxruntime_lib_path() -> Option<String> {
    if !cohere_coreml_requested() {
        return None;
    }

    [
        "/opt/homebrew/lib/libonnxruntime.dylib",
        "/usr/local/lib/libonnxruntime.dylib",
    ]
    .into_iter()
    .find(|path| Path::new(path).is_file())
    .map(str::to_string)
}

#[cfg(not(target_os = "macos"))]
fn default_macos_coreml_onnxruntime_lib_path() -> Option<String> {
    None
}

fn session_from_providers(path: &Path, providers: &[ExecutionProviderDispatch]) -> Result<Session> {
    let optimization_level = if cohere_coreml_requested() {
        GraphOptimizationLevel::Disable
    } else {
        GraphOptimizationLevel::Level3
    };
    let intra_threads = cohere_intra_threads();
    let inter_threads = cohere_inter_threads();

    let mut builder = Session::builder()
        .map_err(ort_error)?
        .with_optimization_level(optimization_level)
        .map_err(ort_error)?
        .with_log_level(LogLevel::Info)
        .map_err(ort_error)?
        .with_execution_providers(providers)
        .map_err(ort_error)?;

    if let Some(intra_threads) = intra_threads {
        builder = builder
            .with_intra_threads(intra_threads)
            .map_err(ort_error)?;
    }

    if env_var_truthy("ASR_COHERE_PARALLEL_EXECUTION") {
        builder = builder.with_parallel_execution(true).map_err(ort_error)?;
    }

    if let Some(inter_threads) = inter_threads {
        builder = builder
            .with_inter_threads(inter_threads)
            .map_err(ort_error)?;
    }

    builder.commit_from_file(path).map_err(ort_error)
}

fn cohere_coreml_requested() -> bool {
    let provider =
        env_var_nonempty("ASR_COHERE_EXECUTION_PROVIDER").map(|value| value.to_ascii_lowercase());
    env_var_truthy("ASR_COHERE_COREML")
        || matches!(provider.as_deref(), Some("coreml" | "metal" | "apple"))
}

fn concrete_shape(value_type: &ValueType) -> Option<Vec<usize>> {
    let shape = value_type.tensor_shape()?;
    Some(
        shape
            .iter()
            .map(|dim| if *dim > 0 { *dim as usize } else { 1 })
            .collect(),
    )
}

fn input_shapes(path: &Path) -> Result<Vec<(String, Vec<usize>)>> {
    let session = session_from_providers(path, &[CPUExecutionProvider::default().build()])?;
    Ok(session
        .inputs()
        .iter()
        .map(|input| {
            (
                input.name().to_string(),
                concrete_shape(input.dtype()).unwrap_or_default(),
            )
        })
        .collect())
}

fn inspect_shapes(
    path: &Path,
    enabled: bool,
    component_name: &str,
) -> Option<Vec<(String, Vec<usize>)>> {
    if !enabled {
        return None;
    }

    match input_shapes(path) {
        Ok(shapes) => Some(shapes),
        Err(error) => {
            warn!(
                path = %path.display(),
                error = %error,
                "{component_name} shapes unavailable for TensorRT; continuing without explicit profiles"
            );
            None
        }
    }
}

fn format_profile_entry(name: &str, dims: &[usize]) -> String {
    format!(
        "{}:{}",
        name,
        dims.iter()
            .map(|dim| dim.to_string())
            .collect::<Vec<_>>()
            .join("x")
    )
}

fn frame_steps_for_duration(sample_rate: usize, hop_size: usize, seconds: usize) -> usize {
    ((sample_rate * seconds.max(1)) / hop_size.max(1)).max(1)
}

fn encoded_steps_for(feature_steps: usize, subsampling_factor: usize) -> usize {
    feature_steps.div_ceil(subsampling_factor.max(1)).max(1)
}

fn provider_chain(
    component: ModelComponent,
    device_id: Option<usize>,
    runtime: &CohereRuntimeConfig,
    shapes: Option<&[(String, Vec<usize>)]>,
) -> Vec<ExecutionProviderDispatch> {
    if runtime.force_cpu {
        return vec![CPUExecutionProvider::default().build()];
    }

    if runtime.coreml.enabled {
        return vec![coreml_provider(&runtime.coreml)];
    }

    match device_id {
        Some(device_id) => {
            let mut providers = Vec::new();
            if runtime.trt.enabled_for(component) {
                let cache_dir = runtime.trt.cache_dir.to_string_lossy().into_owned();
                let mut tensorrt = TensorRTExecutionProvider::default()
                    .with_device_id(device_id as i32)
                    .with_engine_cache(true)
                    .with_engine_cache_path(&cache_dir)
                    .with_engine_cache_prefix(component.cache_prefix())
                    .with_timing_cache(true)
                    .with_timing_cache_path(&cache_dir)
                    .with_max_workspace_size(runtime.trt.workspace_bytes)
                    .with_builder_optimization_level(runtime.trt.builder_optimization_level)
                    .with_force_sequential_engine_build(true)
                    .with_layer_norm_fp32_fallback(true)
                    .with_detailed_build_log(runtime.trt.detailed_build_log);
                if runtime.trt.fp16 {
                    tensorrt = tensorrt.with_fp16(true);
                }
                if let Some(shapes) = shapes {
                    if let Some(min_shapes) =
                        runtime
                            .trt
                            .profile_for(component, CohereProfileTarget::Min, shapes)
                    {
                        tensorrt = tensorrt.with_profile_min_shapes(min_shapes);
                    }
                    if let Some(opt_shapes) =
                        runtime
                            .trt
                            .profile_for(component, CohereProfileTarget::Opt, shapes)
                    {
                        tensorrt = tensorrt.with_profile_opt_shapes(opt_shapes);
                    }
                    if let Some(max_shapes) =
                        runtime
                            .trt
                            .profile_for(component, CohereProfileTarget::Max, shapes)
                    {
                        tensorrt = tensorrt.with_profile_max_shapes(max_shapes);
                    }
                }
                providers.push(tensorrt.build().error_on_failure());
            }
            providers.push(
                CUDAExecutionProvider::default()
                    .with_device_id(device_id as i32)
                    .build()
                    .error_on_failure(),
            );
            providers
        }
        None => vec![CPUExecutionProvider::default().build()],
    }
}

fn coreml_provider(config: &CohereCoreMlConfig) -> ExecutionProviderDispatch {
    let mut coreml = CoreML::default()
        .with_compute_units(config.compute_units.into())
        .with_static_input_shapes(true)
        .with_specialization_strategy(coreml::SpecializationStrategy::FastPrediction);
    if let Some(cache_dir) = &config.cache_dir {
        coreml = coreml.with_model_cache_dir(cache_dir.display().to_string());
    }
    if config.low_precision_accumulation_on_gpu {
        coreml = coreml.with_low_precision_accumulation_on_gpu(true);
    }
    coreml.build().error_on_failure()
}

impl CohereRuntimeConfig {
    fn from_env(
        model_dir: &Path,
        preprocessor: &CoherePreprocessorConfig,
        model_config: &CohereModelConfig,
        prompt_len: usize,
        max_new_tokens: usize,
    ) -> Self {
        Self {
            force_cpu: env_var_truthy("ASR_COHERE_FORCE_CPU"),
            coreml: CohereCoreMlConfig::from_env(),
            trt: CohereTensorRtConfig::from_env(
                model_dir,
                preprocessor,
                model_config,
                prompt_len,
                max_new_tokens,
            ),
        }
    }

    fn requires_device_ids(&self) -> bool {
        !self.force_cpu && !self.coreml.enabled
    }

    fn uses_single_execution_target(&self) -> bool {
        self.force_cpu || self.coreml.enabled
    }

    fn validate(&self, device_id: Option<usize>) -> Result<()> {
        if self.force_cpu {
            return Ok(());
        }

        if self.coreml.enabled {
            anyhow::ensure!(
                CoreML::default().is_available().unwrap_or(false),
                "Cohere CoreML requested, but the CoreML execution provider is unavailable in the linked ONNX Runtime build"
            );
            if let Some(cache_dir) = &self.coreml.cache_dir {
                fs::create_dir_all(cache_dir).with_context(|| {
                    format!(
                        "failed to create Cohere CoreML cache dir {}",
                        cache_dir.display()
                    )
                })?;
            }
            return Ok(());
        }

        if self.trt.any_enabled() {
            anyhow::ensure!(
                device_id.is_some(),
                "Cohere TensorRT requires a GPU device id"
            );
            anyhow::ensure!(
                TensorRTExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false),
                "Cohere TensorRT requested via ASR_COHERE_TRT_COMPONENTS={}, but TensorRT execution provider is unavailable; check the linked ONNX Runtime build and TensorRT shared libraries",
                self.trt.components_raw
            );
            fs::create_dir_all(&self.trt.cache_dir).with_context(|| {
                format!(
                    "failed to create Cohere TensorRT cache dir {}",
                    self.trt.cache_dir.display()
                )
            })?;
        }

        Ok(())
    }
}

impl CohereCoreMlConfig {
    fn from_env() -> Self {
        let enabled = cohere_coreml_requested();
        let compute_units = env_var_nonempty("ASR_COHERE_COREML_COMPUTE_UNITS")
            .as_deref()
            .and_then(parse_coreml_compute_units)
            .unwrap_or(CohereCoreMlComputeUnits::CpuAndGpu);
        let cache_dir = env_var_nonempty("ASR_COHERE_COREML_CACHE_DIR").map(PathBuf::from);
        let low_precision_accumulation_on_gpu =
            env_var_truthy("ASR_COHERE_COREML_LOW_PRECISION_ACCUMULATION_ON_GPU");

        Self {
            enabled,
            compute_units,
            cache_dir,
            low_precision_accumulation_on_gpu,
        }
    }
}

fn parse_coreml_compute_units(value: &str) -> Option<CohereCoreMlComputeUnits> {
    let normalized = value.trim().to_ascii_lowercase().replace(['_', ' '], "-");
    match normalized.as_str() {
        "all" => Some(CohereCoreMlComputeUnits::All),
        "cpu-and-neural-engine" | "cpu-neural-engine" | "cpu-and-ane" | "ane" => {
            Some(CohereCoreMlComputeUnits::CpuAndNeuralEngine)
        }
        "cpu-and-gpu" | "cpu-gpu" | "gpu" | "metal" => Some(CohereCoreMlComputeUnits::CpuAndGpu),
        "cpu-only" | "cpu" => Some(CohereCoreMlComputeUnits::CpuOnly),
        _ => None,
    }
}

impl CohereTensorRtConfig {
    fn from_env(
        model_dir: &Path,
        preprocessor: &CoherePreprocessorConfig,
        model_config: &CohereModelConfig,
        prompt_len: usize,
        max_new_tokens: usize,
    ) -> Self {
        let components_raw =
            env_var_nonempty("ASR_COHERE_TRT_COMPONENTS").unwrap_or_else(|| "none".to_string());
        let enabled_components = components_raw
            .split(',')
            .map(|value| value.trim().to_ascii_lowercase())
            .filter(|value| !value.is_empty() && value != "none")
            .collect::<HashSet<_>>();
        let default_max_s = model_config
            .max_audio_clip_s
            .map(|seconds| seconds.ceil() as usize)
            .unwrap_or(35)
            .max(1);
        let min_duration_s = env_var_usize("ASR_COHERE_TRT_PROFILE_MIN_S")
            .unwrap_or(1)
            .max(1);
        let max_duration_s = env_var_usize("ASR_COHERE_TRT_PROFILE_MAX_S")
            .or_else(|| env_var_usize("ASR_COHERE_TRT_PROFILE_SECONDS"))
            .unwrap_or(default_max_s)
            .max(min_duration_s);
        let opt_duration_s = env_var_usize("ASR_COHERE_TRT_PROFILE_OPT_S")
            .unwrap_or_else(|| cmp::min(max_duration_s, cmp::max(min_duration_s, 4)))
            .clamp(min_duration_s, max_duration_s);
        let default_min_feature_steps = frame_steps_for_duration(
            preprocessor.sampling_rate as usize,
            preprocessor.n_window_stride,
            min_duration_s,
        );
        let default_opt_feature_steps = frame_steps_for_duration(
            preprocessor.sampling_rate as usize,
            preprocessor.n_window_stride,
            opt_duration_s,
        );
        let default_max_feature_steps = frame_steps_for_duration(
            preprocessor.sampling_rate as usize,
            preprocessor.n_window_stride,
            max_duration_s,
        );
        let min_feature_steps = env_var_usize("ASR_COHERE_TRT_PROFILE_MIN_FRAMES")
            .unwrap_or(default_min_feature_steps)
            .max(1);
        let max_feature_steps = env_var_usize("ASR_COHERE_TRT_PROFILE_MAX_FRAMES")
            .unwrap_or(default_max_feature_steps)
            .max(min_feature_steps);
        let opt_feature_steps = env_var_usize("ASR_COHERE_TRT_PROFILE_OPT_FRAMES")
            .unwrap_or(default_opt_feature_steps)
            .clamp(min_feature_steps, max_feature_steps);

        Self {
            components_raw,
            enabled_components,
            cache_dir: env_var_nonempty("ASR_COHERE_TRT_CACHE_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| model_dir.join(".trt_cache")),
            min_duration_s,
            opt_duration_s,
            max_duration_s,
            min_feature_steps,
            opt_feature_steps,
            max_feature_steps,
            workspace_bytes: env_var_usize("ASR_COHERE_TRT_WORKSPACE_BYTES")
                .unwrap_or(4 * 1024 * 1024 * 1024)
                .max(1 << 20),
            builder_optimization_level: env_var_u8("ASR_COHERE_TRT_BUILDER_OPT_LEVEL")
                .unwrap_or(5)
                .min(5),
            fp16: env::var("ASR_COHERE_TRT_FP16")
                .map(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "YES"))
                .unwrap_or(true),
            detailed_build_log: env_var_truthy("ASR_COHERE_TRT_DETAILED_BUILD_LOG"),
            feature_size: preprocessor.feature_size,
            subsampling_factor: model_config
                .encoder
                .as_ref()
                .and_then(|encoder| encoder.subsampling_factor)
                .unwrap_or(8)
                .max(1),
            prompt_len,
            max_new_tokens: max_new_tokens.max(1),
        }
    }

    fn any_enabled(&self) -> bool {
        !self.enabled_components.is_empty()
    }

    fn enabled_for(&self, component: ModelComponent) -> bool {
        self.enabled_components.contains("all")
            || self.enabled_components.contains(component.config_token())
    }

    fn feature_steps_for(&self, target: CohereProfileTarget) -> usize {
        match target {
            CohereProfileTarget::Min => self.min_feature_steps,
            CohereProfileTarget::Opt => self.opt_feature_steps,
            CohereProfileTarget::Max => self.max_feature_steps,
        }
    }

    fn self_steps_for(&self, target: CohereProfileTarget) -> usize {
        let extra_tokens = match target {
            CohereProfileTarget::Min => 0,
            CohereProfileTarget::Opt => cmp::max(1, self.max_new_tokens.div_ceil(2)),
            CohereProfileTarget::Max => self.max_new_tokens,
        };
        self.prompt_len + extra_tokens
    }

    fn profile_for(
        &self,
        component: ModelComponent,
        target: CohereProfileTarget,
        shapes: &[(String, Vec<usize>)],
    ) -> Option<String> {
        let feature_steps = self.feature_steps_for(target);
        let encoded_steps = encoded_steps_for(feature_steps, self.subsampling_factor);
        let self_steps = self.self_steps_for(target);
        let mut entries = Vec::new();

        for (name, shape) in shapes {
            if shape.is_empty() {
                continue;
            }

            let mut dims = shape.clone();
            match component {
                ModelComponent::Encoder if name == "input_features" && dims.len() >= 3 => {
                    dims[0] = 1;
                    dims[1] = self.feature_size;
                    dims[2] = feature_steps;
                    entries.push(format_profile_entry(name, &dims));
                }
                ModelComponent::Encoder if name == "length" && !dims.is_empty() => {
                    dims[0] = 1;
                    entries.push(format_profile_entry(name, &dims));
                }
                ModelComponent::DecoderPrefill
                    if name == "encoder_hidden_states" && dims.len() >= 3 =>
                {
                    dims[0] = 1;
                    dims[1] = encoded_steps;
                    entries.push(format_profile_entry(name, &dims));
                }
                ModelComponent::DecoderPrefill
                    if (name == "decoder_input_ids" || name == "decoder_attention_mask")
                        && dims.len() >= 2 =>
                {
                    dims[0] = 1;
                    dims[1] = self.prompt_len;
                    entries.push(format_profile_entry(name, &dims));
                }
                ModelComponent::DecoderPrefill if name == "length" && !dims.is_empty() => {
                    dims[0] = 1;
                    entries.push(format_profile_entry(name, &dims));
                }
                ModelComponent::DecoderCachedStep
                    if name == "decoder_input_ids" && dims.len() >= 2 =>
                {
                    dims[0] = 1;
                    dims[1] = 1;
                    entries.push(format_profile_entry(name, &dims));
                }
                ModelComponent::DecoderCachedStep
                    if name == "encoded_length" && !dims.is_empty() =>
                {
                    dims[0] = 1;
                    entries.push(format_profile_entry(name, &dims));
                }
                ModelComponent::DecoderCachedStep
                    if (name.starts_with("self_key_") || name.starts_with("self_value_"))
                        && dims.len() >= 3 =>
                {
                    dims[0] = 1;
                    dims[2] = self_steps;
                    entries.push(format_profile_entry(name, &dims));
                }
                ModelComponent::DecoderCachedStep
                    if (name.starts_with("cross_key_") || name.starts_with("cross_value_"))
                        && dims.len() >= 3 =>
                {
                    dims[0] = 1;
                    dims[2] = encoded_steps;
                    entries.push(format_profile_entry(name, &dims));
                }
                _ => {}
            }
        }

        if entries.is_empty() {
            None
        } else {
            Some(entries.join(","))
        }
    }
}

fn apply_side_model_timestamps(
    result: &mut WindowTranscription,
    timestamped_words: &[TimedWord],
) -> Result<()> {
    anyhow::ensure!(
        result.words.len() == timestamped_words.len(),
        "timestamp word count {} did not match Cohere word count {}",
        timestamped_words.len(),
        result.words.len()
    );
    anyhow::ensure!(
        result
            .words
            .iter()
            .zip(timestamped_words)
            .all(|(cohere, timestamped)| cohere.word == timestamped.word),
        "timestamp word sequence did not match the Cohere word sequence"
    );

    let stitch_words = result.words.clone();
    for (cohere, timestamped) in result.words.iter_mut().zip(timestamped_words) {
        cohere.start_ms = timestamped.start_ms;
        cohere.end_ms = timestamped.end_ms;
    }
    result.stitch_words = Some(stitch_words);
    Ok(())
}

#[cfg(test)]
mod timestamp_tests {
    use super::*;

    #[test]
    fn applying_side_model_timestamps_preserves_cohere_text_and_words() {
        let mut result = WindowTranscription {
            text: "Keep  Cohere’s text.\nExactly.".to_string(),
            words: vec![
                TimedWord {
                    word: "Keep".to_string(),
                    start_ms: 0,
                    end_ms: 100,
                },
                TimedWord {
                    word: "Cohere’s".to_string(),
                    start_ms: 100,
                    end_ms: 200,
                },
                TimedWord {
                    word: "text.".to_string(),
                    start_ms: 200,
                    end_ms: 300,
                },
                TimedWord {
                    word: "Exactly.".to_string(),
                    start_ms: 300,
                    end_ms: 400,
                },
            ],
            stitch_words: None,
        };
        let original_text = result.text.clone();
        let original_words = result
            .words
            .iter()
            .map(|word| word.word.clone())
            .collect::<Vec<_>>();
        let timestamped_words = result
            .words
            .iter()
            .enumerate()
            .map(|(index, word)| TimedWord {
                word: word.word.clone(),
                start_ms: 500 + index as u32 * 200,
                end_ms: 600 + index as u32 * 200,
            })
            .collect::<Vec<_>>();

        apply_side_model_timestamps(&mut result, &timestamped_words).unwrap();

        assert_eq!(result.text.as_bytes(), original_text.as_bytes());
        assert_eq!(
            result
                .words
                .iter()
                .map(|word| word.word.clone())
                .collect::<Vec<_>>(),
            original_words
        );
        assert_eq!(result.words[0].start_ms, 500);
        assert_eq!(result.words[3].end_ms, 1_200);
        assert_eq!(result.stitch_words.as_ref().unwrap()[0].start_ms, 0);
    }

    #[test]
    fn side_model_word_changes_are_rejected() {
        let mut result = WindowTranscription {
            text: "Cohere".to_string(),
            words: vec![TimedWord {
                word: "Cohere".to_string(),
                start_ms: 0,
                end_ms: 100,
            }],
            stitch_words: None,
        };
        let timestamped_words = vec![TimedWord {
            word: "Parakeet".to_string(),
            start_ms: 10,
            end_ms: 90,
        }];

        assert!(apply_side_model_timestamps(&mut result, &timestamped_words).is_err());
        assert_eq!(result.words[0].word, "Cohere");
        assert_eq!((result.words[0].start_ms, result.words[0].end_ms), (0, 100));
        assert!(result.stitch_words.is_none());
    }
}
