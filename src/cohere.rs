use crate::asr::WindowTranscription;
use crate::chunking::TimedWord;
use crate::config::DEFAULT_LANGUAGE;
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
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;
use serde::Deserialize;
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::env;
use std::f32::consts::PI;
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
struct PreprocessorConfig {
    dither: f32,
    feature_size: usize,
    n_fft: usize,
    n_window_size: usize,
    n_window_stride: usize,
    normalize: String,
    padding_value: f32,
    sampling_rate: u32,
    window: String,
}

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
}

impl CohereBackend {
    pub fn new(
        model_dir: &Path,
        device_ids: &[usize],
        onnx_sessions: usize,
        max_new_tokens: usize,
    ) -> Result<Self> {
        let preprocessor =
            load_json::<PreprocessorConfig>(&model_dir.join("preprocessor_config.json"))
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
        Ok(Self { frontend, decoder })
    }

    pub async fn transcribe_window(
        &self,
        samples: Vec<f32>,
        _seq: u32,
    ) -> Result<WindowTranscription> {
        let features = self.frontend.compute(&samples)?;
        let text = self.decoder.decode(features).await?;
        Ok(WindowTranscription {
            text,
            words: Vec::<TimedWord>::new(),
        })
    }
}

struct CohereDecoderClient {
    next_id: AtomicU64,
    job_tx: Sender<CohereJob>,
    state: Arc<Mutex<CohereDecoderState>>,
}

struct CohereDecoderState {
    pending: HashMap<u64, oneshot::Sender<std::result::Result<String, String>>>,
    completed: HashMap<u64, std::result::Result<String, String>>,
}

struct CohereJob {
    job_id: u64,
    features: Array2<f32>,
}

struct CohereJobResult {
    job_id: u64,
    result: std::result::Result<String, String>,
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
        for device_id in effective_device_ids {
            for _ in 0..onnx_sessions.max(1) {
                let worker = CohereWorker::new(
                    model_dir,
                    device_id,
                    tokenizer.clone(),
                    decode.clone(),
                    runtime.clone(),
                )?;
                let worker_job_rx = job_rx.clone();
                let worker_result_tx = result_tx.clone();
                thread::spawn(move || worker_loop(worker, worker_job_rx, worker_result_tx));
            }
        }
        drop(result_tx);

        let dispatch_state = Arc::clone(&state);
        thread::spawn(move || {
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
            job_tx,
            state,
        })
    }

    async fn decode(&self, features: Array2<f32>) -> Result<String> {
        let job_id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        {
            let mut guard = self.state.lock().expect("cohere state mutex poisoned");
            if let Some(result) = guard.completed.remove(&job_id) {
                return result.map_err(anyhow::Error::msg);
            }
            guard.pending.insert(job_id, tx);
        }

        if let Err(error) = self.job_tx.send(CohereJob { job_id, features }) {
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

fn worker_loop(
    mut worker: CohereWorker,
    job_rx: Receiver<CohereJob>,
    result_tx: Sender<CohereJobResult>,
) {
    for job in job_rx {
        let result = worker
            .decode(job.features)
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

    fn decode(&mut self, features: Array2<f32>) -> Result<String> {
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
        let text = self
            .tokenizer
            .decode(&generated_ids, true)
            .map(|text| text.trim().to_string())
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .context("failed to decode Cohere token ids")?;
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

        Ok(text)
    }
}

fn extract_array_f32(value: &ort::value::Value) -> Result<ArrayD<f32>> {
    value
        .try_extract_array::<f32>()
        .map(|array| array.to_owned())
        .map_err(|error| anyhow::anyhow!(error.to_string()))
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
        preprocessor: &PreprocessorConfig,
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
        preprocessor: &PreprocessorConfig,
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

struct CohereFrontend {
    sample_rate: u32,
    n_fft: usize,
    window_size: usize,
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
    fn new(config: PreprocessorConfig) -> Result<Self> {
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
            sample_rate: config.sampling_rate,
            n_fft: config.n_fft,
            window_size: config.n_window_size,
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

    fn compute(&self, samples: &[f32]) -> Result<Array2<f32>> {
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
