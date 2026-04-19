use crate::asr::WindowTranscription;
use crate::chunking::TimedWord;
use crate::config::DEFAULT_LANGUAGE;
use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender, bounded};
use ndarray::{Array2, Array3, ArrayD, Axis, Ix1};
use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, ExecutionProviderDispatch,
};
use ort::logging::LogLevel;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor as OrtTensor;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;
use serde::Deserialize;
use std::env;
use std::collections::HashMap;
use std::f32::consts::PI;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use tokio::sync::oneshot;
use tokenizers::Tokenizer;
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
        let preprocessor = load_json::<PreprocessorConfig>(&model_dir.join("preprocessor_config.json"))
            .context("failed to load Cohere preprocessor_config.json")?;
        let generation = load_json::<GenerationConfig>(&model_dir.join("generation_config.json"))
            .context("failed to load Cohere generation_config.json")?;
        let export = load_json::<ExportMetadata>(&model_dir.join("export.json")).unwrap_or_default();
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
        let decoder = CohereDecoderClient::new(model_dir, device_ids, onnx_sessions, tokenizer, decode)?;
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
    ) -> Result<Self> {
        let force_cpu = env_var_truthy("ASR_COHERE_FORCE_CPU");
        if !force_cpu {
            anyhow::ensure!(
                !device_ids.is_empty(),
                "Cohere backend requires at least one GPU device id; set ASR_DEVICE_IDS or use ASR_COHERE_FORCE_CPU=true for explicit CPU compare mode"
            );
        }

        let worker_count = device_ids.len().max(1) * onnx_sessions.max(1);
        let (job_tx, job_rx) = bounded::<CohereJob>(worker_count * 2);
        let (result_tx, result_rx) = bounded::<CohereJobResult>(worker_count * 2);
        let state = Arc::new(Mutex::new(CohereDecoderState {
            pending: HashMap::new(),
            completed: HashMap::new(),
        }));

        let effective_device_ids = if force_cpu {
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
    ) -> Result<Self> {
        ensure_ort_initialized()?;
        let providers = provider_chain(device_id);
        let encoder = session_from_providers(&model_dir.join("encoder.onnx"), &providers)
            .context("failed to initialize Cohere encoder session")?;
        let decoder_prefill =
            session_from_providers(&model_dir.join("decoder_prefill.onnx"), &providers)
                .context("failed to initialize Cohere decoder_prefill session")?;
        let decoder_cached_step =
            session_from_providers(&model_dir.join("decoder_cached_step.onnx"), &providers)
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
        let feature_shape = features.dim();
        let raw_feature_length = feature_shape.1 as i64;
        let (feature_data, feature_offset) = features.into_raw_vec_and_offset();
        anyhow::ensure!(
            feature_offset.unwrap_or(0) == 0,
            "Cohere feature tensor had a non-zero storage offset"
        );
        let feature_tensor = Array3::from_shape_vec(
            (1, feature_shape.0, feature_shape.1),
            feature_data,
        )?;
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
        let encoder_outputs = self
            .encoder
            .run(encoder_inputs)
            .context("Cohere encoder session failed")?;
        anyhow::ensure!(
            encoder_outputs.len() >= 2,
            "Cohere encoder did not return encoder_hidden_states and encoded_length"
        );
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

        let prompt_len = self.decode.prompt_ids.len();
        let prompt_ids = OrtTensor::from_array((
            [1i64, prompt_len as i64],
            self.decode.prompt_ids.clone(),
        ))?;
        let prompt_mask = OrtTensor::from_array(([1i64, prompt_len as i64], vec![1i64; prompt_len]))?;
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
        let prefill_outputs = self
            .decoder_prefill
            .run(prefill_inputs)
            .context("Cohere decoder_prefill session failed")?;
        anyhow::ensure!(
            prefill_outputs.len() == 1 + (self.decoder_num_layers * 4),
            "unexpected Cohere decoder_prefill output count {}",
            prefill_outputs.len()
        );

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

        for _ in 0..self.decode.max_new_tokens {
            if current_token == self.decode.eos_token_id {
                break;
            }
            generated_ids.push(current_token as u32);

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

            let cached_outputs = self
                .decoder_cached_step
                .run(inputs)
                .context("Cohere decoder_cached_step session failed")?;
            anyhow::ensure!(
                cached_outputs.len() == 1 + (self.decoder_num_layers * 2),
                "unexpected Cohere decoder_cached_step output count {}",
                cached_outputs.len()
            );
            current_token = argmax_last_token(&extract_array_f32(&cached_outputs[0])?)?;
            for layer_idx in 0..self.decoder_num_layers {
                let base = 1 + (layer_idx * 2);
                self_keys[layer_idx] = extract_array_f32(&cached_outputs[base])?;
                self_values[layer_idx] = extract_array_f32(&cached_outputs[base + 1])?;
            }
        }

        self.tokenizer
            .decode(&generated_ids, true)
            .map(|text| text.trim().to_string())
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .context("failed to decode Cohere token ids")
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
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", path.display()))
}

fn ort_error<E: std::fmt::Display>(error: E) -> anyhow::Error {
    anyhow::anyhow!(error.to_string())
}

fn env_var_truthy(name: &str) -> bool {
    env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "True"))
        .unwrap_or(false)
}

fn ensure_ort_initialized() -> Result<()> {
    let result = ORT_INIT.get_or_init(|| {
        let _created = ort::init().commit();
        Ok(())
    });
    result
        .as_ref()
        .map_err(|error| anyhow::anyhow!(error.clone()))?;
    Ok(())
}

fn session_from_providers(path: &Path, providers: &[ExecutionProviderDispatch]) -> Result<Session> {
    Session::builder()
        .map_err(ort_error)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_error)?
        .with_log_level(LogLevel::Info)
        .map_err(ort_error)?
        .with_execution_providers(providers)
        .map_err(ort_error)?
        .with_intra_threads(1)
        .map_err(ort_error)?
        .commit_from_file(path)
        .map_err(ort_error)
}

fn provider_chain(device_id: Option<usize>) -> Vec<ExecutionProviderDispatch> {
    match device_id {
        Some(device_id) => vec![
            CUDAExecutionProvider::default()
                .with_device_id(device_id as i32)
                .build()
                .error_on_failure(),
        ],
        None => vec![CPUExecutionProvider::default().build()],
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
                let filter = &self.mel_filters[(mel_idx * self.fft_bins)..((mel_idx + 1) * self.fft_bins)];
                let mut energy = 0.0f32;
                for (weight, bin_power) in filter.iter().zip(power.iter()) {
                    energy += *weight * *bin_power;
                }
                let logged = (energy + 2f32.powi(-24)).ln();
                features[(mel_idx * seq_len) + frame_idx] = logged;
            }
        }

        normalize_per_feature(&mut features, self.feature_size, seq_len, self.padding_value);
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
