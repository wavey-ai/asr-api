use crate::chunking::TimedWord;
#[cfg(feature = "cohere-backend")]
use crate::cohere::CohereBackend as CohereAsrBackend;
use crate::config::AsrModelProvider;
#[cfg(feature = "nemo-backend")]
use anyhow::Context;
use anyhow::Result;
#[cfg(feature = "nemo-backend")]
use asr_onnx::{
    Config as OnnxConfig, JobMeta, TranscriberPool as OnnxDecoderPool,
    TranscriptionResult as OnnxResult,
};
#[cfg(feature = "nemo-backend")]
use asr_torch::{FeaturizerPool, MelFeaturesPayload};
#[cfg(feature = "nemo-backend")]
use crossbeam_channel::Sender;
#[cfg(feature = "nemo-backend")]
use ndarray::Array2;
#[cfg(feature = "nemo-backend")]
use std::collections::HashMap;
use std::path::Path;
#[cfg(feature = "nemo-backend")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "nemo-backend")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "nemo-backend")]
use std::thread;
#[cfg(feature = "nemo-backend")]
use std::time::Duration;
#[cfg(feature = "nemo-backend")]
use tokio::sync::oneshot;
#[cfg(feature = "nemo-backend")]
use tracing::warn;

#[cfg(feature = "nemo-backend")]
type FeaturizerJob = (u64, u32, Vec<f32>);

#[derive(Debug, Clone)]
pub struct WindowTranscription {
    pub text: String,
    pub words: Vec<TimedWord>,
}

pub struct AsrBackend {
    inner: BackendImpl,
}

enum BackendImpl {
    #[cfg(feature = "nemo-backend")]
    Nemo(NemoAsrBackend),
    #[cfg(feature = "cohere-backend")]
    Cohere(CohereAsrBackend),
}

impl AsrBackend {
    pub fn new(
        model_dir: &Path,
        vocab_path: Option<&Path>,
        model_provider: AsrModelProvider,
        device_ids: &[usize],
        torch_sessions: usize,
        onnx_sessions: usize,
        cohere_max_new_tokens: usize,
    ) -> Result<Self> {
        let inner = match model_provider {
            AsrModelProvider::Nemo => {
                #[cfg(feature = "nemo-backend")]
                {
                    BackendImpl::Nemo(NemoAsrBackend::new(
                        model_dir,
                        vocab_path.context("vocab path is required for the NeMo backend")?,
                        device_ids,
                        torch_sessions,
                        onnx_sessions,
                    )?)
                }
                #[cfg(not(feature = "nemo-backend"))]
                {
                    let _ = (model_dir, vocab_path, device_ids, torch_sessions, onnx_sessions);
                    anyhow::bail!("this asr-api build does not include the NeMo backend");
                }
            }
            AsrModelProvider::Cohere => {
                #[cfg(feature = "cohere-backend")]
                {
                    let _ = torch_sessions;
                    BackendImpl::Cohere(CohereAsrBackend::new(
                        model_dir,
                        device_ids,
                        onnx_sessions,
                        cohere_max_new_tokens,
                    )?)
                }
                #[cfg(not(feature = "cohere-backend"))]
                {
                    let _ = (model_dir, device_ids, onnx_sessions, cohere_max_new_tokens);
                    anyhow::bail!("this asr-api build does not include the Cohere backend");
                }
            }
            AsrModelProvider::Auto => {
                anyhow::bail!("AsrBackend::new requires a concrete ASR model provider")
            }
        };
        Ok(Self { inner })
    }

    pub async fn transcribe_window(
        &self,
        samples: Vec<f32>,
        seq: u32,
    ) -> Result<WindowTranscription> {
        match &self.inner {
            #[cfg(feature = "nemo-backend")]
            BackendImpl::Nemo(backend) => backend.transcribe_window(samples, seq).await,
            #[cfg(feature = "cohere-backend")]
            BackendImpl::Cohere(backend) => backend.transcribe_window(samples, seq).await,
        }
    }
}

#[cfg(feature = "nemo-backend")]
struct NemoAsrBackend {
    featurizer: FeaturizerClient,
    decoder: OnnxDecoderClient,
}

#[cfg(feature = "nemo-backend")]
impl NemoAsrBackend {
    fn new(
        model_dir: &Path,
        vocab_path: &Path,
        device_ids: &[usize],
        torch_sessions: usize,
        onnx_sessions: usize,
    ) -> Result<Self> {
        let featurizer_pool = FeaturizerPool::new(device_ids, torch_sessions)
            .context("failed to initialize asr-torch featurizer pool")?;
        let featurizer = FeaturizerClient::new(featurizer_pool);

        let onnx_config = OnnxConfig::default().with_num_sessions(onnx_sessions);
        let decoder_pool = OnnxDecoderPool::new(model_dir, vocab_path, device_ids, onnx_config)
            .context("failed to initialize asr-onnx decoder pool")?;
        let expected_ready = device_ids.len().max(1) * onnx_sessions;
        for ready_idx in 0..expected_ready {
            decoder_pool
                .ready()
                .recv_timeout(Duration::from_secs(120))
                .with_context(|| {
                    format!("timed out waiting for ASR ONNX decoder session {ready_idx}")
                })?;
        }
        let decoder = OnnxDecoderClient::new(decoder_pool);

        Ok(Self {
            featurizer,
            decoder,
        })
    }

    async fn transcribe_window(&self, samples: Vec<f32>, seq: u32) -> Result<WindowTranscription> {
        let features = self.featurizer.featurize(seq, samples).await?;
        let features_array = Array2::from_shape_vec((features.rows, features.cols), features.data)
            .context("invalid mel feature tensor shape from asr-torch")?;
        let result = self
            .decoder
            .decode(
                format!("chunk-{seq}"),
                features_array,
                JobMeta {
                    seq,
                    chunk_id: u64::from(seq),
                },
            )
            .await?;

        Ok(WindowTranscription {
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
        })
    }
}

#[cfg(feature = "nemo-backend")]
struct FeaturizerClient {
    next_id: AtomicU64,
    job_sender: Sender<FeaturizerJob>,
    pending:
        Arc<Mutex<HashMap<u64, oneshot::Sender<std::result::Result<MelFeaturesPayload, String>>>>>,
}

#[cfg(feature = "nemo-backend")]
impl FeaturizerClient {
    fn new(pool: FeaturizerPool) -> Self {
        let pending: Arc<
            Mutex<HashMap<u64, oneshot::Sender<std::result::Result<MelFeaturesPayload, String>>>>,
        > = Arc::new(Mutex::new(HashMap::new()));
        let result_rx = pool.result_receiver_bin.clone();
        let dispatch_pending = Arc::clone(&pending);

        thread::spawn(move || {
            for payload in result_rx {
                let sender = dispatch_pending
                    .lock()
                    .expect("featurizer pending mutex poisoned")
                    .remove(&payload.chunk_id);
                if let Some(sender) = sender {
                    let _ = sender.send(Ok(payload));
                } else {
                    warn!(
                        chunk_id = payload.chunk_id,
                        "dropping unmatched featurizer result"
                    );
                }
            }

            let mut guard = dispatch_pending
                .lock()
                .expect("featurizer pending mutex poisoned");
            for (_, sender) in guard.drain() {
                let _ = sender.send(Err("featurizer result channel closed".into()));
            }
        });

        Self {
            next_id: AtomicU64::new(0),
            job_sender: pool.job_sender.clone(),
            pending,
        }
    }

    async fn featurize(&self, seq: u32, samples: Vec<f32>) -> Result<MelFeaturesPayload> {
        let chunk_id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.pending
            .lock()
            .expect("featurizer pending mutex poisoned")
            .insert(chunk_id, tx);

        if let Err(error) = self.job_sender.send((chunk_id, seq, samples)) {
            self.pending
                .lock()
                .expect("featurizer pending mutex poisoned")
                .remove(&chunk_id);
            anyhow::bail!("failed to enqueue featurizer job: {error}");
        }

        match rx.await {
            Ok(Ok(payload)) => Ok(payload),
            Ok(Err(error)) => anyhow::bail!(error),
            Err(_) => anyhow::bail!("featurizer request was canceled"),
        }
    }
}

#[cfg(feature = "nemo-backend")]
struct OnnxDecoderClient {
    pool: Arc<OnnxDecoderPool>,
    state: Arc<Mutex<OnnxDecoderState>>,
}

#[cfg(feature = "nemo-backend")]
struct OnnxDecoderState {
    pending: HashMap<u64, oneshot::Sender<std::result::Result<OnnxResult, String>>>,
    completed: HashMap<u64, std::result::Result<OnnxResult, String>>,
}

#[cfg(feature = "nemo-backend")]
impl OnnxDecoderClient {
    fn new(pool: OnnxDecoderPool) -> Self {
        let pool = Arc::new(pool);
        let state = Arc::new(Mutex::new(OnnxDecoderState {
            pending: HashMap::new(),
            completed: HashMap::new(),
        }));

        let result_rx = pool.result_rx().clone();
        let dispatch_state = Arc::clone(&state);
        thread::spawn(move || {
            for result in result_rx {
                dispatch_onnx_decoder_result(&dispatch_state, result);
            }

            let mut guard = dispatch_state.lock().expect("onnx state mutex poisoned");
            for (_, sender) in guard.pending.drain() {
                let _ = sender.send(Err("asr-onnx result channel closed".into()));
            }
            guard.completed.clear();
        });

        Self { pool, state }
    }

    async fn decode(
        &self,
        name: String,
        features: Array2<f32>,
        meta: JobMeta,
    ) -> Result<OnnxResult> {
        let job_id = self
            .pool
            .submit(name, features, meta)
            .context("failed to enqueue asr-onnx job")?;
        let (tx, rx) = oneshot::channel();

        {
            let mut guard = self.state.lock().expect("onnx state mutex poisoned");
            if let Some(result) = guard.completed.remove(&job_id) {
                return result.map_err(anyhow::Error::msg);
            }
            guard.pending.insert(job_id, tx);
        }

        match rx.await {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(error)) => anyhow::bail!(error),
            Err(_) => anyhow::bail!("asr-onnx request was canceled"),
        }
    }
}

#[cfg(feature = "nemo-backend")]
fn dispatch_onnx_decoder_result(state: &Arc<Mutex<OnnxDecoderState>>, result: OnnxResult) {
    let mut guard = state.lock().expect("onnx state mutex poisoned");
    if let Some(sender) = guard.pending.remove(&result.job_id) {
        let _ = sender.send(Ok(result));
    } else {
        guard.completed.insert(result.job_id, Ok(result));
    }
}
