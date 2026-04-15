use crate::chunking::TimedWord;
use anyhow::{Context, Result};
use asr_onnx::{
    Config as OnnxConfig, JobMeta, TranscriberPool as OnnxDecoderPool,
    TranscriptionResult as OnnxResult,
};
use asr_torch::{FeaturizerPool, MelFeaturesPayload};
use crossbeam_channel::Sender;
use ndarray::Array2;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tokio::sync::oneshot;
use tracing::warn;

type FeaturizerJob = (u64, u32, Vec<f32>);

#[derive(Debug, Clone)]
pub struct WindowTranscription {
    pub text: String,
    pub words: Vec<TimedWord>,
}

pub struct AsrBackend {
    featurizer: FeaturizerClient,
    decoder: OnnxDecoderClient,
}

impl AsrBackend {
    pub fn new(
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

    pub async fn transcribe_window(
        &self,
        samples: Vec<f32>,
        seq: u32,
    ) -> Result<WindowTranscription> {
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

struct FeaturizerClient {
    next_id: AtomicU64,
    job_sender: Sender<FeaturizerJob>,
    pending:
        Arc<Mutex<HashMap<u64, oneshot::Sender<std::result::Result<MelFeaturesPayload, String>>>>>,
}

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

struct OnnxDecoderClient {
    pool: Arc<OnnxDecoderPool>,
    state: Arc<Mutex<OnnxDecoderState>>,
}

struct OnnxDecoderState {
    pending: HashMap<u64, oneshot::Sender<std::result::Result<OnnxResult, String>>>,
    completed: HashMap<u64, std::result::Result<OnnxResult, String>>,
}

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

fn dispatch_onnx_decoder_result(state: &Arc<Mutex<OnnxDecoderState>>, result: OnnxResult) {
    let mut guard = state.lock().expect("onnx state mutex poisoned");
    if let Some(sender) = guard.pending.remove(&result.job_id) {
        let _ = sender.send(Ok(result));
    } else {
        guard.completed.insert(result.job_id, Ok(result));
    }
}
