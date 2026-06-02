use crate::chunking::TimedWord;
#[cfg(feature = "cohere-backend")]
use crate::cohere::CohereBackend as CohereAsrBackend;
#[cfg(feature = "cohere-mlx")]
use crate::cohere_mlx::CohereMlxBackend;
use crate::config::AsrModelProvider;
#[cfg(feature = "parakeet-backend")]
use crate::parakeet::ParakeetBackend;
use anyhow::Result;
use std::env;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct WindowTranscription {
    pub text: String,
    pub words: Vec<TimedWord>,
}

pub struct AsrBackend {
    inner: BackendImpl,
}

enum BackendImpl {
    #[cfg(feature = "cohere-backend")]
    Cohere(CohereAsrBackend),
    #[cfg(feature = "cohere-mlx")]
    CohereMlx(CohereMlxBackend),
    #[cfg(feature = "parakeet-backend")]
    Parakeet(ParakeetBackend),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CohereRuntime {
    Onnx,
    Mlx,
}

impl AsrBackend {
    pub fn new(
        model_dir: &Path,
        model_provider: AsrModelProvider,
        device_ids: &[usize],
        onnx_sessions: usize,
        cohere_max_new_tokens: usize,
    ) -> Result<Self> {
        let inner = match model_provider {
            AsrModelProvider::Auto | AsrModelProvider::Cohere => match cohere_runtime()? {
                CohereRuntime::Onnx => {
                    #[cfg(feature = "cohere-backend")]
                    {
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
                        anyhow::bail!(
                            "ASR_COHERE_BACKEND=onnx requested, but this asr-api build does not include the cohere-backend feature"
                        );
                    }
                }
                CohereRuntime::Mlx => {
                    #[cfg(feature = "cohere-mlx")]
                    {
                        let _ = (device_ids, onnx_sessions);
                        BackendImpl::CohereMlx(CohereMlxBackend::new(
                            model_dir,
                            cohere_max_new_tokens,
                        )?)
                    }
                    #[cfg(not(feature = "cohere-mlx"))]
                    {
                        let _ = (model_dir, device_ids, onnx_sessions, cohere_max_new_tokens);
                        anyhow::bail!(
                            "ASR_COHERE_BACKEND=mlx requested, but this asr-api build does not include the cohere-mlx feature"
                        );
                    }
                }
            },
            AsrModelProvider::Parakeet => {
                #[cfg(feature = "parakeet-backend")]
                {
                    BackendImpl::Parakeet(ParakeetBackend::new(
                        model_dir,
                        device_ids,
                        onnx_sessions,
                    )?)
                }
                #[cfg(not(feature = "parakeet-backend"))]
                {
                    let _ = (model_dir, device_ids, onnx_sessions);
                    anyhow::bail!(
                        "ASR_MODEL_PROVIDER=parakeet requested, but this asr-api build does not include the parakeet-backend feature"
                    );
                }
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
            #[cfg(feature = "cohere-backend")]
            BackendImpl::Cohere(backend) => backend.transcribe_window(samples, seq).await,
            #[cfg(feature = "cohere-mlx")]
            BackendImpl::CohereMlx(backend) => backend.transcribe_window(samples, seq).await,
            #[cfg(feature = "parakeet-backend")]
            BackendImpl::Parakeet(backend) => backend.transcribe_window(samples, seq).await,
        }
    }
}

fn cohere_runtime() -> Result<CohereRuntime> {
    match env::var("ASR_COHERE_BACKEND")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .as_deref()
    {
        Some("onnx" | "ort") => Ok(CohereRuntime::Onnx),
        Some("mlx") => Ok(CohereRuntime::Mlx),
        Some(value) => {
            anyhow::bail!("unsupported ASR_COHERE_BACKEND={value}; expected onnx or mlx")
        }
        None => Ok(default_cohere_runtime()),
    }
}

#[cfg(feature = "cohere-backend")]
fn default_cohere_runtime() -> CohereRuntime {
    CohereRuntime::Onnx
}

#[cfg(all(not(feature = "cohere-backend"), feature = "cohere-mlx"))]
fn default_cohere_runtime() -> CohereRuntime {
    CohereRuntime::Mlx
}

#[cfg(not(any(feature = "cohere-backend", feature = "cohere-mlx")))]
fn default_cohere_runtime() -> CohereRuntime {
    CohereRuntime::Onnx
}
