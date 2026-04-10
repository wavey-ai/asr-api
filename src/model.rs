use anyhow::{Context, Result};
use parakeet_rs::{ParakeetTDT, TimestampMode, Transcriber};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

pub struct ModelPool {
    models: Vec<Arc<Mutex<ParakeetTDT>>>,
    next: AtomicUsize,
}

impl ModelPool {
    pub fn new(model_dir: &Path, instances: usize) -> Result<Self> {
        let mut models = Vec::with_capacity(instances);
        for _ in 0..instances {
            let model = ParakeetTDT::from_pretrained(model_dir, None).with_context(|| {
                format!("failed to load Parakeet TDT from {}", model_dir.display())
            })?;
            models.push(Arc::new(Mutex::new(model)));
        }

        Ok(Self {
            models,
            next: AtomicUsize::new(0),
        })
    }

    pub async fn transcribe_sentences(
        &self,
        audio: Vec<f32>,
    ) -> Result<parakeet_rs::TranscriptionResult> {
        let index = self.next.fetch_add(1, Ordering::Relaxed) % self.models.len();
        let model = Arc::clone(&self.models[index]);
        tokio::task::spawn_blocking(move || {
            let mut guard = model.lock().expect("parakeet model mutex poisoned");
            guard
                .transcribe_samples(audio, 16_000, 1, Some(TimestampMode::Sentences))
                .context("Parakeet transcription failed")
        })
        .await
        .context("Parakeet transcription task join failed")?
    }
}
