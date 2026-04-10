use crate::chunking::{AudioChunker, SegmentCommitter, TimedSegment};
use crate::config::AppConfig;
use crate::events::{encode_event, TranscriptEvent};
use crate::model::ModelPool;
use anyhow::Result;
use bytes::Bytes;
use http_pack::stream::{StreamHeaders, StreamResponseHeaders};
use http_pack::HeaderField;
use soundkit::audio_pipeline::{deserialize_audio, vec_i16_to_f32, vec_i32_to_f32};
use soundkit::audio_types::{AudioData, PcmData};
use soundkit_decoder::{DecodeOptions, DecodePipeline, DecodePipelineHandle};
use std::sync::Arc;
use tokio::time::{interval, Duration};
use tracing::{debug, error};
use upload_response::{TailSlot, UploadResponseService};

#[derive(Clone)]
pub struct WorkerState {
    config: AppConfig,
    models: Arc<ModelPool>,
}

impl WorkerState {
    pub fn new(config: AppConfig, models: Arc<ModelPool>) -> Self {
        Self { config, models }
    }

    pub async fn process_stream(&self, service: Arc<UploadResponseService>, stream_id: u64) {
        let worker_id = format!("parakeet-tdt-{stream_id}");
        if let Err(error) = self
            .process_stream_inner(service.clone(), stream_id, &worker_id)
            .await
        {
            error!(stream_id, error = %error, "stream processing failed");
            let _ = self
                .send_error_response(service.clone(), stream_id, &worker_id, error.to_string())
                .await;
        }
    }

    async fn process_stream_inner(
        &self,
        service: Arc<UploadResponseService>,
        stream_id: u64,
        worker_id: &str,
    ) -> Result<()> {
        service.register_reader(stream_id, worker_id).await;
        let mut claimed = false;
        let mut last_slot = 0usize;
        let mut poll = interval(Duration::from_millis(1));
        let mut decoder = Self::new_decoder();
        let mut chunker = AudioChunker::new(
            self.config.chunk_samples(),
            self.config.overlap_samples(),
            self.config.min_final_samples(),
        );
        let mut committer = SegmentCommitter::default();

        loop {
            poll.tick().await;
            let current_last = service.request_last(stream_id).unwrap_or(0);
            if current_last <= last_slot {
                continue;
            }

            for slot_id in (last_slot + 1)..=current_last {
                match service.tail_request(stream_id, slot_id).await {
                    Some(TailSlot::Headers(_headers)) => {
                        if !claimed {
                            anyhow::ensure!(
                                service.try_claim_response(stream_id, worker_id).await,
                                "response already claimed for stream {stream_id}"
                            );
                            claimed = true;
                            self.write_started_response(&service, stream_id).await?;
                        }
                    }
                    Some(TailSlot::Body(data)) => {
                        self.feed_decoder(&mut decoder, data, &mut chunker).await?;
                        self.process_ready_windows(
                            &service,
                            stream_id,
                            &mut chunker,
                            &mut committer,
                        )
                        .await?;
                    }
                    Some(TailSlot::End) => {
                        self.finish_decoder(&mut decoder, &mut chunker).await?;
                        self.process_ready_windows(
                            &service,
                            stream_id,
                            &mut chunker,
                            &mut committer,
                        )
                        .await?;
                        if let Some(window) = chunker.take_final_window() {
                            self.transcribe_window(
                                &service,
                                stream_id,
                                &window.samples,
                                window.start_sample,
                                true,
                                chunker.stride_samples(),
                                &mut committer,
                            )
                            .await?;
                        }
                        let done = TranscriptEvent::Done {
                            segment_count: committer.emitted_count(),
                        };
                        map_upload_response(
                            service
                                .append_response_body(stream_id, encode_event(&done)?)
                                .await,
                            "failed to append done event",
                        )?;
                        map_upload_response(
                            service.end_response(stream_id).await,
                            "failed to end response",
                        )?;
                        service.unregister_reader(stream_id, worker_id).await;
                        if claimed {
                            service.release_response(stream_id, worker_id).await;
                        }
                        return Ok(());
                    }
                    None => {}
                }
            }

            last_slot = current_last;
        }
    }

    async fn send_error_response(
        &self,
        service: Arc<UploadResponseService>,
        stream_id: u64,
        worker_id: &str,
        message: String,
    ) -> Result<()> {
        let already_claimed = service.is_response_claimed_by(stream_id, worker_id).await;
        if !already_claimed && !service.try_claim_response(stream_id, worker_id).await {
            return Ok(());
        }

        if service.get_response_headers(stream_id).await.is_none() {
            let headers = StreamHeaders::Response(StreamResponseHeaders {
                stream_id,
                version: http_pack::HttpVersion::Http11,
                status: 500,
                headers: response_headers(stream_id),
            });
            map_upload_response(
                service.write_response_headers(stream_id, headers).await,
                "failed to write error response headers",
            )?;
        }

        let event = TranscriptEvent::Error { message };
        map_upload_response(
            service
                .append_response_body(stream_id, encode_event(&event)?)
                .await,
            "failed to append error event",
        )?;
        map_upload_response(
            service.end_response(stream_id).await,
            "failed to end error response",
        )?;
        service.unregister_reader(stream_id, worker_id).await;
        service.release_response(stream_id, worker_id).await;
        Ok(())
    }

    async fn write_started_response(
        &self,
        service: &UploadResponseService,
        stream_id: u64,
    ) -> Result<()> {
        let headers = StreamHeaders::Response(StreamResponseHeaders {
            stream_id,
            version: http_pack::HttpVersion::Http11,
            status: 200,
            headers: response_headers(stream_id),
        });
        map_upload_response(
            service.write_response_headers(stream_id, headers).await,
            "failed to write response headers",
        )?;

        let started = TranscriptEvent::Started {
            stream_id,
            chunk_ms: (self.config.chunk_seconds * 1000.0).round() as u64,
            overlap_ms: (self.config.overlap_seconds * 1000.0).round() as u64,
        };
        map_upload_response(
            service
                .append_response_body(stream_id, encode_event(&started)?)
                .await,
            "failed to append started event",
        )?;
        Ok(())
    }

    fn new_decoder() -> DecodePipelineHandle {
        DecodePipeline::spawn_with_options(DecodeOptions {
            output_bits_per_sample: Some(16),
            output_sample_rate: Some(16_000),
            output_channels: Some(1),
        })
    }

    async fn feed_decoder(
        &self,
        decoder: &mut DecodePipelineHandle,
        data: Bytes,
        chunker: &mut AudioChunker,
    ) -> Result<()> {
        loop {
            match decoder.send(data.clone()) {
                Ok(()) => break,
                Err(soundkit_decoder::DecodeError::InputBufferFull) => {
                    tokio::task::yield_now().await;
                }
                Err(error) => return Err(anyhow::anyhow!("decoder send failed: {error}")),
            }
        }

        self.drain_decoder(decoder, chunker)
    }

    async fn finish_decoder(
        &self,
        decoder: &mut DecodePipelineHandle,
        chunker: &mut AudioChunker,
    ) -> Result<()> {
        loop {
            match decoder.send(Bytes::new()) {
                Ok(()) => break,
                Err(soundkit_decoder::DecodeError::InputBufferFull) => {
                    tokio::task::yield_now().await;
                }
                Err(error) => return Err(anyhow::anyhow!("decoder EOF send failed: {error}")),
            }
        }
        while let Some(output) = decoder.recv() {
            let audio = output.map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;
            let samples = audio_to_mono_f32(&audio)?;
            chunker.push(&samples);
        }
        Ok(())
    }

    fn drain_decoder(
        &self,
        decoder: &mut DecodePipelineHandle,
        chunker: &mut AudioChunker,
    ) -> Result<()> {
        while let Some(output) = decoder.try_recv() {
            let audio = output.map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;
            let samples = audio_to_mono_f32(&audio)?;
            chunker.push(&samples);
        }
        Ok(())
    }

    async fn process_ready_windows(
        &self,
        service: &UploadResponseService,
        stream_id: u64,
        chunker: &mut AudioChunker,
        committer: &mut SegmentCommitter,
    ) -> Result<()> {
        for window in chunker.take_ready_windows() {
            self.transcribe_window(
                service,
                stream_id,
                &window.samples,
                window.start_sample,
                window.is_final,
                chunker.stride_samples(),
                committer,
            )
            .await?;
        }
        Ok(())
    }

    async fn transcribe_window(
        &self,
        service: &UploadResponseService,
        stream_id: u64,
        samples: &[f32],
        start_sample: usize,
        is_final: bool,
        stable_samples: usize,
        committer: &mut SegmentCommitter,
    ) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }

        debug!(
            stream_id,
            sample_count = samples.len(),
            start_sample,
            is_final,
            "running Parakeet window"
        );

        let result = self.models.transcribe_sentences(samples.to_vec()).await?;
        let segments: Vec<TimedSegment> = result
            .tokens
            .into_iter()
            .map(|token| TimedSegment {
                text: token.text,
                start_secs: token.start as f64,
                end_secs: token.end as f64,
            })
            .collect();

        for segment in committer.commit(start_sample, stable_samples, is_final, &segments) {
            let event = TranscriptEvent::Segment {
                index: segment.index,
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
                text: segment.text,
                final_segment: true,
            };
            map_upload_response(
                service
                    .append_response_body(stream_id, encode_event(&event)?)
                    .await,
                "failed to append transcript segment",
            )?;
        }

        Ok(())
    }
}

fn response_headers(stream_id: u64) -> Vec<HeaderField> {
    vec![
        HeaderField {
            name: b"content-type".to_vec(),
            value: b"application/x-ndjson".to_vec(),
        },
        HeaderField {
            name: b"cache-control".to_vec(),
            value: b"no-store".to_vec(),
        },
        HeaderField {
            name: b"x-stream-id".to_vec(),
            value: stream_id.to_string().into_bytes(),
        },
    ]
}

fn audio_to_mono_f32(audio: &AudioData) -> Result<Vec<f32>> {
    let channels: Vec<Vec<f32>> =
        match deserialize_audio(audio.data(), audio.bits_per_sample(), audio.channel_count())
            .map_err(|error| anyhow::anyhow!("failed to deserialize PCM: {error}"))?
        {
            PcmData::I16(channels) => channels.into_iter().map(vec_i16_to_f32).collect(),
            PcmData::I32(channels) => channels.into_iter().map(vec_i32_to_f32).collect(),
            PcmData::F32(channels) => channels,
        };

    if channels.is_empty() {
        return Ok(Vec::new());
    }
    if channels.len() == 1 {
        return Ok(channels.into_iter().next().unwrap_or_default());
    }

    let len = channels[0].len();
    let mut mono = vec![0.0f32; len];
    for channel in &channels {
        for (index, sample) in channel.iter().enumerate().take(len) {
            mono[index] += *sample;
        }
    }
    let scale = 1.0 / channels.len() as f32;
    for sample in &mut mono {
        *sample *= scale;
    }
    Ok(mono)
}

fn map_upload_response(result: Result<(), String>, context: &str) -> Result<()> {
    result.map_err(|error| anyhow::anyhow!("{context}: {error}"))
}
