use crate::config::{AppConfig, ASR_SAMPLE_RATE};
use crate::deepgram::ListenOptions;
use crate::ids::{next_request_id, request_id_from_headers};
use crate::processing::{
    encode_processing_head, encode_samples_f32le, ProcessingHead, DECODE_STAGE,
};
use anyhow::Result;
use av_api::linear16::Linear16PcmStream;
use bytes::Bytes;
use gpu_worker::upload_response::{
    run_local_worker_loop, run_remote_worker_loop, LocalJob, LocalJobProcessor, LocalWorkerConfig,
    PipelineSpec, RemoteJob, RemoteJobProcessor, RemoteWorkerConfig, SinkLane, SourceFrame,
    SourceLane,
};
use soundkit::audio_pipeline::audio_to_mono_f32;
use soundkit_decoder::{DecodeOptions, DecodePipeline, DecodePipelineHandle};
use std::sync::Arc;
use tokio::time::Duration;
use tracing::{error, info, info_span, Instrument};
use upload_response::{RemoteIngressClient, RequestControl, UploadResponseService};

#[derive(Clone)]
pub struct DecoderState {
    config: AppConfig,
}

impl DecoderState {
    pub fn new(config: AppConfig) -> Self {
        Self { config }
    }

    pub fn spawn_cache_worker(
        self: Arc<Self>,
        service: Arc<UploadResponseService>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            self.run_cache_worker(service).await;
        })
    }

    pub fn spawn_remote_cache_worker(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            self.run_remote_cache_worker().await;
        })
    }

    async fn run_cache_worker(self: Arc<Self>, service: Arc<UploadResponseService>) {
        run_local_worker_loop(service, self.local_worker_config(), self).await;
    }

    fn local_worker_config(&self) -> LocalWorkerConfig {
        let mut config = LocalWorkerConfig::new(
            self.config.upload_response_worker_id.clone(),
            PipelineSpec {
                source: SourceLane::Request,
                sink: SinkLane::Stage(DECODE_STAGE.to_string()),
            },
        );
        config.heartbeat_stage = "decode".to_string();
        config.max_inflight = self.config.upload_response_max_inflight;
        config.poll_interval =
            Duration::from_millis(self.config.upload_response_worker_poll_ms.max(1));
        config.heartbeat_interval = Duration::from_millis(
            self.config
                .upload_response_worker_heartbeat_interval_ms
                .max(1),
        );
        config
    }

    fn remote_worker_config(&self) -> RemoteWorkerConfig {
        let mut config = RemoteWorkerConfig::new(
            self.config.upload_response_worker_id.clone(),
            PipelineSpec {
                source: SourceLane::Request,
                sink: SinkLane::Stage(DECODE_STAGE.to_string()),
            },
        );
        config.heartbeat_stage = "decode".to_string();
        config.max_inflight = self.config.upload_response_max_inflight;
        config.poll_interval =
            Duration::from_millis(self.config.upload_response_worker_poll_ms.max(1));
        config.discovery_interval =
            Duration::from_millis(self.config.upload_response_discovery_interval_ms.max(1));
        config.heartbeat_interval = Duration::from_millis(
            self.config
                .upload_response_worker_heartbeat_interval_ms
                .max(1),
        );
        config.ingress_urls = self.config.upload_response_ingress_urls.clone();
        config.discovery_dns = self.config.upload_response_discovery_dns.clone();
        config
    }

    async fn run_remote_cache_worker(self: Arc<Self>) {
        let client = match RemoteIngressClient::new(
            self.config.upload_response_config().slot_bytes(),
            self.config.upload_response_insecure_tls,
        ) {
            Ok(client) => client,
            Err(error) => {
                error!(error = %error, "failed to build remote ingress client");
                return;
            }
        };
        run_remote_worker_loop(client, self.remote_worker_config(), self).await;
    }

    async fn process_local_job(&self, job: LocalJob) -> Result<()> {
        let request = job
            .request()
            .await?
            .ok_or_else(|| anyhow::anyhow!("request headers were missing"))?;
        let request_id = request_id_from_headers(request.headers()).unwrap_or_else(next_request_id);
        let span = info_span!(
            "decoder_cached_stream",
            request_id,
            stream_id = job.stream_id,
            worker_id = %job.worker_id(),
            source = "local_cache",
        );

        async move {
            let mut sink = LocalProcessingSink::new(job.clone());
            self.decode_cached_request(&job, &request, &mut sink).await
        }
        .instrument(span)
        .await
    }

    async fn process_remote_job(&self, job: RemoteJob) -> Result<()> {
        let request = job
            .request()
            .await?
            .ok_or_else(|| anyhow::anyhow!("request headers were missing"))?;
        let request_id = request_id_from_headers(request.headers()).unwrap_or_else(next_request_id);
        let span = info_span!(
            "decoder_remote_stream",
            request_id,
            stream_id = job.stream_id,
            worker_id = %job.worker_id(),
            origin = %job.origin,
            source = "remote_cache",
        );

        async move {
            let mut sink = RemoteProcessingSink::new(job.clone());
            self.decode_remote_request(&job, &request, &mut sink).await
        }
        .instrument(span)
        .await
    }
    async fn decode_cached_request<S: ProcessingSink + Send>(
        &self,
        job: &LocalJob,
        request: &http::Request<()>,
        sink: &mut S,
    ) -> Result<()> {
        let mut reader = job.source_reader_from(
            1,
            Duration::from_millis(self.config.upload_response_worker_poll_ms.max(1)),
        );
        let mut received_bytes = 0usize;
        let mut decoder = RequestDecoder::from_request(request, &self.config)?;
        sink.write_head(processing_head_bytes()?).await?;
        info!("started cached decode stream");

        while let Some(frame) = reader.next_frame().await? {
            match frame {
                SourceFrame::RequestHeaders(_) | SourceFrame::StageHead(_) => {}
                SourceFrame::Body(chunk) => {
                    received_bytes += chunk.len();
                    self.decode_request_chunk(&mut decoder, chunk, sink).await?;
                }
                SourceFrame::Control(control) => {
                    sink.append_control(control).await?;
                }
                SourceFrame::End => break,
            }
        }

        anyhow::ensure!(
            received_bytes > 0,
            "request body did not include audio bytes"
        );

        self.finish_request_decoder(&mut decoder, sink).await?;
        sink.finish().await?;
        info!(received_bytes, "finished cached decode stream");
        Ok(())
    }

    async fn decode_remote_request<S: ProcessingSink + Send>(
        &self,
        job: &RemoteJob,
        request: &http::Request<()>,
        sink: &mut S,
    ) -> Result<()> {
        let mut reader = job.source_reader_from(
            1,
            Duration::from_millis(self.config.upload_response_worker_poll_ms.max(1)),
        );
        let mut received_bytes = 0usize;
        let mut decoder = RequestDecoder::from_request(request, &self.config)?;
        sink.write_head(processing_head_bytes()?).await?;
        info!(origin = %job.origin, "started remote decode stream");

        while let Some(frame) = reader.next_frame().await? {
            match frame {
                SourceFrame::RequestHeaders(_) | SourceFrame::StageHead(_) => {}
                SourceFrame::Body(chunk) => {
                    received_bytes += chunk.len();
                    self.decode_request_chunk(&mut decoder, chunk, sink).await?;
                }
                SourceFrame::Control(control) => {
                    sink.append_control(control).await?;
                }
                SourceFrame::End => break,
            }
        }

        anyhow::ensure!(
            received_bytes > 0,
            "request body did not include audio bytes"
        );

        self.finish_request_decoder(&mut decoder, sink).await?;
        sink.finish().await?;
        info!(origin = %job.origin, received_bytes, "finished remote decode stream");
        Ok(())
    }

    async fn decode_request_chunk<S: ProcessingSink + Send>(
        &self,
        decoder: &mut RequestDecoder,
        chunk: Bytes,
        sink: &mut S,
    ) -> Result<()> {
        match decoder {
            RequestDecoder::Linear16(stream) => {
                let samples = stream
                    .push(&chunk)
                    .map_err(|error| anyhow::anyhow!("raw linear16 decode failed: {error}"))?;
                self.append_processing_samples(sink, &samples).await
            }
            RequestDecoder::Compressed(handle) => {
                self.send_decoder_bytes(handle, chunk, sink).await?;
                self.flush_decoder(handle, sink, false).await
            }
        }
    }

    async fn finish_request_decoder<S: ProcessingSink + Send>(
        &self,
        decoder: &mut RequestDecoder,
        sink: &mut S,
    ) -> Result<()> {
        match decoder {
            RequestDecoder::Linear16(stream) => {
                let tail = stream
                    .finish()
                    .map_err(|error| anyhow::anyhow!("raw linear16 decode failed: {error}"))?;
                self.append_processing_samples(sink, &tail).await
            }
            RequestDecoder::Compressed(handle) => {
                self.send_decoder_bytes(handle, Bytes::new(), sink).await?;
                self.flush_decoder(handle, sink, true).await
            }
        }
    }

    async fn send_decoder_bytes<S: ProcessingSink + Send>(
        &self,
        decoder: &mut DecodePipelineHandle,
        data: Bytes,
        sink: &mut S,
    ) -> Result<()> {
        loop {
            match decoder.send(data.clone()) {
                Ok(()) => return Ok(()),
                Err(soundkit_decoder::DecodeError::InputBufferFull) => {
                    self.flush_decoder(decoder, sink, false).await?;
                    tokio::task::yield_now().await;
                }
                Err(error) => {
                    return Err(anyhow::anyhow!("decoder send failed: {error}"));
                }
            }
        }
    }

    async fn flush_decoder<S: ProcessingSink + Send>(
        &self,
        decoder: &mut DecodePipelineHandle,
        sink: &mut S,
        blocking: bool,
    ) -> Result<()> {
        if blocking {
            while let Some(output) = decoder.recv() {
                let audio = output.map_err(|error| anyhow::anyhow!("decode failed: {error}"))?;
                let samples = audio_to_mono_f32(&audio).map_err(anyhow::Error::msg)?;
                self.append_processing_samples(sink, &samples).await?;
            }
            return Ok(());
        }

        while let Some(output) = decoder.try_recv() {
            let audio = output.map_err(|error| anyhow::anyhow!("decode failed: {error}"))?;
            let samples = audio_to_mono_f32(&audio).map_err(anyhow::Error::msg)?;
            self.append_processing_samples(sink, &samples).await?;
        }
        Ok(())
    }

    async fn append_processing_samples<S: ProcessingSink + Send>(
        &self,
        sink: &mut S,
        samples: &[f32],
    ) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }
        sink.append_body(encode_samples_f32le(samples)).await
    }
}

enum RequestDecoder {
    Linear16(Linear16PcmStream),
    Compressed(DecodePipelineHandle),
}

impl RequestDecoder {
    fn from_request(request: &http::Request<()>, config: &AppConfig) -> Result<Self> {
        let options = ListenOptions::from_request(request, config);
        if options.raw_linear16() {
            let sample_rate = options
                .sample_rate_hz
                .ok_or_else(|| anyhow::anyhow!("sample_rate is required for encoding=linear16"))?;
            let channels = options.channels.max(1);
            Ok(Self::Linear16(
                Linear16PcmStream::new(sample_rate, ASR_SAMPLE_RATE, channels)
                    .map_err(|error| anyhow::anyhow!("{error}"))?,
            ))
        } else {
            Ok(Self::Compressed(DecodePipeline::spawn_with_options(
                DecodeOptions {
                    output_bits_per_sample: Some(16),
                    output_sample_rate: Some(ASR_SAMPLE_RATE),
                    output_channels: Some(1),
                },
            )))
        }
    }
}

#[async_trait::async_trait]
trait ProcessingSink {
    async fn write_head(&mut self, head: Bytes) -> Result<()>;
    async fn append_body(&mut self, body: Bytes) -> Result<()>;
    async fn append_control(&mut self, control: RequestControl) -> Result<()>;
    async fn finish(&mut self) -> Result<()>;
}

struct LocalProcessingSink {
    job: LocalJob,
    slot_bytes: usize,
}

impl LocalProcessingSink {
    fn new(job: LocalJob) -> Self {
        Self {
            slot_bytes: job.slot_bytes().max(1),
            job,
        }
    }
}

#[async_trait::async_trait]
impl ProcessingSink for LocalProcessingSink {
    async fn write_head(&mut self, head: Bytes) -> Result<()> {
        self.job
            .write_stage_head(head)
            .await
            .map_err(anyhow::Error::msg)
    }

    async fn append_body(&mut self, body: Bytes) -> Result<()> {
        for chunk in body.chunks(self.slot_bytes) {
            if chunk.is_empty() {
                continue;
            }
            self.job
                .append_body(Bytes::copy_from_slice(chunk))
                .await
                .map_err(anyhow::Error::msg)?;
        }
        Ok(())
    }

    async fn append_control(&mut self, control: RequestControl) -> Result<()> {
        self.job
            .append_control(control)
            .await
            .map_err(anyhow::Error::msg)
    }

    async fn finish(&mut self) -> Result<()> {
        self.job.end().await.map_err(anyhow::Error::msg)
    }
}

struct RemoteProcessingSink {
    job: RemoteJob,
}

impl RemoteProcessingSink {
    fn new(job: RemoteJob) -> Self {
        Self { job }
    }
}

#[async_trait::async_trait]
impl ProcessingSink for RemoteProcessingSink {
    async fn write_head(&mut self, head: Bytes) -> Result<()> {
        self.job.write_stage_head(head).await
    }

    async fn append_body(&mut self, body: Bytes) -> Result<()> {
        self.job.append_body(body).await
    }

    async fn append_control(&mut self, control: RequestControl) -> Result<()> {
        self.job.append_control(control).await
    }

    async fn finish(&mut self) -> Result<()> {
        self.job.end().await
    }
}

fn processing_head_bytes() -> Result<Bytes> {
    encode_processing_head(&ProcessingHead::pcm_mono_f32())
}

#[async_trait::async_trait]
impl LocalJobProcessor for DecoderState {
    async fn process(&self, job: LocalJob) -> Result<()> {
        self.process_local_job(job).await
    }
}

#[async_trait::async_trait]
impl RemoteJobProcessor for DecoderState {
    async fn process(&self, job: RemoteJob) -> Result<()> {
        self.process_remote_job(job).await
    }
}
