use crate::config::{AppConfig, ASR_SAMPLE_RATE};
use crate::deepgram::ListenOptions;
use crate::ids::{next_request_id, request_id_from_headers};
use crate::processing::{
    encode_processing_head, encode_samples_f32le, ProcessingHead, DECODE_STAGE,
};
use anyhow::Result;
use av_api::linear16::Linear16PcmStream;
use bytes::Bytes;
use soundkit::audio_pipeline::audio_to_mono_f32;
use soundkit_decoder::{DecodeOptions, DecodePipeline, DecodePipelineHandle};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::task::JoinSet;
use tokio::time::{interval, Duration};
use tracing::{debug, error, info, info_span, warn, Instrument};
use upload_response::{
    discover_ingress_origins, request_from_stream_headers, RemoteIngressClient, RemoteRequestSlot,
    RequestControl, TailSlot, UploadResponseService, WorkerHeartbeatUpdate,
};

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

    fn worker_heartbeat(&self, inflight: usize) -> WorkerHeartbeatUpdate {
        let max_inflight = self.config.upload_response_max_inflight;
        let inflight = inflight.min(max_inflight);
        WorkerHeartbeatUpdate {
            stage: "decode".to_string(),
            max_inflight,
            inflight,
            available_slots: max_inflight.saturating_sub(inflight),
        }
    }

    async fn publish_local_worker_capacity(
        &self,
        service: &UploadResponseService,
        inflight: usize,
    ) {
        service
            .upsert_worker_heartbeat(
                &self.config.upload_response_worker_id,
                self.worker_heartbeat(inflight),
            )
            .await;
    }

    async fn publish_remote_worker_capacity(
        &self,
        client: &RemoteIngressClient,
        origins: &[String],
        inflight: usize,
    ) {
        let heartbeat = self.worker_heartbeat(inflight);
        for origin in origins {
            if let Err(error) = client
                .heartbeat_worker(origin, &self.config.upload_response_worker_id, &heartbeat)
                .await
            {
                warn!(
                    origin,
                    worker_id = %self.config.upload_response_worker_id,
                    error = %error,
                    "failed to publish remote decoder capacity"
                );
            }
        }
    }

    async fn run_cache_worker(self: Arc<Self>, service: Arc<UploadResponseService>) {
        info!(
            worker_id = %self.config.upload_response_worker_id,
            max_inflight = self.config.upload_response_max_inflight,
            poll_ms = self.config.upload_response_worker_poll_ms,
            heartbeat_ms = self.config.upload_response_worker_heartbeat_interval_ms,
            "local decode worker started"
        );
        let mut poll = interval(Duration::from_millis(
            self.config.upload_response_worker_poll_ms.max(1),
        ));
        let mut heartbeat = interval(Duration::from_millis(
            self.config
                .upload_response_worker_heartbeat_interval_ms
                .max(1),
        ));
        let mut inflight = HashSet::new();
        let mut tasks = JoinSet::new();
        let mut send_heartbeat = true;

        loop {
            tokio::select! {
                _ = poll.tick() => {}
                _ = heartbeat.tick() => {
                    send_heartbeat = true;
                }
            }

            while let Some(joined) = tasks.try_join_next() {
                match joined {
                    Ok(stream_id) => {
                        inflight.remove(&stream_id);
                        send_heartbeat = true;
                    }
                    Err(error) => {
                        error!(%error, "decode worker task failed");
                    }
                }
            }

            if send_heartbeat {
                self.publish_local_worker_capacity(&service, inflight.len())
                    .await;
                send_heartbeat = false;
            }

            if inflight.len() >= self.config.upload_response_max_inflight {
                continue;
            }

            for stream in service.active_streams().await {
                if inflight.len() >= self.config.upload_response_max_inflight {
                    break;
                }
                if inflight.contains(&stream.stream_id)
                    || stream.request_last == 0
                    || stream.stage_last(DECODE_STAGE) != 0
                    || stream.stage_owner(DECODE_STAGE).is_some()
                {
                    continue;
                }

                if !service
                    .try_claim_stage(
                        stream.stream_id,
                        DECODE_STAGE,
                        &self.config.upload_response_worker_id,
                    )
                    .await
                {
                    continue;
                }

                let _ = service
                    .register_reader(stream.stream_id, &self.config.upload_response_worker_id)
                    .await;

                inflight.insert(stream.stream_id);
                send_heartbeat = true;
                let service = service.clone();
                let worker = self.clone();
                let worker_id = self.config.upload_response_worker_id.clone();
                tasks.spawn(async move {
                    let stream_id = stream.stream_id;
                    let result = worker
                        .process_cached_stream(service.clone(), stream_id)
                        .await;
                    if let Err(error) = result {
                        error!(stream_id, error = %error, "cached decode failed");
                    }
                    let _ = service
                        .release_stage(stream_id, DECODE_STAGE, &worker_id)
                        .await;
                    let _ = service.unregister_reader(stream_id, &worker_id).await;
                    stream_id
                });
            }
        }
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

        let mut poll = interval(Duration::from_millis(
            self.config.upload_response_worker_poll_ms.max(1),
        ));
        let mut discovery = interval(Duration::from_millis(
            self.config.upload_response_discovery_interval_ms.max(1),
        ));
        let mut heartbeat = interval(Duration::from_millis(
            self.config
                .upload_response_worker_heartbeat_interval_ms
                .max(1),
        ));
        let mut inflight = HashSet::new();
        let mut tasks = JoinSet::new();
        let mut origins: Vec<String> = Vec::new();
        let mut refresh_origins = true;
        let mut send_heartbeat = true;

        info!(
            worker_id = %self.config.upload_response_worker_id,
            max_inflight = self.config.upload_response_max_inflight,
            poll_ms = self.config.upload_response_worker_poll_ms,
            discovery_ms = self.config.upload_response_discovery_interval_ms,
            heartbeat_ms = self.config.upload_response_worker_heartbeat_interval_ms,
            "remote decode worker started"
        );

        loop {
            tokio::select! {
                _ = poll.tick() => {}
                _ = discovery.tick() => {
                    refresh_origins = true;
                }
                _ = heartbeat.tick() => {
                    send_heartbeat = true;
                }
            }

            if refresh_origins {
                match discover_ingress_origins(
                    &self.config.upload_response_ingress_urls,
                    self.config.upload_response_discovery_dns.as_deref(),
                )
                .await
                {
                    Ok(next) => {
                        if next != origins {
                            debug!(origins = ?next, "updated ingress origins");
                        }
                        origins = next;
                    }
                    Err(error) => {
                        warn!(error = %error, "failed to discover ingress origins");
                    }
                }
                refresh_origins = false;
                send_heartbeat = true;
            }

            while let Some(joined) = tasks.try_join_next() {
                match joined {
                    Ok(key) => {
                        inflight.remove(&key);
                        send_heartbeat = true;
                    }
                    Err(error) => {
                        error!(%error, "remote decode worker task failed");
                    }
                }
            }

            if send_heartbeat {
                self.publish_remote_worker_capacity(&client, &origins, inflight.len())
                    .await;
                send_heartbeat = false;
            }

            if inflight.len() >= self.config.upload_response_max_inflight {
                continue;
            }

            for origin in &origins {
                if inflight.len() >= self.config.upload_response_max_inflight {
                    break;
                }

                let streams = match client.list_streams(origin).await {
                    Ok(streams) => streams,
                    Err(error) => {
                        warn!(origin, error = %error, "failed to list remote streams");
                        continue;
                    }
                };

                for stream in streams {
                    if inflight.len() >= self.config.upload_response_max_inflight {
                        break;
                    }
                    if stream.request_last == 0
                        || stream.stage_last(DECODE_STAGE) != 0
                        || stream.stage_owner(DECODE_STAGE).is_some()
                    {
                        continue;
                    }

                    let inflight_key = format!("{}#{}", origin, stream.stream_id);
                    if inflight.contains(&inflight_key) {
                        continue;
                    }

                    match client
                        .try_claim_stage(
                            origin,
                            stream.stream_id,
                            DECODE_STAGE,
                            &self.config.upload_response_worker_id,
                        )
                        .await
                    {
                        Ok(true) => {}
                        Ok(false) => continue,
                        Err(error) => {
                            warn!(
                                origin,
                                stream_id = stream.stream_id,
                                error = %error,
                                "failed to claim remote processing stream"
                            );
                            continue;
                        }
                    }

                    let _ = client
                        .register_reader(
                            origin,
                            stream.stream_id,
                            &self.config.upload_response_worker_id,
                        )
                        .await;

                    inflight.insert(inflight_key.clone());
                    send_heartbeat = true;
                    let worker = self.clone();
                    let client = client.clone();
                    let worker_id = self.config.upload_response_worker_id.clone();
                    let origin = origin.clone();
                    tasks.spawn(async move {
                        let result = worker
                            .process_remote_stream(&client, &origin, stream.stream_id)
                            .await;
                        if let Err(error) = result {
                            error!(
                                origin,
                                stream_id = stream.stream_id,
                                error = %error,
                                "remote cached decode failed"
                            );
                        }
                        let _ = client
                            .release_stage(&origin, stream.stream_id, DECODE_STAGE, &worker_id)
                            .await;

                        let _ = client
                            .unregister_reader(&origin, stream.stream_id, &worker_id)
                            .await;
                        inflight_key
                    });
                }
            }
        }
    }

    async fn process_cached_stream(
        &self,
        service: Arc<UploadResponseService>,
        stream_id: u64,
    ) -> Result<()> {
        let request = self
            .read_cached_request_headers(&service, stream_id)
            .await?;
        let request_id = request_id_from_headers(request.headers()).unwrap_or_else(next_request_id);
        let span = info_span!(
            "decoder_cached_stream",
            request_id,
            stream_id,
            worker_id = %self.config.upload_response_worker_id,
            source = "local_cache",
        );

        async move {
            let mut sink = LocalProcessingSink::new(
                service.clone(),
                stream_id,
                self.config.upload_response_config().slot_bytes(),
            );
            self.decode_cached_request(&service, stream_id, &request, &mut sink)
                .await
        }
        .instrument(span)
        .await
    }

    async fn process_remote_stream(
        &self,
        client: &RemoteIngressClient,
        origin: &str,
        stream_id: u64,
    ) -> Result<()> {
        let request = client
            .request_headers(origin, stream_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("request headers were missing"))?;
        let request_id = request_id_from_headers(request.headers()).unwrap_or_else(next_request_id);
        let span = info_span!(
            "decoder_remote_stream",
            request_id,
            stream_id,
            worker_id = %self.config.upload_response_worker_id,
            origin = %origin,
            source = "remote_cache",
        );

        async move {
            let mut sink = RemoteProcessingSink::new(client.clone(), origin.to_string(), stream_id);
            self.decode_remote_request(client, origin, stream_id, &request, &mut sink)
                .await
        }
        .instrument(span)
        .await
    }

    async fn read_cached_request_headers(
        &self,
        service: &UploadResponseService,
        stream_id: u64,
    ) -> Result<http::Request<()>> {
        match service.tail_request(stream_id, 1).await {
            Some(TailSlot::Headers(headers)) => request_from_stream_headers(headers),
            Some(_) | None => Err(anyhow::anyhow!("request headers were missing")),
        }
    }

    async fn decode_cached_request<S: ProcessingSink + Send>(
        &self,
        service: &UploadResponseService,
        stream_id: u64,
        request: &http::Request<()>,
        sink: &mut S,
    ) -> Result<()> {
        let mut poll = interval(Duration::from_millis(
            self.config.upload_response_worker_poll_ms.max(1),
        ));
        let mut last_slot = 0usize;
        let mut received_bytes = 0usize;
        let mut decoder = RequestDecoder::from_request(request, &self.config)?;
        sink.write_head(processing_head_bytes()?).await?;
        info!("started cached decode stream");

        'stream: loop {
            poll.tick().await;
            let current_last = service
                .request_last(stream_id)
                .ok_or_else(|| anyhow::anyhow!("request stream {stream_id} disappeared"))?;

            if current_last <= last_slot {
                continue;
            }

            for slot_id in (last_slot + 1)..=current_last {
                match service.tail_request(stream_id, slot_id).await {
                    Some(TailSlot::Headers(_)) => {}
                    Some(TailSlot::Body(chunk)) => {
                        received_bytes += chunk.len();
                        self.decode_request_chunk(&mut decoder, chunk, sink).await?;
                    }
                    Some(TailSlot::Control(control)) => {
                        sink.append_control(control).await?;
                    }
                    Some(TailSlot::End) => break 'stream,
                    None => {}
                }
            }

            last_slot = current_last;
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
        client: &RemoteIngressClient,
        origin: &str,
        stream_id: u64,
        request: &http::Request<()>,
        sink: &mut S,
    ) -> Result<()> {
        let mut poll = interval(Duration::from_millis(
            self.config.upload_response_worker_poll_ms.max(1),
        ));
        let mut last_slot = 0usize;
        let mut received_bytes = 0usize;
        let mut decoder = RequestDecoder::from_request(request, &self.config)?;
        sink.write_head(processing_head_bytes()?).await?;
        info!(origin, "started remote decode stream");

        'stream: loop {
            poll.tick().await;
            let current_last = client.request_last(origin, stream_id).await?;
            if current_last <= last_slot {
                continue;
            }

            for slot_id in (last_slot + 1)..=current_last {
                match client.request_slot(origin, stream_id, slot_id).await? {
                    Some(RemoteRequestSlot::Headers(_)) => {}
                    Some(RemoteRequestSlot::Body(chunk)) => {
                        received_bytes += chunk.len();
                        self.decode_request_chunk(&mut decoder, chunk, sink).await?;
                    }
                    Some(RemoteRequestSlot::Control(control)) => {
                        sink.append_control(control).await?;
                    }
                    Some(RemoteRequestSlot::End) => break 'stream,
                    None => {}
                }
            }

            last_slot = current_last;
        }

        anyhow::ensure!(
            received_bytes > 0,
            "request body did not include audio bytes"
        );

        self.finish_request_decoder(&mut decoder, sink).await?;
        sink.finish().await?;
        info!(origin, received_bytes, "finished remote decode stream");
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
                self.send_decoder_bytes(handle, chunk).await?;
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
                self.send_decoder_bytes(handle, Bytes::new()).await?;
                self.flush_decoder(handle, sink, true).await
            }
        }
    }

    async fn send_decoder_bytes(
        &self,
        decoder: &mut DecodePipelineHandle,
        data: Bytes,
    ) -> Result<()> {
        loop {
            match decoder.send(data.clone()) {
                Ok(()) => return Ok(()),
                Err(soundkit_decoder::DecodeError::InputBufferFull) => {
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
    service: Arc<UploadResponseService>,
    stream_id: u64,
    slot_bytes: usize,
}

impl LocalProcessingSink {
    fn new(service: Arc<UploadResponseService>, stream_id: u64, slot_bytes: usize) -> Self {
        Self {
            service,
            stream_id,
            slot_bytes: slot_bytes.max(1),
        }
    }
}

#[async_trait::async_trait]
impl ProcessingSink for LocalProcessingSink {
    async fn write_head(&mut self, head: Bytes) -> Result<()> {
        self.service
            .write_stage_head(self.stream_id, DECODE_STAGE, head)
            .await
            .map_err(anyhow::Error::msg)
    }

    async fn append_body(&mut self, body: Bytes) -> Result<()> {
        for chunk in body.chunks(self.slot_bytes) {
            if chunk.is_empty() {
                continue;
            }
            self.service
                .append_stage_body(self.stream_id, DECODE_STAGE, Bytes::copy_from_slice(chunk))
                .await
                .map_err(anyhow::Error::msg)?;
        }
        Ok(())
    }

    async fn append_control(&mut self, control: RequestControl) -> Result<()> {
        self.service
            .append_stage_control(self.stream_id, DECODE_STAGE, control)
            .await
            .map_err(anyhow::Error::msg)
    }

    async fn finish(&mut self) -> Result<()> {
        self.service
            .end_stage(self.stream_id, DECODE_STAGE)
            .await
            .map_err(anyhow::Error::msg)
    }
}

struct RemoteProcessingSink {
    client: RemoteIngressClient,
    origin: String,
    stream_id: u64,
}

impl RemoteProcessingSink {
    fn new(client: RemoteIngressClient, origin: String, stream_id: u64) -> Self {
        Self {
            client,
            origin,
            stream_id,
        }
    }
}

#[async_trait::async_trait]
impl ProcessingSink for RemoteProcessingSink {
    async fn write_head(&mut self, head: Bytes) -> Result<()> {
        self.client
            .write_stage_head(&self.origin, self.stream_id, DECODE_STAGE, head)
            .await
    }

    async fn append_body(&mut self, body: Bytes) -> Result<()> {
        self.client
            .append_stage_body(&self.origin, self.stream_id, DECODE_STAGE, body)
            .await
    }

    async fn append_control(&mut self, control: RequestControl) -> Result<()> {
        self.client
            .append_stage_control(&self.origin, self.stream_id, DECODE_STAGE, control)
            .await
    }

    async fn finish(&mut self) -> Result<()> {
        self.client
            .end_stage(&self.origin, self.stream_id, DECODE_STAGE)
            .await
    }
}

fn processing_head_bytes() -> Result<Bytes> {
    encode_processing_head(&ProcessingHead::pcm_mono_f32())
}
