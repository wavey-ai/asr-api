use crate::asr::AsrBackend;
use crate::chunking::{AudioChunker, CommittedWord, WordCommitter};
use crate::config::{AppConfig, ASR_SAMPLE_RATE};
use crate::deepgram::{build_response, ListenOptions};
use anyhow::Result;
use bytes::Bytes;
use futures_util::StreamExt;
use http::{header::CONTENT_TYPE, HeaderName, HeaderValue, Request, StatusCode};
use http_pack::stream::{decode_frame, encode_frame, StreamFrame, StreamHeaders};
use reqwest::{Client, StatusCode as ReqwestStatusCode};
use serde::Serialize;
use sha2::{Digest, Sha256};
use soundkit::audio_pipeline::{deserialize_audio, vec_i16_to_f32, vec_i32_to_f32};
use soundkit::audio_types::{AudioData, PcmData};
use soundkit_decoder::{DecodeOptions, DecodePipeline, DecodePipelineHandle};
use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;
use tokio::net::lookup_host;
use tokio::task::JoinSet;
use tokio::time::{interval, Duration};
use tracing::{debug, error, warn};
use upload_response::{TailSlot, UploadResponseService};
use uuid::Uuid;
use web_service::{BodyStream, HandlerResponse};

#[derive(Clone)]
pub struct WorkerState {
    config: AppConfig,
    backend: Arc<AsrBackend>,
}

struct PreparedTranscript {
    committed_words: Vec<CommittedWord>,
    fallback_fragments: Vec<String>,
    duration_secs: f64,
    sha256: String,
}

impl WorkerState {
    pub fn new(config: AppConfig, backend: Arc<AsrBackend>) -> Self {
        Self { config, backend }
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

    pub async fn handle_listen(&self, req: Request<()>, body: BodyStream) -> HandlerResponse {
        match self.handle_listen_inner(req, body).await {
            Ok(response) => response,
            Err(error) => error_response(classify_error(&error), error.to_string()),
        }
    }

    async fn run_cache_worker(self: Arc<Self>, service: Arc<UploadResponseService>) {
        let mut poll = interval(Duration::from_millis(
            self.config.upload_response_worker_poll_ms.max(1),
        ));
        let mut inflight = HashSet::new();
        let mut tasks = JoinSet::new();

        loop {
            poll.tick().await;

            while let Some(joined) = tasks.try_join_next() {
                match joined {
                    Ok(stream_id) => {
                        inflight.remove(&stream_id);
                    }
                    Err(error) => {
                        error!(%error, "cache worker task failed");
                    }
                }
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
                    || stream.response_owner.is_some()
                {
                    continue;
                }

                if !service
                    .try_claim_response(stream.stream_id, &self.config.upload_response_worker_id)
                    .await
                {
                    continue;
                }

                let _ = service
                    .register_reader(stream.stream_id, &self.config.upload_response_worker_id)
                    .await;

                inflight.insert(stream.stream_id);
                let service = service.clone();
                let worker = self.clone();
                let worker_id = self.config.upload_response_worker_id.clone();
                tasks.spawn(async move {
                    let stream_id = stream.stream_id;
                    let result = worker.process_cached_stream(service.clone(), stream_id).await;
                    if let Err(error) = result {
                        error!(stream_id, error = %error, "cached transcription failed");
                        let response = error_response(classify_error(&error), error.to_string());
                        if let Err(write_error) = service
                            .write_handler_response(stream_id, response)
                            .await
                            .map_err(anyhow::Error::msg)
                        {
                            error!(
                                stream_id,
                                error = %write_error,
                                "failed to write cached error response"
                            );
                            let _ = service.release_response(stream_id, &worker_id).await;
                        }
                    }
                    stream_id
                });
            }
        }
    }

    async fn run_remote_cache_worker(self: Arc<Self>) {
        let client = match RemoteIngressClient::new(&self.config) {
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
        let mut inflight = HashSet::new();
        let mut tasks = JoinSet::new();
        let mut origins: Vec<String> = Vec::new();
        let mut refresh_origins = true;

        loop {
            tokio::select! {
                _ = poll.tick() => {}
                _ = discovery.tick() => {
                    refresh_origins = true;
                }
            }

            if refresh_origins {
                match discover_ingress_origins(&self.config).await {
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
            }

            while let Some(joined) = tasks.try_join_next() {
                match joined {
                    Ok(key) => {
                        inflight.remove(&key);
                    }
                    Err(error) => {
                        error!(%error, "remote cache worker task failed");
                    }
                }
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
                    if stream.request_last == 0 || stream.response_owner.is_some() {
                        continue;
                    }

                    let inflight_key = format!("{}#{}", origin, stream.stream_id);
                    if inflight.contains(&inflight_key) {
                        continue;
                    }

                    match client
                        .try_claim_response(
                            origin,
                            stream.stream_id,
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
                                "failed to claim remote response"
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
                                "remote cached transcription failed"
                            );
                            let response =
                                error_response(classify_error(&error), error.to_string());
                            if let Err(write_error) = client
                                .write_handler_response(&origin, stream.stream_id, response)
                                .await
                            {
                                error!(
                                    origin,
                                    stream_id = stream.stream_id,
                                    error = %write_error,
                                    "failed to write remote cached error response"
                                );
                                let _ = client
                                    .release_response(&origin, stream.stream_id, &worker_id)
                                    .await;
                            }
                        }

                        let _ = client
                            .unregister_reader(&origin, stream.stream_id, &worker_id)
                            .await;
                        inflight_key
                    });
                }
            }
        }
    }

    async fn handle_listen_inner(
        &self,
        req: Request<()>,
        body: BodyStream,
    ) -> Result<HandlerResponse> {
        reject_json_requests(&req)?;
        let options = ListenOptions::from_query(req.uri().query(), &self.config);
        let prepared = self.transcribe_upload(body).await?;
        let fallback_transcript = prepared.fallback_fragments.join(" ");
        let payload = build_response(
            Uuid::new_v4().to_string(),
            prepared.sha256,
            prepared.duration_secs,
            &prepared.committed_words,
            &fallback_transcript,
            &options,
        );
        json_response(StatusCode::OK, &payload)
    }

    async fn process_cached_stream(
        &self,
        service: Arc<UploadResponseService>,
        stream_id: u64,
    ) -> Result<()> {
        let (request, prepared) = self.transcribe_cached_upload(&service, stream_id).await?;
        let options = ListenOptions::from_query(request.uri().query(), &self.config);
        let fallback_transcript = prepared.fallback_fragments.join(" ");
        let payload = build_response(
            Uuid::new_v4().to_string(),
            prepared.sha256,
            prepared.duration_secs,
            &prepared.committed_words,
            &fallback_transcript,
            &options,
        );
        let response = json_response(StatusCode::OK, &payload)?;
        service
            .write_handler_response(stream_id, response)
            .await
            .map_err(anyhow::Error::msg)?;
        Ok(())
    }

    async fn process_remote_stream(
        &self,
        client: &RemoteIngressClient,
        origin: &str,
        stream_id: u64,
    ) -> Result<()> {
        let (request, prepared) = self.transcribe_remote_upload(client, origin, stream_id).await?;
        let options = ListenOptions::from_query(request.uri().query(), &self.config);
        let fallback_transcript = prepared.fallback_fragments.join(" ");
        let payload = build_response(
            Uuid::new_v4().to_string(),
            prepared.sha256,
            prepared.duration_secs,
            &prepared.committed_words,
            &fallback_transcript,
            &options,
        );
        let response = json_response(StatusCode::OK, &payload)?;
        client.write_handler_response(origin, stream_id, response).await?;
        Ok(())
    }

    async fn transcribe_cached_upload(
        &self,
        service: &UploadResponseService,
        stream_id: u64,
    ) -> Result<(Request<()>, PreparedTranscript)> {
        let mut poll = interval(Duration::from_millis(
            self.config.upload_response_worker_poll_ms.max(1),
        ));
        let mut chunker = AudioChunker::new(
            self.config.chunk_samples(),
            self.config.overlap_samples(),
            self.config.min_final_samples(),
        );
        let mut committer = WordCommitter::default();
        let mut committed_words = Vec::new();
        let mut fallback_fragments = Vec::new();
        let mut window_seq = 0u32;
        let mut total_samples = 0usize;
        let mut received_bytes = 0usize;
        let mut hasher = Sha256::new();
        let mut last_slot = 0usize;
        let mut request = None;

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
                    Some(TailSlot::Headers(headers)) => {
                        let built = build_request_from_parts(
                            headers.method,
                            headers.path,
                            headers.authority,
                            headers
                                .headers
                                .into_iter()
                                .map(|header| (header.name, header.value))
                                .collect(),
                        )?;
                        reject_json_requests(&built)?;
                        request = Some(built);
                    }
                    Some(TailSlot::Body(chunk)) => {
                        received_bytes += chunk.len();
                        hasher.update(&chunk);
                        self.push_pcm_bytes(&chunk, &mut chunker, &mut total_samples)?;
                        self.process_ready_windows(
                            &mut chunker,
                            &mut committer,
                            &mut committed_words,
                            &mut fallback_fragments,
                            &mut window_seq,
                        )
                        .await?;
                    }
                    Some(TailSlot::End) => {
                        break 'stream;
                    }
                    None => {}
                }
            }

            last_slot = current_last;
        }

        anyhow::ensure!(received_bytes > 0, "request body did not include audio bytes");
        let request = request.ok_or_else(|| anyhow::anyhow!("request headers were missing"))?;
        self.process_ready_windows(
            &mut chunker,
            &mut committer,
            &mut committed_words,
            &mut fallback_fragments,
            &mut window_seq,
        )
        .await?;

        if let Some(window) = chunker.take_final_window() {
            self.transcribe_window(
                &window.samples,
                window.start_sample,
                window.is_final,
                chunker.stride_samples(),
                window_seq,
                &mut committer,
                &mut committed_words,
                &mut fallback_fragments,
            )
            .await?;
        }

        Ok((
            request,
            PreparedTranscript {
                committed_words,
                fallback_fragments,
                duration_secs: total_samples as f64 / ASR_SAMPLE_RATE as f64,
                sha256: format!("{:x}", hasher.finalize()),
            },
        ))
    }

    async fn transcribe_remote_upload(
        &self,
        client: &RemoteIngressClient,
        origin: &str,
        stream_id: u64,
    ) -> Result<(Request<()>, PreparedTranscript)> {
        let mut poll = interval(Duration::from_millis(
            self.config.upload_response_worker_poll_ms.max(1),
        ));
        let mut chunker = AudioChunker::new(
            self.config.chunk_samples(),
            self.config.overlap_samples(),
            self.config.min_final_samples(),
        );
        let mut committer = WordCommitter::default();
        let mut committed_words = Vec::new();
        let mut fallback_fragments = Vec::new();
        let mut window_seq = 0u32;
        let mut total_samples = 0usize;
        let mut received_bytes = 0usize;
        let mut hasher = Sha256::new();
        let mut last_slot = 0usize;
        let mut request = None;

        'stream: loop {
            poll.tick().await;
            let current_last = client.request_last(origin, stream_id).await?;
            if current_last <= last_slot {
                continue;
            }

            for slot_id in (last_slot + 1)..=current_last {
                match client.request_slot(origin, stream_id, slot_id).await? {
                    Some(RemoteRequestSlot::Headers(bytes)) => {
                        let built = decode_request_headers_frame(&bytes)?;
                        reject_json_requests(&built)?;
                        request = Some(built);
                    }
                    Some(RemoteRequestSlot::Body(chunk)) => {
                        received_bytes += chunk.len();
                        hasher.update(&chunk);
                        self.push_pcm_bytes(&chunk, &mut chunker, &mut total_samples)?;
                        self.process_ready_windows(
                            &mut chunker,
                            &mut committer,
                            &mut committed_words,
                            &mut fallback_fragments,
                            &mut window_seq,
                        )
                        .await?;
                    }
                    Some(RemoteRequestSlot::End) => {
                        break 'stream;
                    }
                    None => {}
                }
            }

            last_slot = current_last;
        }

        anyhow::ensure!(received_bytes > 0, "request body did not include audio bytes");
        let request = request.ok_or_else(|| anyhow::anyhow!("request headers were missing"))?;
        self.process_ready_windows(
            &mut chunker,
            &mut committer,
            &mut committed_words,
            &mut fallback_fragments,
            &mut window_seq,
        )
        .await?;

        if let Some(window) = chunker.take_final_window() {
            self.transcribe_window(
                &window.samples,
                window.start_sample,
                window.is_final,
                chunker.stride_samples(),
                window_seq,
                &mut committer,
                &mut committed_words,
                &mut fallback_fragments,
            )
            .await?;
        }

        Ok((
            request,
            PreparedTranscript {
                committed_words,
                fallback_fragments,
                duration_secs: total_samples as f64 / ASR_SAMPLE_RATE as f64,
                sha256: format!("{:x}", hasher.finalize()),
            },
        ))
    }

    async fn transcribe_upload(&self, mut body: BodyStream) -> Result<PreparedTranscript> {
        let mut decoder = Self::new_decoder();
        let mut chunker = AudioChunker::new(
            self.config.chunk_samples(),
            self.config.overlap_samples(),
            self.config.min_final_samples(),
        );
        let mut committer = WordCommitter::default();
        let mut committed_words = Vec::new();
        let mut fallback_fragments = Vec::new();
        let mut window_seq = 0u32;
        let mut total_samples = 0usize;
        let mut received_bytes = 0usize;
        let mut hasher = Sha256::new();

        while let Some(next) = body.next().await {
            let chunk =
                next.map_err(|error| anyhow::anyhow!("failed to read request body: {error}"))?;
            if chunk.is_empty() {
                continue;
            }

            received_bytes += chunk.len();
            hasher.update(&chunk);
            self.feed_decoder(&mut decoder, chunk, &mut chunker, &mut total_samples)
                .await?;
            self.process_ready_windows(
                &mut chunker,
                &mut committer,
                &mut committed_words,
                &mut fallback_fragments,
                &mut window_seq,
            )
            .await?;
        }

        anyhow::ensure!(
            received_bytes > 0,
            "request body did not include audio bytes"
        );

        self.finish_decoder(&mut decoder, &mut chunker, &mut total_samples)
            .await?;
        self.process_ready_windows(
            &mut chunker,
            &mut committer,
            &mut committed_words,
            &mut fallback_fragments,
            &mut window_seq,
        )
        .await?;

        if let Some(window) = chunker.take_final_window() {
            self.transcribe_window(
                &window.samples,
                window.start_sample,
                window.is_final,
                chunker.stride_samples(),
                window_seq,
                &mut committer,
                &mut committed_words,
                &mut fallback_fragments,
            )
            .await?;
        }

        Ok(PreparedTranscript {
            committed_words,
            fallback_fragments,
            duration_secs: total_samples as f64 / ASR_SAMPLE_RATE as f64,
            sha256: format!("{:x}", hasher.finalize()),
        })
    }

    fn new_decoder() -> DecodePipelineHandle {
        DecodePipeline::spawn_with_options(DecodeOptions {
            output_bits_per_sample: Some(16),
            output_sample_rate: Some(ASR_SAMPLE_RATE),
            output_channels: Some(1),
        })
    }

    async fn feed_decoder(
        &self,
        decoder: &mut DecodePipelineHandle,
        data: Bytes,
        chunker: &mut AudioChunker,
        total_samples: &mut usize,
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

        self.drain_decoder(decoder, chunker, total_samples)
    }

    async fn finish_decoder(
        &self,
        decoder: &mut DecodePipelineHandle,
        chunker: &mut AudioChunker,
        total_samples: &mut usize,
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
            let audio = output.map_err(|error| anyhow::anyhow!("decode failed: {error}"))?;
            let samples = audio_to_mono_f32(&audio)?;
            *total_samples += samples.len();
            chunker.push(&samples);
        }

        Ok(())
    }

    fn drain_decoder(
        &self,
        decoder: &mut DecodePipelineHandle,
        chunker: &mut AudioChunker,
        total_samples: &mut usize,
    ) -> Result<()> {
        while let Some(output) = decoder.try_recv() {
            let audio = output.map_err(|error| anyhow::anyhow!("decode failed: {error}"))?;
            let samples = audio_to_mono_f32(&audio)?;
            *total_samples += samples.len();
            chunker.push(&samples);
        }
        Ok(())
    }

    fn push_pcm_bytes(
        &self,
        chunk: &[u8],
        chunker: &mut AudioChunker,
        total_samples: &mut usize,
    ) -> Result<()> {
        let samples = pcm_f32le_bytes_to_vec(chunk)?;
        *total_samples += samples.len();
        chunker.push(&samples);
        Ok(())
    }

    async fn process_ready_windows(
        &self,
        chunker: &mut AudioChunker,
        committer: &mut WordCommitter,
        committed_words: &mut Vec<CommittedWord>,
        fallback_fragments: &mut Vec<String>,
        window_seq: &mut u32,
    ) -> Result<()> {
        let stable_samples = chunker.stride_samples();
        for window in chunker.take_ready_windows() {
            self.transcribe_window(
                &window.samples,
                window.start_sample,
                window.is_final,
                stable_samples,
                *window_seq,
                committer,
                committed_words,
                fallback_fragments,
            )
            .await?;
            *window_seq += 1;
        }
        Ok(())
    }

    async fn transcribe_window(
        &self,
        samples: &[f32],
        start_sample: usize,
        is_final: bool,
        stable_samples: usize,
        seq: u32,
        committer: &mut WordCommitter,
        committed_words: &mut Vec<CommittedWord>,
        fallback_fragments: &mut Vec<String>,
    ) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }

        let result = self
            .backend
            .transcribe_window(samples.to_vec(), seq)
            .await?;
        let trimmed = result.text.trim();
        if !trimmed.is_empty()
            && fallback_fragments
                .last()
                .map(|last| last != trimmed)
                .unwrap_or(true)
        {
            fallback_fragments.push(trimmed.to_string());
        }

        committed_words.extend(committer.commit(
            start_sample,
            stable_samples,
            is_final,
            &result.words,
        ));
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct RemoteStreamInfo {
    stream_id: u64,
    request_last: usize,
    response_owner: Option<String>,
}

#[derive(Debug)]
enum RemoteRequestSlot {
    Headers(Bytes),
    Body(Bytes),
    End,
}

#[derive(Clone)]
struct RemoteIngressClient {
    client: Client,
    slot_bytes: usize,
}

impl RemoteIngressClient {
    fn new(config: &AppConfig) -> Result<Self> {
        let client = Client::builder()
            .danger_accept_invalid_certs(config.upload_response_insecure_tls)
            .http2_adaptive_window(true)
            .build()
            .map_err(|error| anyhow::anyhow!("failed to build reqwest client: {error}"))?;
        Ok(Self {
            client,
            slot_bytes: config.upload_response_config().slot_bytes().max(1),
        })
    }

    async fn list_streams(&self, origin: &str) -> Result<Vec<RemoteStreamInfo>> {
        let response = self
            .client
            .get(format!("{origin}/_upload_response/streams"))
            .send()
            .await
            .map_err(|error| anyhow::anyhow!("failed to list streams from {origin}: {error}"))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| anyhow::anyhow!("failed to read stream list from {origin}: {error}"))?;
        anyhow::ensure!(
            status.is_success(),
            "unexpected stream list status {status} from {origin}: {body}"
        );

        let mut streams = Vec::new();
        for (index, line) in body.lines().enumerate() {
            if index == 0 || line.trim().is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() < 6 {
                continue;
            }
            streams.push(RemoteStreamInfo {
                stream_id: fields[0]
                    .parse()
                    .map_err(|error| anyhow::anyhow!("invalid stream id in {origin}: {error}"))?,
                request_last: fields[2]
                    .parse()
                    .map_err(|error| anyhow::anyhow!("invalid request_last in {origin}: {error}"))?,
                response_owner: match fields[5] {
                    "" | "-" => None,
                    value => Some(value.to_string()),
                },
            });
        }
        Ok(streams)
    }

    async fn request_last(&self, origin: &str, stream_id: u64) -> Result<usize> {
        let response = self
            .client
            .get(format!(
                "{origin}/_upload_response/streams/{stream_id}/request/last"
            ))
            .send()
            .await
            .map_err(|error| anyhow::anyhow!("failed to read request_last for {stream_id}: {error}"))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| anyhow::anyhow!("failed to read request_last body for {stream_id}: {error}"))?;
        anyhow::ensure!(
            status.is_success(),
            "unexpected request_last status {status} for stream {stream_id}: {body}"
        );
        body.trim()
            .parse()
            .map_err(|error| anyhow::anyhow!("invalid request_last for stream {stream_id}: {error}"))
    }

    async fn request_slot(
        &self,
        origin: &str,
        stream_id: u64,
        slot_id: usize,
    ) -> Result<Option<RemoteRequestSlot>> {
        let response = self
            .client
            .get(format!(
                "{origin}/_upload_response/streams/{stream_id}/request/slots/{slot_id}"
            ))
            .send()
            .await
            .map_err(|error| anyhow::anyhow!("failed to fetch stream {stream_id} slot {slot_id}: {error}"))?;
        if response.status() == ReqwestStatusCode::NOT_FOUND {
            return Ok(None);
        }
        let status = response.status();
        let slot_type = response
            .headers()
            .get("x-upload-response-slot-type")
            .and_then(|value| value.to_str().ok())
            .unwrap_or("body")
            .to_string();
        let body = response
            .bytes()
            .await
            .map_err(|error| anyhow::anyhow!("failed to read stream {stream_id} slot {slot_id}: {error}"))?;
        anyhow::ensure!(
            status.is_success(),
            "unexpected slot status {status} for stream {stream_id} slot {slot_id}"
        );

        Ok(Some(match slot_type.as_str() {
            "headers" => RemoteRequestSlot::Headers(body),
            "end" => RemoteRequestSlot::End,
            _ => RemoteRequestSlot::Body(body),
        }))
    }

    async fn register_reader(&self, origin: &str, stream_id: u64, worker_id: &str) -> Result<()> {
        let response = self
            .client
            .put(format!(
                "{origin}/_upload_response/streams/{stream_id}/readers/{worker_id}"
            ))
            .send()
            .await
            .map_err(|error| anyhow::anyhow!("failed to register reader for {stream_id}: {error}"))?;
        anyhow::ensure!(
            response.status().is_success() || response.status() == ReqwestStatusCode::NO_CONTENT,
            "reader registration failed for stream {stream_id} with status {}",
            response.status()
        );
        Ok(())
    }

    async fn unregister_reader(
        &self,
        origin: &str,
        stream_id: u64,
        worker_id: &str,
    ) -> Result<()> {
        let response = self
            .client
            .delete(format!(
                "{origin}/_upload_response/streams/{stream_id}/readers/{worker_id}"
            ))
            .send()
            .await
            .map_err(|error| anyhow::anyhow!("failed to unregister reader for {stream_id}: {error}"))?;
        anyhow::ensure!(
            response.status().is_success()
                || response.status() == ReqwestStatusCode::NO_CONTENT
                || response.status() == ReqwestStatusCode::CONFLICT
                || response.status() == ReqwestStatusCode::NOT_FOUND,
            "reader unregister failed for stream {stream_id} with status {}",
            response.status()
        );
        Ok(())
    }

    async fn try_claim_response(
        &self,
        origin: &str,
        stream_id: u64,
        worker_id: &str,
    ) -> Result<bool> {
        let response = self
            .client
            .put(format!(
                "{origin}/_upload_response/streams/{stream_id}/response/claim/{worker_id}"
            ))
            .send()
            .await
            .map_err(|error| anyhow::anyhow!("failed to claim response for {stream_id}: {error}"))?;
        match response.status() {
            ReqwestStatusCode::OK => Ok(true),
            ReqwestStatusCode::CONFLICT | ReqwestStatusCode::NO_CONTENT | ReqwestStatusCode::NOT_FOUND => Ok(false),
            status => {
                let body = response.text().await.unwrap_or_default();
                Err(anyhow::anyhow!(
                    "unexpected claim status {status} for stream {stream_id}: {body}"
                ))
            }
        }
    }

    async fn release_response(
        &self,
        origin: &str,
        stream_id: u64,
        worker_id: &str,
    ) -> Result<()> {
        let response = self
            .client
            .delete(format!(
                "{origin}/_upload_response/streams/{stream_id}/response/claim/{worker_id}"
            ))
            .send()
            .await
            .map_err(|error| anyhow::anyhow!("failed to release response for {stream_id}: {error}"))?;
        anyhow::ensure!(
            response.status().is_success()
                || response.status() == ReqwestStatusCode::CONFLICT
                || response.status() == ReqwestStatusCode::NOT_FOUND,
            "release response failed for stream {stream_id} with status {}",
            response.status()
        );
        Ok(())
    }

    async fn write_handler_response(
        &self,
        origin: &str,
        stream_id: u64,
        response: HandlerResponse,
    ) -> Result<()> {
        let mut builder = http::Response::builder().status(response.status);
        if let Some(content_type) = &response.content_type {
            builder = builder.header(http::header::CONTENT_TYPE, content_type);
        }
        if let Some(etag) = response.etag {
            builder = builder.header(http::header::ETAG, etag.to_string());
        }
        for (name, value) in &response.headers {
            builder = builder.header(name, value);
        }

        let response_head = builder
            .body(())
            .map_err(|error| anyhow::anyhow!("failed to build response head: {error}"))?;
        let frame = StreamFrame::Headers(
            StreamHeaders::from_response(stream_id, &response_head)
                .map_err(|error| anyhow::anyhow!("failed to encode response headers: {error}"))?,
        );
        let payload = encode_frame(&frame);

        self.expect_ok(
            self.client
                .put(format!(
                    "{origin}/_upload_response/streams/{stream_id}/response/headers"
                ))
                .body(payload)
                .send()
                .await,
            format!("write response headers for stream {stream_id}"),
        )
        .await?;

        if let Some(body) = response.body {
            for chunk in body.chunks(self.slot_bytes) {
                if chunk.is_empty() {
                    continue;
                }
                self.expect_ok(
                    self.client
                        .put(format!(
                            "{origin}/_upload_response/streams/{stream_id}/response/body"
                        ))
                        .body(chunk.to_vec())
                        .send()
                        .await,
                    format!("write response body for stream {stream_id}"),
                )
                .await?;
            }
        }

        self.expect_ok(
            self.client
                .put(format!(
                    "{origin}/_upload_response/streams/{stream_id}/response/end"
                ))
                .send()
                .await,
            format!("write response end for stream {stream_id}"),
        )
        .await?;

        Ok(())
    }

    async fn expect_ok(
        &self,
        response: std::result::Result<reqwest::Response, reqwest::Error>,
        context: String,
    ) -> Result<()> {
        let response = response.map_err(|error| anyhow::anyhow!("{context}: {error}"))?;
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::ensure!(status.is_success(), "{context}: unexpected status {status}: {body}");
        Ok(())
    }
}

async fn discover_ingress_origins(config: &AppConfig) -> Result<Vec<String>> {
    let mut origins = BTreeSet::new();

    for origin in &config.upload_response_ingress_urls {
        let trimmed = origin.trim().trim_end_matches('/');
        if !trimmed.is_empty() {
            origins.insert(trimmed.to_string());
        }
    }

    if let Some(discovery_dns) = &config.upload_response_discovery_dns {
        let discovery_dns = discovery_dns.trim();
        if !discovery_dns.is_empty() {
            for socket in lookup_host(discovery_dns)
                .await
                .map_err(|error| anyhow::anyhow!("failed to resolve {discovery_dns}: {error}"))?
            {
                origins.insert(format!("https://{}", socket));
            }
        }
    }

    Ok(origins.into_iter().collect())
}

fn pcm_f32le_bytes_to_vec(chunk: &[u8]) -> Result<Vec<f32>> {
    anyhow::ensure!(
        chunk.len() % std::mem::size_of::<f32>() == 0,
        "invalid cached PCM chunk length {}; expected multiple of 4",
        chunk.len()
    );

    let mut samples = Vec::with_capacity(chunk.len() / std::mem::size_of::<f32>());
    for bytes in chunk.chunks_exact(std::mem::size_of::<f32>()) {
        samples.push(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
    }
    Ok(samples)
}

fn decode_request_headers_frame(bytes: &[u8]) -> Result<Request<()>> {
    let frame = decode_frame(bytes)
        .map_err(|error| anyhow::anyhow!("failed to decode cached request headers: {error}"))?;
    match frame {
        StreamFrame::Headers(StreamHeaders::Request(headers)) => build_request_from_parts(
            headers.method,
            headers.path,
            headers.authority,
            headers
                .headers
                .into_iter()
                .map(|header| (header.name, header.value))
                .collect(),
        ),
        _ => anyhow::bail!("unexpected cached request frame; expected request headers"),
    }
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

fn reject_json_requests(req: &Request<()>) -> Result<()> {
    let is_json = req
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| {
            let value = value.to_ascii_lowercase();
            value.starts_with("application/json") || value.ends_with("+json")
        })
        .unwrap_or(false);

    if is_json {
        anyhow::bail!(
            "JSON URL payloads are not supported yet; upload audio bytes in the request body"
        );
    }

    Ok(())
}

fn json_response<T: Serialize>(status: StatusCode, value: &T) -> Result<HandlerResponse> {
    Ok(HandlerResponse {
        status,
        body: Some(Bytes::from(serde_json::to_vec(value)?)),
        content_type: Some("application/json".into()),
        headers: vec![("cache-control".into(), "no-store".into())],
        etag: None,
    })
}

fn error_response(status: StatusCode, message: String) -> HandlerResponse {
    HandlerResponse {
        status,
        body: Some(Bytes::from(
            serde_json::to_vec(&serde_json::json!({ "error": message }))
                .unwrap_or_else(|_| b"{\"error\":\"serialization failure\"}".to_vec()),
        )),
        content_type: Some("application/json".into()),
        headers: vec![("cache-control".into(), "no-store".into())],
        etag: None,
    }
}

fn classify_error(error: &anyhow::Error) -> StatusCode {
    let message = error.to_string().to_ascii_lowercase();
    if message.contains("request body")
        || message.contains("not supported")
        || message.contains("decode")
        || message.contains("deserialize")
        || message.contains("audio bytes")
    {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    }
}

fn build_request_from_parts(
    method: Vec<u8>,
    path: Vec<u8>,
    authority: Option<Vec<u8>>,
    headers: Vec<(Vec<u8>, Vec<u8>)>,
) -> Result<Request<()>> {
    let method = String::from_utf8(method)
        .map_err(|error| anyhow::anyhow!("invalid request method bytes: {error}"))?;
    let uri = String::from_utf8(path)
        .map_err(|error| anyhow::anyhow!("invalid request path bytes: {error}"))?;

    let mut builder = Request::builder().method(method.as_str()).uri(uri.as_str());
    if let Some(authority) = authority {
        builder = builder.header(
            http::header::HOST,
            HeaderValue::from_bytes(&authority)
                .map_err(|error| anyhow::anyhow!("invalid authority header: {error}"))?,
        );
    }

    for (name_bytes, value_bytes) in headers {
        let name = HeaderName::from_bytes(&name_bytes)
            .map_err(|error| anyhow::anyhow!("invalid header name: {error}"))?;
        let value = HeaderValue::from_bytes(&value_bytes)
            .map_err(|error| anyhow::anyhow!("invalid header value: {error}"))?;
        builder = builder.header(name, value);
    }

    builder
        .body(())
        .map_err(|error| anyhow::anyhow!("failed to build request: {error}"))
}
