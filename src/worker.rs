use crate::asr::AsrBackend;
use crate::chunking::{AudioChunker, CommittedWord, WordCommitter};
use crate::config::{AppConfig, ASR_SAMPLE_RATE};
use crate::deepgram::{
    append_word, build_response, default_model_info, ends_sentence, join_words,
    words_from_committed, ListenOptions, ModelInfo, Word,
};
use crate::pcm::{rms_level, Linear16PcmStream};
use crate::protocol::{INTERNAL_STREAMING_MODE_HEADER, INTERNAL_STREAMING_MODE_JSONL};
use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use http::{header::CONTENT_TYPE, HeaderName, HeaderValue, Request, StatusCode};
use http_pack::stream::{decode_frame, encode_frame, StreamFrame, StreamHeaders};
use hyper::upgrade::Upgraded;
use hyper_util::rt::TokioIo;
use reqwest::{Client, StatusCode as ReqwestStatusCode};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use soundkit::audio_pipeline::{deserialize_audio, vec_i16_to_f32, vec_i32_to_f32};
use soundkit::audio_types::{AudioData, PcmData};
use soundkit_decoder::{DecodeOptions, DecodePipeline, DecodePipelineHandle};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::Arc;
use tokio::net::lookup_host;
use tokio::task::JoinSet;
use tokio::time::{interval, Duration};
use tokio_tungstenite::{tungstenite::Message, WebSocketStream};
use tracing::{debug, error, warn};
use upload_response::{RequestControl, TailSlot, UploadResponseService};
use uuid::Uuid;
use web_service::{BodyStream, HandlerResponse, HandlerResult, ServerError, WebSocketHandler};

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

const WS_STREAM_CHUNK_SECONDS: f32 = 6.0;
const WS_STREAM_OVERLAP_SECONDS: f32 = 0.8;
const WS_STREAM_FINAL_MIN_SECONDS: f32 = 0.2;
const WS_INTERIM_MIN_SECONDS: f32 = 0.6;
const WS_INTERIM_INTERVAL_MS: u64 = 400;
const WS_SPEECH_RMS_THRESHOLD: f32 = 0.01;
const WS_WORD_DEDUPE_EPSILON_MS: u64 = 25;

#[derive(Clone)]
pub struct ListenWebSocketHandler {
    worker: Arc<WorkerState>,
}

#[async_trait]
trait JsonEventSink {
    async fn send_json(&mut self, json: String) -> Result<()>;

    async fn finish(&mut self) -> Result<()> {
        Ok(())
    }
}

enum ResponseCacheTarget {
    Local(Arc<UploadResponseService>),
    Remote {
        client: RemoteIngressClient,
        origin: String,
    },
}

struct JsonLineResponseWriter {
    target: ResponseCacheTarget,
    stream_id: u64,
    slot_bytes: usize,
    started: bool,
    finished: bool,
}

struct WebSocketJsonSink<'a> {
    stream: &'a mut WebSocketStream<TokioIo<Upgraded>>,
}

#[derive(Debug)]
struct WsTranscriptState {
    options: ListenOptions,
    request_id: String,
    model_id: String,
    model_info: BTreeMap<String, ModelInfo>,
    chunker: AudioChunker,
    committer: WordCommitter,
    pending_final_words: Vec<CommittedWord>,
    completed_transcript: String,
    total_samples: usize,
    next_seq: u32,
    last_interim_total_samples: usize,
    speech_started_sent: bool,
    gap_threshold_ms: u64,
}

#[derive(Debug, Serialize)]
struct WsMetadataEvent {
    #[serde(rename = "type")]
    event_type: &'static str,
    request_id: String,
    sha256: String,
    created: String,
    duration: f64,
    channels: u8,
    model_info: ModelInfo,
    model_uuid: String,
}

#[derive(Debug, Serialize)]
struct WsResultsEvent {
    #[serde(rename = "type")]
    event_type: &'static str,
    channel_index: [u8; 1],
    duration: f64,
    start: f64,
    is_final: bool,
    speech_final: bool,
    from_finalize: bool,
    channel: WsChannel,
    metadata: WsResultsMetadata,
}

#[derive(Debug, Serialize)]
struct WsChannel {
    alternatives: Vec<WsAlternative>,
}

#[derive(Debug, Serialize)]
struct WsAlternative {
    transcript: String,
    confidence: f64,
    words: Vec<Word>,
}

#[derive(Debug, Serialize)]
struct WsResultsMetadata {
    request_id: String,
    model_info: ModelInfo,
    model_uuid: String,
    device_id: usize,
}

#[derive(Debug, Serialize)]
struct WsSpeechStartedEvent {
    #[serde(rename = "type")]
    event_type: &'static str,
    channel: [u8; 1],
    timestamp: f64,
}

#[derive(Debug, Serialize)]
struct WsUtteranceEndEvent {
    #[serde(rename = "type")]
    event_type: &'static str,
    channel: [u8; 1],
    last_word_end: f64,
}

#[derive(Debug, Deserialize)]
struct WsClientEvent {
    #[serde(rename = "type")]
    event_type: String,
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

    async fn handle_listen_websocket(
        &self,
        req: Request<()>,
        mut stream: WebSocketStream<TokioIo<Upgraded>>,
    ) -> HandlerResult<()> {
        let options = ListenOptions::from_request(&req, &self.config);
        let sample_rate = options.sample_rate_hz.unwrap_or(ASR_SAMPLE_RATE);
        let channels = options.channels.max(1);
        let mut pcm_stream = Linear16PcmStream::new(sample_rate, channels)
            .map_err(|error| ServerError::Config(error.to_string()))?;
        let request_id = Uuid::new_v4().to_string();
        let (model_id, model_info) = default_model_info(&options.model);
        let mut state = WsTranscriptState::new(options, request_id.clone(), model_id, model_info);

        let metadata = WsMetadataEvent {
            event_type: "Metadata",
            request_id,
            sha256: String::new(),
            created: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            duration: 0.0,
            channels: 1,
            model_info: state.primary_model_info().clone(),
            model_uuid: state.model_id.clone(),
        };
        let mut sink = WebSocketJsonSink {
            stream: &mut stream,
        };
        send_json_event(&mut sink, &metadata)
            .await
            .map_err(anyhow_to_server_error)?;

        while let Some(frame) = stream.next().await {
            match frame {
                Ok(Message::Binary(bytes)) => {
                    let samples = pcm_stream
                        .push(&bytes)
                        .map_err(|error| ServerError::Config(error.to_string()))?;

                    if !state.speech_started_sent
                        && !samples.is_empty()
                        && rms_level(&samples) >= WS_SPEECH_RMS_THRESHOLD
                    {
                        state.speech_started_sent = true;
                        let event = WsSpeechStartedEvent {
                            event_type: "SpeechStarted",
                            channel: [0],
                            timestamp: state.total_duration_secs(),
                        };
                        let mut sink = WebSocketJsonSink {
                            stream: &mut stream,
                        };
                        send_json_event(&mut sink, &event)
                            .await
                            .map_err(anyhow_to_server_error)?;
                    }

                    let mut sink = WebSocketJsonSink {
                        stream: &mut stream,
                    };
                    self.process_streaming_samples(&mut state, &samples, &mut sink)
                        .await
                        .map_err(anyhow_to_server_error)?;
                }
                Ok(Message::Text(text)) => {
                    let event = serde_json::from_str::<WsClientEvent>(&text).map_err(|error| {
                        ServerError::Config(format!("invalid websocket control message: {error}"))
                    })?;
                    match event.event_type.as_str() {
                        "KeepAlive" => {}
                        "Finalize" => {
                            let mut sink = WebSocketJsonSink {
                                stream: &mut stream,
                            };
                            self.flush_streaming_session(&mut state, None, false, &mut sink)
                                .await
                                .map_err(anyhow_to_server_error)?;
                        }
                        "CloseStream" => {
                            let mut sink = WebSocketJsonSink {
                                stream: &mut stream,
                            };
                            self.flush_streaming_session(
                                &mut state,
                                Some(&mut pcm_stream),
                                true,
                                &mut sink,
                            )
                            .await
                            .map_err(anyhow_to_server_error)?;
                            let _ = stream.close(None).await;
                            return Ok(());
                        }
                        other => {
                            return Err(ServerError::Config(format!(
                                "unsupported websocket control type: {other}"
                            )));
                        }
                    }
                }
                Ok(Message::Ping(payload)) => {
                    stream
                        .send(Message::Pong(payload))
                        .await
                        .map_err(|error| ServerError::Handler(Box::new(error)))?;
                }
                Ok(Message::Pong(_)) => {}
                Ok(Message::Close(frame)) => {
                    let _ = stream.close(frame).await;
                    return Ok(());
                }
                Ok(Message::Frame(_)) => {}
                Err(error) => return Err(ServerError::Handler(Box::new(error))),
            }
        }

        Ok(())
    }

    async fn process_streaming_samples<S: JsonEventSink + Send>(
        &self,
        state: &mut WsTranscriptState,
        samples: &[f32],
        sink: &mut S,
    ) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }

        state.total_samples += samples.len();
        state.chunker.push(samples);
        let stable_samples = state.chunker.stride_samples();

        for window in state.chunker.take_ready_windows() {
            let result = self
                .backend
                .transcribe_window(window.samples, state.next_seq())
                .await?;
            let committed = commit_absolute_words(
                &mut state.committer,
                window.start_sample,
                stable_samples,
                window.is_final,
                &result.words,
            );
            self.emit_streaming_committed(state, committed, true, sink)
                .await?;
        }

        self.maybe_send_streaming_interim(state, sink).await
    }

    async fn flush_streaming_session<S: JsonEventSink + Send>(
        &self,
        state: &mut WsTranscriptState,
        pcm_stream: Option<&mut Linear16PcmStream>,
        close_stream: bool,
        sink: &mut S,
    ) -> Result<()> {
        if let Some(pcm_stream) = pcm_stream {
            let tail = pcm_stream
                .finish()
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            if !tail.is_empty() {
                state.total_samples += tail.len();
                state.chunker.push(&tail);
            }
        }

        if let Some(window) = state.chunker.take_final_window() {
            let result = self
                .backend
                .transcribe_window(window.samples, state.next_seq())
                .await?;
            let committed = commit_absolute_words(
                &mut state.committer,
                window.start_sample,
                state.chunker.stride_samples(),
                true,
                &result.words,
            );
            self.emit_streaming_committed(state, committed, false, sink)
                .await?;
        }

        if let Some(segment) = state.take_pending_segment() {
            self.send_streaming_result(state, segment, true, close_stream, false, sink)
                .await?;
        }

        Ok(())
    }

    async fn emit_streaming_committed<S: JsonEventSink + Send>(
        &self,
        state: &mut WsTranscriptState,
        committed: Vec<CommittedWord>,
        from_finalize: bool,
        sink: &mut S,
    ) -> Result<()> {
        if committed.is_empty() {
            return Ok(());
        }

        state.pending_final_words.extend(committed);
        while let Some(segment) = state.take_auto_finalized_segment() {
            self.send_streaming_result(state, segment, true, true, from_finalize, sink)
                .await?;
        }
        Ok(())
    }

    async fn maybe_send_streaming_interim<S: JsonEventSink + Send>(
        &self,
        state: &mut WsTranscriptState,
        sink: &mut S,
    ) -> Result<()> {
        if !state.options.interim_results {
            return Ok(());
        }

        let min_samples = seconds_to_samples(WS_INTERIM_MIN_SECONDS);
        let interval_samples = ((ASR_SAMPLE_RATE as u64 * WS_INTERIM_INTERVAL_MS) / 1000) as usize;
        if state.chunker.pending_samples().len() < min_samples
            || state
                .total_samples
                .saturating_sub(state.last_interim_total_samples)
                < interval_samples
        {
            return Ok(());
        }

        let result = self
            .backend
            .transcribe_window(state.chunker.pending_samples().to_vec(), state.next_seq())
            .await?;
        let preview_words =
            preview_absolute_words(state.chunker.pending_start_sample(), &result.words);
        let words = state.preview_words(preview_words);
        let transcript = state.preview_transcript(&words, result.text.trim());
        if transcript.is_empty() {
            return Ok(());
        }

        state.last_interim_total_samples = state.total_samples;
        let event = WsResultsEvent {
            event_type: "Results",
            channel_index: [0],
            duration: state.total_duration_secs(),
            start: 0.0,
            is_final: false,
            speech_final: false,
            from_finalize: false,
            channel: WsChannel {
                alternatives: vec![WsAlternative {
                    transcript,
                    confidence: 0.0,
                    words: if state.options.wants_word_timestamps() {
                        words_from_committed(&words)
                    } else {
                        Vec::new()
                    },
                }],
            },
            metadata: state.results_metadata(),
        };
        send_json_event(sink, &event).await
    }

    async fn send_streaming_result<S: JsonEventSink + Send>(
        &self,
        state: &mut WsTranscriptState,
        segment: Vec<CommittedWord>,
        is_final: bool,
        speech_final: bool,
        from_finalize: bool,
        sink: &mut S,
    ) -> Result<()> {
        let transcript = join_words(segment.iter().map(|word| word.word.as_str()));
        if transcript.is_empty() {
            return Ok(());
        }

        state.append_completed_transcript(&transcript);
        let words = if state.options.wants_word_timestamps() {
            words_from_committed(&segment)
        } else {
            Vec::new()
        };
        let start = segment
            .first()
            .map(|word| word.start_ms as f64 / 1000.0)
            .unwrap_or(0.0);
        let duration = segment
            .last()
            .map(|word| word.end_ms as f64 / 1000.0)
            .unwrap_or(start);

        let event = WsResultsEvent {
            event_type: "Results",
            channel_index: [0],
            duration,
            start,
            is_final,
            speech_final,
            from_finalize,
            channel: WsChannel {
                alternatives: vec![WsAlternative {
                    transcript,
                    confidence: 0.0,
                    words,
                }],
            },
            metadata: state.results_metadata(),
        };
        send_json_event(sink, &event).await?;

        if state.options.wants_word_timestamps() {
            if let Some(last_word) = segment.last() {
                let utterance_end = WsUtteranceEndEvent {
                    event_type: "UtteranceEnd",
                    channel: [0],
                    last_word_end: last_word.end_ms as f64 / 1000.0,
                };
                send_json_event(sink, &utterance_end).await?;
            }
        }

        Ok(())
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
                    let result = worker
                        .process_cached_stream(service.clone(), stream_id)
                        .await;
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
        let options = ListenOptions::from_request(&req, &self.config);
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
        if let Some(request) = self
            .read_cached_request_headers(&service, stream_id)
            .await?
        {
            if is_streaming_request(&request) {
                let mut writer = JsonLineResponseWriter::local(
                    Arc::clone(&service),
                    stream_id,
                    self.config.upload_response_config().slot_bytes(),
                );
                return self
                    .stream_cached_upload(service, stream_id, request, &mut writer)
                    .await;
            }
        }

        let (request, prepared) = self.transcribe_cached_upload(&service, stream_id).await?;
        let options = ListenOptions::from_request(&request, &self.config);
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
        if let Some(request) = client.request_headers(origin, stream_id).await? {
            if is_streaming_request(&request) {
                let mut writer = JsonLineResponseWriter::remote(
                    client.clone(),
                    origin.to_string(),
                    stream_id,
                    self.config.upload_response_config().slot_bytes(),
                );
                return self
                    .stream_remote_upload(client, origin, stream_id, request, &mut writer)
                    .await;
            }
        }

        let (request, prepared) = self
            .transcribe_remote_upload(client, origin, stream_id)
            .await?;
        let options = ListenOptions::from_request(&request, &self.config);
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
        client
            .write_handler_response(origin, stream_id, response)
            .await?;
        Ok(())
    }

    async fn read_cached_request_headers(
        &self,
        service: &UploadResponseService,
        stream_id: u64,
    ) -> Result<Option<Request<()>>> {
        match service.tail_request(stream_id, 1).await {
            Some(TailSlot::Headers(headers)) => Ok(Some(build_request_from_parts(
                headers.method,
                headers.path,
                headers.authority,
                headers
                    .headers
                    .into_iter()
                    .map(|header| (header.name, header.value))
                    .collect(),
            )?)),
            Some(_) | None => Ok(None),
        }
    }

    async fn stream_cached_upload(
        &self,
        service: Arc<UploadResponseService>,
        stream_id: u64,
        request: Request<()>,
        writer: &mut JsonLineResponseWriter,
    ) -> Result<()> {
        let mut poll = interval(Duration::from_millis(
            self.config.upload_response_worker_poll_ms.max(1),
        ));
        let options = ListenOptions::from_request(&request, &self.config);
        let (model_id, model_info) = default_model_info(&options.model);
        let request_id = Uuid::new_v4().to_string();
        let mut state = WsTranscriptState::new(options, request_id.clone(), model_id, model_info);
        let mut received_bytes = 0usize;
        let mut last_slot = 0usize;

        let metadata = WsMetadataEvent {
            event_type: "Metadata",
            request_id,
            sha256: String::new(),
            created: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            duration: 0.0,
            channels: 1,
            model_info: state.primary_model_info().clone(),
            model_uuid: state.model_id.clone(),
        };
        send_json_event(writer, &metadata).await?;

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
                        let samples = pcm_f32le_bytes_to_vec(&chunk)?;
                        if !state.speech_started_sent
                            && !samples.is_empty()
                            && rms_level(&samples) >= WS_SPEECH_RMS_THRESHOLD
                        {
                            state.speech_started_sent = true;
                            let event = WsSpeechStartedEvent {
                                event_type: "SpeechStarted",
                                channel: [0],
                                timestamp: state.total_duration_secs(),
                            };
                            send_json_event(writer, &event).await?;
                        }
                        self.process_streaming_samples(&mut state, &samples, writer)
                            .await?;
                    }
                    Some(TailSlot::Control(RequestControl::Finalize)) => {
                        self.flush_streaming_session(&mut state, None, false, writer)
                            .await?;
                    }
                    Some(TailSlot::Control(RequestControl::KeepAlive)) => {}
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

        self.flush_streaming_session(&mut state, None, true, writer)
            .await?;
        writer.finish().await
    }

    async fn stream_remote_upload(
        &self,
        client: &RemoteIngressClient,
        origin: &str,
        stream_id: u64,
        request: Request<()>,
        writer: &mut JsonLineResponseWriter,
    ) -> Result<()> {
        let mut poll = interval(Duration::from_millis(
            self.config.upload_response_worker_poll_ms.max(1),
        ));
        let options = ListenOptions::from_request(&request, &self.config);
        let (model_id, model_info) = default_model_info(&options.model);
        let request_id = Uuid::new_v4().to_string();
        let mut state = WsTranscriptState::new(options, request_id.clone(), model_id, model_info);
        let mut received_bytes = 0usize;
        let mut last_slot = 0usize;

        let metadata = WsMetadataEvent {
            event_type: "Metadata",
            request_id,
            sha256: String::new(),
            created: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            duration: 0.0,
            channels: 1,
            model_info: state.primary_model_info().clone(),
            model_uuid: state.model_id.clone(),
        };
        send_json_event(writer, &metadata).await?;

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
                        let samples = pcm_f32le_bytes_to_vec(&chunk)?;
                        if !state.speech_started_sent
                            && !samples.is_empty()
                            && rms_level(&samples) >= WS_SPEECH_RMS_THRESHOLD
                        {
                            state.speech_started_sent = true;
                            let event = WsSpeechStartedEvent {
                                event_type: "SpeechStarted",
                                channel: [0],
                                timestamp: state.total_duration_secs(),
                            };
                            send_json_event(writer, &event).await?;
                        }
                        self.process_streaming_samples(&mut state, &samples, writer)
                            .await?;
                    }
                    Some(RemoteRequestSlot::Control(RequestControl::Finalize)) => {
                        self.flush_streaming_session(&mut state, None, false, writer)
                            .await?;
                    }
                    Some(RemoteRequestSlot::Control(RequestControl::KeepAlive)) => {}
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

        self.flush_streaming_session(&mut state, None, true, writer)
            .await?;
        writer.finish().await
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
                    Some(TailSlot::Control(_)) => {}
                    Some(TailSlot::End) => {
                        break 'stream;
                    }
                    None => {}
                }
            }

            last_slot = current_last;
        }

        anyhow::ensure!(
            received_bytes > 0,
            "request body did not include audio bytes"
        );
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
                    Some(RemoteRequestSlot::Control(_)) => {}
                    Some(RemoteRequestSlot::End) => {
                        break 'stream;
                    }
                    None => {}
                }
            }

            last_slot = current_last;
        }

        anyhow::ensure!(
            received_bytes > 0,
            "request body did not include audio bytes"
        );
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

impl ListenWebSocketHandler {
    pub fn new(worker: Arc<WorkerState>) -> Self {
        Self { worker }
    }
}

#[async_trait]
impl WebSocketHandler for ListenWebSocketHandler {
    async fn handle_websocket(
        &self,
        req: Request<()>,
        stream: WebSocketStream<TokioIo<Upgraded>>,
    ) -> HandlerResult<()> {
        self.worker.handle_listen_websocket(req, stream).await
    }

    fn can_handle(&self, path: &str) -> bool {
        path == "/v1/listen"
    }
}

impl WsTranscriptState {
    fn new(
        options: ListenOptions,
        request_id: String,
        model_id: String,
        model_info: BTreeMap<String, ModelInfo>,
    ) -> Self {
        Self {
            gap_threshold_ms: options
                .endpointing_ms
                .unwrap_or_else(|| (options.utterance_split_secs.max(0.0) * 1000.0).round() as u64),
            options,
            request_id,
            model_id,
            model_info,
            chunker: AudioChunker::new(
                seconds_to_samples(WS_STREAM_CHUNK_SECONDS),
                seconds_to_samples(WS_STREAM_OVERLAP_SECONDS),
                seconds_to_samples(WS_STREAM_FINAL_MIN_SECONDS),
            ),
            committer: WordCommitter::default(),
            pending_final_words: Vec::new(),
            completed_transcript: String::new(),
            total_samples: 0,
            next_seq: 0,
            last_interim_total_samples: 0,
            speech_started_sent: false,
        }
    }

    fn next_seq(&mut self) -> u32 {
        let next = self.next_seq;
        self.next_seq = self.next_seq.saturating_add(1);
        next
    }

    fn total_duration_secs(&self) -> f64 {
        self.total_samples as f64 / ASR_SAMPLE_RATE as f64
    }

    fn primary_model_info(&self) -> &ModelInfo {
        self.model_info
            .get(&self.model_id)
            .expect("primary model info should exist")
    }

    fn results_metadata(&self) -> WsResultsMetadata {
        WsResultsMetadata {
            request_id: self.request_id.clone(),
            model_info: self.primary_model_info().clone(),
            model_uuid: self.model_id.clone(),
            device_id: 0,
        }
    }

    fn take_auto_finalized_segment(&mut self) -> Option<Vec<CommittedWord>> {
        let boundary_index =
            self.pending_final_words
                .iter()
                .enumerate()
                .find_map(|(index, word)| {
                    let is_last = index + 1 == self.pending_final_words.len();
                    if is_last {
                        return None;
                    }

                    let next = &self.pending_final_words[index + 1];
                    let gap = next.start_ms.saturating_sub(word.end_ms);
                    (gap >= self.gap_threshold_ms || ends_sentence(&word.word)).then_some(index)
                })?;

        Some(self.pending_final_words.drain(..=boundary_index).collect())
    }

    fn take_pending_segment(&mut self) -> Option<Vec<CommittedWord>> {
        (!self.pending_final_words.is_empty()).then(|| self.pending_final_words.drain(..).collect())
    }

    fn append_completed_transcript(&mut self, transcript: &str) {
        append_word(&mut self.completed_transcript, transcript);
    }

    fn preview_words(&self, preview_words: Vec<CommittedWord>) -> Vec<CommittedWord> {
        let mut combined = self.pending_final_words.clone();
        let last_pending_end_ms = combined.last().map(|word| word.end_ms);
        for preview in preview_words {
            if last_pending_end_ms
                .map(|end_ms| preview.end_ms > end_ms.saturating_add(WS_WORD_DEDUPE_EPSILON_MS))
                .unwrap_or(true)
            {
                combined.push(preview);
            }
        }
        combined
    }

    fn preview_transcript(
        &self,
        preview_words: &[CommittedWord],
        fallback_partial: &str,
    ) -> String {
        let preview = if preview_words.is_empty() {
            fallback_partial.to_string()
        } else {
            join_words(preview_words.iter().map(|word| word.word.as_str()))
        };

        if self.completed_transcript.is_empty() {
            preview
        } else if preview.is_empty() {
            self.completed_transcript.clone()
        } else {
            let mut transcript = self.completed_transcript.clone();
            append_word(&mut transcript, &preview);
            transcript
        }
    }
}

fn commit_absolute_words(
    committer: &mut WordCommitter,
    start_sample: usize,
    stable_samples: usize,
    is_final: bool,
    words: &[crate::chunking::TimedWord],
) -> Vec<CommittedWord> {
    committer.commit(start_sample, stable_samples, is_final, words)
}

fn preview_absolute_words(
    start_sample: usize,
    words: &[crate::chunking::TimedWord],
) -> Vec<CommittedWord> {
    let start_ms_offset = ((start_sample as f64 / ASR_SAMPLE_RATE as f64) * 1000.0).round() as u64;
    words
        .iter()
        .filter(|word| !word.word.trim().is_empty())
        .map(|word| CommittedWord {
            index: 0,
            start_ms: start_ms_offset + u64::from(word.start_ms),
            end_ms: start_ms_offset + u64::from(word.end_ms),
            word: word.word.clone(),
        })
        .collect()
}

fn seconds_to_samples(seconds: f32) -> usize {
    ((seconds.max(0.0) * ASR_SAMPLE_RATE as f32).round() as usize).max(1)
}

async fn send_json_event<S: JsonEventSink + Send, T: Serialize>(
    sink: &mut S,
    value: &T,
) -> Result<()> {
    let payload = serde_json::to_string(value)
        .map_err(|error| anyhow::anyhow!("failed to serialize streaming event: {error}"))?;
    sink.send_json(payload).await
}

fn anyhow_to_server_error(error: anyhow::Error) -> ServerError {
    ServerError::Config(error.to_string())
}

#[async_trait]
impl<'a> JsonEventSink for WebSocketJsonSink<'a> {
    async fn send_json(&mut self, json: String) -> Result<()> {
        self.stream
            .send(Message::Text(json.into()))
            .await
            .map_err(|error| anyhow::anyhow!("failed to write websocket event: {error}"))
    }
}

#[async_trait]
impl JsonEventSink for JsonLineResponseWriter {
    async fn send_json(&mut self, json: String) -> Result<()> {
        self.ensure_started().await?;
        let mut payload = json.into_bytes();
        payload.push(b'\n');
        self.append_body(Bytes::from(payload)).await
    }

    async fn finish(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        self.ensure_started().await?;
        match &self.target {
            ResponseCacheTarget::Local(service) => service
                .end_response(self.stream_id)
                .await
                .map_err(anyhow::Error::msg)?,
            ResponseCacheTarget::Remote { client, origin } => {
                client.end_response(origin, self.stream_id).await?;
            }
        }
        self.finished = true;
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
    Control(RequestControl),
    End,
}

#[derive(Clone)]
struct RemoteIngressClient {
    client: Client,
    slot_bytes: usize,
}

impl JsonLineResponseWriter {
    fn local(service: Arc<UploadResponseService>, stream_id: u64, slot_bytes: usize) -> Self {
        Self {
            target: ResponseCacheTarget::Local(service),
            stream_id,
            slot_bytes: slot_bytes.max(1),
            started: false,
            finished: false,
        }
    }

    fn remote(
        client: RemoteIngressClient,
        origin: String,
        stream_id: u64,
        slot_bytes: usize,
    ) -> Self {
        Self {
            target: ResponseCacheTarget::Remote { client, origin },
            stream_id,
            slot_bytes: slot_bytes.max(1),
            started: false,
            finished: false,
        }
    }

    async fn ensure_started(&mut self) -> Result<()> {
        if self.started {
            return Ok(());
        }

        let builder = http::Response::builder()
            .status(StatusCode::OK)
            .header(CONTENT_TYPE, "application/x-ndjson; charset=utf-8")
            .header("cache-control", "no-store");
        let response_head = builder
            .body(())
            .map_err(|error| anyhow::anyhow!("failed to build streaming response head: {error}"))?;
        let headers =
            StreamHeaders::from_response(self.stream_id, &response_head).map_err(|error| {
                anyhow::anyhow!("failed to encode streaming response headers: {error}")
            })?;

        match &self.target {
            ResponseCacheTarget::Local(service) => service
                .write_response_headers(self.stream_id, headers)
                .await
                .map_err(anyhow::Error::msg)?,
            ResponseCacheTarget::Remote { client, origin } => {
                client
                    .write_response_headers(origin, self.stream_id, headers)
                    .await?;
            }
        }

        self.started = true;
        Ok(())
    }

    async fn append_body(&self, body: Bytes) -> Result<()> {
        match &self.target {
            ResponseCacheTarget::Local(service) => {
                for chunk in body.chunks(self.slot_bytes) {
                    if chunk.is_empty() {
                        continue;
                    }
                    service
                        .append_response_body(self.stream_id, Bytes::copy_from_slice(chunk))
                        .await
                        .map_err(anyhow::Error::msg)?;
                }
            }
            ResponseCacheTarget::Remote { client, origin } => {
                client
                    .append_response_body(origin, self.stream_id, body)
                    .await?;
            }
        }
        Ok(())
    }
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
        let body = response.text().await.map_err(|error| {
            anyhow::anyhow!("failed to read stream list from {origin}: {error}")
        })?;
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
                request_last: fields[2].parse().map_err(|error| {
                    anyhow::anyhow!("invalid request_last in {origin}: {error}")
                })?,
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
            .map_err(|error| {
                anyhow::anyhow!("failed to read request_last for {stream_id}: {error}")
            })?;
        let status = response.status();
        let body = response.text().await.map_err(|error| {
            anyhow::anyhow!("failed to read request_last body for {stream_id}: {error}")
        })?;
        anyhow::ensure!(
            status.is_success(),
            "unexpected request_last status {status} for stream {stream_id}: {body}"
        );
        body.trim().parse().map_err(|error| {
            anyhow::anyhow!("invalid request_last for stream {stream_id}: {error}")
        })
    }

    async fn request_headers(&self, origin: &str, stream_id: u64) -> Result<Option<Request<()>>> {
        match self.request_slot(origin, stream_id, 1).await? {
            Some(RemoteRequestSlot::Headers(bytes)) => {
                Ok(Some(decode_request_headers_frame(&bytes)?))
            }
            Some(_) | None => Ok(None),
        }
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
            .map_err(|error| {
                anyhow::anyhow!("failed to fetch stream {stream_id} slot {slot_id}: {error}")
            })?;
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
        let body = response.bytes().await.map_err(|error| {
            anyhow::anyhow!("failed to read stream {stream_id} slot {slot_id}: {error}")
        })?;
        anyhow::ensure!(
            status.is_success(),
            "unexpected slot status {status} for stream {stream_id} slot {slot_id}"
        );

        Ok(Some(match slot_type.as_str() {
            "headers" => RemoteRequestSlot::Headers(body),
            "control-finalize" => RemoteRequestSlot::Control(RequestControl::Finalize),
            "control-keepalive" => RemoteRequestSlot::Control(RequestControl::KeepAlive),
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
            .map_err(|error| {
                anyhow::anyhow!("failed to register reader for {stream_id}: {error}")
            })?;
        anyhow::ensure!(
            response.status().is_success() || response.status() == ReqwestStatusCode::NO_CONTENT,
            "reader registration failed for stream {stream_id} with status {}",
            response.status()
        );
        Ok(())
    }

    async fn unregister_reader(&self, origin: &str, stream_id: u64, worker_id: &str) -> Result<()> {
        let response = self
            .client
            .delete(format!(
                "{origin}/_upload_response/streams/{stream_id}/readers/{worker_id}"
            ))
            .send()
            .await
            .map_err(|error| {
                anyhow::anyhow!("failed to unregister reader for {stream_id}: {error}")
            })?;
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
            .map_err(|error| {
                anyhow::anyhow!("failed to claim response for {stream_id}: {error}")
            })?;
        match response.status() {
            ReqwestStatusCode::OK => Ok(true),
            ReqwestStatusCode::CONFLICT
            | ReqwestStatusCode::NO_CONTENT
            | ReqwestStatusCode::NOT_FOUND => Ok(false),
            status => {
                let body = response.text().await.unwrap_or_default();
                Err(anyhow::anyhow!(
                    "unexpected claim status {status} for stream {stream_id}: {body}"
                ))
            }
        }
    }

    async fn release_response(&self, origin: &str, stream_id: u64, worker_id: &str) -> Result<()> {
        let response = self
            .client
            .delete(format!(
                "{origin}/_upload_response/streams/{stream_id}/response/claim/{worker_id}"
            ))
            .send()
            .await
            .map_err(|error| {
                anyhow::anyhow!("failed to release response for {stream_id}: {error}")
            })?;
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
        let headers = StreamHeaders::from_response(stream_id, &response_head)
            .map_err(|error| anyhow::anyhow!("failed to encode response headers: {error}"))?;
        self.write_response_headers(origin, stream_id, headers)
            .await?;

        if let Some(body) = response.body {
            self.append_response_body(origin, stream_id, body).await?;
        }

        self.end_response(origin, stream_id).await?;

        Ok(())
    }

    async fn write_response_headers(
        &self,
        origin: &str,
        stream_id: u64,
        headers: StreamHeaders,
    ) -> Result<()> {
        let payload = encode_frame(&StreamFrame::Headers(headers));
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
        .await
    }

    async fn append_response_body(&self, origin: &str, stream_id: u64, body: Bytes) -> Result<()> {
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
        Ok(())
    }

    async fn end_response(&self, origin: &str, stream_id: u64) -> Result<()> {
        self.expect_ok(
            self.client
                .put(format!(
                    "{origin}/_upload_response/streams/{stream_id}/response/end"
                ))
                .send()
                .await,
            format!("write response end for stream {stream_id}"),
        )
        .await
    }

    async fn expect_ok(
        &self,
        response: std::result::Result<reqwest::Response, reqwest::Error>,
        context: String,
    ) -> Result<()> {
        let response = response.map_err(|error| anyhow::anyhow!("{context}: {error}"))?;
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::ensure!(
            status.is_success(),
            "{context}: unexpected status {status}: {body}"
        );
        Ok(())
    }
}

fn is_streaming_request(req: &Request<()>) -> bool {
    req.headers()
        .get(INTERNAL_STREAMING_MODE_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.eq_ignore_ascii_case(INTERNAL_STREAMING_MODE_JSONL))
        .unwrap_or(false)
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
