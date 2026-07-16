use crate::asr::AsrBackend;
use crate::chunking::{AudioChunker, CommittedWord, WordCommitter};
use crate::config::{AppConfig, ASR_SAMPLE_RATE};
use crate::deepgram::{
    append_word, build_response, default_model_info, ends_sentence, join_words,
    words_from_committed, ListenOptions, ModelInfo, Word,
};
#[cfg(feature = "audio-decoder")]
use crate::ids::ensure_request_id;
use crate::ids::{next_request_id, request_id_from_headers};
use crate::pcm::rms_level;
use crate::processing::{decode_processing_head, decode_samples_f32le, DECODE_STAGE};
use crate::protocol::{INTERNAL_STREAMING_MODE_HEADER, INTERNAL_STREAMING_MODE_JSONL};
use anyhow::Result;
use async_trait::async_trait;
#[cfg(feature = "audio-decoder")]
use av_api::linear16::Linear16PcmStream;
use bytes::Bytes;
#[cfg(feature = "audio-decoder")]
use futures_util::{SinkExt, StreamExt};
use gpu_worker::upload_response::{
    run_local_worker_loop, run_remote_worker_loop, LocalJob, LocalJobProcessor, LocalWorkerConfig,
    PipelineSpec, RemoteJob, RemoteJobProcessor, RemoteWorkerConfig, SinkLane, SourceFrame,
    SourceLane,
};
use http::{header::CONTENT_TYPE, Request, StatusCode};
#[cfg(feature = "audio-decoder")]
use hyper::upgrade::Upgraded;
#[cfg(feature = "audio-decoder")]
use hyper_util::rt::TokioIo;
#[cfg(feature = "audio-decoder")]
use serde::Deserialize;
use serde::Serialize;
use sha2::{Digest, Sha256};
#[cfg(feature = "audio-decoder")]
use soundkit::audio_pipeline::audio_to_mono_f32;
#[cfg(feature = "audio-decoder")]
use soundkit_decoder::{DecodeOptions, DecodePipeline, DecodePipelineHandle};
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::time::Duration;
#[cfg(feature = "audio-decoder")]
use tokio_tungstenite::{tungstenite::Message, WebSocketStream};
use tracing::{debug, error, info, info_span, trace, Instrument};
use upload_response::{
    RemoteIngressClient, RequestControl, ResponseCacheWriter, UploadResponseService,
};
use web_service::HandlerResponse;
#[cfg(feature = "audio-decoder")]
use web_service::{BodyStream, HandlerResult, ServerError, WebSocketHandler};

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
const FALLBACK_OVERLAP_WORD_LIMIT: usize = 24;
const FALLBACK_MIN_OVERLAP_WORDS: usize = 2;

#[derive(Clone)]
#[cfg(feature = "audio-decoder")]
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

struct JsonLineResponseWriter {
    writer: ResponseCacheWriter,
    started: bool,
    finished: bool,
}

#[cfg(feature = "audio-decoder")]
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
    fallback_fragments: Vec<String>,
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
#[cfg(feature = "audio-decoder")]
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

    fn local_worker_config(&self) -> LocalWorkerConfig {
        let mut config = LocalWorkerConfig::new(
            self.config.upload_response_worker_id.clone(),
            PipelineSpec {
                source: SourceLane::Stage(DECODE_STAGE.to_string()),
                sink: SinkLane::Response,
            },
        );
        config.heartbeat_stage = "response".to_string();
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
                source: SourceLane::Stage(DECODE_STAGE.to_string()),
                sink: SinkLane::Response,
            },
        );
        config.heartbeat_stage = "response".to_string();
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

    #[cfg(feature = "audio-decoder")]
    pub async fn handle_listen(&self, req: Request<()>, body: BodyStream) -> HandlerResponse {
        match self.handle_listen_inner(req, body).await {
            Ok(response) => response,
            Err(error) => error_response(classify_error(&error), error.to_string()),
        }
    }

    #[cfg(feature = "audio-decoder")]
    async fn handle_listen_websocket(
        &self,
        req: Request<()>,
        mut stream: WebSocketStream<TokioIo<Upgraded>>,
    ) -> HandlerResult<()> {
        let request_id = ensure_request_id(req.headers());
        let options = ListenOptions::from_request(&req, &self.config);
        let sample_rate = options.sample_rate_hz.unwrap_or(ASR_SAMPLE_RATE);
        let channels = options.channels.max(1);
        let span = info_span!(
            "worker_listen_websocket",
            request_id,
            role = ?self.config.role,
            transport = "websocket",
            method = %req.method(),
            path = %req.uri().path(),
            sample_rate,
            channels,
        );

        async move {
            let mut pcm_stream = Linear16PcmStream::new(sample_rate, ASR_SAMPLE_RATE, channels)
                .map_err(|error| ServerError::Config(error.to_string()))?;
            let request_id_text = request_id.to_string();
            let (model_id, model_info) = default_model_info(&options.model);
            let mut state =
                WsTranscriptState::new(options, request_id_text.clone(), model_id, model_info);

            let metadata = WsMetadataEvent {
                event_type: "Metadata",
                request_id: request_id_text,
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
            info!("worker websocket session started");

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
                        let event =
                            serde_json::from_str::<WsClientEvent>(&text).map_err(|error| {
                                ServerError::Config(format!(
                                    "invalid websocket control message: {error}"
                                ))
                            })?;
                        match event.event_type.as_str() {
                            "KeepAlive" => {}
                            "Finalize" => {
                                let mut sink = WebSocketJsonSink {
                                    stream: &mut stream,
                                };
                                self.flush_streaming_session(&mut state, false, &mut sink)
                                    .await
                                    .map_err(anyhow_to_server_error)?;
                            }
                            "CloseStream" => {
                                let mut sink = WebSocketJsonSink {
                                    stream: &mut stream,
                                };
                                self.flush_streaming_session_with_pcm(
                                    &mut state,
                                    &mut pcm_stream,
                                    true,
                                    &mut sink,
                                )
                                .await
                                .map_err(anyhow_to_server_error)?;
                                let _ = stream.close(None).await;
                                info!("worker websocket session closed");
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
                        info!("worker websocket session closed by peer");
                        return Ok(());
                    }
                    Ok(Message::Frame(_)) => {}
                    Err(error) => return Err(ServerError::Handler(Box::new(error))),
                }
            }

            info!("worker websocket session completed");
            Ok(())
        }
        .instrument(span)
        .await
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
            if result.words.is_empty() {
                state.push_fallback_fragment(result.text.trim());
                continue;
            }
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
        close_stream: bool,
        sink: &mut S,
    ) -> Result<()> {
        if let Some(window) = state.chunker.take_final_window() {
            let result = self
                .backend
                .transcribe_window(window.samples, state.next_seq())
                .await?;
            if result.words.is_empty() {
                state.push_fallback_fragment(result.text.trim());
            } else {
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
        }

        if let Some(segment) = state.take_pending_segment() {
            self.send_streaming_result(state, segment, true, close_stream, false, sink)
                .await?;
        } else if let Some(transcript) = state.take_fallback_transcript() {
            self.send_streaming_text_result(state, &transcript, true, close_stream, false, sink)
                .await?;
        }

        Ok(())
    }

    #[cfg(feature = "audio-decoder")]
    async fn flush_streaming_session_with_pcm<S: JsonEventSink + Send>(
        &self,
        state: &mut WsTranscriptState,
        pcm_stream: &mut Linear16PcmStream,
        close_stream: bool,
        sink: &mut S,
    ) -> Result<()> {
        let tail = pcm_stream
            .finish()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        if !tail.is_empty() {
            state.total_samples += tail.len();
            state.chunker.push(&tail);
        }
        self.flush_streaming_session(state, close_stream, sink)
            .await
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

    async fn send_streaming_text_result<S: JsonEventSink + Send>(
        &self,
        state: &mut WsTranscriptState,
        transcript: &str,
        is_final: bool,
        speech_final: bool,
        from_finalize: bool,
        sink: &mut S,
    ) -> Result<()> {
        let transcript = transcript.trim();
        if transcript.is_empty() {
            return Ok(());
        }

        if is_final {
            state.append_completed_transcript(transcript);
        }

        let event = WsResultsEvent {
            event_type: "Results",
            channel_index: [0],
            duration: state.total_duration_secs(),
            start: 0.0,
            is_final,
            speech_final,
            from_finalize,
            channel: WsChannel {
                alternatives: vec![WsAlternative {
                    transcript: transcript.to_string(),
                    confidence: 0.0,
                    words: Vec::new(),
                }],
            },
            metadata: state.results_metadata(),
        };
        send_json_event(sink, &event).await
    }

    async fn run_cache_worker(self: Arc<Self>, service: Arc<UploadResponseService>) {
        run_local_worker_loop(service, self.local_worker_config(), self).await;
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

    #[cfg(feature = "audio-decoder")]
    async fn handle_listen_inner(
        &self,
        req: Request<()>,
        body: BodyStream,
    ) -> Result<HandlerResponse> {
        let request_id = ensure_request_id(req.headers());
        let span = info_span!(
            "worker_listen_http",
            request_id,
            role = ?self.config.role,
            transport = "http",
            method = %req.method(),
            path = %req.uri().path(),
        );
        async move {
            reject_json_requests(&req)?;
            let options = ListenOptions::from_request(&req, &self.config);
            let prepared = self.transcribe_upload(body).await?;
            let fallback_transcript = merge_fallback_fragments(&prepared.fallback_fragments);
            let payload = build_response(
                request_id.to_string(),
                prepared.sha256,
                prepared.duration_secs,
                &prepared.committed_words,
                &fallback_transcript,
                &options,
            );
            info!(
                duration_secs = prepared.duration_secs,
                committed_words = prepared.committed_words.len(),
                fallback_fragments = prepared.fallback_fragments.len(),
                "worker http transcription completed"
            );
            json_response(StatusCode::OK, &payload)
        }
        .instrument(span)
        .await
    }

    async fn process_local_job(&self, job: LocalJob) -> Result<()> {
        let service = Arc::clone(job.service());
        let stream_id = job.stream_id;
        match self.process_cached_stream(&job).await {
            Ok(()) => Ok(()),
            Err(error) => {
                let response = error_response(classify_error(&error), error.to_string());
                service
                    .write_handler_response(stream_id, response)
                    .await
                    .map_err(anyhow::Error::msg)?;
                Ok(())
            }
        }
    }

    async fn process_remote_job(&self, job: RemoteJob) -> Result<()> {
        match self.process_remote_stream(&job).await {
            Ok(()) => Ok(()),
            Err(error) => {
                let response = error_response(classify_error(&error), error.to_string());
                job.client()
                    .write_handler_response(&job.origin, job.stream_id, response)
                    .await?;
                Ok(())
            }
        }
    }

    async fn process_cached_stream(&self, job: &LocalJob) -> Result<()> {
        let service = Arc::clone(job.service());
        let stream_id = job.stream_id;
        let initial_request = job.request().await?;
        let request_id = initial_request
            .as_ref()
            .and_then(|request| request_id_from_headers(request.headers()))
            .unwrap_or_else(next_request_id);
        let span = info_span!(
            "worker_cached_stream",
            request_id,
            stream_id,
            worker_id = %self.config.upload_response_worker_id,
            source = "local_cache",
        );

        async move {
            if let Some(request) = initial_request {
                if is_streaming_request(&request) {
                    let mut writer = JsonLineResponseWriter::local(
                        Arc::clone(&service),
                        stream_id,
                        self.config.upload_response_config().slot_bytes(),
                    );
                    return self.stream_cached_upload(job, request, &mut writer).await;
                }
            }

            let (request, prepared) = self.transcribe_cached_upload(job).await?;
            let options = ListenOptions::from_request(&request, &self.config);
            let fallback_transcript = merge_fallback_fragments(&prepared.fallback_fragments);
            let payload = build_response(
                request_id.to_string(),
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
            info!(
                duration_secs = prepared.duration_secs,
                committed_words = prepared.committed_words.len(),
                fallback_fragments = prepared.fallback_fragments.len(),
                "cached transcription completed"
            );
            Ok(())
        }
        .instrument(span)
        .await
    }

    async fn process_remote_stream(&self, job: &RemoteJob) -> Result<()> {
        let client = job.client();
        let origin = job.origin.as_str();
        let stream_id = job.stream_id;
        let initial_request = job.request().await?;
        let request_id = initial_request
            .as_ref()
            .and_then(|request| request_id_from_headers(request.headers()))
            .unwrap_or_else(next_request_id);
        let span = info_span!(
            "worker_remote_stream",
            request_id,
            stream_id,
            worker_id = %self.config.upload_response_worker_id,
            origin = %origin,
            source = "remote_cache",
        );

        async move {
            if let Some(request) = initial_request {
                if is_streaming_request(&request) {
                    let mut writer = JsonLineResponseWriter::remote(
                        client.clone(),
                        origin.to_string(),
                        stream_id,
                        client.slot_bytes(),
                    );
                    return self.stream_remote_upload(job, request, &mut writer).await;
                }
            }

            let (request, prepared) = self.transcribe_remote_upload(job).await?;
            let options = ListenOptions::from_request(&request, &self.config);
            let fallback_transcript = merge_fallback_fragments(&prepared.fallback_fragments);
            let payload = build_response(
                request_id.to_string(),
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
            info!(
                duration_secs = prepared.duration_secs,
                committed_words = prepared.committed_words.len(),
                fallback_fragments = prepared.fallback_fragments.len(),
                "remote cached transcription completed"
            );
            Ok(())
        }
        .instrument(span)
        .await
    }

    async fn stream_cached_upload(
        &self,
        job: &LocalJob,
        request: Request<()>,
        writer: &mut JsonLineResponseWriter,
    ) -> Result<()> {
        let poll_interval =
            Duration::from_millis(self.config.upload_response_worker_poll_ms.max(1));
        let options = ListenOptions::from_request(&request, &self.config);
        let (model_id, model_info) = default_model_info(&options.model);
        let request_id = request_id_from_headers(request.headers())
            .unwrap_or_else(next_request_id)
            .to_string();
        let mut state = WsTranscriptState::new(options, request_id.clone(), model_id, model_info);
        let mut received_bytes = 0usize;
        let mut reader = job.source_reader_from(1, poll_interval);

        let head = job
            .stage_head()
            .await
            .ok_or_else(|| anyhow::anyhow!("processing head was missing"))?;
        decode_processing_head(&head)?.validate_for_asr()?;

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
        info!("started cached streaming transcription");

        while let Some(frame) = reader.next_frame().await? {
            match frame {
                SourceFrame::StageHead(_) | SourceFrame::RequestHeaders(_) => {}
                SourceFrame::Body(chunk) => {
                    received_bytes += chunk.len();
                    let samples = decode_samples_f32le(&chunk)?;
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
                SourceFrame::Control(RequestControl::Finalize) => {
                    self.flush_streaming_session(&mut state, false, writer)
                        .await?;
                }
                SourceFrame::Control(RequestControl::KeepAlive) => {}
                SourceFrame::End => break,
            }
        }

        anyhow::ensure!(
            received_bytes > 0,
            "request body did not include audio bytes"
        );

        self.flush_streaming_session(&mut state, true, writer)
            .await?;
        info!(
            received_bytes,
            duration_secs = state.total_duration_secs(),
            "finished cached streaming transcription"
        );
        writer.finish().await
    }

    async fn stream_remote_upload(
        &self,
        job: &RemoteJob,
        request: Request<()>,
        writer: &mut JsonLineResponseWriter,
    ) -> Result<()> {
        let poll_interval =
            Duration::from_millis(self.config.upload_response_worker_poll_ms.max(1));
        let origin = job.origin.as_str();
        let options = ListenOptions::from_request(&request, &self.config);
        let (model_id, model_info) = default_model_info(&options.model);
        let request_id = request_id_from_headers(request.headers())
            .unwrap_or_else(next_request_id)
            .to_string();
        let mut state = WsTranscriptState::new(options, request_id.clone(), model_id, model_info);
        let mut received_bytes = 0usize;
        let mut reader = job.source_reader_from(1, poll_interval);

        let head = job
            .stage_head()
            .await?
            .ok_or_else(|| anyhow::anyhow!("processing head was missing"))?;
        decode_processing_head(&head)?.validate_for_asr()?;

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
        info!(origin, "started remote cached streaming transcription");

        while let Some(frame) = reader.next_frame().await? {
            match frame {
                SourceFrame::StageHead(_) | SourceFrame::RequestHeaders(_) => {}
                SourceFrame::Body(chunk) => {
                    received_bytes += chunk.len();
                    let samples = decode_samples_f32le(&chunk)?;
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
                SourceFrame::Control(RequestControl::Finalize) => {
                    self.flush_streaming_session(&mut state, false, writer)
                        .await?;
                }
                SourceFrame::Control(RequestControl::KeepAlive) => {}
                SourceFrame::End => break,
            }
        }

        anyhow::ensure!(
            received_bytes > 0,
            "request body did not include audio bytes"
        );

        self.flush_streaming_session(&mut state, true, writer)
            .await?;
        info!(
            origin,
            received_bytes,
            duration_secs = state.total_duration_secs(),
            "finished remote cached streaming transcription"
        );
        writer.finish().await
    }

    async fn transcribe_cached_upload(
        &self,
        job: &LocalJob,
    ) -> Result<(Request<()>, PreparedTranscript)> {
        let stream_id = job.stream_id;
        let poll_interval =
            Duration::from_millis(self.config.upload_response_worker_poll_ms.max(1));
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
        let mut reader = job.source_reader_from(1, poll_interval);

        let head = job
            .stage_head()
            .await
            .ok_or_else(|| anyhow::anyhow!("processing head was missing"))?;
        decode_processing_head(&head)?.validate_for_asr()?;

        while let Some(frame) = reader.next_frame().await? {
            match frame {
                SourceFrame::StageHead(_) | SourceFrame::RequestHeaders(_) => {}
                SourceFrame::Body(chunk) => {
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
                SourceFrame::Control(_) => {}
                SourceFrame::End => break,
            }
        }

        anyhow::ensure!(
            received_bytes > 0,
            "request body did not include audio bytes"
        );
        let request = job
            .request()
            .await?
            .ok_or_else(|| anyhow::anyhow!("request headers were missing"))?;
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

        let prepared = PreparedTranscript {
            committed_words,
            fallback_fragments,
            duration_secs: total_samples as f64 / ASR_SAMPLE_RATE as f64,
            sha256: format!("{:x}", hasher.finalize()),
        };
        debug!(
            stream_id,
            received_bytes,
            duration_secs = prepared.duration_secs,
            committed_words = prepared.committed_words.len(),
            fallback_fragments = prepared.fallback_fragments.len(),
            "finished cached upload transcription"
        );
        Ok((request, prepared))
    }

    async fn transcribe_remote_upload(
        &self,
        job: &RemoteJob,
    ) -> Result<(Request<()>, PreparedTranscript)> {
        let origin = job.origin.as_str();
        let stream_id = job.stream_id;
        let poll_interval =
            Duration::from_millis(self.config.upload_response_worker_poll_ms.max(1));
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
        let mut reader = job.source_reader_from(1, poll_interval);

        let head = job
            .stage_head()
            .await?
            .ok_or_else(|| anyhow::anyhow!("processing head was missing"))?;
        decode_processing_head(&head)?.validate_for_asr()?;

        while let Some(frame) = reader.next_frame().await? {
            match frame {
                SourceFrame::StageHead(_) | SourceFrame::RequestHeaders(_) => {}
                SourceFrame::Body(chunk) => {
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
                SourceFrame::Control(_) => {}
                SourceFrame::End => break,
            }
        }

        anyhow::ensure!(
            received_bytes > 0,
            "request body did not include audio bytes"
        );
        let request = job
            .request()
            .await?
            .ok_or_else(|| anyhow::anyhow!("request headers were missing"))?;
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

        let prepared = PreparedTranscript {
            committed_words,
            fallback_fragments,
            duration_secs: total_samples as f64 / ASR_SAMPLE_RATE as f64,
            sha256: format!("{:x}", hasher.finalize()),
        };
        debug!(
            origin,
            stream_id,
            received_bytes,
            duration_secs = prepared.duration_secs,
            committed_words = prepared.committed_words.len(),
            fallback_fragments = prepared.fallback_fragments.len(),
            "finished remote upload transcription"
        );
        Ok((request, prepared))
    }

    #[cfg(feature = "audio-decoder")]
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

        let prepared = PreparedTranscript {
            committed_words,
            fallback_fragments,
            duration_secs: total_samples as f64 / ASR_SAMPLE_RATE as f64,
            sha256: format!("{:x}", hasher.finalize()),
        };
        debug!(
            received_bytes,
            duration_secs = prepared.duration_secs,
            committed_words = prepared.committed_words.len(),
            fallback_fragments = prepared.fallback_fragments.len(),
            "finished direct upload transcription"
        );
        Ok(prepared)
    }

    #[cfg(feature = "audio-decoder")]
    fn new_decoder() -> DecodePipelineHandle {
        DecodePipeline::spawn_with_options(DecodeOptions {
            output_bits_per_sample: Some(16),
            output_sample_rate: Some(ASR_SAMPLE_RATE),
            output_channels: Some(1),
        })
    }

    #[cfg(feature = "audio-decoder")]
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
                    self.drain_decoder(decoder, chunker, total_samples)?;
                    tokio::task::yield_now().await;
                }
                Err(error) => return Err(anyhow::anyhow!("decoder send failed: {error}")),
            }
        }

        self.drain_decoder(decoder, chunker, total_samples)
    }

    #[cfg(feature = "audio-decoder")]
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
                    self.drain_decoder(decoder, chunker, total_samples)?;
                    tokio::task::yield_now().await;
                }
                Err(error) => return Err(anyhow::anyhow!("decoder EOF send failed: {error}")),
            }
        }

        while let Some(output) = decoder.recv() {
            let audio = output.map_err(|error| anyhow::anyhow!("decode failed: {error}"))?;
            let samples = audio_to_mono_f32(&audio).map_err(anyhow::Error::msg)?;
            *total_samples += samples.len();
            chunker.push(&samples);
        }

        Ok(())
    }

    #[cfg(feature = "audio-decoder")]
    fn drain_decoder(
        &self,
        decoder: &mut DecodePipelineHandle,
        chunker: &mut AudioChunker,
        total_samples: &mut usize,
    ) -> Result<()> {
        while let Some(output) = decoder.try_recv() {
            let audio = output.map_err(|error| anyhow::anyhow!("decode failed: {error}"))?;
            let samples = audio_to_mono_f32(&audio).map_err(anyhow::Error::msg)?;
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
        let samples = decode_samples_f32le(chunk)?;
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

        trace!(
            seq,
            start_sample,
            sample_count = samples.len(),
            is_final,
            stable_samples,
            "transcribing window"
        );

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

#[cfg(feature = "audio-decoder")]
impl ListenWebSocketHandler {
    pub fn new(worker: Arc<WorkerState>) -> Self {
        Self { worker }
    }
}

#[async_trait]
#[cfg(feature = "audio-decoder")]
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
            fallback_fragments: Vec::new(),
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

    fn push_fallback_fragment(&mut self, transcript: &str) {
        let transcript = transcript.trim();
        if transcript.is_empty() {
            return;
        }
        if self
            .fallback_fragments
            .last()
            .map(|last| last == transcript)
            .unwrap_or(false)
        {
            return;
        }
        self.fallback_fragments.push(transcript.to_string());
    }

    fn take_fallback_transcript(&mut self) -> Option<String> {
        if self.fallback_fragments.is_empty() {
            return None;
        }
        let fragments = self.fallback_fragments.drain(..).collect::<Vec<_>>();
        Some(merge_fallback_fragments(&fragments))
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

fn merge_fallback_fragments(fragments: &[String]) -> String {
    let mut merged = Vec::<String>::new();
    for fragment in fragments {
        let words = fragment
            .split_whitespace()
            .filter(|word| !word.is_empty())
            .collect::<Vec<_>>();
        if words.is_empty() {
            continue;
        }
        if merged.is_empty() {
            merged.extend(words.into_iter().map(ToOwned::to_owned));
            continue;
        }

        let overlap = fallback_word_overlap(&merged, &words);
        merged.extend(words.into_iter().skip(overlap).map(ToOwned::to_owned));
    }
    merged.join(" ")
}

fn fallback_word_overlap(existing: &[String], incoming: &[&str]) -> usize {
    let max_overlap = existing
        .len()
        .min(incoming.len())
        .min(FALLBACK_OVERLAP_WORD_LIMIT);
    for overlap in (FALLBACK_MIN_OVERLAP_WORDS..=max_overlap).rev() {
        let start = existing.len() - overlap;
        let matches = existing[start..]
            .iter()
            .zip(incoming.iter().take(overlap))
            .all(|(left, right)| normalize_fallback_word(left) == normalize_fallback_word(right));
        if matches {
            return overlap;
        }
    }
    0
}

fn normalize_fallback_word(word: &str) -> String {
    word.trim_matches(|ch: char| !ch.is_alphanumeric())
        .to_ascii_lowercase()
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

#[async_trait]
impl LocalJobProcessor for WorkerState {
    async fn process(&self, job: LocalJob) -> Result<()> {
        self.process_local_job(job).await
    }
}

#[async_trait]
impl RemoteJobProcessor for WorkerState {
    async fn process(&self, job: RemoteJob) -> Result<()> {
        self.process_remote_job(job).await
    }
}

async fn send_json_event<S: JsonEventSink + Send, T: Serialize>(
    sink: &mut S,
    value: &T,
) -> Result<()> {
    let payload = serde_json::to_string(value)
        .map_err(|error| anyhow::anyhow!("failed to serialize streaming event: {error}"))?;
    sink.send_json(payload).await
}

#[cfg(feature = "audio-decoder")]
fn anyhow_to_server_error(error: anyhow::Error) -> ServerError {
    ServerError::Config(error.to_string())
}

#[async_trait]
#[cfg(feature = "audio-decoder")]
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
        self.writer.send_body(Bytes::from(payload)).await
    }

    async fn finish(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        self.ensure_started().await?;
        self.writer.finish().await?;
        self.finished = true;
        Ok(())
    }
}

impl JsonLineResponseWriter {
    fn local(service: Arc<UploadResponseService>, stream_id: u64, slot_bytes: usize) -> Self {
        Self {
            writer: ResponseCacheWriter::local(service, stream_id, slot_bytes),
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
            writer: ResponseCacheWriter::remote(client, origin, stream_id, slot_bytes),
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
        self.writer.ensure_started(response_head).await?;
        self.started = true;
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

#[cfg(feature = "audio-decoder")]
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

#[cfg(test)]
mod fallback_tests {
    use super::merge_fallback_fragments;

    #[test]
    fn merges_overlapping_fallback_fragments() {
        let fragments = vec![
            "And so, my fellow Americans, ask not what your".to_string(),
            "What your country can do for you, ask what you can do for your country.".to_string(),
        ];

        assert_eq!(
            merge_fallback_fragments(&fragments),
            "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
        );
    }

    #[test]
    fn keeps_distinct_fallback_fragments() {
        let fragments = vec![
            "The first sentence is here.".to_string(),
            "A different sentence follows.".to_string(),
        ];

        assert_eq!(
            merge_fallback_fragments(&fragments),
            "The first sentence is here. A different sentence follows."
        );
    }
}
