use crate::config::{AppConfig, ASR_SAMPLE_RATE};
use crate::deepgram::ListenOptions;
use av_api::cached_audio::CachedMonoPcmWriter;
use av_api::linear16::Linear16PcmStream;
use crate::protocol::{INTERNAL_STREAMING_MODE_HEADER, INTERNAL_STREAMING_MODE_JSONL};
use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use futures_util::{stream::SplitSink, SinkExt, StreamExt};
use http::{header::CONTENT_TYPE, HeaderName, HeaderValue, Request, StatusCode};
use hyper::upgrade::Upgraded;
use hyper_util::rt::TokioIo;
use serde::Deserialize;
use serde_json::json;
use soundkit_decoder::{DecodeOptions, DecodePipeline, DecodePipelineHandle};
use std::sync::Arc;
use tokio::time::{interval, timeout, Duration};
use tokio_tungstenite::{tungstenite::Message, WebSocketStream};
use upload_response::{
    response_content_type, CachedIngress, IngressProxyConfig, RequestControl, TailSlot,
    UploadResponseService,
};
use web_service::{
    BodyStream, HandlerResponse, HandlerResult, ServerError, StreamWriter, WebSocketHandler,
};

#[derive(Clone)]
pub struct ListenIngress {
    config: AppConfig,
    cached: CachedIngress,
    service: Arc<UploadResponseService>,
}

#[derive(Clone)]
pub struct ListenIngressWebSocketHandler {
    ingress: Arc<ListenIngress>,
}

#[derive(Debug, Deserialize)]
struct WsClientEvent {
    #[serde(rename = "type")]
    event_type: String,
}

impl ListenIngress {
    pub fn new(config: AppConfig, service: Arc<UploadResponseService>) -> Self {
        let cached = CachedIngress::new(
            service.clone(),
            IngressProxyConfig {
                response_timeout_ms: config.upload_response_timeout_ms,
                watch_poll_ms: config.upload_response_watch_poll_ms,
            },
        );
        Self {
            config,
            cached,
            service,
        }
    }

    pub async fn handle_listen(&self, req: Request<()>, body: BodyStream) -> HandlerResponse {
        match self.handle_listen_inner(req, body).await {
            Ok(response) => response,
            Err(error) => error_response(classify_error(&error), error.to_string()),
        }
    }

    pub async fn handle_listen_stream(
        &self,
        req: Request<()>,
        body: BodyStream,
        stream_writer: Box<dyn StreamWriter>,
    ) -> HandlerResult<()> {
        reject_json_requests(&req).map_err(anyhow_to_server_error)?;
        let options = ListenOptions::from_request(&req, &self.config);

        let guard = self
            .cached
            .open_streaming_request()
            .await
            .map_err(anyhow_to_server_error)?;
        let stream_id = guard.stream_id();

        self.write_cached_request_headers(stream_id, &req, true)
            .await
            .map_err(anyhow_to_server_error)?;

        let ingress = self.clone();
        let body_task = tokio::spawn(async move {
            let result: Result<()> = async {
                ingress
                    .transcode_request_body(stream_id, body, &options)
                    .await?;
                ingress.cached.end_request(stream_id).await?;
                Ok(())
            }
            .await;

            if let Err(error) = &result {
                ingress.write_stream_error_response(stream_id, error).await;
            }

            result
        });

        let proxy_result = self.cached.proxy_streaming_response(stream_id, stream_writer).await;
        let _ = body_task.await;
        guard.close().await;
        proxy_result
    }

    pub async fn handle_listen_websocket(
        &self,
        req: Request<()>,
        stream: WebSocketStream<TokioIo<Upgraded>>,
    ) -> HandlerResult<()> {
        let options = ListenOptions::from_request(&req, &self.config);
        let sample_rate = options.sample_rate_hz.unwrap_or(ASR_SAMPLE_RATE);
        let channels = options.channels.max(1);
        let mut pcm_stream = Linear16PcmStream::new(sample_rate, ASR_SAMPLE_RATE, channels)
            .map_err(|error| ServerError::Config(error.to_string()))?;

        let upload_stream = self
            .cached
            .open_streaming_request()
            .await
            .map_err(anyhow_to_server_error)?;
        let stream_id = upload_stream.stream_id();

        self.write_cached_request_headers(stream_id, &req, true)
            .await
            .map_err(anyhow_to_server_error)?;

        let (sink, mut source) = stream.split();
        let response_task = tokio::spawn(self.clone().proxy_websocket_response(stream_id, sink));

        while let Some(frame) = source.next().await {
            match frame {
                Ok(Message::Binary(bytes)) => {
                    let samples = match pcm_stream.push(&bytes) {
                        Ok(samples) => samples,
                        Err(error) => {
                            let error = anyhow::anyhow!("raw linear16 decode failed: {error}");
                            self.write_stream_error_response(stream_id, &error).await;
                            let _ = self.service.end_request(stream_id).await;
                            break;
                        }
                    };

                    if let Err(error) = self.append_samples(stream_id, &samples).await {
                        self.write_stream_error_response(stream_id, &error).await;
                        let _ = self.service.end_request(stream_id).await;
                        break;
                    }
                }
                Ok(Message::Text(text)) => {
                    let event =
                        serde_json::from_str::<WsClientEvent>(&text).unwrap_or(WsClientEvent {
                            event_type: String::new(),
                        });
                    match event.event_type.as_str() {
                        "KeepAlive" => {
                            let _ = self
                                .service
                                .append_request_control(stream_id, RequestControl::KeepAlive)
                                .await;
                        }
                        "Finalize" => {
                            let _ = self
                                .service
                                .append_request_control(stream_id, RequestControl::Finalize)
                                .await;
                        }
                        "CloseStream" => {
                            let _ = self
                                .finish_linear16_request(stream_id, &mut pcm_stream)
                                .await;
                            break;
                        }
                        _ => {}
                    }
                }
                Ok(Message::Ping(payload)) => {
                    // The tungstenite stream is already split; respond through the cached path only.
                    let _ = payload;
                }
                Ok(Message::Pong(_)) => {}
                Ok(Message::Close(_)) => {
                    let _ = self
                        .finish_linear16_request(stream_id, &mut pcm_stream)
                        .await;
                    break;
                }
                Ok(Message::Frame(_)) => {}
                Err(error) => {
                    let error = anyhow::anyhow!("websocket receive failed: {error}");
                    self.write_stream_error_response(stream_id, &error).await;
                    let _ = self.service.end_request(stream_id).await;
                    break;
                }
            }
        }

        let _ = response_task.await;
        upload_stream.close().await;
        Ok(())
    }

    async fn handle_listen_inner(
        &self,
        req: Request<()>,
        body: BodyStream,
    ) -> Result<HandlerResponse> {
        reject_json_requests(&req)?;
        let options = ListenOptions::from_request(&req, &self.config);

        let mut guard = self.cached.open_buffered_request().await?;
        let stream_id = guard.stream_id();

        let result = async {
            self.write_cached_request_headers(stream_id, &req, false)
                .await?;

            self.transcode_request_body(stream_id, body, &options)
                .await?;
            self.cached.end_request(stream_id).await?;

            let rx = guard
                .take_response_receiver()
                .ok_or_else(|| anyhow::anyhow!("response receiver missing for buffered request"))?;
            self.cached.await_response(stream_id, rx).await
        }
        .await;

        guard.close().await;
        result
    }

    async fn write_cached_request_headers(
        &self,
        stream_id: u64,
        req: &Request<()>,
        streaming: bool,
    ) -> Result<()> {
        self.cached
            .write_request_headers_with(stream_id, req, |headers| {
                if streaming {
                    headers.insert(
                        HeaderName::from_static(INTERNAL_STREAMING_MODE_HEADER),
                        HeaderValue::from_static(INTERNAL_STREAMING_MODE_JSONL),
                    );
                }
                Ok(())
            })
            .await
    }

    async fn transcode_request_body(
        &self,
        stream_id: u64,
        mut body: BodyStream,
        options: &ListenOptions,
    ) -> Result<()> {
        if options.raw_linear16() {
            return self
                .transcode_linear16_request_body(stream_id, body, options)
                .await;
        }

        let mut decoder = Self::new_decoder();
        let mut received_bytes = 0usize;

        while let Some(next) = body.next().await {
            let chunk =
                next.map_err(|error| anyhow::anyhow!("failed to read request body: {error}"))?;
            if chunk.is_empty() {
                continue;
            }

            received_bytes += chunk.len();
            self.send_decoder_bytes(&mut decoder, chunk).await?;
            self.flush_decoder(stream_id, &mut decoder, false).await?;
        }

        anyhow::ensure!(
            received_bytes > 0,
            "request body did not include audio bytes"
        );

        self.send_decoder_bytes(&mut decoder, Bytes::new()).await?;
        self.flush_decoder(stream_id, &mut decoder, true).await?;
        Ok(())
    }

    async fn transcode_linear16_request_body(
        &self,
        stream_id: u64,
        mut body: BodyStream,
        options: &ListenOptions,
    ) -> Result<()> {
        let sample_rate = options
            .sample_rate_hz
            .ok_or_else(|| anyhow::anyhow!("sample_rate is required for encoding=linear16"))?;
        let channels = options.channels.max(1);
        let mut received_bytes = 0usize;
        let mut pcm = Linear16PcmStream::new(sample_rate, ASR_SAMPLE_RATE, channels)
            .map_err(|error| anyhow::anyhow!("{error}"))?;

        while let Some(next) = body.next().await {
            let chunk =
                next.map_err(|error| anyhow::anyhow!("failed to read request body: {error}"))?;
            if chunk.is_empty() {
                continue;
            }

            received_bytes += chunk.len();
            let samples = pcm
                .push(&chunk)
                .map_err(|error| anyhow::anyhow!("raw linear16 decode failed: {error}"))?;
            self.append_samples(stream_id, &samples).await?;
        }

        anyhow::ensure!(
            received_bytes > 0,
            "request body did not include audio bytes"
        );

        let tail = pcm
            .finish()
            .map_err(|error| anyhow::anyhow!("raw linear16 decode failed: {error}"))?;
        self.append_samples(stream_id, &tail).await?;
        Ok(())
    }

    async fn finish_linear16_request(
        &self,
        stream_id: u64,
        pcm_stream: &mut Linear16PcmStream,
    ) -> Result<()> {
        let tail = pcm_stream
            .finish()
            .map_err(|error| anyhow::anyhow!("raw linear16 decode failed: {error}"))?;
        self.append_samples(stream_id, &tail).await?;
        self.service
            .end_request(stream_id)
            .await
            .map_err(anyhow::Error::msg)
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

    async fn flush_decoder(
        &self,
        stream_id: u64,
        decoder: &mut DecodePipelineHandle,
        blocking: bool,
    ) -> Result<()> {
        if blocking {
            while let Some(output) = decoder.recv() {
                let audio = output.map_err(|error| anyhow::anyhow!("decode failed: {error}"))?;
                self.append_audio(stream_id, &audio).await?;
            }
            return Ok(());
        }

        while let Some(output) = decoder.try_recv() {
            let audio = output.map_err(|error| anyhow::anyhow!("decode failed: {error}"))?;
            self.append_audio(stream_id, &audio).await?;
        }
        Ok(())
    }

    fn cached_pcm_writer(&self, stream_id: u64) -> CachedMonoPcmWriter {
        CachedMonoPcmWriter::new(
            self.service.clone(),
            stream_id,
            self.config.upload_response_config().slot_bytes().max(1),
        )
    }

    async fn append_audio(
        &self,
        stream_id: u64,
        audio: &soundkit::audio_types::AudioData,
    ) -> Result<()> {
        self.cached_pcm_writer(stream_id)
            .append_audio(audio)
            .await
            .map_err(anyhow::Error::msg)
    }

    async fn append_samples(&self, stream_id: u64, samples: &[f32]) -> Result<()> {
        self.cached_pcm_writer(stream_id)
            .append_samples(samples)
            .await
            .map_err(anyhow::Error::msg)
    }

    async fn proxy_websocket_response(
        self,
        stream_id: u64,
        mut sink: SplitSink<WebSocketStream<TokioIo<Upgraded>>, Message>,
    ) -> HandlerResult<()> {
        let timeout_duration = Duration::from_millis(self.config.upload_response_timeout_ms);
        timeout(timeout_duration, async {
            let mut poll = interval(Duration::from_millis(
                self.config.upload_response_watch_poll_ms.max(1),
            ));
            let mut last_slot = 0usize;
            let mut headers_seen = false;
            let mut stream_json_lines = false;
            let mut line_buffer = Vec::new();

            loop {
                poll.tick().await;

                if !headers_seen {
                    if let Some(headers) = self.service.get_response_headers(stream_id).await {
                        stream_json_lines = response_content_type(&headers)
                            .map(|value| value.contains("ndjson"))
                            .unwrap_or(false);
                        headers_seen = true;
                        last_slot = 1;
                    } else {
                        continue;
                    }
                }

                let current_last = self.service.response_last(stream_id).unwrap_or(0);
                if current_last <= last_slot {
                    continue;
                }

                for slot_id in (last_slot + 1)..=current_last {
                    match self.service.tail_response(stream_id, slot_id).await {
                        Some(TailSlot::Body(bytes)) if stream_json_lines => {
                            line_buffer.extend_from_slice(&bytes);
                            while let Some(position) =
                                line_buffer.iter().position(|byte| *byte == b'\n')
                            {
                                let line = line_buffer.drain(..=position).collect::<Vec<u8>>();
                                let line = trim_newline(&line);
                                if line.is_empty() {
                                    continue;
                                }
                                sink.send(Message::Text(
                                    String::from_utf8_lossy(line).into_owned().into(),
                                ))
                                .await
                                .map_err(|error| ServerError::Handler(Box::new(error)))?;
                            }
                        }
                        Some(TailSlot::Body(bytes)) => {
                            send_ws_body_chunk(&mut sink, bytes).await?;
                        }
                        Some(TailSlot::End) => {
                            if stream_json_lines && !line_buffer.is_empty() {
                                let line = trim_newline(&line_buffer);
                                if !line.is_empty() {
                                    sink.send(Message::Text(
                                        String::from_utf8_lossy(line).into_owned().into(),
                                    ))
                                    .await
                                    .map_err(|error| ServerError::Handler(Box::new(error)))?;
                                }
                            }
                            let _ = sink.close().await;
                            return Ok(());
                        }
                        _ => {}
                    }
                }

                last_slot = current_last;
            }
        })
        .await
        .map_err(|_| ServerError::Config("response timeout".into()))?
    }

    async fn write_stream_error_response(&self, stream_id: u64, error: &anyhow::Error) {
        self.cached
            .write_handler_response(
                stream_id,
                error_response(classify_error(error), error.to_string()),
            )
            .await;
    }

    fn new_decoder() -> DecodePipelineHandle {
        DecodePipeline::spawn_with_options(DecodeOptions {
            output_bits_per_sample: Some(16),
            output_sample_rate: Some(ASR_SAMPLE_RATE),
            output_channels: Some(1),
        })
    }
}

impl ListenIngressWebSocketHandler {
    pub fn new(ingress: Arc<ListenIngress>) -> Self {
        Self { ingress }
    }
}

#[async_trait]
impl WebSocketHandler for ListenIngressWebSocketHandler {
    async fn handle_websocket(
        &self,
        req: Request<()>,
        stream: WebSocketStream<TokioIo<Upgraded>>,
    ) -> HandlerResult<()> {
        self.ingress.handle_listen_websocket(req, stream).await
    }

    fn can_handle(&self, path: &str) -> bool {
        path == "/v1/listen"
    }
}

async fn send_ws_body_chunk(
    sink: &mut SplitSink<WebSocketStream<TokioIo<Upgraded>>, Message>,
    bytes: Bytes,
) -> HandlerResult<()> {
    if let Ok(text) = std::str::from_utf8(&bytes) {
        sink.send(Message::Text(text.to_string().into()))
            .await
            .map_err(|error| ServerError::Handler(Box::new(error)))
    } else {
        sink.send(Message::Binary(bytes))
            .await
            .map_err(|error| ServerError::Handler(Box::new(error)))
    }
}

fn trim_newline(bytes: &[u8]) -> &[u8] {
    let bytes = bytes.strip_suffix(b"\n").unwrap_or(bytes);
    bytes.strip_suffix(b"\r").unwrap_or(bytes)
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

fn error_response(status: StatusCode, message: String) -> HandlerResponse {
    HandlerResponse {
        status,
        body: Some(Bytes::from(
            serde_json::to_vec(&json!({ "error": message }))
                .unwrap_or_else(|_| b"{\"error\":\"serialization failure\"}".to_vec()),
        )),
        content_type: Some("application/json".into()),
        headers: vec![("cache-control".into(), "no-store".into())],
        etag: None,
    }
}

fn anyhow_to_server_error(error: anyhow::Error) -> ServerError {
    ServerError::Config(error.to_string())
}
