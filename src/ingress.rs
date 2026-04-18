use crate::config::AppConfig;
use crate::ids::ensure_request_id;
use crate::protocol::{
    INTERNAL_REQUEST_ID_HEADER, INTERNAL_STREAMING_MODE_HEADER, INTERNAL_STREAMING_MODE_JSONL,
};
use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use futures_util::{stream::SplitSink, SinkExt, StreamExt};
use http::{header::CONTENT_TYPE, HeaderName, HeaderValue, Request, Response, StatusCode};
use hyper::upgrade::Upgraded;
use hyper_util::rt::TokioIo;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use tokio::time::{interval, timeout, Duration};
use tokio_tungstenite::{tungstenite::Message, WebSocketStream};
use tracing::{debug, error, field, info, info_span, Instrument};
use upload_response::{
    response_content_type, CachedIngress, IngressProxyConfig, RequestControl, TailSlot,
    UploadResponseService, WorkerCapacitySummary,
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
        mut stream_writer: Box<dyn StreamWriter>,
    ) -> HandlerResult<()> {
        let request_id = ensure_request_id(req.headers());
        let method = req.method().clone();
        let path = req.uri().path().to_string();
        let span = info_span!(
            "listen_http_stream",
            request_id,
            role = ?self.config.role,
            method = %method,
            path = %path,
            transport = "http",
            stream_id = field::Empty,
        );

        async move {
            if let Err(error) = reject_json_requests(&req) {
                write_stream_error(
                    &mut *stream_writer,
                    error_response(classify_error(&error), error.to_string()),
                )
                .await?;
                return Ok(());
            }
            let capacity = match self.ensure_worker_capacity().await {
                Ok(capacity) => capacity,
                Err(error) => {
                    write_stream_error(
                        &mut *stream_writer,
                        error_response(classify_error(&error), error.to_string()),
                    )
                    .await?;
                    return Ok(());
                }
            };
            debug!(
                workers = capacity.workers,
                total_inflight = capacity.total_inflight,
                total_available_slots = capacity.total_available_slots,
                "worker capacity accepted request"
            );
            let guard = self
                .cached
                .open_streaming_request()
                .await
                .map_err(anyhow_to_server_error)?;
            let stream_id = guard.stream_id();
            tracing::Span::current().record("stream_id", field::display(stream_id));

            self.write_cached_request_headers(stream_id, &req, true, request_id)
                .await
                .map_err(anyhow_to_server_error)?;
            info!("opened streaming listen request");

            let ingress = self.clone();
            let body_span = info_span!(
                "listen_http_stream_body",
                request_id,
                stream_id,
                transport = "http",
            );
            let body_task = tokio::spawn(
                async move {
                    let result: Result<()> = async {
                        ingress.cached.copy_request_body(stream_id, body).await?;
                        ingress.cached.end_request(stream_id).await?;
                        Ok(())
                    }
                    .await;

                    if let Err(error) = &result {
                        ingress.write_stream_error_response(stream_id, error).await;
                    }

                    result
                }
                .instrument(body_span),
            );

            let proxy_result = self
                .cached
                .proxy_streaming_response(stream_id, stream_writer)
                .await;
            if let Err(error) = body_task.await {
                error!(error = %error, "streaming body task join failed");
            }
            guard.close().await;
            if proxy_result.is_ok() {
                info!("streaming listen request completed");
            }
            proxy_result
        }
        .instrument(span)
        .await
    }

    pub async fn handle_listen_websocket(
        &self,
        req: Request<()>,
        stream: WebSocketStream<TokioIo<Upgraded>>,
    ) -> HandlerResult<()> {
        let request_id = ensure_request_id(req.headers());
        let method = req.method().clone();
        let path = req.uri().path().to_string();
        let span = info_span!(
            "listen_websocket_ingress",
            request_id,
            role = ?self.config.role,
            method = %method,
            path = %path,
            transport = "websocket",
            stream_id = field::Empty,
        );

        async move {
            let capacity = self
                .ensure_worker_capacity()
                .await
                .map_err(anyhow_to_server_error)?;
            debug!(
                workers = capacity.workers,
                total_inflight = capacity.total_inflight,
                total_available_slots = capacity.total_available_slots,
                "worker capacity accepted websocket request"
            );
            let upload_stream = self
                .cached
                .open_streaming_request()
                .await
                .map_err(anyhow_to_server_error)?;
            let stream_id = upload_stream.stream_id();
            tracing::Span::current().record("stream_id", field::display(stream_id));

            self.write_cached_request_headers(stream_id, &req, true, request_id)
                .await
                .map_err(anyhow_to_server_error)?;
            info!("opened websocket listen request");

            let (sink, mut source) = stream.split();
            let response_task = tokio::spawn(
                self.clone()
                    .proxy_websocket_response(stream_id, sink)
                    .instrument(info_span!(
                        "listen_websocket_proxy",
                        request_id,
                        stream_id,
                        transport = "websocket",
                    )),
            );

            while let Some(frame) = source.next().await {
                match frame {
                    Ok(Message::Binary(bytes)) => {
                        if let Err(error) = self
                            .cached
                            .append_request_body_sliced(
                                stream_id,
                                &bytes,
                                self.config.upload_response_config().slot_bytes(),
                            )
                            .await
                        {
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
                                    .cached
                                    .append_request_control(stream_id, RequestControl::KeepAlive)
                                    .await;
                            }
                            "Finalize" => {
                                let _ = self
                                    .cached
                                    .append_request_control(stream_id, RequestControl::Finalize)
                                    .await;
                            }
                            "CloseStream" => {
                                let _ = self.service.end_request(stream_id).await;
                                break;
                            }
                            _ => {}
                        }
                    }
                    Ok(Message::Ping(payload)) => {
                        let _ = payload;
                    }
                    Ok(Message::Pong(_)) => {}
                    Ok(Message::Close(_)) => {
                        let _ = self.service.end_request(stream_id).await;
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

            if let Err(error) = response_task.await {
                error!(error = %error, "websocket response task join failed");
            }
            upload_stream.close().await;
            info!("websocket listen request completed");
            Ok(())
        }
        .instrument(span)
        .await
    }

    async fn handle_listen_inner(
        &self,
        req: Request<()>,
        body: BodyStream,
    ) -> Result<HandlerResponse> {
        let request_id = ensure_request_id(req.headers());
        let method = req.method().clone();
        let path = req.uri().path().to_string();
        let span = info_span!(
            "listen_http_buffered",
            request_id,
            role = ?self.config.role,
            method = %method,
            path = %path,
            transport = "http",
            stream_id = field::Empty,
        );
        async move {
            reject_json_requests(&req)?;
            let capacity = self.ensure_worker_capacity().await?;
            debug!(
                workers = capacity.workers,
                total_inflight = capacity.total_inflight,
                total_available_slots = capacity.total_available_slots,
                "worker capacity accepted buffered request"
            );
            let mut guard = self.cached.open_buffered_request().await?;
            let stream_id = guard.stream_id();
            tracing::Span::current().record("stream_id", field::display(stream_id));

            let result = async {
                self.write_cached_request_headers(stream_id, &req, false, request_id)
                    .await?;

                self.cached.copy_request_body(stream_id, body).await?;
                self.cached.end_request(stream_id).await?;

                let rx = guard.take_response_receiver().ok_or_else(|| {
                    anyhow::anyhow!("response receiver missing for buffered request")
                })?;
                self.cached.await_response(stream_id, rx).await
            }
            .await;

            guard.close().await;
            if result.is_ok() {
                info!("buffered listen request completed");
            }
            result
        }
        .instrument(span)
        .await
    }

    async fn ensure_worker_capacity(&self) -> Result<WorkerCapacitySummary> {
        let workers = self
            .service
            .list_workers(Some(self.config.upload_response_worker_ttl_ms))
            .await;
        let hinted_processing = workers
            .iter()
            .any(|worker| worker_role_hint(&worker.worker_id) == Some(WorkerRoleHint::Processing));
        let hinted_response = workers
            .iter()
            .any(|worker| worker_role_hint(&worker.worker_id) == Some(WorkerRoleHint::Response));
        let has_any_workers = !workers.is_empty();
        let has_processing = hinted_processing || (!hinted_response && has_any_workers);
        let has_response = hinted_response || (!hinted_processing && has_any_workers);
        anyhow::ensure!(has_processing, "no live decoder workers available");
        anyhow::ensure!(has_response, "no live response workers available");
        let summary = WorkerCapacitySummary {
            workers: workers.len(),
            total_max_inflight: workers.iter().map(|worker| worker.max_inflight).sum(),
            total_inflight: workers.iter().map(|worker| worker.inflight).sum(),
            total_available_slots: workers.iter().map(|worker| worker.available_slots).sum(),
        };
        Ok(summary)
    }

    async fn write_cached_request_headers(
        &self,
        stream_id: u64,
        req: &Request<()>,
        streaming: bool,
        request_id: i64,
    ) -> Result<()> {
        self.cached
            .write_request_headers_with(stream_id, req, |headers| {
                if streaming {
                    headers.insert(
                        HeaderName::from_static(INTERNAL_STREAMING_MODE_HEADER),
                        HeaderValue::from_static(INTERNAL_STREAMING_MODE_JSONL),
                    );
                }
                headers.insert(
                    HeaderName::from_static(INTERNAL_REQUEST_ID_HEADER),
                    HeaderValue::from_str(&request_id.to_string())
                        .map_err(|error| anyhow::anyhow!("invalid request id header: {error}"))?,
                );
                Ok(())
            })
            .await
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
        error!(stream_id, error = %error, "writing cached error response");
        self.cached
            .write_handler_response(
                stream_id,
                error_response(classify_error(error), error.to_string()),
            )
            .await;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkerRoleHint {
    Processing,
    Response,
}

fn worker_role_hint(worker_id: &str) -> Option<WorkerRoleHint> {
    let worker_id = worker_id.trim().to_ascii_lowercase();
    if worker_id.contains("decode") || worker_id.contains("decoder") {
        return Some(WorkerRoleHint::Processing);
    }
    if worker_id.contains("response") || worker_id.contains("gpu") || worker_id.contains("worker")
    {
        return Some(WorkerRoleHint::Response);
    }
    None
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
    } else if message.contains("no live worker capacity")
        || message.contains("no live workers available")
        || message.contains("no live decoder workers available")
        || message.contains("no live response workers available")
    {
        StatusCode::SERVICE_UNAVAILABLE
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

async fn write_stream_error(
    stream_writer: &mut dyn StreamWriter,
    handler_response: HandlerResponse,
) -> HandlerResult<()> {
    let mut response = Response::builder().status(handler_response.status);
    if let Some(content_type) = handler_response.content_type {
        response = response.header("content-type", content_type);
    }
    if let Some(etag) = handler_response.etag {
        response = response.header("etag", etag.to_string());
    }
    for (key, value) in handler_response.headers {
        response = response.header(&key, &value);
    }

    stream_writer.send_response(response.body(())?).await?;
    if let Some(body) = handler_response.body {
        stream_writer.send_data(body).await?;
    }
    stream_writer.finish().await
}

fn anyhow_to_server_error(error: anyhow::Error) -> ServerError {
    ServerError::Config(error.to_string())
}
