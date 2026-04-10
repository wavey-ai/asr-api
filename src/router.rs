use crate::config::AppConfig;
use crate::worker::WorkerState;
use async_trait::async_trait;
use bytes::Bytes;
use futures_util::StreamExt;
use http::{Method, Request, Response, StatusCode};
use std::sync::Arc;
use tokio::time::{sleep, Duration, Instant};
use tracing::{debug, warn};
use upload_response::{TailSlot, UploadResponseConfig, UploadResponseService};
use web_service::{
    BodyStream, HandlerResponse, HandlerResult, Router, ServerError, StreamWriter,
    WebSocketHandler, WebTransportHandler,
};

#[derive(Clone)]
pub struct AppRouter {
    config: AppConfig,
    service: Arc<UploadResponseService>,
    workers: Arc<WorkerState>,
}

impl AppRouter {
    pub fn new(config: AppConfig, workers: Arc<WorkerState>) -> Self {
        let service = Arc::new(UploadResponseService::new(UploadResponseConfig {
            num_streams: config.num_streams,
            slot_size_kb: config.slot_size_kb,
            slots_per_stream: config.slots_per_stream,
            response_timeout_ms: config.response_timeout_ms,
        }));
        Self {
            config,
            service,
            workers,
        }
    }

    async fn handle_health(&self) -> HandlerResult<HandlerResponse> {
        Ok(HandlerResponse {
            status: StatusCode::OK,
            body: Some(Bytes::from(
                "{\"status\":\"ok\",\"service\":\"bag-of-beats\"}",
            )),
            content_type: Some("application/json".to_string()),
            headers: vec![("cache-control".to_string(), "no-store".to_string())],
            etag: None,
        })
    }

    async fn handle_options(&self) -> HandlerResult<HandlerResponse> {
        Ok(HandlerResponse {
            status: StatusCode::NO_CONTENT,
            body: None,
            content_type: None,
            headers: vec![("cache-control".to_string(), "no-store".to_string())],
            etag: None,
        })
    }

    async fn handle_transcribe_stream(
        &self,
        req: Request<()>,
        body: BodyStream,
        mut stream_writer: Box<dyn StreamWriter>,
    ) -> HandlerResult<()> {
        let _permit = self
            .service
            .acquire_stream()
            .await
            .map_err(ServerError::Config)?;
        let stream_id = self.service.next_id();

        let headers = http_pack::stream::StreamHeaders::from_request(stream_id, &req)
            .map_err(|error| ServerError::Config(error.to_string()))?;
        self.service
            .write_request_headers(stream_id, headers)
            .await
            .map_err(ServerError::Config)?;

        let service = Arc::clone(&self.service);
        let worker_state = Arc::clone(&self.workers);
        let upload_service = Arc::clone(&service);
        let upload_fut = stream_request_body(upload_service, stream_id, body);
        let worker_fut = worker_state.process_stream(service, stream_id);
        let response_fut = self.stream_response(stream_id, &mut stream_writer);

        let (upload_result, _, stream_result) = tokio::join!(upload_fut, worker_fut, response_fut);

        if let Err(error) = upload_result {
            return Err(ServerError::Config(error));
        }

        stream_result.map_err(|error| {
            warn!(stream_id, error = %error, "response streaming failed");
            error
        })
    }

    async fn stream_response(
        &self,
        stream_id: u64,
        stream_writer: &mut Box<dyn StreamWriter>,
    ) -> HandlerResult<()> {
        let response_headers = wait_for_response_headers(
            Arc::clone(&self.service),
            stream_id,
            Duration::from_millis(self.config.response_timeout_ms),
        )
        .await?;

        let mut response = Response::builder().status(response_headers.status);
        for header in response_headers.headers {
            let name = String::from_utf8_lossy(&header.name).to_string();
            let value = String::from_utf8_lossy(&header.value).to_string();
            response = response.header(name, value);
        }
        let response = response.body(()).map_err(ServerError::Http)?;
        stream_writer.send_response(response).await?;

        let mut last_slot = 1usize;
        loop {
            let current_last = wait_for_response_slot(
                Arc::clone(&self.service),
                stream_id,
                last_slot,
                Duration::from_millis(self.config.response_timeout_ms),
            )
            .await?;

            for slot_id in (last_slot + 1)..=current_last {
                match self.service.tail_response(stream_id, slot_id).await {
                    Some(TailSlot::Body(data)) => {
                        debug!(
                            stream_id,
                            slot_id,
                            len = data.len(),
                            "streaming response chunk"
                        );
                        stream_writer.send_data(data).await?;
                    }
                    Some(TailSlot::End) => {
                        stream_writer.finish().await?;
                        return Ok(());
                    }
                    Some(TailSlot::Headers(_)) | None => {}
                }
            }
            last_slot = current_last;
        }
    }
}

#[async_trait]
impl Router for AppRouter {
    async fn route(&self, req: Request<()>) -> HandlerResult<HandlerResponse> {
        match (req.method(), req.uri().path()) {
            (&Method::GET, "/") | (&Method::GET, "/health") | (&Method::GET, "/healthz") => {
                self.handle_health().await
            }
            (&Method::OPTIONS, "/transcribe") => self.handle_options().await,
            _ => Ok(HandlerResponse {
                status: StatusCode::NOT_FOUND,
                body: Some(Bytes::from("{\"error\":\"not found\"}")),
                content_type: Some("application/json".to_string()),
                headers: vec![("cache-control".to_string(), "no-store".to_string())],
                etag: None,
            }),
        }
    }

    fn has_body_stream_handler(&self, path: &str) -> bool {
        path == "/transcribe"
    }

    fn is_streaming(&self, _path: &str) -> bool {
        false
    }

    async fn route_body_stream(
        &self,
        req: Request<()>,
        body: BodyStream,
        stream_writer: Box<dyn StreamWriter>,
    ) -> HandlerResult<()> {
        if req.method() != Method::POST || req.uri().path() != "/transcribe" {
            return Err(ServerError::Config("unsupported streaming route".into()));
        }
        self.handle_transcribe_stream(req, body, stream_writer)
            .await
    }

    async fn route_stream(
        &self,
        _req: Request<()>,
        _stream_writer: Box<dyn StreamWriter>,
    ) -> HandlerResult<()> {
        Err(ServerError::Config(
            "stream-only routes are not supported".into(),
        ))
    }

    fn webtransport_handler(&self) -> Option<&dyn WebTransportHandler> {
        None
    }

    fn websocket_handler(&self, _path: &str) -> Option<&dyn WebSocketHandler> {
        None
    }
}

async fn stream_request_body(
    service: Arc<UploadResponseService>,
    stream_id: u64,
    mut body: BodyStream,
) -> Result<(), String> {
    let slot_bytes = service.config().slot_bytes();

    while let Some(chunk) = body.next().await {
        let chunk = chunk.map_err(|error| error.to_string())?;
        if chunk.is_empty() {
            continue;
        }
        if chunk.len() <= slot_bytes {
            service.append_request_body(stream_id, chunk).await?;
            continue;
        }

        let mut remaining = chunk;
        while !remaining.is_empty() {
            let take = remaining.len().min(slot_bytes);
            let piece = remaining.split_to(take);
            service.append_request_body(stream_id, piece).await?;
        }
    }

    service.end_request(stream_id).await
}

async fn wait_for_response_headers(
    service: Arc<UploadResponseService>,
    stream_id: u64,
    timeout: Duration,
) -> HandlerResult<http_pack::stream::StreamResponseHeaders> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(headers) = service.get_response_headers(stream_id).await {
            return Ok(headers);
        }
        if Instant::now() >= deadline {
            return Err(ServerError::Config("response header timeout".into()));
        }
        sleep(Duration::from_millis(1)).await;
    }
}

async fn wait_for_response_slot(
    service: Arc<UploadResponseService>,
    stream_id: u64,
    last_seen: usize,
    timeout: Duration,
) -> HandlerResult<usize> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(current_last) = service.response_last(stream_id) {
            if current_last > last_seen {
                return Ok(current_last);
            }
        }
        if Instant::now() >= deadline {
            return Err(ServerError::Config("response body timeout".into()));
        }
        sleep(Duration::from_millis(1)).await;
    }
}
