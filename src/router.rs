use crate::config::AppConfig;
use crate::ingress::{ListenIngress, ListenIngressWebSocketHandler};
use async_trait::async_trait;
use bytes::Bytes;
use http::{Method, Request, StatusCode};
use std::sync::Arc;
use upload_response::UploadResponseRouter;
use web_service::{
    BodyStream, HandlerResponse, HandlerResult, Router, ServerError, StreamWriter,
    WebSocketHandler, WebTransportHandler,
};

#[derive(Clone)]
pub struct AppRouter {
    config: AppConfig,
    upload: Option<Arc<UploadResponseRouter>>,
    listen: Option<Arc<ListenIngress>>,
    listen_ws: Option<Arc<ListenIngressWebSocketHandler>>,
}

impl AppRouter {
    pub fn new(
        config: AppConfig,
        upload: Option<Arc<UploadResponseRouter>>,
        listen: Option<Arc<ListenIngress>>,
        listen_ws: Option<Arc<ListenIngressWebSocketHandler>>,
    ) -> Self {
        Self {
            config,
            upload,
            listen,
            listen_ws,
        }
    }

    async fn handle_health(&self) -> HandlerResult<HandlerResponse> {
        Ok(HandlerResponse {
            status: StatusCode::OK,
            body: Some(Bytes::from("{\"status\":\"ok\",\"service\":\"asr-api\"}")),
            content_type: Some("application/json".into()),
            headers: vec![("cache-control".into(), "no-store".into())],
            etag: None,
        })
    }

    async fn handle_status(&self) -> HandlerResult<HandlerResponse> {
        let body = format!(
            "{{\"status\":\"ok\",\"service\":\"asr-api\",\"role\":\"{:?}\",\"model_name\":\"{}\"}}",
            self.config.role,
            self.config.default_model_name()
        );
        Ok(HandlerResponse {
            status: StatusCode::OK,
            body: Some(Bytes::from(body)),
            content_type: Some("application/json".into()),
            headers: vec![("cache-control".into(), "no-store".into())],
            etag: None,
        })
    }

    async fn handle_model_info(&self) -> HandlerResult<HandlerResponse> {
        Ok(HandlerResponse {
            status: StatusCode::OK,
            body: Some(Bytes::from(format!(
                "{{\"model_name\":\"{}\"}}",
                self.config.default_model_name()
            ))),
            content_type: Some("application/json".into()),
            headers: vec![("cache-control".into(), "no-store".into())],
            etag: None,
        })
    }

    async fn handle_live(&self, include_body: bool) -> HandlerResult<HandlerResponse> {
        Ok(HandlerResponse {
            status: StatusCode::OK,
            body: include_body.then(|| Bytes::from("OK")),
            content_type: include_body.then_some("text/plain".into()),
            headers: vec![("cache-control".into(), "no-store".into())],
            etag: None,
        })
    }

    async fn handle_available(&self) -> HandlerResult<HandlerResponse> {
        Ok(HandlerResponse {
            status: StatusCode::OK,
            body: None,
            content_type: None,
            headers: vec![("cache-control".into(), "no-store".into())],
            etag: None,
        })
    }

    async fn handle_version(&self) -> HandlerResult<HandlerResponse> {
        let sha = option_env!("GIT_SHA").unwrap_or("dev");
        let device_id = self.config.device_ids.first().copied().unwrap_or(0);
        Ok(HandlerResponse {
            status: StatusCode::OK,
            body: Some(Bytes::from(format!(
                "{{\"sha\":\"{}\",\"device_id\":{},\"model\":\"{}\"}}",
                sha,
                device_id,
                self.config.default_model_name()
            ))),
            content_type: Some("application/json".into()),
            headers: vec![("cache-control".into(), "no-store".into())],
            etag: None,
        })
    }

    async fn handle_options(&self) -> HandlerResult<HandlerResponse> {
        Ok(HandlerResponse {
            status: StatusCode::NO_CONTENT,
            body: None,
            content_type: None,
            headers: vec![("cache-control".into(), "no-store".into())],
            etag: None,
        })
    }

    fn not_found() -> HandlerResponse {
        HandlerResponse {
            status: StatusCode::NOT_FOUND,
            body: Some(Bytes::from("{\"error\":\"not found\"}")),
            content_type: Some("application/json".into()),
            headers: vec![("cache-control".into(), "no-store".into())],
            etag: None,
        }
    }

    fn is_upload_path(path: &str) -> bool {
        path.starts_with("/_upload_response/")
    }

    fn is_listen_path(path: &str) -> bool {
        path == "/v1/listen"
    }
}

#[async_trait]
impl Router for AppRouter {
    async fn route(&self, req: Request<()>) -> HandlerResult<HandlerResponse> {
        match (req.method(), req.uri().path()) {
            (&Method::GET, "/") | (&Method::GET, "/health") | (&Method::GET, "/healthz") => {
                self.handle_health().await
            }
            (&Method::GET, "/status") => self.handle_status().await,
            (&Method::GET, "/info/model") => self.handle_model_info().await,
            (&Method::GET, "/ha/live") => self.handle_live(true).await,
            (&Method::HEAD, "/ha/live") => self.handle_live(false).await,
            (&Method::GET, "/ha/available") | (&Method::HEAD, "/ha/available") => {
                self.handle_available().await
            }
            (&Method::GET, "/ha/version") | (&Method::HEAD, "/ha/version") => {
                self.handle_version().await
            }
            (&Method::OPTIONS, "/v1/listen") if self.listen.is_some() => {
                self.handle_options().await
            }
            _ if Self::is_upload_path(req.uri().path()) => {
                if let Some(upload) = &self.upload {
                    upload.route(req).await
                } else {
                    Ok(Self::not_found())
                }
            }
            _ => Ok(Self::not_found()),
        }
    }

    async fn route_body(
        &self,
        req: Request<()>,
        body: BodyStream,
    ) -> HandlerResult<HandlerResponse> {
        if Self::is_upload_path(req.uri().path()) {
            if let Some(upload) = &self.upload {
                upload.route_body(req, body).await
            } else {
                Ok(Self::not_found())
            }
        } else if Self::is_listen_path(req.uri().path()) {
            let _ = body;
            Err(ServerError::Config(
                "listen requires streaming request/response handling".into(),
            ))
        } else {
            Ok(Self::not_found())
        }
    }

    fn has_body_handler(&self, path: &str) -> bool {
        self.upload.is_some() && Self::is_upload_path(path)
    }

    fn has_body_stream_handler(&self, path: &str) -> bool {
        self.listen.is_some() && Self::is_listen_path(path)
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
        if Self::is_listen_path(req.uri().path()) {
            if let Some(listen) = &self.listen {
                listen.handle_listen_stream(req, body, stream_writer).await
            } else {
                Err(ServerError::Config("listen ingress not configured".into()))
            }
        } else {
            Err(ServerError::Config(
                "streaming request/response is not supported for this route".into(),
            ))
        }
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

    fn websocket_handler(&self, path: &str) -> Option<&dyn WebSocketHandler> {
        if let Some(handler) = &self.listen_ws {
            if handler.can_handle(path) {
                return Some(handler.as_ref());
            }
        }
        None
    }
}
