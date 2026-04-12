use async_trait::async_trait;
use bytes::Bytes;
use http::{Method, Request, StatusCode};
use std::sync::Arc;
use upload_response::UploadResponseRouter;
use crate::ingress::ListenIngress;
use web_service::{
    BodyStream, HandlerResponse, HandlerResult, Router, ServerError, StreamWriter,
    WebSocketHandler, WebTransportHandler,
};

#[derive(Clone)]
pub struct AppRouter {
    upload: Option<Arc<UploadResponseRouter>>,
    listen: Option<Arc<ListenIngress>>,
}

impl AppRouter {
    pub fn new(upload: Option<Arc<UploadResponseRouter>>, listen: Option<Arc<ListenIngress>>) -> Self {
        Self { upload, listen }
    }

    async fn handle_health(&self) -> HandlerResult<HandlerResponse> {
        Ok(HandlerResponse {
            status: StatusCode::OK,
            body: Some(Bytes::from(
                "{\"status\":\"ok\",\"service\":\"transcriber\"}",
            )),
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
            (&Method::OPTIONS, "/v1/listen") if self.listen.is_some() => self.handle_options().await,
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
        if Self::is_listen_path(req.uri().path()) {
            if let Some(listen) = &self.listen {
                Ok(listen.handle_listen(req, body).await)
            } else {
                Ok(Self::not_found())
            }
        } else if Self::is_upload_path(req.uri().path()) {
            if let Some(upload) = &self.upload {
                upload.route_body(req, body).await
            } else {
                Ok(Self::not_found())
            }
        } else {
            Ok(Self::not_found())
        }
    }

    fn has_body_handler(&self, path: &str) -> bool {
        (self.upload.is_some() && Self::is_upload_path(path))
            || (self.listen.is_some() && Self::is_listen_path(path))
    }

    fn is_streaming(&self, _path: &str) -> bool {
        false
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
