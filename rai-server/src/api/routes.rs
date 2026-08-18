use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::uri::Authority;
use axum::http::{header, HeaderValue, StatusCode, Uri};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore};

use crate::api::handlers;
use crate::state::AppState;

const MAX_REQUEST_BYTES: usize = 64 * 1024;
const MAX_CONCURRENT_REQUESTS: usize = 16;
const MAX_REQUESTS_PER_MINUTE: u32 = 240;

/// Wall-clock ceiling for handling one request, embedding round-trip included.
///
/// The default OpenAI embedding request already carries a 30-second timeout, so this is set just
/// above it: a stuck request is abandoned instead of holding a concurrency permit forever.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone)]
struct AuthState {
    token: Option<Arc<str>>,
    port: u16,
    concurrent_requests: Arc<Semaphore>,
    rate_window: Arc<Mutex<RateWindow>>,
}

struct RateWindow {
    started_at: Instant,
    requests: u32,
}

/// Build the REST API router.
pub fn build_router(state: AppState, api_token: Option<String>, port: u16) -> Router {
    let auth = AuthState {
        token: api_token.map(Arc::<str>::from),
        port,
        concurrent_requests: Arc::new(Semaphore::new(MAX_CONCURRENT_REQUESTS)),
        rate_window: Arc::new(Mutex::new(RateWindow {
            started_at: Instant::now(),
            requests: 0,
        })),
    };

    Router::new()
        .route("/v1/store", post(handlers::store))
        .route("/v1/forget", post(handlers::forget))
        .route("/v1/recall", post(handlers::recall))
        .route("/v1/intersect", post(handlers::intersect))
        .route("/v1/contradict", post(handlers::contradict))
        .route("/v1/surprise", post(handlers::surprise))
        .route("/v1/confidence", post(handlers::confidence))
        .route("/v1/snapshot", post(handlers::snapshot))
        .route("/v1/health", get(handlers::health))
        .route_layer(middleware::from_fn_with_state(auth, require_auth))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BYTES))
        // Outermost, so the deadline also covers authentication, body reading, and the
        // concurrency permit held inside `require_auth`.
        .layer(middleware::from_fn(enforce_timeout))
        .with_state(state)
}

/// Abandon a request that outlives [`REQUEST_TIMEOUT`].
///
/// Dropping the handler future releases whatever it held, including the concurrency permit, so a
/// wedged upstream cannot exhaust the server's request slots.
async fn enforce_timeout(request: Request, next: Next) -> Response {
    run_with_deadline(REQUEST_TIMEOUT, request, next).await
}

async fn run_with_deadline(limit: Duration, request: Request, next: Next) -> Response {
    match tokio::time::timeout(limit, next.run(request)).await {
        Ok(response) => response,
        Err(_) => {
            log::error!("request exceeded the {limit:?} handling timeout");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "error": format!(
                        "request exceeded the {}-second handling timeout",
                        REQUEST_TIMEOUT.as_secs()
                    )
                })),
            )
                .into_response()
        }
    }
}

async fn require_auth(State(auth): State<AuthState>, request: Request, next: Next) -> Response {
    // Account for every request before performing cheap rejection checks. Otherwise an
    // attacker can send invalid credentials (or Host headers) without ever consuming the
    // rate/concurrency budget.
    let Some(_permit) = acquire_request_slot(&auth).await else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"error": "request limit exceeded"})),
        )
            .into_response();
    };

    let host = request
        .headers()
        .get(header::HOST)
        .and_then(|value| value.to_str().ok());
    if !host.is_some_and(|host| is_allowed_host(host, auth.port)) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({
                "error": format!(
                    "invalid Host header; send a loopback host such as '127.0.0.1:{port}', \
                     '[::1]:{port}', or 'localhost' (a port, when present, must be {port})",
                    port = auth.port
                )
            })),
        )
            .into_response();
    }

    if request
        .headers()
        .get(header::ORIGIN)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|origin| !is_allowed_origin(origin, auth.port))
    {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"error": "cross-origin requests are not allowed"})),
        )
            .into_response();
    }

    let Some(expected) = auth.token.as_deref() else {
        return next.run(request).await;
    };

    let supplied = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));

    if supplied.is_some_and(|token| constant_time_eq(token.as_bytes(), expected.as_bytes())) {
        return next.run(request).await;
    }

    let mut response = (
        StatusCode::UNAUTHORIZED,
        Json(json!({"error": "missing or invalid bearer token"})),
    )
        .into_response();
    response
        .headers_mut()
        .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    response
}

/// Accept only loopback `Host` values, with or without a port.
///
/// The DNS-rebinding defence is the host *name* check: a rebound browser request always carries
/// the attacker's own hostname. A port is therefore optional — a reverse proxy or a hand-written
/// `curl -H "Host: localhost"` is legitimate — but when one is present it must be this server's.
fn is_allowed_host(host: &str, port: u16) -> bool {
    host.parse::<Authority>().is_ok_and(|authority| {
        !authority.as_str().contains('@')
            && authority.port_u16().is_none_or(|actual| actual == port)
            && is_loopback_name(authority.host())
    })
}

fn is_allowed_origin(origin: &str, port: u16) -> bool {
    origin.parse::<Uri>().is_ok_and(|uri| {
        uri.scheme_str() == Some("http")
            && uri.path_and_query().is_some_and(|path_and_query| {
                path_and_query.path() == "/" && path_and_query.query().is_none()
            })
            && uri.authority().is_some_and(|authority| {
                !authority.as_str().contains('@')
                    && authority.port_u16() == Some(port)
                    && is_loopback_name(authority.host())
            })
    })
}

fn is_loopback_name(host: &str) -> bool {
    let host = host
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(host);
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

async fn acquire_request_slot(auth: &AuthState) -> Option<tokio::sync::OwnedSemaphorePermit> {
    {
        let mut window = auth.rate_window.lock().await;
        if window.started_at.elapsed() >= Duration::from_secs(60) {
            window.started_at = Instant::now();
            window.requests = 0;
        }
        if window.requests >= MAX_REQUESTS_PER_MINUTE {
            return None;
        }
        window.requests += 1;
    }

    Arc::clone(&auth.concurrent_requests)
        .try_acquire_owned()
        .ok()
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    // Always inspect exactly the configured-secret length. A supplied token with the wrong
    // length must still perform the same amount of comparison work as a matching token.
    let mut difference = left.len() ^ right.len();
    for (index, &expected) in right.iter().enumerate() {
        let supplied = left.get(index).copied().unwrap_or_default();
        difference |= usize::from(supplied ^ expected);
    }
    difference == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use rai_core::embedding::{EmbeddingBridge, MockEmbedder};
    use rai_core::MemoryManager;
    use tower::ServiceExt;

    fn test_state() -> AppState {
        let embedder = Arc::new(MockEmbedder::new(384));
        let bridge = Arc::new(EmbeddingBridge::new(embedder, 32, 32, 64));
        AppState::new(
            Arc::new(MemoryManager::try_new(bridge).expect("valid test manager")),
            None,
        )
    }

    #[tokio::test]
    async fn protected_routes_require_bearer_token() {
        let router = build_router(test_state(), Some("a".repeat(32)), 3000);
        let request = Request::builder()
            .method("POST")
            .uri("/v1/store")
            .header(header::HOST, "127.0.0.1:3000")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"content":"test"}"#))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            response.headers().get(header::WWW_AUTHENTICATE).unwrap(),
            "Bearer"
        );
    }

    #[tokio::test]
    async fn invalid_credentials_are_rate_limited() {
        let router = build_router(test_state(), Some("a".repeat(32)), 3000);
        for _ in 0..MAX_REQUESTS_PER_MINUTE {
            let response = router
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/v1/health")
                        .header(header::HOST, "127.0.0.1:3000")
                        .header(header::AUTHORIZATION, "Bearer wrong")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        }

        let response = router
            .oneshot(
                Request::builder()
                    .uri("/v1/health")
                    .header(header::HOST, "127.0.0.1:3000")
                    .header(header::AUTHORIZATION, "Bearer wrong")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[test]
    fn token_comparison_rejects_length_mismatches() {
        assert!(constant_time_eq(b"correct", b"correct"));
        assert!(!constant_time_eq(b"correct-extra", b"correct"));
        assert!(!constant_time_eq(b"short", b"correct"));
    }

    #[tokio::test]
    async fn oversized_json_is_rejected() {
        let token = "a".repeat(32);
        let router = build_router(test_state(), Some(token.clone()), 3000);
        let body = format!(r#"{{"content":"{}"}}"#, "x".repeat(MAX_REQUEST_BYTES));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/store")
            .header(header::HOST, "127.0.0.1:3000")
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::AUTHORIZATION, format!("Bearer {token}"))
            .body(Body::from(body))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn global_request_rate_is_bounded() {
        let router = build_router(test_state(), None, 3000);
        for _ in 0..MAX_REQUESTS_PER_MINUTE {
            let response = router
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/v1/health")
                        .header(header::HOST, "127.0.0.1:3000")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }

        let response = router
            .oneshot(
                Request::builder()
                    .uri("/v1/health")
                    .header(header::HOST, "127.0.0.1:3000")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    /// Training was removed rather than left as a permanent 501, so the route must be gone.
    #[tokio::test]
    async fn the_removed_training_route_is_not_found() {
        let router = build_router(test_state(), None, 3000);
        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/train")
                    .header(header::HOST, "127.0.0.1:3000")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    /// A handler that never finishes must be abandoned with a JSON error rather than hanging on
    /// to its concurrency permit forever.
    #[tokio::test]
    async fn a_handler_that_outlives_the_deadline_returns_a_json_error() {
        let deadline = Duration::from_millis(10);
        let router = Router::new()
            .route(
                "/slow",
                get(|| async {
                    std::future::pending::<()>().await;
                    StatusCode::OK
                }),
            )
            .layer(middleware::from_fn(move |request, next| {
                run_with_deadline(deadline, request, next)
            }));

        let response = router
            .oneshot(Request::builder().uri("/slow").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let error: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(
            error["error"].as_str().unwrap().contains("timeout"),
            "{error}"
        );
    }

    /// A request that finishes inside the deadline must pass through untouched.
    #[tokio::test]
    async fn a_prompt_handler_is_not_affected_by_the_deadline() {
        let router = build_router(test_state(), None, 3000);
        let response = router
            .oneshot(
                Request::builder()
                    .uri("/v1/health")
                    .header(header::HOST, "127.0.0.1:3000")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn rejects_dns_rebinding_and_cross_origin_requests() {
        let router = build_router(test_state(), None, 3000);
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/health")
                    .header(header::HOST, "attacker.example:3000")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);

        let response = router
            .oneshot(
                Request::builder()
                    .uri("/v1/health")
                    .header(header::HOST, "127.0.0.1:3000")
                    .header(header::ORIGIN, "https://attacker.example")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    /// A rejected Host must say what form is expected instead of a bare 403.
    #[tokio::test]
    async fn a_rejected_host_names_the_required_form() {
        let router = build_router(test_state(), None, 3000);
        let response = router
            .oneshot(
                Request::builder()
                    .uri("/v1/health")
                    .header(header::HOST, "attacker.example")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let error: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let message = error["error"].as_str().unwrap();
        assert!(message.contains("127.0.0.1:3000"), "{message}");
        assert!(message.contains("localhost"), "{message}");
    }

    #[test]
    fn accepts_all_loopback_host_forms_with_the_exact_port() {
        assert!(is_allowed_host("localhost:3000", 3000));
        assert!(is_allowed_host("127.0.0.2:3000", 3000));
        assert!(is_allowed_host("[::1]:3000", 3000));
        assert!(is_allowed_origin("http://127.23.4.5:3000", 3000));
        assert!(is_allowed_origin("http://127.23.4.5:3000/", 3000));
        assert!(is_allowed_origin("http://[::1]:3000", 3000));

        assert!(!is_allowed_host("127.0.0.1:4000", 3000));
        assert!(!is_allowed_host("192.168.1.2:3000", 3000));
        assert!(!is_allowed_origin("https://127.0.0.1:3000", 3000));
        assert!(!is_allowed_origin("http://127.0.0.1:4000", 3000));
        assert!(!is_allowed_origin("http://127.0.0.1:3000/path", 3000));
        assert!(!is_allowed_origin("http://127.0.0.1:3000/?query", 3000));
        assert!(!is_allowed_origin("http://user@127.0.0.1:3000", 3000));
        assert!(!is_allowed_origin("http://attacker.example:3000", 3000));
    }

    /// A portless loopback Host is legitimate — it names a loopback address, which is what the
    /// DNS-rebinding check is actually about — and must not be a silent 403.
    #[test]
    fn accepts_portless_loopback_hosts_but_not_portless_foreign_ones() {
        assert!(is_allowed_host("localhost", 3000));
        assert!(is_allowed_host("127.0.0.1", 3000));
        assert!(is_allowed_host("[::1]", 3000));

        assert!(!is_allowed_host("attacker.example", 3000));
        assert!(!is_allowed_host("192.168.1.2", 3000));
        assert!(!is_allowed_host("user@localhost", 3000));
    }
}
