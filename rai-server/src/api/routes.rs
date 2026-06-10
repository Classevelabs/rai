use axum::routing::{get, post};
use axum::Router;

use crate::api::handlers::{self, AppState};

/// Build the REST API router.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/store", post(handlers::store))
        .route("/v1/recall", post(handlers::recall))
        .route("/v1/intersect", post(handlers::intersect))
        .route("/v1/contradict", post(handlers::contradict))
        .route("/v1/surprise", post(handlers::surprise))
        .route("/v1/confidence", post(handlers::confidence))
        .route("/v1/train", post(handlers::train))
        .route("/v1/snapshot", post(handlers::snapshot))
        .route("/v1/health", get(handlers::health))
        .with_state(state)
}
