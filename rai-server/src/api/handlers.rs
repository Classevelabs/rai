use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::state::AppState;
use crate::validate;

type ApiError = (StatusCode, Json<ErrorResponse>);

// --- Request/Response types ---

#[derive(Deserialize)]
pub struct StoreRequest {
    pub content: String,
}

#[derive(Serialize)]
pub struct StoreResponse {
    pub status: String,
    pub interference: rai_core::InterferenceReport,
}

#[derive(Deserialize)]
pub struct RecallRequest {
    pub query: String,
}

#[derive(Deserialize)]
pub struct IntersectRequest {
    pub concepts: Vec<String>,
}

#[derive(Deserialize)]
pub struct ContradictRequest {
    pub fact: String,
}

#[derive(Deserialize)]
pub struct SurpriseRequest {
    pub content: String,
}

#[derive(Deserialize)]
pub struct ConfidenceRequest {
    pub query: String,
}

#[derive(Serialize)]
pub struct SnapshotEntry {
    pub index: usize,
    pub energy: f64,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

// --- Handlers ---

pub async fn store(
    State(state): State<AppState>,
    Json(req): Json<StoreRequest>,
) -> Result<Json<StoreResponse>, ApiError> {
    validate::validate_text("content", &req.content).map_err(client_error)?;
    let interference = state
        .store(&req.content)
        .await
        .map_err(|error| rai_error("store", error))?;

    Ok(Json(StoreResponse {
        status: "stored".to_string(),
        interference,
    }))
}

pub async fn recall(
    State(state): State<AppState>,
    Json(req): Json<RecallRequest>,
) -> Result<Json<rai_core::RetrievalResult>, ApiError> {
    validate::validate_text("query", &req.query).map_err(client_error)?;
    let result = state
        .manager
        .recall(&req.query)
        .await
        .map_err(|error| rai_error("recall", error))?;

    Ok(Json(result))
}

pub async fn intersect(
    State(state): State<AppState>,
    Json(req): Json<IntersectRequest>,
) -> Result<Json<rai_core::IntersectionResult>, ApiError> {
    validate::validate_concepts(&req.concepts).map_err(client_error)?;
    let result = state
        .manager
        .intersect(&req.concepts)
        .await
        .map_err(|error| rai_error("intersect", error))?;

    Ok(Json(result))
}

pub async fn contradict(
    State(state): State<AppState>,
    Json(req): Json<ContradictRequest>,
) -> Result<Json<rai_core::InterferenceReport>, ApiError> {
    validate::validate_text("fact", &req.fact).map_err(client_error)?;
    let result = state
        .manager
        .check_contradiction(&req.fact)
        .await
        .map_err(|error| rai_error("crowding check", error))?;

    Ok(Json(result))
}

pub async fn surprise(
    State(state): State<AppState>,
    Json(req): Json<SurpriseRequest>,
) -> Result<Json<rai_core::SurpriseResult>, ApiError> {
    validate::validate_text("content", &req.content).map_err(client_error)?;
    let result = state
        .manager
        .measure_surprise(&req.content)
        .await
        .map_err(|error| rai_error("surprise measurement", error))?;

    Ok(Json(result))
}

pub async fn confidence(
    State(state): State<AppState>,
    Json(req): Json<ConfidenceRequest>,
) -> Result<Json<rai_core::ConfidenceExplanation>, ApiError> {
    validate::validate_text("query", &req.query).map_err(client_error)?;
    let result = state
        .manager
        .explain_confidence(&req.query)
        .await
        .map_err(|error| rai_error("confidence explanation", error))?;

    Ok(Json(result))
}

pub async fn snapshot(State(state): State<AppState>) -> Result<Json<Vec<SnapshotEntry>>, ApiError> {
    let snap = state.manager.energy_snapshot().await;
    let entries: Vec<SnapshotEntry> = snap
        .into_iter()
        .enumerate()
        .map(|(i, (_, energy))| SnapshotEntry { index: i, energy })
        .collect();

    Ok(Json(entries))
}

pub async fn health(
    State(state): State<AppState>,
) -> Result<Json<rai_core::HealthReport>, ApiError> {
    let report = state
        .manager
        .health()
        .await
        .map_err(|error| rai_error("health check", error))?;

    Ok(Json(report))
}

fn client_error(message: impl Into<String>) -> ApiError {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: message.into(),
        }),
    )
}

fn rai_error(operation: &'static str, error: rai_core::RaiError) -> ApiError {
    match error {
        rai_core::RaiError::InvalidInput(message) => client_error(message),
        // The store being full is the caller's problem to resolve, not an internal fault.
        error @ rai_core::RaiError::CapacityExhausted { .. } => (
            StatusCode::CONFLICT,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        ),
        error => internal_error(operation, "internal server error", error),
    }
}

fn internal_error(
    operation: &'static str,
    public_message: &'static str,
    error: impl std::fmt::Display,
) -> ApiError {
    log::error!("{operation} failed: {error}");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
            error: public_message.to_string(),
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_error_details_are_not_returned() {
        let secret = r#"C:\\private\\memory.json"#;
        let (status, Json(response)) = internal_error(
            "test operation",
            "internal server error",
            format!("could not read {secret}"),
        );
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(response.error, "internal server error");
        assert!(!response.error.contains(secret));
    }

    #[test]
    fn invalid_input_is_a_client_error() {
        let (status, Json(response)) = rai_error(
            "test operation",
            rai_core::RaiError::InvalidInput("bad query".to_string()),
        );
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(response.error, "bad query");
    }

    #[test]
    fn a_full_store_is_a_conflict_not_an_internal_error() {
        let (status, Json(response)) = rai_error(
            "store",
            rai_core::RaiError::CapacityExhausted { limit: 512 },
        );
        assert_eq!(status, StatusCode::CONFLICT);
        assert!(response.error.contains("512"), "{}", response.error);
        assert!(response.error.contains("capacity"), "{}", response.error);
    }
}
