use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::state::AppState;

const MAX_TEXT_BYTES: usize = 16 * 1024;
const MAX_INTERSECTION_CONCEPTS: usize = 32;

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
pub struct TrainResponse {
    pub status: String,
    pub final_loss: Option<f64>,
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
    validate_text("content", &req.content)?;
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
    validate_text("query", &req.query)?;
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
    validate_concepts(&req.concepts)?;
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
    validate_text("fact", &req.fact)?;
    let result = state
        .manager
        .check_contradiction(&req.fact)
        .await
        .map_err(|error| rai_error("contradiction check", error))?;

    Ok(Json(result))
}

pub async fn surprise(
    State(state): State<AppState>,
    Json(req): Json<SurpriseRequest>,
) -> Result<Json<rai_core::SurpriseResult>, ApiError> {
    validate_text("content", &req.content)?;
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
    validate_text("query", &req.query)?;
    let result = state
        .manager
        .explain_confidence(&req.query)
        .await
        .map_err(|error| rai_error("confidence explanation", error))?;

    Ok(Json(result))
}

pub async fn train(State(state): State<AppState>) -> Result<Json<TrainResponse>, ApiError> {
    let _training = state.try_training_lock().ok_or_else(|| {
        (
            StatusCode::CONFLICT,
            Json(ErrorResponse {
                error: "training is already in progress".to_string(),
            }),
        )
    })?;
    let losses = state
        .train_nra()
        .await
        .map_err(|error| rai_error("training", error))?;

    Ok(Json(TrainResponse {
        status: "trained".to_string(),
        final_loss: losses.last().copied(),
    }))
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

fn validate_text(field: &str, value: &str) -> Result<(), ApiError> {
    if value.trim().is_empty() {
        return Err(client_error(format!("{field} must not be empty")));
    }
    if value.len() > MAX_TEXT_BYTES {
        return Err(client_error(format!(
            "{field} exceeds the {MAX_TEXT_BYTES}-byte limit"
        )));
    }
    Ok(())
}

fn validate_concepts(concepts: &[String]) -> Result<(), ApiError> {
    if concepts.len() < 2 {
        return Err(client_error("at least two concepts are required"));
    }
    if concepts.len() > MAX_INTERSECTION_CONCEPTS {
        return Err(client_error(format!(
            "concepts exceeds the {MAX_INTERSECTION_CONCEPTS}-item limit"
        )));
    }

    let mut total_bytes = 0usize;
    for concept in concepts {
        validate_text("concept", concept)?;
        total_bytes = total_bytes
            .checked_add(concept.len())
            .ok_or_else(|| client_error("combined concept text is too large"))?;
    }
    if total_bytes > MAX_TEXT_BYTES {
        return Err(client_error(format!(
            "combined concept text exceeds the {MAX_TEXT_BYTES}-byte limit"
        )));
    }
    Ok(())
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
        rai_core::RaiError::TrainingError(_) => (
            StatusCode::NOT_IMPLEMENTED,
            Json(ErrorResponse {
                error: "training is unavailable because this build does not implement parameter optimization"
                    .to_string(),
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
    fn validation_rejects_empty_and_amplifying_inputs() {
        assert!(validate_text("query", "   ").is_err());
        assert!(validate_text("query", &"x".repeat(MAX_TEXT_BYTES + 1)).is_err());
        assert!(validate_concepts(&[]).is_err());
        assert!(validate_concepts(&vec!["x".to_string(); MAX_INTERSECTION_CONCEPTS + 1]).is_err());
        assert!(validate_concepts(&["valid".to_string(), " ".to_string()]).is_err());
    }

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
    fn unavailable_training_is_reported_honestly() {
        let (status, Json(response)) = rai_error(
            "training",
            rai_core::RaiError::TrainingError("internal training details".to_string()),
        );
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(
            response.error,
            "training is unavailable because this build does not implement parameter optimization"
        );
        assert!(!response.error.contains("internal training details"));
    }
}
