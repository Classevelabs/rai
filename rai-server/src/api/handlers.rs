use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use rai_core::MemoryManager;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub type AppState = Arc<MemoryManager>;

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
) -> Result<Json<StoreResponse>, (StatusCode, Json<ErrorResponse>)> {
    let interference = state.store(&req.content).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    Ok(Json(StoreResponse {
        status: "stored".to_string(),
        interference,
    }))
}

pub async fn recall(
    State(state): State<AppState>,
    Json(req): Json<RecallRequest>,
) -> Result<Json<rai_core::RetrievalResult>, (StatusCode, Json<ErrorResponse>)> {
    let result = state.recall(&req.query).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    Ok(Json(result))
}

pub async fn intersect(
    State(state): State<AppState>,
    Json(req): Json<IntersectRequest>,
) -> Result<Json<rai_core::IntersectionResult>, (StatusCode, Json<ErrorResponse>)> {
    let result = state.intersect(&req.concepts).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    Ok(Json(result))
}

pub async fn contradict(
    State(state): State<AppState>,
    Json(req): Json<ContradictRequest>,
) -> Result<Json<rai_core::InterferenceReport>, (StatusCode, Json<ErrorResponse>)> {
    let result = state.check_contradiction(&req.fact).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    Ok(Json(result))
}

pub async fn surprise(
    State(state): State<AppState>,
    Json(req): Json<SurpriseRequest>,
) -> Result<Json<rai_core::SurpriseResult>, (StatusCode, Json<ErrorResponse>)> {
    let result = state.measure_surprise(&req.content).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    Ok(Json(result))
}

pub async fn confidence(
    State(state): State<AppState>,
    Json(req): Json<ConfidenceRequest>,
) -> Result<Json<rai_core::ConfidenceExplanation>, (StatusCode, Json<ErrorResponse>)> {
    let result = state.explain_confidence(&req.query).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    Ok(Json(result))
}

pub async fn train(
    State(state): State<AppState>,
) -> Result<Json<TrainResponse>, (StatusCode, Json<ErrorResponse>)> {
    let losses = state.train_nra().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    Ok(Json(TrainResponse {
        status: "trained".to_string(),
        final_loss: losses.last().copied(),
    }))
}

pub async fn snapshot(
    State(state): State<AppState>,
) -> Result<Json<Vec<SnapshotEntry>>, (StatusCode, Json<ErrorResponse>)> {
    let snap = state.energy_snapshot().await;
    let entries: Vec<SnapshotEntry> = snap
        .into_iter()
        .enumerate()
        .map(|(i, (_, energy))| SnapshotEntry { index: i, energy })
        .collect();

    Ok(Json(entries))
}

pub async fn health(
    State(state): State<AppState>,
) -> Result<Json<rai_core::HealthReport>, (StatusCode, Json<ErrorResponse>)> {
    let report = state.health().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    Ok(Json(report))
}
