use crate::embedding::bridge::TextIndex;
use crate::embedding::projection::Projection;
use crate::RaiError;
use rem_nra::nra::{NRAConfig, NRAParams};
use rem_nra::rem::REMConfig;
use rem_nra::Vec64;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Serializable snapshot of the full RAI memory state.
#[derive(Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// NRA parameters.
    pub nra_params: NRAParams,
    /// NRA config.
    pub nra_config: NRAConfig,
    /// NRA stored items (omega, value).
    pub nra_items: Vec<(Vec64, Vec64)>,
    /// REM config.
    pub rem_config: REMConfig,
    /// REM encoder params.
    pub rem_encoder: rem_nra::rem::encoder::EncoderParams,
    /// REM decoder params.
    pub rem_decoder: rem_nra::rem::decoder::DecoderParams,
    /// REM memory state.
    pub rem_memory_state: Vec64,
    /// REM stored items (key, value).
    pub rem_items: Vec<(Vec64, Vec64)>,
    /// Text index for text <-> vector mapping.
    pub text_index: TextIndex,
    /// Omega projection.
    pub omega_proj: Projection,
    /// Key projection.
    pub key_proj: Projection,
    /// Value projection.
    pub value_proj: Projection,
    /// Total items stored.
    pub total_items: usize,
}

impl MemorySnapshot {
    /// Save snapshot to a JSON file.
    pub fn save(&self, path: &Path) -> Result<(), RaiError> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| RaiError::PersistenceError(format!("serialize: {e}")))?;
        std::fs::write(path, json)
            .map_err(|e| RaiError::PersistenceError(format!("write: {e}")))?;
        Ok(())
    }

    /// Load snapshot from a JSON file.
    pub fn load(path: &Path) -> Result<Self, RaiError> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| RaiError::PersistenceError(format!("read: {e}")))?;
        let snapshot: Self = serde_json::from_str(&json)
            .map_err(|e| RaiError::PersistenceError(format!("deserialize: {e}")))?;
        Ok(snapshot)
    }
}
