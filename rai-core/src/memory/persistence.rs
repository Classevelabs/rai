use crate::embedding::bridge::TextIndex;
use crate::embedding::projection::Projection;
use crate::RaiError;
use rem_nra::nra::{NRAConfig, NRAParams};
use rem_nra::rem::REMConfig;
use rem_nra::Vec64;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

const MAX_SNAPSHOT_BYTES: u64 = 64 * 1024 * 1024;
pub(crate) const MAX_STORED_ITEMS: usize = 100_000;
pub(crate) const MAX_VECTOR_DIMENSION: usize = 16_384;
/// Upper bound on a persisted text label.
///
/// This is deliberately larger than [`crate::MAX_TEXT_BYTES`], the limit enforced when new
/// content enters the store: snapshots written by earlier releases accepted labels up to this
/// size and must still load. New stores can never exceed the smaller live limit.
const MAX_SNAPSHOT_TEXT_BYTES: usize = 64 * 1024;
const MAX_ABS_COMPONENT: f64 = 1.0e100;
pub(crate) const CURRENT_SNAPSHOT_VERSION: u32 = 1;

/// Serializable snapshot of the full RAI memory state.
///
/// The written payload is a strict subset of what earlier 0.1.x builds wrote: the REM encoder
/// and decoder biases, the REM rolling memory state, the training loss, and the NRA value basis
/// backed no live behaviour and are no longer emitted. Snapshots that still contain those keys
/// load unchanged — serde ignores them.
#[derive(Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// On-disk schema version. Missing values identify legacy version 0 snapshots.
    #[serde(default)]
    pub version: u32,
    /// NRA parameters.
    pub nra_params: NRAParams,
    /// NRA config.
    pub nra_config: NRAConfig,
    /// NRA stored items (omega, value).
    pub nra_items: Vec<(Vec64, Vec64)>,
    /// REM config.
    pub rem_config: REMConfig,
    /// REM stored items (key, value).
    pub rem_items: Vec<(Vec64, Vec64)>,
    /// REM mean residual norm at snapshot time.
    #[serde(default)]
    pub rem_residual_norm: f64,
    /// Text index for text <-> vector mapping.
    pub text_index: TextIndex,
    /// Text labels parallel to NRA/REM items (duplicates preserved).
    #[serde(default)]
    pub texts: Vec<String>,
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
        self.validate()?;
        if path.file_name().is_none() {
            return Err(RaiError::PersistenceError(
                "snapshot path must name a file".into(),
            ));
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| RaiError::PersistenceError(format!("serialize: {e}")))?;
        if json.len() as u64 > MAX_SNAPSHOT_BYTES {
            return Err(RaiError::PersistenceError(format!(
                "snapshot exceeds the {MAX_SNAPSHOT_BYTES}-byte limit"
            )));
        }

        let temp_path = temporary_path(path);
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }

        let write_result = (|| -> std::io::Result<()> {
            let mut file = options.open(&temp_path)?;
            file.write_all(json.as_bytes())?;
            file.sync_all()?;
            drop(file);
            atomic_replace(&temp_path, path)?;

            #[cfg(unix)]
            if let Some(parent) = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                std::fs::File::open(parent)?.sync_all()?;
            }
            Ok(())
        })();

        if let Err(error) = write_result {
            let _ = std::fs::remove_file(&temp_path);
            return Err(RaiError::PersistenceError(format!("atomic write: {error}")));
        }
        Ok(())
    }

    /// Load snapshot from a JSON file.
    pub fn load(path: &Path) -> Result<Self, RaiError> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| RaiError::PersistenceError(format!("open: {e}")))?;
        let metadata = file
            .metadata()
            .map_err(|e| RaiError::PersistenceError(format!("metadata: {e}")))?;
        if metadata.len() > MAX_SNAPSHOT_BYTES {
            return Err(RaiError::PersistenceError(format!(
                "snapshot exceeds the {MAX_SNAPSHOT_BYTES}-byte limit"
            )));
        }

        let mut json = Vec::with_capacity(metadata.len() as usize);
        Read::by_ref(&mut file)
            .take(MAX_SNAPSHOT_BYTES + 1)
            .read_to_end(&mut json)
            .map_err(|e| RaiError::PersistenceError(format!("read: {e}")))?;
        if json.len() as u64 > MAX_SNAPSHOT_BYTES {
            return Err(RaiError::PersistenceError(format!(
                "snapshot exceeds the {MAX_SNAPSHOT_BYTES}-byte limit"
            )));
        }

        let snapshot: Self = serde_json::from_slice(&json)
            .map_err(|e| RaiError::PersistenceError(format!("deserialize: {e}")))?;
        snapshot.validate()?;
        Ok(snapshot)
    }

    fn validate(&self) -> Result<(), RaiError> {
        let invalid = |message: &str| RaiError::PersistenceError(message.to_string());
        if self.version > CURRENT_SNAPSHOT_VERSION {
            return Err(invalid(
                "snapshot was created by a newer unsupported version",
            ));
        }
        let dims = [
            self.nra_config.dim_state,
            self.nra_config.dim_omega,
            self.nra_config.dim_value,
            self.rem_config.dim_memory,
            self.rem_config.dim_key,
            self.rem_config.dim_value,
            self.omega_proj.source_dim,
        ];
        if dims
            .iter()
            .any(|dimension| *dimension == 0 || *dimension > MAX_VECTOR_DIMENSION)
        {
            return Err(invalid("snapshot dimensions are outside supported bounds"));
        }
        if self.nra_config.num_units == 0 || self.nra_config.num_units > MAX_STORED_ITEMS {
            return Err(invalid(
                "snapshot configuration is outside supported bounds",
            ));
        }
        if self.total_items > self.nra_config.num_units
            || self.total_items > MAX_STORED_ITEMS
            || self.nra_items.len() != self.total_items
            || self.rem_items.len() != self.total_items
            || (!self.texts.is_empty() && self.texts.len() != self.total_items)
        {
            return Err(invalid(
                "snapshot item counts are incoherent or exceed capacity",
            ));
        }
        if self.nra_config.dim_value != self.rem_config.dim_value
            || self.omega_proj.source_dim != self.key_proj.source_dim
            || self.omega_proj.source_dim != self.value_proj.source_dim
            || self.omega_proj.target_dim != self.nra_config.dim_omega
            || self.key_proj.target_dim != self.rem_config.dim_key
            || self.value_proj.target_dim != self.nra_config.dim_value
            || !self.omega_proj.validate_shape()
            || !self.key_proj.validate_shape()
            || !self.value_proj.validate_shape()
        {
            return Err(invalid("snapshot projection dimensions are inconsistent"));
        }
        if self.nra_params.omega_basis.nrows() != self.nra_config.dim_state
            || self.nra_params.omega_basis.ncols() != self.nra_config.dim_omega
            || !self
                .nra_params
                .omega_basis
                .iter()
                .all(|value| number_is_valid(*value))
        {
            return Err(invalid("snapshot NRA parameters are invalid"));
        }
        if !vectors_are_valid(
            self.nra_items.iter().map(|(omega, _)| omega.as_slice()),
            self.nra_config.dim_omega,
        ) || !vectors_are_valid(
            self.nra_items.iter().map(|(_, value)| value.as_slice()),
            self.nra_config.dim_value,
        ) || !vectors_are_valid(
            self.rem_items.iter().map(|(key, _)| key.as_slice()),
            self.rem_config.dim_key,
        ) || !vectors_are_valid(
            self.rem_items.iter().map(|(_, value)| value.as_slice()),
            self.rem_config.dim_value,
        ) || !number_is_valid(self.rem_residual_norm)
            || self.rem_residual_norm < 0.0
        {
            return Err(invalid("snapshot REM state or item vectors are invalid"));
        }

        let mut indexed_texts = HashSet::with_capacity(self.text_index.entries.len());
        if self.text_index.entries.len() > self.total_items
            || self.text_index.text_to_id.len() != self.text_index.entries.len()
            || self
                .text_index
                .entries
                .iter()
                .enumerate()
                .any(|(position, entry)| {
                    entry.id != position
                        || entry.text.trim().is_empty()
                        || entry.text.len() > MAX_SNAPSHOT_TEXT_BYTES
                        || entry.embedding.len() != self.omega_proj.source_dim
                        || entry.embedding.iter().any(|value| !number_is_valid(*value))
                        || !indexed_texts.insert(entry.text.as_str())
                        || self.text_index.text_to_id.get(&entry.text) != Some(&entry.id)
                })
            || self
                .texts
                .iter()
                .any(|text| text.is_empty() || text.len() > MAX_SNAPSHOT_TEXT_BYTES)
        {
            return Err(invalid("snapshot text index is invalid"));
        }

        // Version 0 snapshots predate the parallel `texts` field, so loading them still
        // reconstructs labels from the text index. Every current-version snapshot writes
        // `texts`; require that it covers every stored item and that the index represents
        // exactly the unique label set. Otherwise a structurally valid but incoherent file
        // could silently substitute or omit the labels returned by recall.
        if self.version == CURRENT_SNAPSHOT_VERSION {
            let stored_texts: HashSet<&str> = self.texts.iter().map(String::as_str).collect();
            if self.texts.len() != self.total_items || indexed_texts != stored_texts {
                return Err(invalid(
                    "snapshot text labels do not match the current text index",
                ));
            }
        }

        Ok(())
    }
}

fn vectors_are_valid<'a>(
    mut vectors: impl Iterator<Item = &'a [f64]>,
    expected_dimension: usize,
) -> bool {
    vectors.all(|vector| {
        vector.len() == expected_dimension && vector.iter().all(|value| number_is_valid(*value))
    })
}

fn number_is_valid(value: f64) -> bool {
    value.is_finite() && value.abs() <= MAX_ABS_COMPONENT
}

fn temporary_path(path: &Path) -> PathBuf {
    let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let mut name = OsString::from(".");
    name.push(path.file_name().unwrap_or_default());
    name.push(format!(".tmp-{}-{timestamp}-{counter}", std::process::id()));
    path.with_file_name(name)
}

#[cfg(not(windows))]
fn atomic_replace(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::fs::rename(source, destination)
}

#[cfg(windows)]
fn atomic_replace(source: &Path, destination: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
    let destination: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();
    let result = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if result == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_snapshot_is_rejected_before_reading() {
        let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rai-oversized-snapshot-{}-{counter}.json",
            std::process::id()
        ));
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.set_len(MAX_SNAPSHOT_BYTES + 1).unwrap();
        drop(file);

        let error = match MemorySnapshot::load(&path) {
            Err(error) => error,
            Ok(_) => panic!("oversized snapshot unexpectedly loaded"),
        };
        assert!(error.to_string().contains("exceeds"));
        std::fs::remove_file(path).unwrap();
    }
}
