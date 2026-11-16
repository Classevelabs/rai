//! How many cores the default thread pool should use.
//!
//! The engine's default is the number of *performance* cores, plus one: decode
//! is bandwidth-bound, and a second hardware thread on a core shares that
//! core's load/store units without adding bandwidth, while the extra worker
//! keeps a core busy while another waits on memory. The previous default,
//! half the logical CPUs plus one, encoded "two threads per core" — true on
//! most x86 laptops and false on Apple Silicon (one thread per core: half the
//! cores sat idle), on most ARM servers, and on any x86 machine with SMT off.

/// Performance cores this machine has, if the platform can say.
///
/// On a hybrid CPU this counts the fast cores only (Apple's performance
/// cluster, Intel's P-cores on Windows): a parallel GEMM splits rows evenly
/// across workers, and a worker on an efficiency core finishes last.
pub(crate) fn performance_cores() -> Option<usize> {
    platform::performance_cores().filter(|&cores| cores > 0)
}

/// The default worker count: performance cores plus one, never more than the
/// CPUs this process may run on (an affinity mask or a container's quota can
/// make that fewer than the machine has), and at least two.
pub(crate) fn default_threads() -> usize {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let cores = performance_cores().unwrap_or_else(|| available.div_ceil(2));
    (cores + 1).min(available).max(2)
}

#[cfg(target_os = "linux")]
mod platform {
    use std::collections::HashSet;
    use std::path::Path;

    pub(super) fn performance_cores() -> Option<usize> {
        count_cores(Path::new("/sys/devices/system/cpu"))
    }

    /// Distinct `(package, core)` pairs among the CPUs sysfs describes.
    pub(super) fn count_cores(root: &Path) -> Option<usize> {
        let read = |path: std::path::PathBuf| -> Option<String> {
            std::fs::read_to_string(path)
                .ok()
                .map(|value| value.trim().to_string())
        };
        let mut cores = HashSet::new();
        for entry in std::fs::read_dir(root).ok()?.flatten() {
            let name = entry.file_name();
            let Some(index) = name.to_str().and_then(|name| name.strip_prefix("cpu")) else {
                continue;
            };
            if index.is_empty() || !index.bytes().all(|byte| byte.is_ascii_digit()) {
                continue;
            }
            // An offline CPU has no topology directory; it is skipped, not an
            // error, and does not count.
            let topology = entry.path().join("topology");
            let (Some(package), Some(core)) = (
                read(topology.join("physical_package_id")),
                read(topology.join("core_id")),
            ) else {
                continue;
            };
            cores.insert((package, core));
        }
        (!cores.is_empty()).then_some(cores.len())
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use std::ffi::CStr;

    pub(super) fn performance_cores() -> Option<usize> {
        // `hw.perflevel0` is the performance cluster on Apple Silicon; Intel
        // Macs have no perflevels, and every core there is a performance core.
        sysctl_count(c"hw.perflevel0.physicalcpu").or_else(|| sysctl_count(c"hw.physicalcpu"))
    }

    fn sysctl_count(name: &CStr) -> Option<usize> {
        let mut value: libc::c_int = 0;
        let mut size = std::mem::size_of::<libc::c_int>();
        // SAFETY: `name` is NUL-terminated, and `value`/`size` describe a
        // writable `c_int`, which is what these integer sysctls return.
        let status = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                (&mut value as *mut libc::c_int).cast(),
                &mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        (status == 0 && value > 0).then_some(value as usize)
    }
}

#[cfg(windows)]
mod platform {
    use windows_sys::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, RelationProcessorCore,
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };

    pub(super) fn performance_cores() -> Option<usize> {
        let mut length = 0u32;
        // SAFETY: a null buffer with length 0 asks only for the size needed.
        unsafe {
            GetLogicalProcessorInformationEx(
                RelationProcessorCore,
                std::ptr::null_mut(),
                &mut length,
            )
        };
        if length == 0 {
            return None;
        }
        // u64 storage keeps the records 8-byte aligned, as their fields need.
        let mut buffer = vec![0u64; (length as usize).div_ceil(8)];
        // SAFETY: `buffer` is `length` bytes, suitably aligned.
        let ok = unsafe {
            GetLogicalProcessorInformationEx(
                RelationProcessorCore,
                buffer.as_mut_ptr().cast(),
                &mut length,
            )
        };
        if ok == 0 {
            return None;
        }
        let bytes = buffer.as_ptr().cast::<u8>();
        let mut classes = Vec::new();
        let mut offset = 0usize;
        while offset + std::mem::size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>()
            <= length as usize
        {
            // SAFETY: the system wrote a sequence of records into `length`
            // bytes, each starting with its own `Size`; the loop condition
            // keeps the header read in bounds, and only the processor
            // variant is read, as `RelationProcessorCore` guarantees.
            let (size, class) = unsafe {
                let record = &*bytes
                    .add(offset)
                    .cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>();
                (
                    record.Size as usize,
                    record.Anonymous.Processor.EfficiencyClass,
                )
            };
            if size == 0 {
                break;
            }
            classes.push(class);
            offset += size;
        }
        // Higher efficiency classes are the faster cores; a CPU that is not
        // hybrid reports one class for every core.
        let fastest = *classes.iter().max()?;
        Some(classes.iter().filter(|&&class| class == fastest).count())
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
mod platform {
    pub(super) fn performance_cores() -> Option<usize> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_bounded_by_what_the_process_may_use() {
        let available = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let threads = default_threads();
        assert!(threads >= 2);
        assert!(threads <= available.max(2), "{threads} > {available}");
    }

    #[cfg(any(target_os = "linux", target_os = "macos", windows))]
    #[test]
    fn this_machine_reports_its_cores() {
        let cores = performance_cores().expect("the platform reports a core count");
        let available = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(cores);
        assert!(cores >= 1);
        assert!(
            cores <= available.max(cores),
            "{cores} cores reported, {available} CPUs available"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn sysfs_counting_collapses_hyperthreads_and_skips_offline_cpus() {
        let root = std::env::temp_dir().join(format!("rai-topology-{}", std::process::id()));
        let cpu = |index: usize, package: &str, core: Option<&str>| {
            let topology = root.join(format!("cpu{index}")).join("topology");
            std::fs::create_dir_all(&topology).unwrap();
            if let Some(core) = core {
                std::fs::write(topology.join("physical_package_id"), package).unwrap();
                std::fs::write(topology.join("core_id"), core).unwrap();
            }
        };
        // Two cores with two threads each, plus an offline CPU.
        cpu(0, "0", Some("0"));
        cpu(1, "0", Some("1"));
        cpu(2, "0", Some("0"));
        cpu(3, "0", Some("1"));
        cpu(4, "0", None);
        std::fs::create_dir_all(root.join("cpufreq")).unwrap();
        assert_eq!(platform::count_cores(&root), Some(2));
        std::fs::remove_dir_all(&root).ok();
    }
}
