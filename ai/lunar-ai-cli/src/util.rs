use anyhow::{Context, Result, anyhow};
use sha2::{Digest, Sha256};
use std::env;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::cli::args_train::CliModel;

pub fn resolve_data_path(data: Option<&str>) -> Result<PathBuf> {
    if let Some(p) = data {
        return Ok(PathBuf::from(p));
    }
    for candidate in [
        "ai_data/clean_stars2.parquet",
        "ai_data/clean_stars.parquet",
    ] {
        let path = Path::new(candidate);
        if path.exists() {
            println!("Using dataset: {}", path.display());
            return Ok(path.to_path_buf());
        }
    }
    anyhow::bail!(
        "No --data given and no cleaned parquet found in ai_data/. \
         Run 'lnaicli fetch' and 'lnaicli clean' first, or pass --data <path>."
    )
}

pub fn find_lnai_binary(override_path: Option<&str>, model: CliModel) -> Result<PathBuf> {
    if let Some(p) = override_path {
        let path = PathBuf::from(p);
        if !path.exists() {
            anyhow::bail!("--lnai-bin does not exist: {}", path.display());
        }
        return Ok(path);
    }

    let exe_suffix = env::consts::EXE_SUFFIX;
    let bin_name = format!("{}{}", model.worker_binary(), exe_suffix);

    if model == CliModel::Pinn
        && let Ok(path_env) = env::var("LNAI_BIN")
    {
        let p = PathBuf::from(path_env);
        if p.exists() {
            return Ok(p);
        }
    }

    if let Ok(paths) = env::var("PATH") {
        for dir in paths.split(std::path::MAIN_SEPARATOR_STR) {
            if dir.is_empty() {
                continue;
            }
            let candidate = Path::new(dir).join(&bin_name);
            if candidate.exists() {
                return Ok(candidate);
            }
        }
    }

    let workspace_root = env::current_dir().ok();
    if let Some(cwd) = workspace_root {
        for profile in ["release", "debug"] {
            for sub in ["", "ai/lnai"] {
                let candidate = cwd.join("target").join(profile);
                let candidate = if sub.is_empty() {
                    candidate.join(&bin_name)
                } else {
                    candidate.join(sub).join(&bin_name)
                };
                if candidate.exists() {
                    return Ok(candidate);
                }
            }
        }
    }

    anyhow::bail!(
        "Could not find the '{}' binary. Build it with `cargo build -p {} --release` \
         or pass --lnai-bin /path/to/{}.",
        bin_name,
        match model {
            CliModel::Pinn => "lnai",
            CliModel::Gnn => "lnai-gnn",
            CliModel::Siren => "lnai-siren",
        },
        bin_name,
    )
}

pub fn copy_file_if_present(src: &Path, dst: &Path) -> Result<()> {
    if !src.exists() {
        return Ok(());
    }
    if same_file(src, dst) {
        return Ok(());
    }
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::copy(src, dst)
        .with_context(|| format!("failed to copy {} -> {}", src.display(), dst.display()))?;
    Ok(())
}

pub fn same_file(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

pub fn sha256_file(path: &str) -> Result<String> {
    let mut file =
        File::open(path).map_err(|e| anyhow!("Cannot open {} for hashing: {}", path, e))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    Ok(hex_encode(&digest))
}

pub fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}
