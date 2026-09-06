use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Streaming SHA-256 of an arbitrary file. Used for both shard files and the
/// assembled dataset so every artifact in a collection directory carries a
/// reproducible digest.
pub fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    use std::io::Read;
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("hash read failed: {e}"))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_encode(&hasher.finalize()))
}

/// Stable canonical-schema digest: hashing the rendered golden column list ties
/// collected shards to the exact schema they were produced under.
pub fn schema_hash() -> String {
    let columns = crate::schema::canonical_columns();
    let rendered: Vec<String> = columns
        .iter()
        .map(|c| {
            format!(
                "{}:{:?}:nullable={}:units={}",
                c.name,
                c.data_type,
                c.nullable,
                c.units.unwrap_or("-")
            )
        })
        .collect();
    let mut hasher = Sha256::new();
    hasher.update(rendered.join("\n"));
    hex_encode(&hasher.finalize())
}

/// In-memory SHA-256 for query hashing and recorded fixture manifests
/// (stage 4A); file-level counterparts live above.
pub fn sha256_hex(data: &[u8]) -> String {
    hex_encode(&Sha256::digest(data))
}

/// Counts data rows in a Gaia/NASA TAP CSV export (skips the single header line;
/// tolerates a trailing newline).
pub fn count_tap_csv_rows(path: &Path) -> Result<u64, String> {
    let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut first_line = String::new();
    reader
        .read_line(&mut first_line)
        .map_err(|e| format!("read header failed: {e}"))?;
    if first_line.trim().is_empty() && !first_line.contains(',') {
        // Only possible for truly empty bodies; treat as zero-row payload.
        return Ok(0);
    }
    let mut rows: u64 = 0;
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .map_err(|e| format!("read row failed: {e}"))?;
        if n == 0 {
            break;
        }
        if !line.trim().is_empty() {
            rows += 1;
        }
    }
    Ok(rows)
}

pub(crate) fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn sha256_is_stable_and_matches_known_vector() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.txt");
        std::fs::write(&p, b"hello").unwrap();
        // echo -n hello | sha256sum
        assert_eq!(
            sha256_file(&p).unwrap(),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn schema_hash_is_deterministic_across_calls() {
        assert_eq!(schema_hash(), schema_hash());
        assert_eq!(schema_hash().len(), 64);
    }

    #[test]
    fn tap_csv_row_counts_skip_header_and_blank_lines() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.csv");
        let mut f = File::create(&p).unwrap();
        writeln!(f, "source_id,ra,dec").unwrap();
        writeln!(f, "1,10.0,20.0").unwrap();
        writeln!(f).unwrap();
        writeln!(f, "2,11.0,21.0").unwrap();
        f.flush().unwrap();
        assert_eq!(count_tap_csv_rows(&p).unwrap(), 2);
    }
}
