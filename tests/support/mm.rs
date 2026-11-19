//! A minimal Matrix Market reader and the reader of the ARPACK reference
//! files of `tests/data` (task L3.4).
//!
//! Test support only: the contract (D-0127) has no Matrix Market reader in
//! the public surface, so this lives with the tests. It reads the
//! `coordinate` format with `real`, `integer` or `pattern` entries and
//! `general` or `symmetric` symmetry (the symmetric ones are expanded to
//! the full matrix), 1-based indices, `%` comments; anything else is an
//! error message (`Err(String)`). The reference files are described in
//! `tests/data/make_fixtures.py`.

use std::fs;
use std::path::PathBuf;

use xingo_sparse::{Col, CscMatrix, Row};

/// The directory of the fixtures.
pub fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

/// A Matrix Market matrix: the full CSC matrix and whether the file was
/// `symmetric`.
pub struct Mm {
    pub a: CscMatrix<f64>,
    pub symmetric: bool,
}

/// Parse a Matrix Market `coordinate` file.
pub fn parse_mm(text: &str) -> Result<Mm, String> {
    let mut lines = text.lines();
    let header = lines.next().ok_or("empty file")?;
    let h: Vec<String> = header.split_whitespace().map(str::to_lowercase).collect();
    if h.len() != 5 || h[0] != "%%matrixmarket" || h[1] != "matrix" {
        return Err(format!("bad header: {header}"));
    }
    if h[2] != "coordinate" {
        return Err(format!("unsupported format {}", h[2]));
    }
    let pattern = match h[3].as_str() {
        "real" | "integer" => false,
        "pattern" => true,
        f => return Err(format!("unsupported field {f}")),
    };
    let symmetric = match h[4].as_str() {
        "general" => false,
        "symmetric" => true,
        s => return Err(format!("unsupported symmetry {s}")),
    };
    let mut data = lines.filter(|l| !l.trim_start().starts_with('%') && !l.trim().is_empty());
    let size = data.next().ok_or("no size line")?;
    let dims: Vec<usize> = size
        .split_whitespace()
        .map(|t| t.parse().map_err(|e| format!("size line: {e}")))
        .collect::<Result<_, _>>()?;
    if dims.len() != 3 {
        return Err(format!("bad size line: {size}"));
    }
    let (m, n, nnz) = (dims[0], dims[1], dims[2]);
    if symmetric && m != n {
        return Err("symmetric but not square".into());
    }
    let mut trip = Vec::with_capacity(if symmetric { 2 * nnz } else { nnz });
    let mut count = 0;
    for l in data {
        let t: Vec<&str> = l.split_whitespace().collect();
        let want = if pattern { 2 } else { 3 };
        if t.len() < want {
            return Err(format!("bad entry: {l}"));
        }
        let i: usize = t[0].parse().map_err(|e| format!("row: {e}"))?;
        let j: usize = t[1].parse().map_err(|e| format!("col: {e}"))?;
        if i == 0 || j == 0 || i > m || j > n {
            return Err(format!("index out of range: {l}"));
        }
        let v: f64 = if pattern {
            1.0
        } else {
            t[2].parse().map_err(|e| format!("value: {e}"))?
        };
        trip.push((Row::from_usize(i - 1), Col::from_usize(j - 1), v));
        if symmetric && i != j {
            trip.push((Row::from_usize(j - 1), Col::from_usize(i - 1), v));
        }
        count += 1;
    }
    if count != nnz {
        return Err(format!("{count} entries, header says {nnz}"));
    }
    let a = CscMatrix::from_triplets(m, n, &trip).map_err(|e| format!("{e:?}"))?;
    Ok(Mm { a, symmetric })
}

/// Read `tests/data/<name>.mtx`.
pub fn read_mm(name: &str) -> Mm {
    let path = data_dir().join(format!("{name}.mtx"));
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    parse_mm(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The records of a reference file: whitespace-separated tokens per line.
pub struct Ref {
    pub lines: Vec<Vec<String>>,
}

impl Ref {
    /// Read `tests/data/<name>.ref`.
    pub fn read(name: &str) -> Self {
        let path = data_dir().join(format!("{name}.ref"));
        let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let lines = text
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .map(|l| l.split_whitespace().map(str::to_owned).collect())
            .collect();
        Self { lines }
    }

    /// The numbers after the key of the single record `key`.
    pub fn values(&self, key: &str) -> Option<Vec<f64>> {
        self.lines
            .iter()
            .find(|l| l[0] == key)
            .map(|l| l[1..].iter().map(|t| t.parse().unwrap()).collect())
    }

    /// The records `key <tag> <k> <values…>`: `(tag, k, values)`.
    pub fn tagged(&self, key: &str) -> Vec<(String, usize, Vec<f64>)> {
        self.lines
            .iter()
            .filter(|l| l[0] == key)
            .map(|l| {
                (
                    l[1].clone(),
                    l[2].parse().unwrap(),
                    l[3..].iter().map(|t| t.parse().unwrap()).collect(),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reader_accepts_and_rejects() {
        let ok = "%%MatrixMarket matrix coordinate real symmetric\n% c\n2 2 2\n1 1 1.5\n2 1 -2\n";
        let m = parse_mm(ok).unwrap();
        assert!(m.symmetric && m.a.nnz() == 3);
        let p = "%%MatrixMarket matrix coordinate pattern general\n2 3 1\n1 3\n";
        assert_eq!(parse_mm(p).unwrap().a.n_cols(), 3);
        for bad in [
            "",
            "%%MatrixMarket matrix array real general\n1 1\n1\n",
            "%%MatrixMarket matrix coordinate complex general\n1 1 1\n1 1 1 0\n",
            "%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 1\n",
            "%%MatrixMarket matrix coordinate real general\n2 2 1\n3 1 1\n",
            "%%MatrixMarket matrix coordinate real symmetric\n2 3 1\n1 1 1\n",
        ] {
            assert!(parse_mm(bad).is_err(), "{bad}");
        }
    }
}
