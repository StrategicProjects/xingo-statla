//! Timing harness of the crate (README, "Measured time"; issue
//! xingo-statla#8): the updatable factorizations `Cholesky` and `Qr` in
//! `f64` and double-double, the iterative drivers `eigsh`, `eigs`,
//! `eigsh_near`, `eigs_near`, `svds` and the randomized `rsvd` in `f64`,
//! and the same drivers in double-double next to their `f64` runs.
//!
//! **The problems** are those the harnesses already use, so that the
//! numbers can be read against the accuracy they come with:
//!
//! * the factorizations and their updates are those of `ACCURACY_TIMING`
//!   in the example `accuracy`: an SPD matrix of order 200 with `κ = 10⁶`
//!   (`spd(200, 1e6, 1)`), the update vector `x` of norm `½`, and a
//!   `400 × 200` matrix with `κ = 10³` (`with_cond(400, 200, 1e3, 3)`),
//!   drawn by the generator of `tests/support` with the same seeds; each
//!   update acts on the factor of the shape it had there (`delete_row` on
//!   the `401 × 200` factor left by `insert_row`, `delete_col` on the
//!   `400 × 201` one left by `insert_col`, the others on `400 × 200`);
//! * the drivers run on the Matrix Market fixtures of `tests/arpack.rs`
//!   and on its three large matrices (rebuilt here, entry for entry as in
//!   `make_fixtures.py`, and checked against the `checksum` record of
//!   their reference files), `k = 6`, default options: `eigsh` with `LA`
//!   on the symmetric ones, `eigs` with `LR` on the others, shift-invert
//!   at the first shift of the reference file, `svds` and `rsvd`;
//! * the double-double runs are the smoke tests: `eigsh` (`LA`, `k = 4`)
//!   and `eigsh_near` (`σ = 3`, `k = 3`) on the graded tridiagonal of
//!   order 200, `eigs` (`LM`, `k = 4`) and `eigs_near` (`σ = 0.25`, `k =
//!   4`) on the random nonsymmetric matrix of order 200 (`tests/arpack.rs`),
//!   `rsvd` on the exactly rank-5 `40 × 30` matrix (`tests/rsvd.rs`) and
//!   `svds` on the `90 × 60` matrix with a geometric spectrum down to
//!   `10⁻²⁰` (`tests/svds.rs`), each also in `f64` on the same matrix
//!   (rounded).
//!
//! **The measurement** (Algorithm T):
//!
//! 1. *Setup*, not timed: the input of one call (a clone of the factor or
//!    of the matrix: every call mutates or may mutate it) is built before
//!    the clock starts, and the result is dropped after it stops.
//! 2. *Calibration*: the number of calls per round doubles from 1 until a
//!    round takes at least half of [`ROUND_NANOS`], and is then scaled to
//!    a round of about [`ROUND_NANOS`] (at least one call: a driver that
//!    takes longer than a round is timed once per round). The calibration
//!    calls are the warm-up.
//! 3. *Rounds*: [`ROUNDS`] rounds; a round's time is the sum of its calls'
//!    times divided by their number.
//! 4. *Report*: the median, the minimum and the maximum of the rounds, in
//!    microseconds (factorizations) or milliseconds (drivers). The results
//!    go through [`std::hint::black_box`].
//!
//! The header records the target and the load average of the machine at
//! the start (`/proc/loadavg`, else `sysctl -n vm.loadavg`, else
//! "unknown"), the footer the load average at the end. The figures depend
//! on the machine and on its load; nothing here asserts a speed, and the
//! correctness of the results is the business of the tests (only the
//! checksums of the rebuilt matrices are checked).
//!
//! Run with `cargo run --release -p xingo-statla --example timing` (no
//! feature is needed; about a minute on an Apple M2 Pro).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]

#[path = "../tests/support/mod.rs"]
mod support;

use std::hint::black_box;
use std::time::Instant;

use support::mm::{Ref, read_mm};
use support::svd::prescribed_dd;
use support::{Rng, Scalar, conv, spd, unit, with_cond};
use xingo_extprec::DoubleDouble as Dd;
use xingo_sparse::{Col, CscMatrix};
use xingo_statla::{
    CholOptions, Cholesky, DenseBlock, EigsOptions, MatRef, Qr, QrOptions, RsvdOptions,
    SvdsOptions, Which, eigs, eigs_near, eigsh, eigsh_near, rsvd, svds,
};

/// Rounds per row (the median, minimum and maximum are over these).
const ROUNDS: usize = 21;
/// Target duration of one round (step 2 of Algorithm T).
const ROUND_NANOS: u128 = 20_000_000;
/// Upper bound of the calls per round.
const MAX_REPS: u128 = 1 << 20;
/// Wanted values of the `f64` drivers (the `k` of the reference files).
const K: usize = 6;

/// One timed row: median, minimum and maximum over the rounds, in seconds
/// per call.
#[derive(Clone, Copy)]
struct Stat {
    median: f64,
    min: f64,
    max: f64,
}

impl Stat {
    /// `median (min–max)` in units of `scale` seconds.
    fn cell(self, scale: f64) -> String {
        format!(
            "{} ({}–{})",
            num(self.median / scale),
            num(self.min / scale),
            num(self.max / scale)
        )
    }
}

/// Three significant figures or more: three decimals below 1, two below
/// 10, one below 100, none above, with the thousands separated by a space.
fn num(v: f64) -> String {
    if v < 1.0 {
        format!("{v:.3}")
    } else if v < 10.0 {
        format!("{v:.2}")
    } else if v < 100.0 {
        format!("{v:.1}")
    } else {
        let digits = format!("{v:.0}");
        let mut out = String::new();
        for (i, ch) in digits.chars().enumerate() {
            if i > 0 && (digits.len() - i) % 3 == 0 {
                out.push(' ');
            }
            out.push(ch);
        }
        out
    }
}

/// Algorithm T: `setup` builds the input of one call (not timed), `op` is
/// the call.
fn measure<S, R>(mut setup: impl FnMut() -> S, mut op: impl FnMut(S) -> R) -> Stat {
    // Step 1: one call, in nanoseconds, its setup and drop outside.
    let mut once = || {
        let s = setup();
        let t = Instant::now();
        let r = op(black_box(s));
        let dt = t.elapsed().as_nanos();
        drop(black_box(r));
        dt
    };
    // Step 2.
    let mut reps: u128 = 1;
    loop {
        let dt: u128 = (0..reps).map(|_| once()).sum();
        if dt >= ROUND_NANOS / 2 || reps >= MAX_REPS {
            reps = (reps * ROUND_NANOS / dt.max(1)).clamp(1, MAX_REPS);
            break;
        }
        reps *= 2;
    }
    // Step 3.
    let mut per_call: Vec<f64> = (0..ROUNDS)
        .map(|_| {
            let total: u128 = (0..reps).map(|_| once()).sum();
            total as f64 * 1e-9 / reps as f64
        })
        .collect();
    // Step 4.
    per_call.sort_by(f64::total_cmp);
    Stat {
        median: per_call[ROUNDS / 2],
        min: per_call[0],
        max: per_call[ROUNDS - 1],
    }
}

/// The rows of the factorizations in scalar `T` (the problems of
/// `ACCURACY_TIMING` in the example `accuracy`, same seeds).
fn factorizations<T: Scalar>(n: usize, m: usize) -> Vec<(String, Stat)> {
    let mut rows = Vec::new();
    let a: Vec<T> = conv(&spd(n, 1e6, 1));
    let mut rng = Rng::new(2);
    let x: Vec<T> = conv(
        &unit(&mut rng, n)
            .iter()
            .map(|v| v * 0.5)
            .collect::<Vec<_>>(),
    );
    let am = MatRef::new(n, n, n, &a).unwrap();
    let opts = CholOptions::default();
    rows.push((
        format!("`Cholesky::factor`, `n = {n}`"),
        measure(|| (), |()| Cholesky::factor(am, &opts).unwrap()),
    ));
    let c = Cholesky::factor(am, &opts).unwrap();
    rows.push((
        "`Cholesky::update`".to_string(),
        measure(
            || c.clone(),
            |mut d| {
                d.update(&x).unwrap();
                d
            },
        ),
    ));
    let mut c_up = c.clone();
    c_up.update(&x).unwrap();
    rows.push((
        "`Cholesky::downdate`".to_string(),
        measure(
            || c_up.clone(),
            |mut d| {
                d.downdate(&x).unwrap();
                d
            },
        ),
    ));

    let b: Vec<T> = conv(&with_cond(m, n, 1e3, 3));
    let bm = MatRef::new(m, n, m, &b).unwrap();
    let qopts = QrOptions::default();
    let w: Vec<T> = conv(&rng.gauss_vec(n));
    let col: Vec<T> = conv(&rng.gauss_vec(m));
    rows.push((
        format!("`Qr::factor`, `{m} × {n}`"),
        measure(|| (), |()| Qr::factor(bm, &qopts).unwrap()),
    ));
    let f = Qr::factor(bm, &qopts).unwrap();
    rows.push((
        "`Qr::insert_row`".to_string(),
        measure(
            || f.clone(),
            |mut g| {
                g.insert_row(m / 2, &w).unwrap();
                g
            },
        ),
    ));
    let mut f_row = f.clone();
    f_row.insert_row(m / 2, &w).unwrap();
    rows.push((
        "`Qr::delete_row`".to_string(),
        measure(
            || f_row.clone(),
            |mut g| {
                g.delete_row(m / 2).unwrap();
                g
            },
        ),
    ));
    rows.push((
        "`Qr::insert_col` (`j = 0`)".to_string(),
        measure(
            || f.clone(),
            |mut g| {
                g.insert_col(0, &col).unwrap();
                g
            },
        ),
    ));
    let mut f_col = f.clone();
    f_col.insert_col(0, &col).unwrap();
    rows.push((
        "`Qr::delete_col` (`j = 0`)".to_string(),
        measure(
            || f_col.clone(),
            |mut g| {
                g.delete_col(0).unwrap();
                g
            },
        ),
    ));
    rows.push((
        "`Qr::rank1_update`".to_string(),
        measure(
            || f.clone(),
            |mut g| {
                g.rank1_update(&col, &w).unwrap();
                g
            },
        ),
    ));
    rows
}

/// A CSC matrix from `(row, col, value)` triplets (distinct positions), as
/// in `tests/arpack.rs`.
fn from_triplets(n: usize, mut t: Vec<(usize, usize, f64)>) -> CscMatrix<f64> {
    t.sort_by_key(|&(i, j, _)| (j, i));
    let mut col_ptr = vec![0usize; n + 1];
    for &(_, j, _) in &t {
        col_ptr[j + 1] += 1;
    }
    for j in 0..n {
        col_ptr[j + 1] += col_ptr[j];
    }
    CscMatrix {
        n_rows: n,
        col_ptr,
        row_idx: t.iter().map(|&(i, _, _)| i as u32).collect(),
        val: t.iter().map(|&(_, _, v)| v).collect(),
    }
}

/// `convdiff(nx, ny, px, py)` of `make_fixtures.py` (`lap2d` is `px = py =
/// 0`), as in `tests/arpack.rs`.
fn grid(nx: usize, ny: usize, px: f64, py: f64) -> CscMatrix<f64> {
    let mut t = Vec::new();
    for iy in 0..ny {
        for ix in 0..nx {
            let p = ix + nx * iy;
            t.push((p, p, 2.0 + 2.0));
            if ix + 1 < nx {
                t.push((p + 1, p, -1.0 - px));
                t.push((p, p + 1, -1.0 + px));
            }
            if iy + 1 < ny {
                t.push((p + nx, p, -1.0 - py));
                t.push((p, p + nx, -1.0 + py));
            }
        }
    }
    from_triplets(nx * ny, t)
}

/// `tridiag_graded(n)` of `make_fixtures.py`: `tridiag(−1, 2 + i/50, −1)`.
fn tridiag_graded(n: usize) -> CscMatrix<f64> {
    let mut t = Vec::new();
    for i in 0..n {
        t.push((i, i, 2.0 + i as f64 / 50.0));
        if i + 1 < n {
            t.push((i + 1, i, -1.0));
            t.push((i, i + 1, -1.0));
        }
    }
    from_triplets(n, t)
}

/// The matrix of a fixture: read from `tests/data`, or, for the large
/// ones, rebuilt and checked against the `checksum` record (`Σ a_ij (i +
/// 1)(2j + 1)`, summed in double-double).
fn fixture(name: &str) -> (CscMatrix<f64>, Ref) {
    let r = Ref::read(name);
    let a = match name {
        "lap2d_60x50" => grid(60, 50, 0.0, 0.0),
        "tridiag_graded_4000" => tridiag_graded(4000),
        "convdiff_60x50" => grid(60, 50, 0.1, 0.05),
        _ => return (read_mm(name).a, r),
    };
    let mut s = Dd::ZERO;
    for j in 0..a.n_cols() {
        let (rows, vals) = a.col(Col::from_usize(j));
        for (&i, &v) in rows.iter().zip(vals) {
            s += Dd::from_f64(v) * Dd::from_f64((f64::from(i) + 1.0) * (2.0 * j as f64 + 1.0));
        }
    }
    let want = r.values("checksum").unwrap()[0];
    assert!(
        (s.to_f64() - want).abs() <= 1e-15 * want.abs(),
        "{name}: rebuilt matrix differs from make_fixtures.py"
    );
    (a, r)
}

/// The cells of one fixture: `eigsh` LA or `eigs` LR, shift-invert at the
/// first shift of the reference file, `svds`, `rsvd`; `None` where the
/// fixture has no such record.
struct DriverRow {
    label: &'static str,
    n: usize,
    krylov: Option<Stat>,
    near: Option<(f64, Stat)>,
    svds: Stat,
    rsvd: Stat,
}

fn drivers(name: &str, label: &'static str) -> DriverRow {
    let (a, r) = fixture(name);
    let n = a.n_cols();
    let o = EigsOptions::default();
    let sym = !r.tagged("eigsh").is_empty();
    let krylov = if sym {
        Some(measure(
            || a.clone(),
            |mut b| eigsh(&mut b, K, Which::LargestAlgebraic, &o).unwrap(),
        ))
    } else if !r.tagged("eigs").is_empty() {
        Some(measure(
            || a.clone(),
            |mut b| eigs(&mut b, K, Which::LargestReal, &o).unwrap(),
        ))
    } else {
        None
    };
    let near = if sym {
        r.tagged("eigsh_near").first().map(|(tag, _, _)| {
            let sigma: f64 = tag.parse().unwrap();
            (
                sigma,
                measure(|| (), |()| eigsh_near(&a, K, sigma, &o).unwrap()),
            )
        })
    } else {
        r.tagged("eigs_near").first().map(|(tag, _, _)| {
            let sigma: f64 = tag.parse().unwrap();
            (
                sigma,
                measure(|| (), |()| eigs_near(&a, K, sigma, &o).unwrap()),
            )
        })
    };
    let k = r.values("svds").unwrap()[0] as usize;
    let so = SvdsOptions::default();
    let svds_t = measure(|| a.clone(), |mut b| svds(&mut b, k, &so).unwrap());
    let ro = RsvdOptions::default();
    let rsvd_t = measure(|| a.clone(), |mut b| rsvd(&mut b, k, &ro).unwrap());
    DriverRow {
        label,
        n,
        krylov,
        near,
        svds: svds_t,
        rsvd: rsvd_t,
    }
}

/// The matrix in double-double (exact conversion).
fn csc_dd(a: &CscMatrix<f64>) -> CscMatrix<Dd> {
    CscMatrix {
        n_rows: a.n_rows,
        col_ptr: a.col_ptr.clone(),
        row_idx: a.row_idx.clone(),
        val: a.val.iter().map(|&v| Dd::from_f64(v)).collect(),
    }
}

/// `(call, f64, double-double)` for the smoke problems in double-double.
fn double_double() -> Vec<(String, Stat, Stat)> {
    let mut rows = Vec::new();
    let a = read_mm("tridiag_graded_200").a;
    let ad = csc_dd(&a);
    let (o, od) = (EigsOptions::default(), EigsOptions::<Dd>::default());
    rows.push((
        "`eigsh` LA, `k = 4`, graded tridiagonal `n = 200`".to_string(),
        measure(
            || a.clone(),
            |mut b| eigsh(&mut b, 4, Which::LargestAlgebraic, &o).unwrap(),
        ),
        measure(
            || ad.clone(),
            |mut b| eigsh(&mut b, 4, Which::LargestAlgebraic, &od).unwrap(),
        ),
    ));
    rows.push((
        "`eigsh_near` `σ = 3`, `k = 3`, the same".to_string(),
        measure(|| (), |()| eigsh_near(&a, 3, 3.0, &o).unwrap()),
        measure(
            || (),
            |()| eigsh_near(&ad, 3, Dd::from_f64(3.0), &od).unwrap(),
        ),
    ));
    let a = read_mm("nonsymrand_200").a;
    let ad = csc_dd(&a);
    rows.push((
        "`eigs` LM, `k = 4`, nonsymmetric random `n = 200`".to_string(),
        measure(
            || a.clone(),
            |mut b| eigs(&mut b, 4, Which::LargestMagnitude, &o).unwrap(),
        ),
        measure(
            || ad.clone(),
            |mut b| eigs(&mut b, 4, Which::LargestMagnitude, &od).unwrap(),
        ),
    ));
    rows.push((
        "`eigs_near` `σ = 0.25`, `k = 4`, the same".to_string(),
        measure(|| (), |()| eigs_near(&a, 4, 0.25, &o).unwrap()),
        measure(
            || (),
            |()| eigs_near(&ad, 4, Dd::from_f64(0.25), &od).unwrap(),
        ),
    ));
    // tests/rsvd.rs, `double_double_smoke`.
    let mut sig = vec![0.0; 30];
    for (i, v) in sig.iter_mut().take(5).enumerate() {
        *v = 2f64.powi(-(i as i32));
    }
    let dd = prescribed_dd(40, 30, &sig, 4, 11);
    let (bd, b) = dense_pair(40, 30, dd);
    let ro = RsvdOptions::default();
    rows.push((
        "`rsvd`, `k = 5`, rank-5 dense `40 × 30`".to_string(),
        measure(|| b.clone(), |mut m| rsvd(&mut m, 5, &ro).unwrap()),
        measure(|| bd.clone(), |mut m| rsvd(&mut m, 5, &ro).unwrap()),
    ));
    // tests/svds.rs, `double_double_smoke`.
    let sigma: Vec<f64> = (0..60)
        .map(|j| 1e-20f64.powf(f64::from(j) / 59.0))
        .collect();
    let dd = prescribed_dd(90, 60, &sigma, 3, 9);
    let (bd, b) = dense_pair(90, 60, dd);
    let (so, sod) = (SvdsOptions::default(), SvdsOptions::<Dd>::default());
    rows.push((
        "`svds`, `k = 5`, dense `90 × 60`, `σ` geometric `1 … 1e-20`".to_string(),
        measure(|| b.clone(), |mut m| svds(&mut m, 5, &so).unwrap()),
        measure(|| bd.clone(), |mut m| svds(&mut m, 5, &sod).unwrap()),
    ));
    rows
}

/// A dense double-double matrix and its rounding to `f64`.
fn dense_pair(rows: usize, cols: usize, data: Vec<Dd>) -> (DenseBlock<Dd>, DenseBlock<f64>) {
    let low = data.iter().map(|x| x.to_f64()).collect();
    (
        DenseBlock { rows, cols, data },
        DenseBlock {
            rows,
            cols,
            data: low,
        },
    )
}

/// The load average, as the system reports it.
fn load_average() -> String {
    if let Ok(s) = std::fs::read_to_string("/proc/loadavg") {
        return s.split_whitespace().take(3).collect::<Vec<_>>().join(" ");
    }
    match std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
    {
        Ok(out) if out.status.success() => {
            let s = String::from_utf8_lossy(&out.stdout);
            let s = s
                .trim()
                .trim_start_matches('{')
                .trim_end_matches('}')
                .trim();
            if s.is_empty() {
                "unknown".to_string()
            } else {
                s.to_string()
            }
        }
        _ => "unknown".to_string(),
    }
}

fn main() {
    let start = Instant::now();
    println!(
        "timing (xingo-statla {}): target {}-{}, load average at start: {}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::ARCH,
        std::env::consts::OS,
        load_average()
    );
    println!(
        "median (min–max) of {ROUNDS} rounds of about {} ms each; setup and drop not timed",
        ROUND_NANOS / 1_000_000
    );

    println!("\n### Updatable factorizations (µs per call)\n");
    println!("| operation | `f64` | double-double | dd / f64 |");
    println!("|---|---:|---:|---:|");
    let f = factorizations::<f64>(200, 400);
    let d = factorizations::<Dd>(200, 400);
    for ((name, x), (_, y)) in f.iter().zip(&d) {
        println!(
            "| {name} | {} | {} | {:.1} |",
            x.cell(1e-6),
            y.cell(1e-6),
            y.median / x.median
        );
    }

    println!(
        "\n### Iterative drivers in `f64` (ms per call, `k = {K}`, default options; shift-invert at the first shift of the reference file)\n"
    );
    println!(
        "| matrix | `n` | `eigsh` LA / `eigs` LR | shift-invert (`eigsh_near` / `eigs_near`) | `svds` | `rsvd` |"
    );
    println!("|---|---:|---:|---:|---:|---:|");
    for (name, label) in [
        ("tridiag_graded_200", "graded tridiagonal"),
        ("symrand_300", "symmetric random"),
        ("lap2d_20x17", "2-D Laplacian `20 × 17`"),
        ("convdiff_20x20", "convection–diffusion `20 × 20`"),
        ("nonsymrand_200", "nonsymmetric random"),
        ("rectrand_300x200", "rectangular random `300 × 200`"),
        ("lap2d_60x50", "2-D Laplacian `60 × 50`"),
        ("tridiag_graded_4000", "graded tridiagonal"),
        ("convdiff_60x50", "convection–diffusion `60 × 50`"),
    ] {
        let r = drivers(name, label);
        let dash = || "—".to_string();
        println!(
            "| {} | {} | {} | {} | {} | {} |",
            r.label,
            r.n,
            r.krylov.map_or_else(dash, |s| s.cell(1e-3)),
            r.near.map_or_else(dash, |(sigma, s)| format!(
                "{}, `σ = {sigma}`",
                s.cell(1e-3)
            )),
            r.svds.cell(1e-3),
            r.rsvd.cell(1e-3)
        );
    }

    println!("\n### Double-double against `f64` on the same matrix (ms per call)\n");
    println!("| call | `f64` | double-double | dd / f64 |");
    println!("|---|---:|---:|---:|");
    for (name, x, y) in double_double() {
        println!(
            "| {name} | {} | {} | {:.1} |",
            x.cell(1e-3),
            y.cell(1e-3),
            y.median / x.median
        );
    }

    println!(
        "\nload average at end: {}; total {:.0} s",
        load_average(),
        start.elapsed().as_secs_f64()
    );
}
