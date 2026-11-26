//! Strongly nonnormal matrices for `eigs` (Chapter 12.6; issue
//! xingo-statla#2): the families on which the coupling check of step KS6
//! — lock a converged wanted Schur block only if *its entries of `b_A`*
//! pass the test of KS5, not only the eigenvector's estimate — refuses
//! blocks, and on which removing it breaks criterion (b).
//!
//! For a normal `T_A` the Schur vector of a leading block is its
//! eigenvector and the two tests agree. For a nonnormal one the Schur
//! vector `q` of a later block differs from the eigenvector `x`, and
//! `|b_Aᵀ q|` can exceed the estimate `|b_Aᵀ x|/‖x‖` by the departure from
//! normality; locking such a block drops a coupling that is not small, a
//! perturbation of `A` that the returned residuals then show. Measured on
//! the families below (task I01): the check refuses blocks a few times per
//! run, and without it the largest residual ratio of criterion (b) goes
//! from `1.1` to `7.4·10³` (Jordan-like blocks, `LM`), `1.0` to `6.3·10⁴`
//! (`SR`) and `5.7` to `95` (convection-dominated, `LR`).
//!
//! `‖A‖₂` is estimated from above by `√(‖A‖₁‖A‖_∞)`, so the ratios here are
//! at most those against the true norm. Every table line is printed with
//! `--nocapture`.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

mod support;

use xingo_extprec::DoubleDouble as Dd;
use xingo_sparse::{Col, CscMatrix};
use xingo_statla::{Complex, EigsOptions, Which, eigs};

/// `c_K` of criterion (b), as in `tests/arpack.rs`.
const C_K: f64 = 8.0;

/// A CSC matrix from a dense column-major one (exact zeros dropped).
fn from_dense(n: usize, d: &[f64]) -> CscMatrix<f64> {
    let mut col_ptr = vec![0usize];
    let (mut row_idx, mut val) = (Vec::new(), Vec::new());
    for j in 0..n {
        for i in 0..n {
            let v = d[i + j * n];
            if v != 0.0 {
                row_idx.push(u32::try_from(i).unwrap());
                val.push(v);
            }
        }
        col_ptr.push(row_idx.len());
    }
    CscMatrix {
        n_rows: n,
        col_ptr,
        row_idx,
        val,
    }
}

/// Jordan-like blocks perturbed: `n/4` blocks `a_b I + N + ε e_4 e_1ᵀ` of
/// order 4 (`N` the shift, `a_b = 1 + U(0, 1)` from `seed`), whose
/// eigenvalues are exactly `a_b + ε^{1/4} iᵏ`, `k = 0..3` (the roots of
/// `(λ − a_b)⁴ = ε`), each with condition number `≈ ε^{−3/4}/4`.
fn jordan_like(n: usize, eps: f64, seed: u64) -> (CscMatrix<f64>, Vec<f64>) {
    let mut d = vec![0.0; n * n];
    let mut rng = support::Rng::new(seed);
    let mut centers = Vec::new();
    for b in 0..n / 4 {
        let i = 4 * b;
        let a = 1.0 + rng.uniform();
        centers.push(a);
        for t in 0..4 {
            d[i + t + (i + t) * n] = a;
            if t < 3 {
                d[i + t + (i + t + 1) * n] = 1.0;
            }
        }
        d[i + 3 + i * n] = eps;
    }
    (from_dense(n, &d), centers)
}

/// `tridiag(−1 − p, 2, −1 + p)` of order `n`: convection-dominated for `p`
/// near 1 (eigenvector condition `((1 + p)/(1 − p))^{(n−1)/2}`).
fn convection(n: usize, p: f64) -> CscMatrix<f64> {
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        d[i + i * n] = 2.0;
        if i + 1 < n {
            d[i + 1 + i * n] = -1.0 - p;
            d[i + (i + 1) * n] = -1.0 + p;
        }
    }
    from_dense(n, &d)
}

/// The Grcar matrix of order `n` with `k` superdiagonals of ones and `−1`
/// on the subdiagonal.
fn grcar(n: usize, k: usize) -> CscMatrix<f64> {
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in i..(i + k + 1).min(n) {
            d[i + j * n] = 1.0;
        }
        if i + 1 < n {
            d[i + 1 + i * n] = -1.0;
        }
    }
    from_dense(n, &d)
}

/// `√(‖A‖₁ ‖A‖_∞) ≥ ‖A‖₂`.
fn norm_bound(a: &CscMatrix<f64>) -> f64 {
    let n = a.n_rows();
    let mut rows = vec![0.0f64; n];
    let mut n1 = 0.0f64;
    for j in 0..a.n_cols() {
        let (ri, vals) = a.col(Col::from_usize(j));
        let mut c = 0.0;
        for (&i, &v) in ri.iter().zip(vals) {
            c += v.abs();
            rows[i as usize] += v.abs();
        }
        n1 = n1.max(c);
    }
    (n1 * rows.iter().copied().fold(0.0, f64::max)).sqrt()
}

/// `‖A y − θ y‖` for a complex pair, in double-double.
fn residual(a: &CscMatrix<f64>, yr: &[f64], yi: &[f64], th: Complex<f64>) -> f64 {
    let n = a.n_rows();
    let (mut ar, mut ai) = (vec![Dd::ZERO; n], vec![Dd::ZERO; n]);
    for j in 0..a.n_cols() {
        let (rows, vals) = a.col(Col::from_usize(j));
        for (&i, &v) in rows.iter().zip(vals) {
            let v = Dd::from_f64(v);
            ar[i as usize] += v * Dd::from_f64(yr[j]);
            ai[i as usize] += v * Dd::from_f64(yi[j]);
        }
    }
    let (tr, ti) = (Dd::from_f64(th.re), Dd::from_f64(th.im));
    let mut s = Dd::ZERO;
    for i in 0..n {
        let (yr, yi) = (Dd::from_f64(yr[i]), Dd::from_f64(yi[i]));
        let er = ar[i] - (tr * yr - ti * yi);
        let ei = ai[i] - (tr * yi + ti * yr);
        s += er * er + ei * ei;
    }
    s.sqrt().to_f64()
}

/// One run: all `k` converge, and the largest residual ratio
/// `(‖r‖/‖A‖ − τ)/(ncv u)` of criterion (b). Returns the values too.
fn run(
    name: &str,
    a: &CscMatrix<f64>,
    which: Which,
    k: usize,
    ncv: usize,
) -> (Vec<Complex<f64>>, f64, usize) {
    let (u, tau) = (f64::EPSILON / 2.0, f64::EPSILON);
    let n = a.n_rows();
    let anorm = norm_bound(a);
    let o = EigsOptions {
        ncv: Some(ncv),
        ..EigsOptions::default()
    };
    let f = eigs(&mut a.clone(), k, which, &o)
        .unwrap_or_else(|e| panic!("{name} {which:?} k = {k} ncv = {ncv}: {e}"));
    assert_eq!(f.values.len(), k);
    let (yr, yi) = (
        f.vectors_re.as_ref().unwrap(),
        f.vectors_im.as_ref().unwrap(),
    );
    let mut worst = 0.0f64;
    for (c, th) in f.values.iter().enumerate() {
        let r = residual(
            a,
            &yr.data[c * n..(c + 1) * n],
            &yi.data[c * n..(c + 1) * n],
            *th,
        );
        worst = worst.max((r / anorm - tau) / (ncv as f64 * u));
    }
    eprintln!(
        "| {name} | {which:?} | k = {k}, ncv = {ncv} | {worst:.3} | {} |",
        f.stats.restarts
    );
    (f.values, worst, f.stats.restarts)
}

/// Jordan-like blocks with `ε = 10⁻¹²` (eigenvalue condition `≈ 2.5·10⁸`):
/// criterion (b) for `LM` and `SR`, `k = 16`, `ncv = 40`, and every value
/// within `10⁻⁶` of an exact eigenvalue (the condition number times `u`
/// is `≈ 3·10⁻⁸`). Without the coupling check of KS6 the residual ratios
/// are `7.4·10³` and `6.3·10⁴`.
#[test]
fn coupling_check_on_jordan_like_blocks() {
    let eps = 1e-12;
    let (a, centers) = jordan_like(200, eps, 3);
    let r4 = eps.powf(0.25);
    for which in [Which::LargestMagnitude, Which::SmallestReal] {
        let (vals, worst, _) = run("jordan_like_200 ε=1e-12", &a, which, 16, 40);
        assert!(worst <= C_K, "{which:?}: residual ratio {worst}");
        for th in &vals {
            let d = centers
                .iter()
                .flat_map(|&c| [(c + r4, 0.0), (c - r4, 0.0), (c, r4), (c, -r4)])
                .map(|(x, y)| (th.re - x).hypot(th.im - y))
                .fold(f64::INFINITY, f64::min);
            assert!(
                d <= 1e-6,
                "{which:?}: θ = {th:?} at {d:e} from the spectrum"
            );
        }
    }
}

/// Convection-dominated, `p = 0.8` (eigenvector condition `9^{99.5}`: the
/// eigenvalues are not determined in `f64`, the residuals are): criterion
/// (b) for `LR`, `k = 16`, `ncv = 40`; without the coupling check, `95`.
#[test]
fn coupling_check_on_a_convection_dominated_operator() {
    let a = convection(200, 0.8);
    let (_, worst, _) = run("convection_200 p=0.8", &a, Which::LargestReal, 16, 40);
    assert!(worst <= C_K, "residual ratio {worst}");
}

/// Grcar matrices, the classical worst case: the residual ratio of
/// criterion (b) grows with the number `R` of restarts (Chapter 12.6, "Long
/// runs", issue xingo-statla#9). A regression check of these four runs,
/// not a guarantee (no form in `R` bounds every measured run): the ratio
/// against `c_K (1 + √R/8) = 8 + √R` and, as before, against `4 c_K`,
/// which is the tighter of the two only at Grcar(200) `LM` (988 restarts,
/// ratio `12.1`; `8 + √R` gives `39.4` there). Measured: `3.78`
/// (`R = 108`), `3.56` (`103`), `12.10` (`988`), `3.81` (`142`), against
/// `18.4`, `18.1`, `39.4`, `19.9`. The coupling check fires on these runs
/// too, without changing the outcome.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "slow in debug builds (Grcar 200 takes ~1000 restarts): run with --ignored"
)]
fn grcar_residuals_within_the_restart_form() {
    for n in [100usize, 200] {
        let a = grcar(n, 3);
        for which in [Which::LargestMagnitude, Which::LargestImag] {
            let (_, worst, restarts) = run(&format!("grcar_{n}"), &a, which, 16, 40);
            let form = C_K * (1.0 + (restarts as f64).sqrt() / 8.0);
            assert!(
                worst <= form && worst <= 4.0 * C_K,
                "grcar {n} {which:?}: {worst} after {restarts} restarts (form {form})"
            );
        }
    }
}
