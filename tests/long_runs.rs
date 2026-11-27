//! Lemma R of Chapter 12.6 on long runs (issue xingo-statla#10): the
//! residual recomputed by `SymEigen::residuals`/`Eigen::residuals`, plus its
//! rounding terms, bounds the true residual (computed in double-double)
//! whatever the number of restarts. Release builds only (hundreds of
//! restarts).
#![allow(clippy::cast_precision_loss)]

use xingo_extprec::DoubleDouble as Dd;
use xingo_statla::{EigsOptions, OperatorError, Which, eigs, eigsh, from_fn};

/// `tridiag(−1, d, −1)` of order `n` applied to `x`, in order (three terms
/// per row: `η_op ≤ γ_3·√(‖A‖₁‖A‖_∞)`).
fn tri(d: f64, x: &[f64], y: &mut [f64]) {
    let n = x.len();
    for i in 0..n {
        let mut s = d * x[i];
        if i > 0 {
            s += -x[i - 1];
        }
        if i + 1 < n {
            s += -x[i + 1];
        }
        y[i] = s;
    }
}

/// The true `‖A y − θ y‖` of a real pair, in double-double.
fn true_residual(d: f64, y: &[f64], theta: f64) -> f64 {
    let n = y.len();
    let mut s = Dd::ZERO;
    for i in 0..n {
        let mut r = Dd::from_f64(d) * Dd::from_f64(y[i]) - Dd::from_f64(theta) * Dd::from_f64(y[i]);
        if i > 0 {
            r -= Dd::from_f64(y[i - 1]);
        }
        if i + 1 < n {
            r -= Dd::from_f64(y[i + 1]);
        }
        s += r * r;
    }
    s.sqrt().to_f64()
}

fn gamma(k: f64) -> f64 {
    let u = f64::EPSILON / 2.0;
    k * u / (1.0 - k * u)
}

fn check(d: f64, n: usize, values: &[f64], vecs: &[f64], rho: &[f64]) -> f64 {
    let anorm = d.abs() + 2.0;
    let eta_op = gamma(3.0) * anorm;
    let mut worst = 0.0f64;
    let mut w = vec![0.0; n];
    for (c, (&th, &r)) in values.iter().zip(rho).enumerate() {
        let y = &vecs[c * n..(c + 1) * n];
        tri(d, y, &mut w);
        let wn = w.iter().map(|v| v * v).sum::<f64>().sqrt();
        let bound = (1.0 + gamma((2 * n + 1) as f64)) * r
            + gamma(3.0) * (wn + 2f64.sqrt() * th.abs())
            + eta_op;
        let t = true_residual(d, y, th);
        assert!(t <= bound, "pair {c}: true {t:e} > bound {bound:e}");
        worst = worst.max(r / anorm);
    }
    worst
}

#[test]
#[cfg_attr(debug_assertions, ignore = "hundreds of restarts: release only")]
fn lemma_r_bounds_long_runs() {
    let u = f64::EPSILON / 2.0;
    // (d, n, driver): the indefinite tridiagonal (d = 0) and the 1-D
    // Laplacian (d = 2), `LM`, `k = 10`, `ncv = 24` (Chapter 12.6, "Long
    // runs"; 987 and 1360 restarts in eigsh since the polish of KS3, 880
    // and 1221 with Algorithm SJ alone).
    for (d, n) in [(0.0, 1000usize), (2.0, 2000)] {
        let o = EigsOptions {
            ncv: Some(24),
            ..EigsOptions::default()
        };
        let mut op = from_fn(
            n,
            n,
            |x: &[f64], y: &mut [f64]| -> Result<(), OperatorError> {
                tri(d, x, y);
                Ok(())
            },
        );
        let f = eigsh(&mut op, 10, Which::LargestMagnitude, &o).unwrap();
        assert!(f.stats.restarts >= 500, "{}", f.stats.restarts);
        let rho = f.residuals(&mut op).unwrap();
        let v = f.vectors.as_ref().unwrap();
        let r1 = check(d, n, &f.values, &v.data, &rho);
        let g = eigs(&mut op, 10, Which::LargestMagnitude, &o).unwrap();
        assert!(g.stats.restarts >= 500, "{}", g.stats.restarts);
        let rho = g.residuals(&mut op).unwrap();
        let re: Vec<f64> = g.values.iter().map(|c| c.re).collect();
        assert!(g.values.iter().all(|c| c.im == 0.0));
        let r2 = check(d, n, &re, &g.vectors_re.as_ref().unwrap().data, &rho);
        eprintln!(
            "d = {d}: eigsh R = {} ρ̂/‖A‖ = {:.1} m u; eigs R = {} {:.1} m u",
            f.stats.restarts,
            r1 / (24.0 * u),
            g.stats.restarts,
            r2 / (24.0 * u)
        );
    }
}
