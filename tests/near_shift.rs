//! Shift-invert with `σ` at a controlled distance from an eigenvalue
//! (Chapter 12.6.2, Algorithm R; issue xingo-statla#1).
//!
//! Algorithm R refines each solve once; one step makes the solve backward
//! stable when `γ·κ(A − σI)·u` is well below one, and the chapter left
//! open what happens when `σ` is within `γ·u·‖A − σI‖` of an eigenvalue.
//! These tests put an eigenvalue at a known distance from `σ`, down to one
//! ulp, and measure what the issue asks for:
//!
//! * `η = ‖x − My‖_∞/(‖M‖_∞‖y‖_∞)` of the applications (`M = A − σI`, the
//!   residual in double-double), after Algorithm R and after a second
//!   refinement step emulated outside the operator;
//! * the eigenvalue nearest `σ` against the exact one, and the residual on
//!   `A` of every returned pair against criterion (b) of `tests/arpack.rs`;
//! * `count_below` against the exact count.
//!
//! The matrices have an eigenvalue known *exactly* in `f64`: the
//! tridiagonal `tridiag(−1 − p, 2, −1 + p)` of odd order `n` has the
//! eigenvalues `2 − 2√(1 − p²) cos(jπ/(n + 1))`, and `j = (n + 1)/2` gives
//! `λ⋆ = 2` exactly, whatever the rounding of the off-diagonal entries
//! (`p = 0`: the symmetric 1-D Laplacian; `p = 1/32`: a nonnormal
//! convection–diffusion operator with a real spectrum). Both are
//! permuted symmetrically by a fixed random permutation (the spectrum is
//! unchanged and the factorizations do not see a band). Every table line
//! is printed with `--nocapture`.

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::cast_possible_truncation
)]

mod support;

use xingo_extprec::DoubleDouble as Dd;
use xingo_sparse::{Col, CscMatrix, LuOptions, NaiveDense, TolScale};
use xingo_statla::{
    Complex, EigsOptions, LinearOperator, ShiftInvert, StatlaError, SymShiftInvert, eigs_near,
    eigsh_near,
};

/// `c_K` of criterion (b), as in `tests/arpack.rs`.
const C_K: f64 = 8.0;
/// Order of the test matrices (odd: `λ⋆ = 2` is an eigenvalue).
const N: usize = 201;

/// `tridiag(−1 − p, 2, −1 + p)` of order `n` (subdiagonal `−1 − p`),
/// permuted symmetrically by `perm`, as a CSC matrix with sorted rows.
fn tridiag(n: usize, p: f64, perm: &[usize]) -> CscMatrix<f64> {
    let mut dense = vec![0.0; n * n];
    for i in 0..n {
        dense[perm[i] + perm[i] * n] = 2.0;
        if i + 1 < n {
            // a_{i+1,i} = −1 − p, a_{i,i+1} = −1 + p.
            dense[perm[i + 1] + perm[i] * n] = -1.0 - p;
            dense[perm[i] + perm[i + 1] * n] = -1.0 + p;
        }
    }
    let mut col_ptr = vec![0usize];
    let (mut row_idx, mut val) = (Vec::new(), Vec::new());
    for j in 0..n {
        for i in 0..n {
            let v = dense[i + j * n];
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

/// A fixed random permutation of `0..n`.
fn permutation(n: usize, seed: u64) -> Vec<usize> {
    let mut rng = support::Rng::new(seed);
    let mut p: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        p.swap(i, rng.below(i + 1));
    }
    p
}

/// The exact spectrum (in `f64`, each value within an ulp or two; `λ⋆ = 2`
/// exactly), ascending.
fn spectrum(n: usize, p: f64) -> Vec<f64> {
    let s = 2.0 * (1.0 - p * p).sqrt();
    let mut v: Vec<f64> = (1..=n)
        .map(|j| {
            if 2 * j == n + 1 {
                2.0
            } else {
                2.0 - s * (j as f64 * std::f64::consts::PI / (n + 1) as f64).cos()
            }
        })
        .collect();
    v.sort_by(f64::total_cmp);
    v
}

/// `‖x − M y‖_∞ / (‖M‖_∞ ‖y‖_∞)`, `M = A − σI`, the residual in
/// double-double.
fn eta(a: &CscMatrix<f64>, sigma: f64, x: &[f64], y: &[f64]) -> f64 {
    let n = a.n_rows();
    let mut r: Vec<Dd> = x.iter().map(|&v| Dd::from_f64(v)).collect();
    let mut rows_abs = vec![0.0f64; n];
    for j in 0..n {
        let (rows, vals) = a.col(Col::from_usize(j));
        for (&i, &v) in rows.iter().zip(vals) {
            let i = i as usize;
            let v = if i == j { v - sigma } else { v };
            r[i] -= Dd::from_f64(v) * Dd::from_f64(y[j]);
            rows_abs[i] += v.abs();
        }
    }
    let rn = r.iter().map(|v| v.to_f64().abs()).fold(0.0, f64::max);
    let mn = rows_abs.iter().copied().fold(0.0, f64::max);
    let yn = y.iter().map(|v| v.abs()).fold(0.0, f64::max);
    rn / (mn * yn)
}

/// `r ← x − M y` in `f64` (the residual of step R2), `M = A − σI`.
fn residual_f64(a: &CscMatrix<f64>, sigma: f64, x: &[f64], y: &[f64]) -> Vec<f64> {
    let mut r = x.to_vec();
    for j in 0..a.n_cols() {
        let (rows, vals) = a.col(Col::from_usize(j));
        for (&i, &v) in rows.iter().zip(vals) {
            let i = i as usize;
            let v = if i == j { v - sigma } else { v };
            r[i] -= v * y[j];
        }
    }
    r
}

/// `η` after Algorithm R and after one more refinement step (`y₂ = y₁ +
/// op(x − M y₁)`, the inner solve itself refined), maxima over 5 random
/// right-hand sides; also `‖y₂ − y₁‖_∞/‖y₁‖_∞`.
fn etas<O: LinearOperator<f64>>(
    op: &mut O,
    a: &CscMatrix<f64>,
    sigma: f64,
    seed: u64,
) -> (f64, f64, f64) {
    let n = a.n_rows();
    let mut rng = support::Rng::new(seed);
    let (mut e1, mut e2, mut corr) = (0.0f64, 0.0f64, 0.0f64);
    for _ in 0..5 {
        let x = rng.gauss_vec(n);
        let mut y = vec![0.0; n];
        op.apply(&x, &mut y).unwrap();
        e1 = e1.max(eta(a, sigma, &x, &y));
        let r = residual_f64(a, sigma, &x, &y);
        let mut d = vec![0.0; n];
        op.apply(&r, &mut d).unwrap();
        let y2: Vec<f64> = y.iter().zip(&d).map(|(a, b)| a + b).collect();
        e2 = e2.max(eta(a, sigma, &x, &y2));
        let dn = d.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let yn = y.iter().map(|v| v.abs()).fold(0.0, f64::max);
        corr = corr.max(dn / yn);
    }
    (e1, e2, corr)
}

/// `‖A y − θ y‖` in double-double, for a complex pair.
fn pair_residual(a: &CscMatrix<f64>, yr: &[f64], yi: &[f64], th: Complex<f64>) -> f64 {
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

/// The shifts: `2 ± δ` for `δ` from `2⁻¹` down to `2⁻⁴⁸`, then 4, 2 and 1
/// ulps of `2` above (`2⁻⁵¹` each) and of the binade below (`2⁻⁵²` each),
/// and `σ = 2` itself.
fn shifts() -> Vec<(String, f64)> {
    let mut v = Vec::new();
    for e in [1, 4, 8, 16, 24, 32, 36, 40, 44, 48] {
        let d = 2f64.powi(-e);
        v.push((format!("2 + 2^-{e}"), 2.0 + d));
        v.push((format!("2 - 2^-{e}"), 2.0 - d));
    }
    for j in [4u64, 2, 1] {
        let up = f64::from_bits(2f64.to_bits() + j);
        let down = f64::from_bits(2f64.to_bits() - j);
        v.push((format!("2 + {j} ulp"), up));
        v.push((format!("2 - {j} ulp"), down));
    }
    v.push(("2".to_owned(), 2.0));
    v
}

/// What one shift gave.
#[derive(Debug)]
enum Outcome {
    /// The operator refused `σ` (zero pivot: `σ` is numerically an
    /// eigenvalue).
    Refused(StatlaError),
    /// The operator was built; the eigensolver may still have failed.
    Measured {
        eta1: f64,
        eta2: f64,
        corr: f64,
        /// The eigensolver's error, if any.
        failed: Option<StatlaError>,
        /// `|λ̃ − 2| / ‖A‖` for the returned value nearest `2` (`∞` when
        /// `2` is not among the returned values).
        err_star: f64,
        /// Largest error of the other returned values against the exact
        /// spectrum, relative to `‖A‖`.
        err_rest: f64,
        /// `(‖r‖/‖A‖ − τ)/(ncv u)` (criterion (b)) of the pair nearest `σ`.
        res_first: f64,
        /// The largest of the same ratio over the other pairs, each divided
        /// by its amplification `|θ_i − σ| / |θ_1 − σ|` (see the chapter).
        res_rest_amp: f64,
        /// The largest unscaled ratio over the other pairs.
        res_rest: f64,
        /// The largest amplification `|θ_i − σ| / |θ_1 − σ|`.
        amp: f64,
    },
}

/// The nearest exact eigenvalue to `x`.
fn nearest(spec: &[f64], x: f64) -> f64 {
    *spec
        .iter()
        .min_by(|a, b| (*a - x).abs().total_cmp(&(*b - x).abs()))
        .unwrap()
}

fn print(name: &str, what: &str, o: &Outcome) {
    let u = f64::EPSILON / 2.0;
    match o {
        Outcome::Refused(e) => eprintln!("| {name} | {what} | refused: {e} | | | | | | | |"),
        Outcome::Measured {
            eta1,
            eta2,
            corr,
            failed: Some(e),
            ..
        } => eprintln!(
            "| {name} | {what} | {:.2} | {:.2} | {corr:.1e} | eigensolver: {e} | | | | |",
            eta1 / u,
            eta2 / u
        ),
        Outcome::Measured {
            eta1,
            eta2,
            corr,
            failed: None,
            err_star,
            err_rest,
            res_first,
            res_rest_amp,
            res_rest,
            ..
        } => eprintln!(
            "| {name} | {what} | {:.2} | {:.2} | {corr:.1e} | {err_star:.1e} | {err_rest:.1e} | {res_first:.2} | {res_rest:.3e} | {res_rest_amp:.3} |",
            eta1 / u,
            eta2 / u
        ),
    }
}

fn header() {
    eprintln!(
        "| matrix | σ | η/u, R | η/u, R + 1 step | ‖Δy‖/‖y‖ of the step | |λ̃ − λ⋆|/‖A‖ | other values, rel ‖A‖ | (b) nearest pair | (b) others | (b) others / amplification |"
    );
    eprintln!("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|");
}

/// The measures of one returned spectrum (values `θ`, residual norms).
fn measure_pairs(
    spec: &[f64],
    sigma: f64,
    vals: &[Complex<f64>],
    resid: &[f64],
    anorm: f64,
    ncv: usize,
) -> (f64, f64, f64, f64, f64, f64) {
    let (u, tau) = (f64::EPSILON / 2.0, f64::EPSILON);
    let (mut err_star, mut err_rest) = (f64::INFINITY, 0.0f64);
    let dist: Vec<f64> = vals.iter().map(|th| (th.re - sigma).hypot(th.im)).collect();
    let d1 = dist.iter().copied().fold(f64::INFINITY, f64::min);
    let (mut res_first, mut res_rest, mut res_rest_amp) = (0.0f64, 0.0f64, 0.0f64);
    for (c, th) in vals.iter().enumerate() {
        // The spectra are real: an imaginary part is an error too.
        let e = (th.re - nearest(spec, th.re)).hypot(th.im) / anorm;
        if nearest(spec, th.re) == 2.0 {
            err_star = e;
        } else {
            err_rest = err_rest.max(e);
        }
        let ratio = (resid[c] / anorm - tau) / (ncv as f64 * u);
        if dist[c] == d1 {
            res_first = res_first.max(ratio);
        } else {
            res_rest = res_rest.max(ratio);
            res_rest_amp = res_rest_amp.max(ratio / (dist[c] / d1));
        }
    }
    let amp = dist.iter().map(|d| d / d1).fold(1.0, f64::max);
    (err_star, err_rest, res_first, res_rest, res_rest_amp, amp)
}

/// The assertions (see the chapter, "σ near an eigenvalue"): `η ≤ 2u` after
/// Algorithm R (invariant I2) at every accepted shift; the eigensolver
/// succeeds; the value nearest `σ` is `λ⋆` to `10⁻¹⁴‖A‖` when `σ` is
/// inside its interval, the others agree with the exact spectrum to
/// `(10⁻¹³ + a u)‖A‖`, `a` the largest amplification `|θ_i − σ|/|θ_1 −
/// σ|`; the pair nearest `σ` meets criterion (b); the others meet it
/// times their amplification.
fn check(name: &str, what: &str, sigma: f64, o: &Outcome) {
    let u = f64::EPSILON / 2.0;
    if let Outcome::Measured {
        eta1,
        failed,
        err_star,
        err_rest,
        res_first,
        res_rest_amp,
        amp,
        ..
    } = o
    {
        assert!(*eta1 <= 2.0 * u, "{name} σ = {what}: η = {} u", eta1 / u);
        assert!(failed.is_none(), "{name} σ = {what}: {failed:?}");
        // λ⋆ is among the k = 3 nearest when σ is well inside its
        // interval (the spacing of the spectrum near 2 is ≈ 0.031).
        if (sigma - 2.0).abs() <= 2f64.powi(-8) || err_star.is_finite() {
            assert!(
                *err_star <= 1e-14,
                "{name} σ = {what}: |λ̃ − 2| = {err_star:e}"
            );
        }
        assert!(
            *err_rest <= 1e-13 + amp * u,
            "{name} σ = {what}: others {err_rest:e}, amplification {amp:e}"
        );
        assert!(
            *res_first <= C_K,
            "{name} σ = {what}: nearest pair {res_first}"
        );
        assert!(
            *res_rest_amp <= C_K,
            "{name} σ = {what}: others {res_rest_amp}"
        );
    }
}

/// `eigsh_near` (Bunch–Kaufman `LBLᵀ`) on the permuted 1-D Laplacian.
#[test]
fn eigsh_near_with_an_eigenvalue_ulps_from_sigma() {
    let a = tridiag(N, 0.0, &permutation(N, 5));
    let spec = spectrum(N, 0.0);
    let anorm = 4.0; // ≥ ‖A‖₂ (Gershgorin), within 0.01 %
    let k = 3;
    let ncv = N.min((2 * k + 1).max(20));
    header();
    let mut accepted = 0;
    for (what, sigma) in shifts() {
        let o = match SymShiftInvert::<f64>::new(&a, sigma, NaiveDense) {
            Err(e) => Outcome::Refused(e),
            Ok(mut op) => {
                // (c) of `tests/arpack.rs`: the exact count below σ.
                assert_eq!(
                    op.count_below(),
                    spec.iter().filter(|&&x| x < sigma).count(),
                    "σ = {what}"
                );
                accepted += 1;
                let (eta1, eta2, corr) = etas(&mut op, &a, sigma, 7);
                match eigsh_near(&a, k, sigma, &EigsOptions::default()) {
                    Err(e) => Outcome::Measured {
                        eta1,
                        eta2,
                        corr,
                        failed: Some(e),
                        err_star: f64::NAN,
                        err_rest: f64::NAN,
                        res_first: f64::NAN,
                        res_rest_amp: f64::NAN,
                        res_rest: f64::NAN,
                        amp: f64::NAN,
                    },
                    Ok((f, _)) => {
                        let y = f.vectors.as_ref().unwrap();
                        let vals: Vec<Complex<f64>> =
                            f.values.iter().map(|&re| Complex { re, im: 0.0 }).collect();
                        let resid: Vec<f64> = vals
                            .iter()
                            .enumerate()
                            .map(|(c, &th)| {
                                pair_residual(&a, &y.data[c * N..(c + 1) * N], &[0.0; N], th)
                            })
                            .collect();
                        let (err_star, err_rest, res_first, res_rest, res_rest_amp, amp) =
                            measure_pairs(&spec, sigma, &vals, &resid, anorm, ncv);
                        Outcome::Measured {
                            eta1,
                            eta2,
                            corr,
                            failed: None,
                            err_star,
                            err_rest,
                            res_first,
                            res_rest_amp,
                            res_rest,
                            amp,
                        }
                    }
                }
            }
        };
        print("lap1d_201 (eigsh_near)", &what, &o);
        check("lap1d_201", &what, sigma, &o);
        // The LBLᵀ refuses σ = λ⋆ only (a zero pivot), down to one ulp.
        if let Outcome::Refused(e) = &o {
            assert!(sigma == 2.0, "σ = {what} refused: {e}");
            assert!(matches!(e, StatlaError::Kkt(_)), "{e:?}");
        }
    }
    assert!(accepted + 1 >= shifts().len(), "{accepted} accepted");
}

/// `eigs_near` (threshold LU) on the permuted nonnormal tridiagonal.
#[test]
fn eigs_near_with_an_eigenvalue_ulps_from_sigma() {
    // p = 1/32: the eigenvector matrix has condition ≈ ((1 + p)/(1 − p))^{(n−1)/2}
    // ≈ 5·10², nonnormal but with eigenvalues still determined to about
    // 10⁻¹³ (p = 1/2 gives 3¹⁰⁰: the eigenvalues are then not determined by
    // the matrix in f64 at all).
    let p = 1.0 / 32.0;
    let a = tridiag(N, p, &permutation(N, 9));
    let spec = spectrum(N, p);
    // ‖A‖₂ ≤ max row sum = 2 + (1 + p) + (1 − p) = 4 (rows and columns alike).
    let anorm = 4.0;
    let k = 3;
    let ncv = N.min((2 * k + 1).max(20));
    header();
    let mut accepted = 0;
    for (what, sigma) in shifts() {
        let o = match ShiftInvert::new(&a, sigma, LuOptions::for_scalar::<f64>()) {
            Err(e) => Outcome::Refused(e),
            Ok(mut op) => {
                accepted += 1;
                let (eta1, eta2, corr) = etas(&mut op, &a, sigma, 7);
                match eigs_near(&a, k, sigma, &EigsOptions::default()) {
                    Err(e) => Outcome::Measured {
                        eta1,
                        eta2,
                        corr,
                        failed: Some(e),
                        err_star: f64::NAN,
                        err_rest: f64::NAN,
                        res_first: f64::NAN,
                        res_rest_amp: f64::NAN,
                        res_rest: f64::NAN,
                        amp: f64::NAN,
                    },
                    Ok(f) => {
                        let (yr, yi) = (
                            f.vectors_re.as_ref().unwrap(),
                            f.vectors_im.as_ref().unwrap(),
                        );
                        let resid: Vec<f64> = f
                            .values
                            .iter()
                            .enumerate()
                            .map(|(c, &th)| {
                                pair_residual(
                                    &a,
                                    &yr.data[c * N..(c + 1) * N],
                                    &yi.data[c * N..(c + 1) * N],
                                    th,
                                )
                            })
                            .collect();
                        let (err_star, err_rest, res_first, res_rest, res_rest_amp, amp) =
                            measure_pairs(&spec, sigma, &f.values, &resid, anorm, ncv);
                        Outcome::Measured {
                            eta1,
                            eta2,
                            corr,
                            failed: None,
                            err_star,
                            err_rest,
                            res_first,
                            res_rest_amp,
                            res_rest,
                            amp,
                        }
                    }
                }
            }
        };
        print("convdiff1d_201 (eigs_near)", &what, &o);
        check("convdiff1d_201", &what, sigma, &o);
        // The threshold LU of `eigs_near` (`LuOptions::for_scalar`, an
        // *absolute* pivot tolerance `τ_abs = 10⁻¹⁰`) declares `A − σI`
        // singular once `|σ − λ⋆|` is below about `τ_abs`: accepted down to
        // `2⁻³⁶ ≈ 1.5·10⁻¹¹`, refused from `2⁻⁴⁰` on (measured; the pivot is
        // not `|σ − λ⋆|` itself; see the chapter).
        let delta = (sigma - 2.0).abs();
        match &o {
            Outcome::Refused(e) => {
                assert!(delta < 2f64.powi(-32), "σ = {what} refused: {e}");
                assert!(matches!(e, StatlaError::Lu(_)), "{e:?}");
            }
            Outcome::Measured { .. } => {
                assert!(delta >= 2f64.powi(-36), "σ = {what} accepted");
            }
        }
    }
    assert!(accepted >= 12, "{accepted} accepted");
}

/// One run of `ShiftInvert` with the options `opts` and `eigs` (`k = 1`)
/// on it, mapped back: `(η after R, η after a second step, |λ̃ − λ⋆|/‖A‖,
/// criterion (b) of the pair)`, or the refusal.
fn lu_run(
    a: &CscMatrix<f64>,
    sigma: f64,
    star: f64,
    anorm: f64,
    opts: LuOptions,
) -> Result<(f64, f64, f64, f64), StatlaError> {
    use xingo_statla::{Which, eigs};
    let (u, tau) = (f64::EPSILON / 2.0, f64::EPSILON);
    let mut op = ShiftInvert::new(a, sigma, opts)?;
    let (eta1, eta2, _) = etas(&mut op, a, sigma, 7);
    let f = eigs(&mut op, 1, Which::LargestMagnitude, &EigsOptions::default())?;
    let nu = f.values[0];
    let w = 1.0 / (nu.re * nu.re + nu.im * nu.im);
    let th = Complex {
        re: sigma + nu.re * w,
        im: -nu.im * w,
    };
    let err = (th.re - star).hypot(th.im) / anorm;
    let r = pair_residual(
        a,
        &f.vectors_re.as_ref().unwrap().data,
        &f.vectors_im.as_ref().unwrap().data,
        th,
    );
    let ncv = a.n_rows().min(20);
    Ok((eta1, eta2, err, (r / anorm - tau) / (ncv as f64 * u)))
}

/// The threshold LU beyond the refusals of `eigs_near` (the chapter, "σ
/// near an eigenvalue"), through `ShiftInvert` and `eigs` directly, on the
/// nonnormal tridiagonal and on the same matrix scaled by `2⁻⁴⁰`
/// (exactly):
///
/// * `eigs_near` (`LuOptions::for_scalar`: `TolScale::Absolute`, `τ_abs =
///   10⁻¹⁰`, `ε_drop = 10⁻¹²`) refuses every shift of the scaled matrix —
///   its tolerances are absolute;
/// * with `TolScale::Column` the scaled matrix behaves as the unscaled one
///   (the same refusals; `η ≤ 2u`, `λ⋆` to `10⁻¹⁴‖A‖` and criterion (b)
///   wherever accepted);
/// * with no absolute pivot tolerance at all (`τ_abs = 0`) the LU accepts
///   `σ` down to one ulp of `λ⋆`, and there Algorithm R is *not* enough:
///   `η` reaches `3.9·10³ u` at `|σ − λ⋆| = 2⁻⁴⁰`, and a second refinement
///   step does not reduce it (its correction is `10⁴` times `y`), as
///   fixed-precision refinement predicts once `γ κ(M) u` is not small.
///   The refusal of the default options falls where one step stops
///   sufficing.
#[test]
fn lu_shift_invert_beyond_the_default_refusals() {
    let p = 1.0 / 32.0;
    let u = f64::EPSILON / 2.0;
    let base = LuOptions::for_scalar::<f64>();
    let column = LuOptions {
        tol_scale: TolScale::Column,
        ..base
    };
    let no_abs = LuOptions {
        abs_pivot_tol: 0.0,
        ..base
    };
    let mut accepted = [Vec::new(), Vec::new()];
    for (si, scale) in [1.0, 2f64.powi(-40)].into_iter().enumerate() {
        let mut a = tridiag(N, p, &permutation(N, 9));
        for v in &mut a.val {
            *v *= scale;
        }
        let (anorm, star) = (4.0 * scale, 2.0 * scale);
        for (what, s) in shifts() {
            let sigma = s * scale;
            let default = eigs_near(&a, 3, sigma, &EigsOptions::default());
            if si == 1 {
                assert!(
                    matches!(default, Err(StatlaError::Lu(_))),
                    "scaled, σ = {what}: {:?}",
                    default.map(|f| f.values)
                );
            }
            match lu_run(&a, sigma, star, anorm, column) {
                Err(e) => {
                    eprintln!("scale {scale:e}, σ = {what}: Column refuses: {e}");
                    assert!(matches!(e, StatlaError::Lu(_)), "{e:?}");
                }
                Ok((eta1, _, err, b)) => {
                    eprintln!(
                        "scale {scale:e}, σ = {what}: eigs_near {}; Column: η = {:.2} u, |λ̃ − λ⋆|/‖A‖ = {err:.1e}, (b) {b:.2}",
                        if default.is_ok() { "ok" } else { "refused" },
                        eta1 / u
                    );
                    assert!(
                        eta1 <= 2.0 * u,
                        "scale {scale:e}, σ = {what}: η = {} u",
                        eta1 / u
                    );
                    if (s - 2.0).abs() <= 2f64.powi(-8) {
                        assert!(err <= 1e-14, "scale {scale:e}, σ = {what}: {err:e}");
                    }
                    assert!(b <= C_K, "scale {scale:e}, σ = {what}: (b) {b}");
                    accepted[si].push(what.clone());
                }
            }
        }
    }
    // Column tolerances are invariant under the (exact) scaling.
    assert_eq!(accepted[0], accepted[1]);
    // τ_abs = 0 on the unscaled matrix: Algorithm R and a second step.
    let a = tridiag(N, p, &permutation(N, 9));
    for e in [40, 44] {
        for sigma in [2.0 + 2f64.powi(-e), 2.0 - 2f64.powi(-e)] {
            let (eta1, eta2, err, b) = lu_run(&a, sigma, 2.0, 4.0, no_abs).unwrap();
            eprintln!(
                "τ_abs = 0, σ = 2 ± 2^-{e}: η = {:.1} u, after a second step {:.1} u, |λ̃ − λ⋆|/‖A‖ = {err:.1e}, (b) {b:.2}",
                eta1 / u,
                eta2 / u
            );
            assert!(eta1 > 100.0 * u, "σ = 2 ± 2^-{e}: η = {} u", eta1 / u);
            assert!(
                eta2 > 0.5 * eta1,
                "a second step helped: {eta1:e} → {eta2:e}"
            );
        }
    }
}
