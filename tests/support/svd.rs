//! Oracles of the acceptance tests of Chapters 12.4 and 12.5 (task L3.3).
//!
//! * **Prescribed spectra, certified.** `A = H_r ⋯ H_1 [diag(σ); 0] G_1 ⋯
//!   G_r` with random Householder reflectors, formed *in double-double*
//!   (`A_dd`) and rounded once to `f64` (`A`). The reflectors are
//!   orthogonal to `≈ 2⁻¹⁰⁴`, so `σ_i(A_dd) = σ_i ± 10⁻²⁹ σ_1`; rounding
//!   perturbs each entry by `≤ u|a_ij|`, so by Weyl `|σ_i(A) − σ_i| ≤ δ :=
//!   u‖A_dd‖_F + 10⁻²⁸ σ_1`. The reference singular values of the matrix the
//!   solver sees are therefore the prescribed ones to `δ` — a dense SVD
//!   *by construction*, valid for any size (task L3.3's oracle for the
//!   `10⁻¹³ σ_1` criterion). A single-vector Krylov method started from a
//!   Gaussian vector behaves, in exact arithmetic, the same for any
//!   orthogonal `U`, `V` (the start vector's coordinates in the singular
//!   basis are again Gaussian), so few reflectors lose nothing.
//! * **Dense reference SVD.** A plain one-sided Jacobi (values only) in any
//!   test scalar, used in double-double on small matrices (cross-check of
//!   the construction, sparse matrices without a prescribed spectrum) and
//!   in `f64` for the spectral norms of the HMT test.
//! * **Residuals** `‖(A v − σ u; Aᵀ u − σ v)‖` evaluated in double-double
//!   from the `f64` (or double-double) data.

use xingo_extprec::DoubleDouble as Dd;
use xingo_statla::DenseBlock;

use super::{Rng, Scalar};

/// A random reflector `I − 2vvᵀ/(vᵀv)` applied to the rows (`left`) or the
/// columns of `a` (`m × n`, column-major), in double-double.
fn reflect(a: &mut [Dd], m: usize, n: usize, v: &[f64], left: bool) {
    let vd: Vec<Dd> = v.iter().map(|&x| Dd::from_f64(x)).collect();
    let vv: Dd = vd.iter().map(|&x| x * x).sum();
    let two = Dd::from_f64(2.0);
    if left {
        for j in 0..n {
            let col = &mut a[j * m..(j + 1) * m];
            let d: Dd = col.iter().zip(&vd).map(|(&x, &y)| x * y).sum();
            let f = two * d / vv;
            for (x, &y) in col.iter_mut().zip(&vd) {
                *x -= f * y;
            }
        }
    } else {
        for i in 0..m {
            let d: Dd = (0..n).map(|j| a[i + j * m] * vd[j]).sum();
            let f = two * d / vv;
            for j in 0..n {
                a[i + j * m] -= f * vd[j];
            }
        }
    }
}

/// `A_dd` (`m × n`) with singular values `sigma` (length `min(m, n)`),
/// `r` reflectors on each side.
pub fn prescribed_dd(m: usize, n: usize, sigma: &[f64], r: usize, seed: u64) -> Vec<Dd> {
    assert_eq!(sigma.len(), m.min(n));
    let mut a = vec![Dd::ZERO; m * n];
    for (i, &s) in sigma.iter().enumerate() {
        a[i + i * m] = Dd::from_f64(s);
    }
    let mut rng = Rng::new(seed);
    for _ in 0..r {
        let v = rng.gauss_vec(m);
        reflect(&mut a, m, n, &v, true);
        let w = rng.gauss_vec(n);
        reflect(&mut a, m, n, &w, false);
    }
    a
}

/// The `f64` matrix and the certified distance `δ` of its singular values
/// from `sigma`.
pub fn prescribed(m: usize, n: usize, sigma: &[f64], seed: u64) -> (DenseBlock<f64>, f64) {
    let a = prescribed_dd(m, n, sigma, 4, seed);
    let fro = a.iter().map(|&x| x * x).sum::<Dd>().sqrt().to_f64();
    let s1 = sigma.iter().fold(0.0f64, |x, &y| x.max(y));
    let delta = f64::EPSILON / 2.0 * fro * (1.0 + 1e-10) + 1e-28 * s1;
    (
        DenseBlock {
            rows: m,
            cols: n,
            data: a.iter().map(|x| x.to_f64()).collect(),
        },
        delta,
    )
}

/// Singular values (nonincreasing) of the `m × n` column-major `a` by a
/// plain cyclic one-sided Jacobi in `T` (the transpose when `m < n`).
pub fn jacobi_values<T: Scalar>(a: &[T], m: usize, n: usize) -> Vec<f64> {
    let (p, q, mut w) = if m >= n {
        (m, n, a.to_vec())
    } else {
        let mut t = vec![T::ZERO; m * n];
        for j in 0..n {
            for i in 0..m {
                t[j + i * n] = a[i + j * m];
            }
        }
        (n, m, t)
    };
    let tol = T::EPSILON * T::from_f64(p as f64);
    for _ in 0..80 {
        let mut off = false;
        for i in 0..q {
            for j in i + 1..q {
                let (mut aa, mut bb, mut g) = (T::ZERO, T::ZERO, T::ZERO);
                for r in 0..p {
                    let (x, y) = (w[r + i * p], w[r + j * p]);
                    aa += x * x;
                    bb += y * y;
                    g += x * y;
                }
                if g.abs() <= tol * (aa * bb).sqrt() {
                    continue;
                }
                off = true;
                let zeta = (bb - aa) / (T::from_f64(2.0) * g);
                let t = if zeta >= T::ZERO {
                    T::ONE / (zeta + (T::ONE + zeta * zeta).sqrt())
                } else {
                    -(T::ONE / (-zeta + (T::ONE + zeta * zeta).sqrt()))
                };
                let c = T::ONE / (T::ONE + t * t).sqrt();
                let s = c * t;
                for r in 0..p {
                    let (x, y) = (w[r + i * p], w[r + j * p]);
                    w[r + i * p] = c * x - s * y;
                    w[r + j * p] = s * x + c * y;
                }
            }
        }
        if !off {
            break;
        }
    }
    let mut s: Vec<f64> = (0..q)
        .map(|j| {
            (0..p)
                .map(|r| w[r + j * p] * w[r + j * p])
                .sum::<T>()
                .sqrt()
                .to_f64()
        })
        .collect();
    s.sort_by(|x, y| y.partial_cmp(x).unwrap());
    s
}

/// `‖x‖₂` of an `m × n` `f64` matrix (largest singular value; Jacobi on
/// the `n × n` triangle of a Householder-free Gram–Schmidt would lose
/// accuracy, so the full Jacobi of `x` itself).
pub fn spectral_norm(x: &[f64], m: usize, n: usize) -> f64 {
    jacobi_values::<f64>(x, m, n)[0]
}

/// A certified **upper** bound of `λ_max` of the symmetric positive
/// semidefinite `n × n` `g`: `(tr g^{2^s})^{1/2^s}`, which lies in
/// `[λ_max, n^{1/2^s} λ_max]` (the trace of a power is at least the power
/// of the largest eigenvalue and at most `n` times it). Each squaring is
/// normalized by the trace (no overflow); rounding adds a relative `O(s n
/// u)`.
pub fn lambda_max_upper(g: &[f64], n: usize, squarings: u32) -> f64 {
    let mut a = g.to_vec();
    let mut log_scale = 0.0f64; // log of the factor removed, per unit power
    let mut power = 1.0f64;
    for _ in 0..squarings {
        let tr: f64 = (0..n).map(|i| a[i + i * n]).sum();
        if tr <= 0.0 {
            return 0.0;
        }
        for v in &mut a {
            *v /= tr;
        }
        log_scale += tr.ln() / power;
        // a² by dot products of columns (a is symmetric): upper half, mirrored.
        let mut b = vec![0.0; n * n];
        for j in 0..n {
            let cj = &a[j * n..(j + 1) * n];
            for i in 0..=j {
                let ci = &a[i * n..(i + 1) * n];
                let d: f64 = ci.iter().zip(cj).map(|(x, y)| x * y).sum();
                b[i + j * n] = d;
                b[j + i * n] = d;
            }
        }
        a = b;
        power *= 2.0;
    }
    let tr: f64 = (0..n).map(|i| a[i + i * n]).sum();
    (log_scale + tr.ln() / power).exp() * (1.0 + 1e-12)
}

/// For every triplet `i`, `‖(A v_i − s_i u_i; Aᵀ u_i − s_i v_i)‖` in
/// double-double (`a` is `m × n`).
pub fn residuals<T: Scalar>(
    a: &[T],
    m: usize,
    n: usize,
    u: &DenseBlock<T>,
    s: &[T],
    v: &DenseBlock<T>,
) -> Vec<f64> {
    (0..s.len())
        .map(|c| {
            let sd = s[c].dd();
            let mut r2 = Dd::ZERO;
            for i in 0..m {
                let av: Dd = (0..n).map(|j| a[i + j * m].dd() * v.get(j, c).dd()).sum();
                let d = av - sd * u.get(i, c).dd();
                r2 += d * d;
            }
            for j in 0..n {
                let au: Dd = (0..m).map(|i| a[i + j * m].dd() * u.get(i, c).dd()).sum();
                let d = au - sd * v.get(j, c).dd();
                r2 += d * d;
            }
            r2.sqrt().to_f64()
        })
        .collect()
}

/// `max |XᵀX − I|` of the columns of `x`, in double-double.
pub fn orth_err<T: Scalar>(x: &DenseBlock<T>) -> f64 {
    let mut e = 0.0f64;
    for i in 0..x.cols {
        for j in 0..=i {
            let d: Dd = (0..x.rows)
                .map(|r| x.get(r, i).dd() * x.get(r, j).dd())
                .sum();
            let t = if i == j { d - Dd::ONE } else { d };
            e = e.max(t.abs().to_f64());
        }
    }
    e
}
