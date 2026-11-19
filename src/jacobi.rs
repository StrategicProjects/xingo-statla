//! # Chapter 12.0.6 — One-sided Jacobi SVD of a small dense matrix
//!
//! ## Purpose
//! Both truncated SVDs end with the SVD of a *small* matrix: the `l × l`
//! triangle of step S5 of [`crate::rsvd()`] and the projected `B` of step B4
//! of [`crate::svds()`] (and `eigsh`/`eigs` of L3.4 will reuse the sorting
//! and completion conventions). `DenseKernel` has no SVD and the crate is
//! generic in `T: Real` (decision S1), so this is the classical one-sided
//! (Hestenes) Jacobi method: it needs only `+ − · /`, `sqrt` and the order,
//! and it is backward stable column by column (Demmel & Veselić), which is
//! what the residual guarantees of 12.4 and 12.5 rest on.
//!
//! ## Notation
//! `B ∈ T^{p×q}`, `p ≥ q`; `w_i` the columns of the working copy `W`; `Y ∈
//! T^{q×q}` the accumulated rotations, so `W = B Y` throughout; at the end
//! `w_i = σ_i x_i` with `‖x_i‖ = 1`, hence `B Y = X Σ`, `B = X Σ Yᵀ`. `tol`
//! and `max_sweeps` are the fields of [`JacobiOptions`].
//!
//! ## Algorithm J
//! J1. \[Scale.\] `μ ← max |b_ij|`. If `μ = 0`: `Σ = 0`, `Y = I`, `X` the
//!     first `q` columns of `I_p`; done. Otherwise `W ← B/2^⌊log₂ μ⌋`
//!     (exact), so the squares below neither overflow nor underflow.
//! J2. \[Sweep.\] First set to zero every column whose `‖w_i‖²` underflows
//!     to 0 (its norm is below `2⁻⁵³⁷` while `max |w_ij| ≥ 1`: far below
//!     `ε_T‖B‖`, so a negligible backward perturbation; without it such a
//!     column, with `a = 0` but `γ ≠ 0`, would be rotated forever). Then for
//!     every pair `i < j` in row-cyclic order: `a ← ‖w_i‖²`,
//!     `b ← ‖w_j‖²`, `γ ← w_iᵀw_j`. If `|γ| > tol·√(a b)`: `ζ ← (b −
//!     a)/(2γ)`, `t ← sign(ζ)/(|ζ| + √(1 + ζ²))` (written as `sign(ζ)/(|ζ|(1 +
//!     √(1 + ζ⁻²)))` when `|ζ| ≥ 1`, so that `ζ²` never overflows), `c ←
//!     1/√(1 + t²)`, `s ← c t`; `(w_i, w_j) ← (c w_i − s w_j, s w_i + c w_j)`,
//!     the same on the columns of `Y`. This makes the new pair exactly
//!     orthogonal in exact arithmetic (`cs(a − b) + (c² − s²)γ = 0`).
//! J3. \[Stop.\] Stop after a sweep without rotation. After `max_sweeps`
//!     sweeps, report non-convergence (never seen: the method converges
//!     quadratically once the off-diagonal mass is small).
//! J4. \[Values.\] Divide `y_i` *and* `w_i` by `‖y_i‖` (this keeps `W = B Y`
//!     and removes the drift of the column norms of `Y`: `c² + s²` rounds
//!     with a bias, and a column touched by `r` rotations drifts by up to
//!     `r·u` — measured `1.3·10⁻¹⁴` for `q = 21` before this step, `8·10⁻¹⁶`
//!     after, in task L3.3; without it the restarts of 12.5, `P ← P Y`,
//!     compound the drift). Then `σ_i ← ‖w_i‖` (unscaled); sort
//!     nonincreasingly (stable: ties keep their column order), permuting `W`
//!     and `Y` alike.
//! J5. \[Left vectors.\] `x_i ← w_i/‖w_i‖` for `σ_i > 0`. For `σ_i = 0` (an
//!     exactly zero column) complete `X` to orthonormal columns: try `e_0,
//!     e_1, …` in turn, orthogonalize against the columns already in `X`
//!     (two Gram–Schmidt passes) and take the first whose remaining norm
//!     exceeds `½`.
//!
//! ## Invariants
//! (J1) `σ` nonincreasing, nonnegative, finite for finite `B`.
//! (J2) `Y` orthogonal and `X` orthonormal to `O(tol)` — by the stopping
//!      test, `|x_iᵀx_j| ≤ tol` *relative*, whatever the sizes of `σ_i`,
//!      `σ_j` (this is why the one-sided method gives small singular values
//!      with orthogonal vectors).
//! (J3) `‖B Y − X Σ‖ ≤ c·q·u·‖B‖` (backward stability of the rotations).
//! All three are checked by the tests below (and debug-asserted for J1).
//!
//! ## References
//! Hestenes, "Inversion of matrices by biorthogonalization", J. SIAM 6
//! (1958); Demmel & Veselić, "Jacobi's method is more accurate than QR",
//! SIMAX 13 (1992); Golub & Van Loan, *Matrix Computations*, 4th ed.,
//! §8.6.3; Drmač & Veselić, "New fast and accurate Jacobi SVD algorithm",
//! SIMAX 29 (2008) (the `√p·ε` tolerance of LAPACK `dgesvj`). Cited from
//! memory.

use xingo_sparse::{DenseBlock, Real};

use crate::StatlaError;
use crate::real::{is_exact_zero, le_exact, lt};
use crate::small::{axpy_sub, dot, norm2};

/// The thresholds of Algorithm J.
#[derive(Clone, Copy, Debug)]
pub(crate) struct JacobiOptions<T> {
    /// Rotate when `|w_iᵀw_j| > tol·‖w_i‖‖w_j‖`.
    pub(crate) tol: T,
    /// Sweep limit of J3.
    pub(crate) max_sweeps: usize,
}

impl<T: Real> JacobiOptions<T> {
    /// The defaults for `p` rows: `tol = √p·ε_T` (as LAPACK `dgesvj`) and
    /// 60 sweeps (more than ten times what quadratic convergence needs at
    /// the sizes of this crate).
    pub(crate) fn for_rows(p: usize) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let pf = T::from_f64(p.max(1) as f64);
        Self {
            tol: T::EPSILON * pf.sqrt(),
            max_sweeps: 60,
        }
    }
}

/// The result `B = X diag(s) Yᵀ` of Algorithm J.
#[derive(Clone, Debug)]
pub(crate) struct SmallSvd<T: Real> {
    /// Left singular vectors, `p × q`.
    pub(crate) x: DenseBlock<T>,
    /// Singular values, nonincreasing.
    pub(crate) s: Vec<T>,
    /// Right singular vectors, `q × q`.
    pub(crate) y: DenseBlock<T>,
    /// Sweeps performed (read by the tests and the measurements).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) sweeps: usize,
}

/// `2^⌊log₂ μ⌋` for a positive normal `μ`, `1` otherwise.
fn pow2_floor(mu: f64) -> f64 {
    let bits = mu.to_bits() & 0x7ff0_0000_0000_0000;
    if bits == 0 || bits == 0x7ff0_0000_0000_0000 {
        1.0
    } else {
        f64::from_bits(bits)
    }
}

/// Rotate columns `i < j` of the `rows`-row array: `(a, b) ← (c a − s b, s
/// a + c b)`.
fn rotate<T: Real>(data: &mut [T], rows: usize, i: usize, j: usize, c: T, s: T) {
    let (left, right) = data.split_at_mut(j * rows);
    let a = &mut left[i * rows..(i + 1) * rows];
    let b = &mut right[..rows];
    for (x, y) in a.iter_mut().zip(b.iter_mut()) {
        let (xa, yb) = (*x, *y);
        *x = c * xa - s * yb;
        *y = s * xa + c * yb;
    }
}

/// Algorithm J on `b` (`p × q`, `p ≥ q`, finite).
#[allow(clippy::too_many_lines)] // steps J1–J5 in one place, as in the chapter
pub(crate) fn jacobi_svd<T: Real>(
    b: &DenseBlock<T>,
    opts: &JacobiOptions<T>,
) -> Result<SmallSvd<T>, StatlaError> {
    let (p, q) = (b.rows, b.cols);
    debug_assert!(p >= q, "Algorithm J needs p >= q");
    let mut y = DenseBlock::zeros(q, q);
    for i in 0..q {
        y.data[i + i * q] = T::ONE;
    }
    // J1.
    let mut mu = T::ZERO;
    for &v in &b.data {
        mu = mu.max(v.abs());
    }
    if is_exact_zero(mu) {
        let mut x = DenseBlock::zeros(p, q);
        for i in 0..q {
            x.data[i + i * p] = T::ONE;
        }
        return Ok(SmallSvd {
            x,
            s: vec![T::ZERO; q],
            y,
            sweeps: 0,
        });
    }
    let scale = pow2_floor(mu.to_f64());
    let inv = T::from_f64(1.0 / scale);
    let mut w = DenseBlock {
        rows: p,
        cols: q,
        data: b.data.iter().map(|&v| v * inv).collect(),
    };
    // J2–J3.
    let one = T::ONE;
    let two = T::from_f64(2.0);
    let mut sweeps = 0;
    loop {
        if sweeps == opts.max_sweeps {
            return Err(StatlaError::NoConvergence {
                requested: q,
                converged: 0,
                restarts: sweeps,
            });
        }
        sweeps += 1;
        let mut rotated = false;
        // A column whose squared norm underflows (below 2⁻¹⁰⁷⁴ after the
        // scaling of J1, i.e. far below ε_T‖B‖) is set to exact zero: a
        // negligible perturbation, and J5 completes it.
        for i in 0..q {
            let wi = &mut w.data[i * p..(i + 1) * p];
            if is_exact_zero(dot(wi, wi)) {
                wi.fill(T::ZERO);
            }
        }
        for i in 0..q {
            for j in i + 1..q {
                let wi = &w.data[i * p..(i + 1) * p];
                let wj = &w.data[j * p..(j + 1) * p];
                let a = dot(wi, wi);
                let bb = dot(wj, wj);
                let g = dot(wi, wj);
                if !lt(opts.tol * a.sqrt() * bb.sqrt(), g.abs()) {
                    continue;
                }
                rotated = true;
                let zeta = (bb - a) / (two * g);
                let az = zeta.abs();
                let mag = if le_exact(one, az) {
                    let r = one / zeta;
                    az * (one + (one + r * r).sqrt())
                } else {
                    az + (one + zeta * zeta).sqrt()
                };
                let t = if lt(zeta, T::ZERO) {
                    -(one / mag)
                } else {
                    one / mag
                };
                let c = one / (one + t * t).sqrt();
                let s = c * t;
                rotate(&mut w.data, p, i, j, c, s);
                rotate(&mut y.data, q, i, j, c, s);
            }
        }
        if !rotated {
            break;
        }
    }
    // J4. Rotations computed in floating point drift the column norms of
    // `Y` away from 1 (c² + s² rounds, with a bias); dividing `y_i` and `w_i`
    // by the same `‖y_i‖` keeps `W = B Y` and removes the drift.
    for i in 0..q {
        let ny = norm2(&y.data[i * q..(i + 1) * q]);
        for v in &mut y.data[i * q..(i + 1) * q] {
            *v = *v / ny;
        }
        for v in &mut w.data[i * p..(i + 1) * p] {
            *v = *v / ny;
        }
    }
    let sc = T::from_f64(scale);
    let norms: Vec<T> = (0..q).map(|i| norm2(&w.data[i * p..(i + 1) * p])).collect();
    let mut order: Vec<usize> = (0..q).collect();
    order.sort_by(|&i, &j| {
        if lt(norms[j], norms[i]) {
            core::cmp::Ordering::Less
        } else if lt(norms[i], norms[j]) {
            core::cmp::Ordering::Greater
        } else {
            i.cmp(&j)
        }
    });
    let mut x = DenseBlock::zeros(p, q);
    let mut ys = DenseBlock::zeros(q, q);
    let mut s = Vec::with_capacity(q);
    for (c, &i) in order.iter().enumerate() {
        s.push(norms[i] * sc);
        ys.data[c * q..(c + 1) * q].copy_from_slice(&y.data[i * q..(i + 1) * q]);
        // J5, nonzero columns.
        if !is_exact_zero(norms[i]) {
            let nr = norms[i];
            for (xv, &wv) in x.data[c * p..(c + 1) * p]
                .iter_mut()
                .zip(&w.data[i * p..(i + 1) * p])
            {
                *xv = wv / nr;
            }
        }
    }
    // J5, completion of the zero columns.
    let half = T::from_f64(0.5);
    let mut next_unit = 0;
    for c in 0..q {
        if !is_exact_zero(s[c]) {
            continue;
        }
        loop {
            debug_assert!(next_unit < p, "(J5) a completion always exists");
            if next_unit >= p {
                break;
            }
            let mut e = vec![T::ZERO; p];
            e[next_unit] = T::ONE;
            next_unit += 1;
            for _ in 0..2 {
                for k in 0..q {
                    if k == c || (is_exact_zero(s[k]) && k > c) {
                        continue;
                    }
                    let xk = &x.data[k * p..(k + 1) * p];
                    let h = dot(xk, &e);
                    axpy_sub(h, xk, &mut e);
                }
            }
            let nu = norm2(&e);
            if lt(half, nu) {
                for (xv, ev) in x.data[c * p..(c + 1) * p].iter_mut().zip(&e) {
                    *xv = *ev / nu;
                }
                break;
            }
        }
    }
    debug_assert!(
        s.windows(2).all(|v| le_exact(v[1], v[0])) && s.iter().all(|v| le_exact(T::ZERO, *v)),
        "(J1) singular values nonincreasing and nonnegative"
    );
    Ok(SmallSvd {
        x,
        s,
        y: ys,
        sweeps,
    })
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
mod tests {
    use super::*;
    use crate::random::Rng;
    use xingo_extprec::DoubleDouble as Dd;

    fn residuals<T: Real>(b: &DenseBlock<T>, f: &SmallSvd<T>) -> (f64, f64, f64) {
        let (p, q) = (b.rows, b.cols);
        let mut back = 0.0f64;
        let mut ox = 0.0f64;
        let mut oy = 0.0f64;
        for j in 0..q {
            for i in 0..p {
                let mut by = T::ZERO;
                for k in 0..q {
                    by += b.get(i, k) * f.y.get(k, j);
                }
                back = back.max((by - f.x.get(i, j) * f.s[j]).abs().to_f64());
            }
            for i in 0..q {
                let e = if i == j { T::ONE } else { T::ZERO };
                let xx: T = (0..p).map(|r| f.x.get(r, i) * f.x.get(r, j)).sum();
                let yy: T = (0..q).map(|r| f.y.get(r, i) * f.y.get(r, j)).sum();
                ox = ox.max((xx - e).abs().to_f64());
                oy = oy.max((yy - e).abs().to_f64());
            }
        }
        (back, ox, oy)
    }

    fn graded<T: Real>(p: usize, q: usize, seed: u64) -> DenseBlock<T> {
        // Columns scaled by 10^{-j}: singular values spread over 10^q.
        let mut r = Rng::new(seed);
        let mut b = DenseBlock::zeros(p, q);
        for j in 0..q {
            for i in 0..p {
                b.data[i + j * p] = T::from_f64(r.gaussian() * 10f64.powi(-(j as i32)));
            }
        }
        b
    }

    fn check<T: Real>(u: f64) {
        for (p, q) in [(1, 1), (5, 3), (12, 12), (30, 17)] {
            let b = graded::<T>(p, q, (p * 100 + q) as u64);
            let f = jacobi_svd(&b, &JacobiOptions::for_rows(p)).unwrap();
            let (back, ox, oy) = residuals(&b, &f);
            let nb = f.s[0].to_f64();
            assert!(back <= 8.0 * q as f64 * u * nb, "{p}x{q}: backward {back}");
            assert!(ox <= 8.0 * (p as f64).sqrt() * u * 2.0, "{p}x{q}: X {ox}");
            assert!(oy <= 8.0 * q as f64 * u, "{p}x{q}: Y {oy}");
            assert!(f.s.windows(2).all(|v| v[1] <= v[0]));
        }
    }

    #[test]
    fn jacobi_is_backward_stable_with_orthogonal_factors() {
        check::<f64>(f64::EPSILON / 2.0);
        check::<Dd>(Dd::EPSILON.to_f64() / 2.0);
    }

    /// (J2) for `Y`: the accumulated rotations stay orthogonal to a small
    /// multiple of `√q·u` (step J4 removes the norm drift, which otherwise
    /// grows like the number of rotations per column).
    #[test]
    fn accumulated_rotations_stay_orthogonal() {
        let mut worst = 0.0f64;
        for seed in 0..4 {
            let b = graded::<f64>(60, 48, 40 + seed);
            let f = jacobi_svd(&b, &JacobiOptions::for_rows(60)).unwrap();
            let (_, _, oy) = residuals(&b, &f);
            worst = worst.max(oy / ((48f64).sqrt() * f64::EPSILON / 2.0));
        }
        eprintln!("(J2) Y: max ‖YᵀY − I‖ = {worst:.2} √q u");
        assert!(worst <= 1.5, "{worst}");
    }

    #[test]
    fn zero_and_rank_deficient_columns_are_completed() {
        // Columns 1 and 3 zero, column 2 = 2·column 0.
        let mut b = DenseBlock::<f64>::zeros(5, 4);
        for i in 0..5 {
            b.set(i, 0, (i + 1) as f64);
            b.set(i, 2, 2.0 * (i + 1) as f64);
        }
        let f = jacobi_svd(&b, &JacobiOptions::for_rows(5)).unwrap();
        let (back, ox, oy) = residuals(&b, &f);
        assert!(back < 1e-14 && ox < 1e-14 && oy < 1e-14, "{back} {ox} {oy}");
        assert!(f.s[1] < 1e-14 && f.s[2] == 0.0 && f.s[3] == 0.0);
        let z = jacobi_svd(&DenseBlock::<f64>::zeros(3, 2), &JacobiOptions::for_rows(3)).unwrap();
        assert_eq!(
            (z.s.clone(), z.x.get(1, 1), z.sweeps),
            (vec![0.0, 0.0], 1.0, 0)
        );
        // A diagonal input is not rotated and keeps exact values.
        let mut d = DenseBlock::<f64>::zeros(3, 3);
        d.set(0, 0, 2.0);
        d.set(1, 1, 3.0);
        d.set(2, 2, 1.0);
        let f = jacobi_svd(&d, &JacobiOptions::for_rows(3)).unwrap();
        assert_eq!(f.s, vec![3.0, 2.0, 1.0]);
        assert_eq!((f.sweeps, f.y.get(1, 0), f.x.get(1, 0)), (1, 1.0, 1.0));
    }

    #[test]
    fn extreme_scales_neither_overflow_nor_underflow() {
        for k in [1e300, 1e-300] {
            let mut b = graded::<f64>(6, 4, 3);
            for v in &mut b.data {
                *v *= k;
            }
            let f = jacobi_svd(&b, &JacobiOptions::for_rows(6)).unwrap();
            let (back, ox, _) = residuals(&b, &f);
            assert!(f.s.iter().all(|v| v.is_finite()));
            assert!(back <= 1e-14 * f.s[0] && ox < 1e-14, "{k}: {back} {ox}");
        }
    }
}
