//! # Chapter 12.0.4 — Small dense routines: norms, Givens rotations, Householder reflectors, DGKS Gram–Schmidt
//!
//! ## Purpose
//! The kernels of `xingo-sparse` (`DenseKernel`) are the BLAS-3 operations
//! of a supernodal factorization; the updating algorithms of Chapters 12.2
//! and 12.3 need the BLAS-1 and LAPACK-auxiliary ones instead — a plane
//! rotation, an elementary reflector, a Gram–Schmidt step. They are written
//! here once, generic in `T: Real` (decision S1 of D-0127), with only `+ −
//! · /`, `sqrt`, `abs` and the order.
//!
//! ## Notation
//! `x, y ∈ T^m`; `‖x‖` the 2-norm; a *rotation* `G = (c, s)` with `c² + s²
//! = 1` acts on a pair as `(x, y) ↦ (c x + s y, −s x + c y)`; a *reflector*
//! `H = I − τ v vᵀ` with `v_0 = 1`; `Q ∈ T^{m×k}` with orthonormal columns
//! (`k` may be 0).
//!
//! ## Algorithm N (2-norm)
//! N1. `μ ← max |x_i|`. If `μ = 0`, return 0.
//! N2. If `2⁻⁵⁰⁰ ≤ μ ≤ 2⁵⁰⁰` (as `f64`), return `√(Σ x_i²)`, summed in index
//!     order.
//! N3. Otherwise let `σ = 2^⌊log₂ μ⌋` (read off the bits of `μ` as `f64`):
//!     return `σ·√(Σ (x_i/σ)²)`. Scaling by a power of two is exact in
//!     `f64` and in double-double, so N2 and N3 round identically; N3 only
//!     avoids overflow and underflow of the squares.
//!
//! ## Algorithm G (Givens rotation; Golub & Van Loan §5.1.8)
//! Given `(a, b)`, find `G = (c, s)` and `r ≥ 0` with `c a + s b = r` and
//! `−s a + c b = 0`:
//! G1. If `b = 0`: `(c, s, r) = (1, 0, a)` if `a ≥ 0`, else `(−1, 0, −a)`.
//! G2. Else `μ ← |a| + |b|`, `r ← μ·√((a/μ)² + (b/μ)²)`, `c ← a/r`, `s ←
//!     b/r` (the scaling is the "safe hypot" of the contract).
//!
//! ## Algorithm H (Householder reflector; GVL §5.1.3)
//! Given `x ∈ T^p`, `p ≥ 1`, find `τ`, `v` (`v_0 = 1`) and `α` with `H x = α
//! e_0`:
//! H1. `σ ← ‖x‖` (Algorithm N). If `σ = 0`: `τ = 0`, `α = 0`, `v = e_0`.
//! H2. `α ← −σ` if `x_0 ≥ 0`, else `σ` (no cancellation in `x_0 − α`).
//! H3. `v_i ← x_i / (x_0 − α)` for `i ≥ 1`, `τ ← (α − x_0)/α`.
//! The callers flip signs afterwards to make the diagonal of `R` positive.
//!
//! ## Algorithm D (classical Gram–Schmidt with DGKS reorthogonalization)
//! Given `Q` and `x`, compute `x ← (I − QQᵀ) x` and the coefficients `h`
//! with `x_in = Q h + x_out`:
//! D1. `ν₀ ← ‖x‖`; `h ← 0`.
//! D2. \[Pass.\] `s_j ← q_jᵀ x` for every `j` (all dot products first: CGS),
//!     then `x ← x − Σ_j s_j q_j`, `h ← h + s`; `ν ← ‖x‖`.
//! D3. \[DGKS.\] If `ν < η ν₀`, the pass cancelled: repeat D2 once (with
//!     `ν₀ ← ν` unused afterwards). **Never a third pass** (Daniel, Gragg,
//!     Kaufman & Stewart 1976: "twice is enough" when the input is not
//!     numerically dependent; a dependent input is caught by the caller's
//!     rank test).
//! D4. Return `ν = ‖x‖` and the number of passes.
//!
//! ## Algorithm Q (thin Householder QR with explicit `Q`, any rank)
//! Given `A ∈ T^{m×l}`, `m ≥ l`: the reflectors of F2 in Chapter 12.3
//! (Algorithm H column by column), `Q ← H_0 ⋯ H_{l−1} [I_l; 0]` applied
//! backwards, and the signs flipped so that `r_kk ≥ 0`. Unlike
//! [`crate::Qr::factor`] there is **no rank test**: a rank-deficient `A`
//! still gives an orthonormal `Q` (the range finder of Chapter 12.4 must
//! accept a sketch `AΩ` of a low-rank `A`).
//!
//! ## Invariants
//! (S1) Algorithm G returns `r ≥ 0` and `|c|, |s| ≤ 1`; for finite input
//!      the output is finite (debug-asserted).
//! (S2) After Algorithm D, `‖Qᵀx‖ ≤ c · m · u · ‖x_in‖` when the second
//!      pass was taken or not needed and `x` is not numerically in the
//!      range of `Q` — the property (Q1) of Chapter 12.3 rests on; tested
//!      in Chapter 12.3 (and in the mutation check that disables D3).
//!
//! ## References
//! Golub & Van Loan, *Matrix Computations*, 4th ed. (2013), §5.1; Daniel,
//! Gragg, Kaufman & Stewart, Math. Comp. 30 (1976); Björck, *Numerical
//! Methods for Least Squares Problems* (1996), §2.4 (cited from memory).

use xingo_sparse::{DenseBlock, Real};

use crate::StatlaError;
use crate::real::{is_exact_zero, le_exact, lt};

/// `2⁵⁰⁰` and `2⁻⁵⁰⁰`: the range in which the squares of Algorithm N can
/// neither overflow nor underflow (a property of the `f64` exponent range
/// — the double-double has the same — not a tuning knob).
const BIG: f64 = 3.273_390_607_896_142e150;
const SMALL: f64 = 3.054_936_363_499_605e-151;

/// `2^⌊log₂ μ⌋` for a positive normal `μ` (its bits with the mantissa
/// cleared); `1` otherwise.
fn pow2_floor(mu: f64) -> f64 {
    let bits = mu.to_bits() & 0x7ff0_0000_0000_0000;
    if bits == 0 || bits == 0x7ff0_0000_0000_0000 {
        1.0
    } else {
        f64::from_bits(bits)
    }
}

/// Algorithm N: `‖x‖`.
pub(crate) fn norm2<T: Real>(x: &[T]) -> T {
    // N1.
    let mut mu = T::ZERO;
    for &v in x {
        mu = mu.max(v.abs());
    }
    if is_exact_zero(mu) {
        return T::ZERO;
    }
    let m = mu.to_f64();
    // N2.
    if le_exact(SMALL, m) && le_exact(m, BIG) {
        return x.iter().map(|&v| v * v).sum::<T>().sqrt();
    }
    // N3.
    let sigma = pow2_floor(m);
    let inv = T::from_f64(1.0 / sigma);
    let s = T::from_f64(sigma);
    s * x
        .iter()
        .map(|&v| {
            let w = v * inv;
            w * w
        })
        .sum::<T>()
        .sqrt()
}

/// `xᵀy`, summed in index order.
#[inline]
pub(crate) fn dot<T: Real>(x: &[T], y: &[T]) -> T {
    x.iter().zip(y).map(|(&a, &b)| a * b).sum()
}

/// `y ← y − α x`.
#[inline]
pub(crate) fn axpy_sub<T: Real>(alpha: T, x: &[T], y: &mut [T]) {
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi -= alpha * xi;
    }
}

/// The first non-finite entry of `x`, as the error of `what`.
pub(crate) fn check_finite<T: Real>(what: &'static str, x: &[T]) -> Result<(), StatlaError> {
    match x.iter().position(|v| !v.is_finite()) {
        Some(index) => Err(StatlaError::NonFinite { what, index }),
        None => Ok(()),
    }
}

/// The length check of a vector argument.
pub(crate) fn check_len(
    what: &'static str,
    expected: usize,
    found: usize,
) -> Result<(), StatlaError> {
    if expected == found {
        Ok(())
    } else {
        Err(StatlaError::Dimension {
            what,
            expected,
            found,
        })
    }
}

/// A plane rotation `(c, s)`: `(x, y) ↦ (c x + s y, −s x + c y)`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Rot<T> {
    pub(crate) c: T,
    pub(crate) s: T,
}

impl<T: Real> Rot<T> {
    /// The identity rotation.
    pub(crate) const IDENTITY: Self = Self {
        c: T::ONE,
        s: T::ZERO,
    };

    /// `G (x, y)`.
    #[inline]
    pub(crate) fn apply(self, x: T, y: T) -> (T, T) {
        (self.c * x + self.s * y, self.c * y - self.s * x)
    }

    /// Rotate columns `j1`, `j2` of the `rows`-row column-major `data`:
    /// `(q_j1, q_j2) ← (c q_j1 + s q_j2, −s q_j1 + c q_j2)` (the right
    /// multiplication `Q ← Q Gᵀ` matching a rotation of rows `j1, j2` of
    /// `R`).
    pub(crate) fn rotate_cols(self, data: &mut [T], rows: usize, j1: usize, j2: usize) {
        debug_assert!(j1 < j2);
        let (left, right) = data.split_at_mut(j2 * rows);
        let a = &mut left[j1 * rows..(j1 + 1) * rows];
        let b = &mut right[..rows];
        for (x, y) in a.iter_mut().zip(b.iter_mut()) {
            let (nx, ny) = self.apply(*x, *y);
            *x = nx;
            *y = ny;
        }
    }

    /// Rotate rows `i1`, `i2` of the block `r` over the columns `from..`.
    pub(crate) fn rotate_rows(self, r: &mut DenseBlock<T>, i1: usize, i2: usize, from: usize) {
        let rows = r.rows;
        for j in from..r.cols {
            let (nx, ny) = self.apply(r.data[i1 + j * rows], r.data[i2 + j * rows]);
            r.data[i1 + j * rows] = nx;
            r.data[i2 + j * rows] = ny;
        }
    }
}

/// Algorithm G: `(G, r)` with `G (a, b) = (r, 0)`, `r ≥ 0`.
pub(crate) fn givens<T: Real>(a: T, b: T) -> (Rot<T>, T) {
    // G1.
    if is_exact_zero(b) {
        return if lt(a, T::ZERO) {
            (
                Rot {
                    c: -T::ONE,
                    s: T::ZERO,
                },
                -a,
            )
        } else {
            (Rot::IDENTITY, a)
        };
    }
    // G2.
    let mu = a.abs() + b.abs();
    let (aa, bb) = (a / mu, b / mu);
    let r = mu * (aa * aa + bb * bb).sqrt();
    let g = Rot { c: a / r, s: b / r };
    debug_assert!(
        !a.is_finite() || !b.is_finite() || (r.is_finite() && le_exact(T::ZERO, r)),
        "(S1) Givens: r must be finite and nonnegative"
    );
    (g, r)
}

/// Algorithm H on `x` (length `p ≥ 1`), in place: on return `x_0 = α`,
/// `x_1.. = v_1..` (with `v_0 = 1` implicit); returns `τ`.
pub(crate) fn householder<T: Real>(x: &mut [T]) -> T {
    debug_assert_ne!(x.len(), 0, "Algorithm H needs p >= 1");
    // H1.
    let sigma = norm2(x);
    if is_exact_zero(sigma) {
        return T::ZERO;
    }
    // H2.
    let x0 = x[0];
    let alpha = if lt(x0, T::ZERO) { sigma } else { -sigma };
    // H3.
    let d = x0 - alpha;
    for v in &mut x[1..] {
        *v = *v / d;
    }
    x[0] = alpha;
    (alpha - x0) / alpha
}

/// Apply `H = I − τ v vᵀ` (`v_0 = 1`, `v_1.. = v_tail`) to `y` (length
/// `1 + v_tail.len()`).
pub(crate) fn apply_householder<T: Real>(tau: T, v_tail: &[T], y: &mut [T]) {
    if is_exact_zero(tau) {
        return;
    }
    let w = tau * (y[0] + dot(v_tail, &y[1..]));
    y[0] -= w;
    axpy_sub(w, v_tail, &mut y[1..]);
}

/// Algorithm D: `x ← (I − QQᵀ)x` against the columns of `q` (`q.rows =
/// x.len()`), accumulating the coefficients into `h` (length `q.cols`).
/// Returns `(‖x‖, passes)`.
pub(crate) fn orthogonalize<T: Real>(
    q: &DenseBlock<T>,
    x: &mut [T],
    h: &mut [T],
    eta: T,
) -> (T, u8) {
    debug_assert_eq!(q.rows, x.len());
    debug_assert_eq!(q.cols, h.len());
    let m = q.rows;
    // D1.
    let nu0 = norm2(x);
    let mut s = vec![T::ZERO; q.cols];
    let mut pass = |x: &mut [T], h: &mut [T]| {
        // D2.
        for (j, sj) in s.iter_mut().enumerate() {
            *sj = dot(&q.data[j * m..(j + 1) * m], x);
        }
        for (j, &sj) in s.iter().enumerate() {
            axpy_sub(sj, &q.data[j * m..(j + 1) * m], x);
            h[j] += sj;
        }
        norm2(x)
    };
    let mut nu = pass(x, h);
    let mut passes = 1;
    // D3.
    if lt(nu, eta * nu0) {
        nu = pass(x, h);
        passes = 2;
    }
    // D4.
    (nu, passes)
}

/// Algorithm Q: `(Q, R)` of the `m × l` block `a` (`m ≥ l`), `r_kk ≥ 0`.
pub(crate) fn householder_qr<T: Real>(mut w: DenseBlock<T>) -> (DenseBlock<T>, DenseBlock<T>) {
    let (m, l) = (w.rows, w.cols);
    debug_assert!(m >= l, "Algorithm Q needs m >= l");
    let mut taus = vec![T::ZERO; l];
    for k in 0..l {
        let (left, right) = w.data.split_at_mut((k + 1) * m);
        let x = &mut left[k * m + k..];
        taus[k] = householder(x);
        let v = &x[1..];
        for j in 0..l - k - 1 {
            apply_householder(taus[k], v, &mut right[j * m + k..(j + 1) * m]);
        }
    }
    let mut r = DenseBlock::zeros(l, l);
    for j in 0..l {
        for i in 0..=j {
            r.data[i + j * l] = w.data[i + j * m];
        }
    }
    let mut q = DenseBlock::zeros(m, l);
    for k in 0..l {
        q.data[k + k * m] = T::ONE;
    }
    for k in (0..l).rev() {
        let v = &w.data[k * m + k + 1..(k + 1) * m];
        for j in k..l {
            apply_householder(taus[k], v, &mut q.data[j * m + k..(j + 1) * m]);
        }
    }
    for k in 0..l {
        if lt(r.data[k + k * l], T::ZERO) {
            for j in k..l {
                r.data[k + j * l] = -r.data[k + j * l];
            }
            for v in &mut q.data[k * m..(k + 1) * m] {
                *v = -*v;
            }
        }
    }
    (q, r)
}

/// Algorithm D on the first `ncols` columns of the column-major array
/// `data` (`rows` rows each) — the growing Krylov bases of Chapter 12.5
/// live in preallocated blocks. The coefficients are discarded. Returns
/// `(‖x‖, passes)`.
pub(crate) fn orthogonalize_against<T: Real>(
    data: &[T],
    rows: usize,
    ncols: usize,
    x: &mut [T],
    eta: T,
) -> (T, u8) {
    orthogonalize_prefix(data, rows, ncols, x, eta, None)
}

/// Algorithm D as [`orthogonalize_against`], accumulating the coefficients
/// of both passes into `h` (length `≥ ncols`) when given — the Arnoldi
/// column of Chapter 12.6. The arithmetic on `x` is the same either way.
pub(crate) fn orthogonalize_prefix<T: Real>(
    data: &[T],
    rows: usize,
    ncols: usize,
    x: &mut [T],
    eta: T,
    mut h: Option<&mut [T]>,
) -> (T, u8) {
    debug_assert_eq!(rows, x.len());
    let nu0 = norm2(x);
    let mut s = vec![T::ZERO; ncols];
    let mut pass = |x: &mut [T], h: &mut Option<&mut [T]>| {
        for (j, sj) in s.iter_mut().enumerate() {
            *sj = dot(&data[j * rows..(j + 1) * rows], x);
        }
        for (j, &sj) in s.iter().enumerate() {
            axpy_sub(sj, &data[j * rows..(j + 1) * rows], x);
        }
        if let Some(h) = h.as_deref_mut() {
            for (hj, &sj) in h.iter_mut().zip(&s) {
                *hj += sj;
            }
        }
        norm2(x)
    };
    let mut nu = pass(x, &mut h);
    let mut passes = 1;
    if lt(nu, eta * nu0) {
        nu = pass(x, &mut h);
        passes = 2;
    }
    (nu, passes)
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // the tests compare exact values
mod tests {
    use super::*;

    #[test]
    fn norm_scales_without_changing_rounding() {
        let x = [3.0, 4.0];
        assert_eq!(norm2(&x), 5.0);
        let big = [3e300, 4e300];
        assert!((norm2(&big) / 5e300 - 1.0).abs() < 4.0 * f64::EPSILON);
        let tiny = [3e-300, 4e-300];
        assert!((norm2(&tiny) / 5e-300 - 1.0).abs() < 4.0 * f64::EPSILON);
        // Exact power-of-two scaling: the same bits as the unscaled sum.
        let y = [0.1_f64, 0.7, -2.3];
        let k = 2f64.powi(600);
        let scaled: Vec<f64> = y.iter().map(|v| v * k).collect();
        assert_eq!(norm2(&scaled), norm2(&y) * k);
        assert_eq!(norm2::<f64>(&[]), 0.0);
        assert_eq!(norm2(&[0.0, -0.0]), 0.0);
    }

    #[test]
    fn givens_annihilates_and_keeps_r_nonnegative() {
        for &(a, b) in &[
            (3.0, 4.0),
            (-3.0, 4.0),
            (3.0, -4.0),
            (0.0, -2.0),
            (-5.0, 0.0),
            (0.0, 0.0),
            (1e300, 1e300),
        ] {
            let (g, r) = givens(a, b);
            let (x, y) = g.apply(a, b);
            assert!(r >= 0.0);
            assert!((x - r).abs() <= 4.0 * f64::EPSILON * r.max(1.0), "{a} {b}");
            assert!(y.abs() <= 4.0 * f64::EPSILON * r.max(1.0), "{a} {b}");
            assert!((g.c * g.c + g.s * g.s - 1.0).abs() <= 4.0 * f64::EPSILON);
        }
    }

    #[test]
    fn householder_maps_to_a_multiple_of_e0() {
        let x0 = [1.0, -2.0, 2.0, 0.5];
        for sign in [1.0, -1.0] {
            let mut x: Vec<f64> = x0.iter().map(|v| v * sign).collect();
            let tau = householder(&mut x);
            let mut y: Vec<f64> = x0.iter().map(|v| v * sign).collect();
            apply_householder(tau, &x[1..], &mut y);
            assert!((y[0] - x[0]).abs() < 1e-15 && (x[0].abs() - norm2(&x0)).abs() < 1e-15);
            assert!(y[1..].iter().all(|v| v.abs() < 1e-15));
        }
        let mut z = [0.0, 0.0];
        assert_eq!(householder(&mut z), 0.0);
    }

    #[test]
    fn dgks_reorthogonalizes_a_nearly_dependent_vector() {
        // Q = e_0, e_1 in R^3; x nearly in their span.
        let q = DenseBlock {
            rows: 3,
            cols: 2,
            data: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        };
        let mut x = [1.0, 1.0, 1e-9];
        let mut h = [0.0; 2];
        let (nu, passes) = orthogonalize(&q, &mut x, &mut h, std::f64::consts::FRAC_1_SQRT_2);
        assert_eq!(passes, 2);
        assert_eq!((nu, h), (1e-9, [1.0, 1.0]));
        let mut x = [0.0, 0.0, 1.0];
        let (_, passes) = orthogonalize(&q, &mut x, &mut [0.0; 2], std::f64::consts::FRAC_1_SQRT_2);
        assert_eq!(passes, 1);
        assert_eq!(
            check_finite("x", &[1.0, f64::NAN]),
            Err(StatlaError::NonFinite {
                what: "x",
                index: 1
            })
        );
        assert!(check_len("x", 2, 3).is_err() && check_len("x", 2, 2).is_ok());
        // The slice form gives the same vector.
        let mut y = [1.0, 1.0, 1e-9];
        let (nu2, p2) =
            orthogonalize_against(&q.data, 3, 2, &mut y, std::f64::consts::FRAC_1_SQRT_2);
        assert_eq!((nu2, p2, y), (1e-9, 2, [0.0, 0.0, 1e-9]));
    }

    #[test]
    fn householder_qr_accepts_rank_deficiency() {
        // Columns 0 and 2 equal: rank 2 of 3.
        let a = DenseBlock {
            rows: 4,
            cols: 3,
            data: vec![1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0],
        };
        let (q, r) = householder_qr(a.clone());
        for i in 0..3 {
            for j in 0..3 {
                let d = dot(&q.data[i * 4..i * 4 + 4], &q.data[j * 4..j * 4 + 4]);
                let e = if i == j { 1.0 } else { 0.0 };
                assert!((d - e).abs() < 1e-15);
            }
            assert!(r.get(i, i) >= 0.0);
        }
        for j in 0..3 {
            for i in 0..4 {
                let qr: f64 = (0..3).map(|k| q.get(i, k) * r.get(k, j)).sum();
                assert!((qr - a.get(i, j)).abs() < 1e-15);
            }
        }
        assert!(r.get(2, 2) < 1e-15);
    }
}
