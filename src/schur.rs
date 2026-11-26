//! # Chapter 12.6.1 — Small dense eigensolvers: symmetric Jacobi, Hessenberg–Francis real Schur, reordering, eigenvectors
//!
//! ## Purpose
//! The restarted drivers of Chapter 12.6 project the operator onto a basis
//! of `ncv` vectors and need, at every restart, the eigen-decomposition of
//! the projected `ncv × ncv` matrix `H` — in the symmetric case its
//! eigenvalues and orthogonal eigenvectors, in the general case a *real
//! Schur form* `H = Z T Zᵀ` (`Z` orthogonal, `T` quasi-upper-triangular)
//! whose leading block holds the wanted Ritz values. `DenseKernel` has no
//! eigensolver and the crate is generic in `T: Real` (decision S1), so the
//! five routines are written here with only `+ − · /`, `sqrt`, `abs` and
//! the order; each is backward stable (orthogonal transformations only),
//! which is what the residual guarantees of 12.6 rest on.
//!
//! ## Notation
//! `A, H, T ∈ T^{p×p}` column-major, `a_ij = data[i + j·p]`; `Z` the
//! accumulated orthogonal transformation (`H_in = Z T Zᵀ` with the `Z`
//! passed in multiplied on the right); `ε = ε_T`; a *block* of `T` is a
//! `1 × 1` diagonal entry or a `2 × 2` diagonal block `[a b; c a]` with `bc
//! < 0` (the *standardized* form: eigenvalues `a ± i√(|b||c|)`), and `T` is
//! zero below the diagonal outside the blocks. `‖·‖` is the largest
//! absolute entry where a scale is needed (`μ`).
//!
//! ## Algorithm SJ (cyclic Jacobi for a symmetric matrix)
//! SJ1. \[Scale.\] `μ ← max |a_ij|`; if `μ = 0` return `d = 0`, `Z = I`.
//!      `W ← A/2^⌊log₂ μ⌋` (exact).
//! SJ2. \[Sweep.\] For every `p < q` in row-cyclic order: if `|w_pq| >
//!      tol·√|w_pp|·√|w_qq|` and `|w_pq| > tol²` (the second test is an
//!      absolute floor `ε²μ`: below it the entry is negligible against
//!      `ε‖A‖` even when both diagonals vanish): `θ ← (w_qq − w_pp)/(2w_pq)`,
//!      `t ← sign(θ)/(|θ| + √(1 + θ²))` (with `θ⁻¹` inside the root when `|θ|
//!      ≥ 1`), `c ← 1/√(1 + t²)`, `s ← t c`; `W ← JᵀWJ` and `Z ← ZJ` with `J`
//!      the rotation in the plane `(p, q)` (`col_p ← c col_p − s col_q`, `col_q
//!      ← s col_p + c col_q`, the same on the rows), then `w_pp ← w_pp −
//!      t w_pq`, `w_qq ← w_qq + t w_pq`, `w_pq = w_qp ← 0` exactly (GVL
//!      §8.5.2, the formulas that keep the diagonal accurate).
//! SJ3. \[Stop.\] After a sweep without rotation, `d_i ← w_ii · 2^⌊log₂ μ⌋`;
//!      the columns of `Z` are renormalized (the drift of `c² + s²`, as J4 of
//!      Chapter 12.0.6). After `max_sweeps`, non-convergence (never seen).
//!
//! The entries left are `≤ tol·√|d_p d_q|` (relative) or `≤ ε²μ`, so every
//! eigenvalue is within `O(p·tol·‖A‖)` of one of `A` (Weyl), and the
//! eigenvectors are orthogonal to `O(tol)`.
//!
//! ## Algorithm HR (Hessenberg reduction)
//! For `k = 0, …, p − 3`: the reflector `P_k = I − τvvᵀ` of Algorithm H
//! (Chapter 12.0.4) on `h_{k+1:p, k}`; `H ← P_k H P_k`, `Z ← Z P_k`; the
//! entries `h_{k+2:p, k}` are set to zero exactly (GVL Alg. 7.4.2).
//!
//! ## Algorithm FQ (Francis double-shift QR to real Schur form)
//! The window is `[l, i]`, initially `[0, p − 1]`; `its` counts the sweeps
//! spent on the current bottom block.
//! FQ1. \[Deflate.\] Find the largest `l ≤ i` with `l = 0` or `|h_{l,l−1}| ≤
//!      ε(|h_{l−1,l−1}| + |h_{ll}|)` (if that sum is `0`, against the largest
//!      absolute entry `μ` of `H`); set `h_{l,l−1} ← 0`.
//! FQ2. \[One converged.\] If `l = i`: `i ← i − 1`, `its ← 0`; go to FQ1 (stop
//!      when the window is empty).
//! FQ3. \[Two converged.\] If `l = i − 1`: standardize the `2 × 2` block by
//!      Algorithm SV, apply its rotation to rows `i − 1, i` (columns `i + 1..`),
//!      to columns `i − 1, i` (rows `..i − 1`) and to `Z`; `i ← i − 2`, `its ←
//!      0`; go to FQ1.
//! FQ4. \[Shifts.\] `its ← its + 1`; if the total exceeds `30·max(10, p)`,
//!      non-convergence. The shifts are the eigenvalues of the trailing `2 × 2`
//!      of the window (trace `s`, determinant `t`); at `its = 10` and `its =
//!      20` the *exceptional* shifts of LAPACK `dlahqr` break a cycle: the
//!      trailing block is replaced by `[0.75w + h_rr, −0.4375w; w, 0.75w +
//!      h_rr]` with `w = |h_{l+1,l}| + |h_{l+2,l+1}|`, `r = l` (at 10) or `w =
//!      |h_{i,i−1}| + |h_{i−1,i−2}|`, `r = i` (at 20).
//! FQ5. \[Sweep.\] `(x, y, z) ←` the first column of `(H − λ₁I)(H − λ₂I)` at
//!      `l`: `x = h_ll² + h_{l,l+1}h_{l+1,l} − s h_ll + t`, `y = h_{l+1,l}(h_ll +
//!      h_{l+1,l+1} − s)`, `z = h_{l+1,l}h_{l+2,l+1}`. For `k = l, …, i − 2`: the
//!      reflector of `(x, y, z)` (length 2 at `k = i − 1`, done after the
//!      loop) applied to rows `k..k+3` (columns `max(l, k − 1)..p`), to
//!      columns `k..k+3` (rows `0..min(k + 4, i + 1)`) and to `Z`; for `k > l`
//!      the bulge entries `h_{k+1,k−1}`, `h_{k+2,k−1}` are set to zero; `(x,
//!      y, z) ← (h_{k+1,k}, h_{k+2,k}, h_{k+3,k})` (GVL Alg. 7.5.1, the
//!      `wantt`/`wantz` form of `dlahqr`). Go to FQ1.
//!
//! ## Algorithm SV (standardization of a 2 × 2 block, LAPACK `dlanv2`)
//! Given `[a b; c d]`, a rotation `R = [cs −sn; sn cs]` with `[a b; c d] = R
//! [a′ b′; c′ d′] Rᵀ` where either `c′ = 0` (real eigenvalues `a′`, `d′`) or `a′
//! = d′`, `b′c′ < 0` (eigenvalues `a′ ± i√(|b′||c′|)`). The branches are those
//! of `dlanv2` (the real case by the stable root `z = p + sign(p)√(p² +
//! bc)`, `p = (a − d)/2`; otherwise the rotation that equalizes the
//! diagonal, then a second one if the result has real eigenvalues),
//! without its rescaling loop (the callers' matrices are scaled to `O(1)`
//! entries by the drivers' normalization of `V`).
//!
//! ## Algorithm SW (swap two adjacent blocks; Bai & Demmel 1993)
//! Blocks `T_11 ∈ T^{n₁×n₁}`, `T_22 ∈ T^{n₂×n₂}` adjacent at `j`, `D = [T_11
//! T_12; 0 T_22]`.
//! SW1. \[Sylvester.\] Solve `T_11 X − X T_22 = T_12` (`n₁n₂ ≤ 4` unknowns,
//!      Gaussian elimination with complete pivoting on the Kronecker form;
//!      a pivot below `ε·μ_D` is replaced by `ε·μ_D` — eigenvalues that
//!      close make the swap ill-conditioned, caught in SW3).
//! SW2. \[Basis.\] Householder QR of `[−X; I_{n₂}] = Q [R; 0]` (its columns
//!      span the invariant subspace of `T_22`), so `QᵀDQ = [T̃_22 *; E T̃_11]`
//!      with `E = 0` in exact arithmetic.
//! SW3. \[Test.\] On a copy: if `max |E| > 10·ε·μ_D`, refuse (return `false`,
//!      nothing modified) — the weak stability test of LAPACK `dlaexc`.
//! SW4. \[Apply.\] `T ← QᵀTQ` on rows `j..j+n` (columns `j..p`), columns
//!      `j..j+n` (rows `0..j+n`), `Z ← ZQ`; `E ← 0` exactly; standardize each
//!      new `2 × 2` block by SV (it may split into two `1 × 1` blocks when
//!      the eigenvalues were nearly real and double).
//! For `n₁ = n₂ = 1` the rotation of `(t_12, t_22 − t_11)` is used instead
//! (it leaves `t_12` unchanged and is exact for diagonal `T`).
//!
//! ## Algorithm RO (reordering by rank)
//! Each block carries an integer rank (lower = wanted earlier) that travels
//! with it through the swaps. For `pos = 0, 1, …` while the leading `need`
//! rows are not settled: pick the block of least rank at or after `pos`
//! (the first on ties), bubble it up to `pos` by SW, `pos ← pos + size`. A
//! refused swap stops the reordering (`false` is returned; the order
//! reached is consistent and the caller treats it as is).
//!
//! ## Algorithm EV (eigenvector of a quasi-triangular matrix, LAPACK `dtrevc`)
//! For the block at `q` (size `s`) with eigenvalue `λ` (for `s = 2`, the one
//! with positive imaginary part `ω = √(|b||c|)`):
//! EV1. `x_q = 1` (`s = 1`); for `s = 2`: `(x_q, x_{q+1}) = (1, iω/b)` if `|b| ≥
//!      |c|`, else `(iω/c, 1)` — a null vector of `[a − λ, b; c, a − λ]`.
//! EV2. For the blocks above `q`, bottom to top: solve `(T_bb − λI) x_b =
//!      −Σ_{j>b} T_bj x_j` (complex `1 × 1` or `2 × 2`, partial pivoting;
//!      a pivot below `smin = max(ε μ, tiny)` is replaced by `smin`). If an
//!      entry exceeds `2⁵⁰⁰`, the whole `x` is scaled down by a power of two.
//! The rows below the block are zero. `T x = λ x` up to `O(ε‖T‖)‖x‖`.
//!
//! ## Invariants
//! (E1) SJ: `‖A Z − Z diag(d)‖ ≤ c p u ‖A‖`, `‖ZᵀZ − I‖ ≤ c p u`.
//! (E2) HR + FQ + SW: `H_in Z_out = Z_out T` to `c p u ‖H‖`, `Z` orthogonal
//!      to `c p u`, `T` quasi-triangular with standardized `2 × 2` blocks
//!      (exact zeros below; [`check_quasi_triangular`]).
//! (E3) EV: `‖T x − λ x‖ ≤ c p u ‖T‖ ‖x‖`.
//! All three are checked by the unit tests below (and (E2)'s shape is
//! debug-asserted after FQ and SW).
//!
//! ## References
//! Golub & Van Loan, *Matrix Computations*, 4th ed. (2013), §7.4 "The
//! Hessenberg and real Schur forms", §7.5 "The practical QR algorithm",
//! §8.5 "Jacobi methods" (the subsection §8.5.2 cited in SJ2 not
//! verified); E. Anderson et al., *LAPACK Users' Guide*, 3rd ed., SIAM
//! (1999), doi:10.1137/1.9780898719604, routines `dlahqr`, `dlanv2`,
//! `dlaexc`, `dtrexc`, `dtrevc` (the routines recalled, not re-read);
//! Z. Bai & J. W. Demmel, "On swapping diagonal blocks in real Schur
//! form", Linear Algebra Appl. 186, 75–95 (1993),
//! doi:10.1016/0024-3795(93)90286-W; Demmel & Veselić, SIAM J. Matrix
//! Anal. Appl. 13(4), 1204–1245 (1992). Checked in task I03 (issue xingo-statla#4) against Crossref and the publishers' tables of contents, task I03.

use xingo_sparse::{DenseBlock, Real};

use crate::StatlaError;
use crate::real::{is_exact_zero, le_exact, lt};
use crate::small::{apply_householder, givens, householder};

/// The thresholds of Algorithms SJ and FQ.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SchurOptions<T> {
    /// Relative rotation threshold of SJ (default `ε_T`).
    pub(crate) jacobi_tol: T,
    /// Sweep limit of SJ (default 60).
    pub(crate) max_sweeps: usize,
    /// Iteration budget of FQ per row (default 30, as `dlahqr`).
    pub(crate) qr_iters_per_row: usize,
    /// Factor of `ε·μ` of the stability test SW3 (default 10, as `dlaexc`).
    pub(crate) swap_factor: T,
}

impl<T: Real> SchurOptions<T> {
    /// The defaults.
    pub(crate) fn new() -> Self {
        Self {
            jacobi_tol: T::EPSILON,
            max_sweeps: 60,
            qr_iters_per_row: 30,
            swap_factor: T::from_f64(10.0),
        }
    }
}

/// `2⁵⁰⁰`: the size above which EV2 rescales (a property of the exponent
/// range, not a tuning knob).
const BIG: f64 = 3.273_390_607_896_142e150;

/// `2^⌊log₂ μ⌋` for a positive normal `μ`, `1` otherwise.
fn pow2_floor(mu: f64) -> f64 {
    let bits = mu.to_bits() & 0x7ff0_0000_0000_0000;
    if bits == 0 || bits == 0x7ff0_0000_0000_0000 {
        1.0
    } else {
        f64::from_bits(bits)
    }
}

/// `sign(1, x)` of Fortran: `−1` for negative `x`, `+1` otherwise.
#[inline]
fn sign1<T: Real>(x: T) -> T {
    if lt(x, T::ZERO) { -T::ONE } else { T::ONE }
}

/// `√(x² + y²)` without overflow.
#[inline]
pub(crate) fn hypot<T: Real>(x: T, y: T) -> T {
    let (ax, ay) = (x.abs(), y.abs());
    let m = ax.max(ay);
    if is_exact_zero(m) {
        return T::ZERO;
    }
    let (a, b) = (ax / m, ay / m);
    m * (a * a + b * b).sqrt()
}

/// Rotate columns `i`, `j` of a column-major array with `rows` rows over the
/// rows `r0..r1`: `(x, y) ← (c x + s y, c y − s x)` (LAPACK `drot`).
fn rot_cols<T: Real>(
    data: &mut [T],
    rows: usize,
    i: usize,
    j: usize,
    r: (usize, usize),
    c: T,
    s: T,
) {
    for k in r.0..r.1 {
        let (x, y) = (data[k + i * rows], data[k + j * rows]);
        data[k + i * rows] = c * x + s * y;
        data[k + j * rows] = c * y - s * x;
    }
}

/// Rotate rows `i`, `j` over the columns `c0..c1` (as [`rot_cols`]).
fn rot_rows<T: Real>(
    data: &mut [T],
    rows: usize,
    i: usize,
    j: usize,
    cr: (usize, usize),
    c: T,
    s: T,
) {
    for k in cr.0..cr.1 {
        let (x, y) = (data[i + k * rows], data[j + k * rows]);
        data[i + k * rows] = c * x + s * y;
        data[j + k * rows] = c * y - s * x;
    }
}

/// The identity of order `p`.
pub(crate) fn identity<T: Real>(p: usize) -> DenseBlock<T> {
    let mut z = DenseBlock::zeros(p, p);
    for i in 0..p {
        z.data[i + i * p] = T::ONE;
    }
    z
}

/// The largest absolute entry.
fn max_abs<T: Real>(data: &[T]) -> T {
    data.iter().fold(T::ZERO, |m, &v| m.max(v.abs()))
}

/// Algorithm SJ: eigenvalues `d` (unsorted) and orthogonal `Z` with `A ≈ Z
/// diag(d) Zᵀ`, for symmetric `a` (its upper and lower triangles are
/// averaged).
pub(crate) fn sym_eig<T: Real>(
    a: &DenseBlock<T>,
    o: &SchurOptions<T>,
) -> Result<(Vec<T>, DenseBlock<T>), StatlaError> {
    let p = a.rows;
    debug_assert_eq!(p, a.cols);
    let mut z = identity::<T>(p);
    // SJ1.
    let mu = max_abs(&a.data);
    if is_exact_zero(mu) {
        return Ok((vec![T::ZERO; p], z));
    }
    let scale = pow2_floor(mu.to_f64());
    let inv = T::from_f64(1.0 / scale);
    let half = T::from_f64(0.5);
    let mut w = DenseBlock::<T>::zeros(p, p);
    for j in 0..p {
        for i in 0..p {
            w.data[i + j * p] = (a.data[i + j * p] + a.data[j + i * p]) * half * inv;
        }
    }
    let (one, two) = (T::ONE, T::from_f64(2.0));
    let tol = o.jacobi_tol;
    let floor = tol * tol;
    // SJ2.
    let mut sweeps = 0;
    loop {
        if sweeps == o.max_sweeps {
            return Err(StatlaError::NoConvergence {
                requested: p,
                converged: 0,
                restarts: sweeps,
            });
        }
        sweeps += 1;
        let mut rotated = false;
        for pp in 0..p {
            for q in pp + 1..p {
                let apq = w.data[pp + q * p];
                let (app, aqq) = (w.data[pp + pp * p], w.data[q + q * p]);
                let rel = tol * app.abs().sqrt() * aqq.abs().sqrt();
                if !lt(rel, apq.abs()) || !lt(floor, apq.abs()) {
                    continue;
                }
                rotated = true;
                let theta = (aqq - app) / (two * apq);
                let at = theta.abs();
                let mag = if le_exact(one, at) {
                    let r = one / theta;
                    at * (one + (one + r * r).sqrt())
                } else {
                    at + (one + theta * theta).sqrt()
                };
                let t = if lt(theta, T::ZERO) {
                    -(one / mag)
                } else {
                    one / mag
                };
                let c = one / (one + t * t).sqrt();
                let s = t * c;
                // W ← W J (columns), then Jᵀ W (rows); col_p ← c col_p − s col_q.
                rot_cols(&mut w.data, p, pp, q, (0, p), c, -s);
                rot_rows(&mut w.data, p, pp, q, (0, p), c, -s);
                rot_cols(&mut z.data, p, pp, q, (0, p), c, -s);
                w.data[pp + pp * p] = app - t * apq;
                w.data[q + q * p] = aqq + t * apq;
                w.data[pp + q * p] = T::ZERO;
                w.data[q + pp * p] = T::ZERO;
            }
        }
        if !rotated {
            break;
        }
    }
    // SJ3.
    let sc = T::from_f64(scale);
    let d: Vec<T> = (0..p).map(|i| w.data[i + i * p] * sc).collect();
    for j in 0..p {
        let col = &mut z.data[j * p..(j + 1) * p];
        let nz = crate::small::norm2(col);
        for v in col.iter_mut() {
            *v = *v / nz;
        }
    }
    Ok((d, z))
}

/// Algorithm HR in place on `h`, accumulating into `z` (`z ← z P`).
pub(crate) fn hessenberg<T: Real>(h: &mut DenseBlock<T>, z: &mut DenseBlock<T>) {
    let p = h.rows;
    let zr = z.rows;
    if p < 3 {
        return;
    }
    let mut row = vec![T::ZERO; p];
    for k in 0..p - 2 {
        let len = p - k - 1;
        let mut x: Vec<T> = h.data[(k + 1) + k * p..p + k * p].to_vec();
        let tau = householder(&mut x);
        h.data[(k + 1) + k * p] = x[0];
        for i in k + 2..p {
            h.data[i + k * p] = T::ZERO;
        }
        if is_exact_zero(tau) {
            continue;
        }
        let v = &x[1..];
        // Left: rows k+1.., columns k+1...
        for j in k + 1..p {
            apply_householder(tau, v, &mut h.data[(k + 1) + j * p..p + j * p]);
        }
        // Right: all rows, columns k+1...
        for i in 0..p {
            for (t, r) in row[..len].iter_mut().enumerate() {
                *r = h.data[i + (k + 1 + t) * p];
            }
            apply_householder(tau, v, &mut row[..len]);
            for (t, &r) in row[..len].iter().enumerate() {
                h.data[i + (k + 1 + t) * p] = r;
            }
        }
        let mut zrow = vec![T::ZERO; len];
        for i in 0..zr {
            for (t, r) in zrow.iter_mut().enumerate() {
                *r = z.data[i + (k + 1 + t) * zr];
            }
            apply_householder(tau, v, &mut zrow);
            for (t, &r) in zrow.iter().enumerate() {
                z.data[i + (k + 1 + t) * zr] = r;
            }
        }
    }
}

/// Algorithm SV on `(a, b, c, d)`: returns the standardized block and
/// `(cs, sn)` with `[a b; c d] = R [a′ b′; c′ d′] Rᵀ`, `R = [cs −sn; sn cs]`.
#[allow(clippy::many_single_char_names)]
pub(crate) fn lanv2<T: Real>(a: T, b: T, c: T, d: T) -> ([T; 4], T, T) {
    let (mut a, mut b, mut c, mut d) = (a, b, c, d);
    let (zero, one, half) = (T::ZERO, T::ONE, T::from_f64(0.5));
    let four_eps = T::from_f64(4.0) * T::EPSILON;
    let (mut cs, mut sn);
    if is_exact_zero(c) {
        cs = one;
        sn = zero;
    } else if is_exact_zero(b) {
        cs = zero;
        sn = one;
        core::mem::swap(&mut a, &mut d);
        b = -c;
        c = zero;
    } else if is_exact_zero(a - d) && !is_exact_zero(sign1(b) - sign1(c)) {
        cs = one;
        sn = zero;
    } else {
        let temp = a - d;
        let p = half * temp;
        let bcmax = b.abs().max(c.abs());
        let bcmis = b.abs().min(c.abs()) * sign1(b) * sign1(c);
        let scale = p.abs().max(bcmax);
        let z = (p / scale) * p + (bcmax / scale) * bcmis;
        if le_exact(four_eps, z) {
            // Real eigenvalues.
            let root = scale.sqrt() * z.sqrt();
            let z = p + if lt(p, zero) { -root } else { root };
            a = d + z;
            d -= (bcmax / z) * bcmis;
            let tau = hypot(c, z);
            cs = z / tau;
            sn = c / tau;
            b -= c;
            c = zero;
        } else {
            // Complex or nearly equal real eigenvalues: equalize the diagonal.
            let sigma = b + c;
            let tau = hypot(sigma, temp);
            cs = (half * (one + sigma.abs() / tau)).sqrt();
            sn = -(p / (tau * cs)) * sign1(sigma);
            let aa = a * cs + b * sn;
            let bb = -a * sn + b * cs;
            let cc = c * cs + d * sn;
            let dd = -c * sn + d * cs;
            a = aa * cs + cc * sn;
            b = bb * cs + dd * sn;
            c = -aa * sn + cc * cs;
            d = -bb * sn + dd * cs;
            let temp = half * (a + d);
            a = temp;
            d = temp;
            if !is_exact_zero(c) {
                if is_exact_zero(b) {
                    b = -c;
                    c = zero;
                    let t = cs;
                    cs = -sn;
                    sn = t;
                } else if is_exact_zero(sign1(b) - sign1(c)) {
                    // Real eigenvalues: reduce to upper triangular.
                    let sab = b.abs().sqrt();
                    let sac = c.abs().sqrt();
                    let pp = if lt(c, zero) { -(sab * sac) } else { sab * sac };
                    let tau = one / (b + c).abs().sqrt();
                    a = temp + pp;
                    d = temp - pp;
                    b -= c;
                    c = zero;
                    let cs1 = sab * tau;
                    let sn1 = sac * tau;
                    let t = cs * cs1 - sn * sn1;
                    sn = cs * sn1 + sn * cs1;
                    cs = t;
                }
            }
        }
    }
    ([a, b, c, d], cs, sn)
}

/// Standardize the `2 × 2` block at `k` of `t` (Algorithm SV) and apply its
/// rotation to the rest of `t` and to `z`.
fn standardize<T: Real>(t: &mut DenseBlock<T>, z: &mut DenseBlock<T>, k: usize) {
    let p = t.rows;
    let at = |i: usize, j: usize| i + j * p;
    let (blk, cs, sn) = lanv2(
        t.data[at(k, k)],
        t.data[at(k, k + 1)],
        t.data[at(k + 1, k)],
        t.data[at(k + 1, k + 1)],
    );
    t.data[at(k, k)] = blk[0];
    t.data[at(k, k + 1)] = blk[1];
    t.data[at(k + 1, k)] = blk[2];
    t.data[at(k + 1, k + 1)] = blk[3];
    rot_rows(&mut t.data, p, k, k + 1, (k + 2, p), cs, sn);
    rot_cols(&mut t.data, p, k, k + 1, (0, k), cs, sn);
    let zr = z.rows;
    rot_cols(&mut z.data, zr, k, k + 1, (0, zr), cs, sn);
}

/// The reflector of a 2- or 3-vector: `(τ, v_tail, β)` with `(I − τvvᵀ)x =
/// β e_0`.
fn small_reflector<T: Real>(x: &[T]) -> (T, [T; 2], T) {
    let mut w = [T::ZERO; 3];
    w[..x.len()].copy_from_slice(x);
    let tau = householder(&mut w[..x.len()]);
    let mut v = [T::ZERO; 2];
    v[..x.len() - 1].copy_from_slice(&w[1..x.len()]);
    (tau, v, w[0])
}

/// Apply a small reflector to rows `k..k+len` over the columns `c0..c1`.
fn reflect_rows<T: Real>(
    t: &mut DenseBlock<T>,
    k: usize,
    len: usize,
    tau: T,
    v: &[T],
    cr: (usize, usize),
) {
    let p = t.rows;
    let mut y = [T::ZERO; 3];
    for j in cr.0..cr.1 {
        for r in 0..len {
            y[r] = t.data[k + r + j * p];
        }
        apply_householder(tau, &v[..len - 1], &mut y[..len]);
        for r in 0..len {
            t.data[k + r + j * p] = y[r];
        }
    }
}

/// Apply a small reflector to columns `k..k+len` over the rows `r0..r1`.
fn reflect_cols<T: Real>(
    t: &mut DenseBlock<T>,
    k: usize,
    len: usize,
    tau: T,
    v: &[T],
    rr: (usize, usize),
) {
    let p = t.rows;
    let mut y = [T::ZERO; 3];
    for i in rr.0..rr.1 {
        for r in 0..len {
            y[r] = t.data[i + (k + r) * p];
        }
        apply_householder(tau, &v[..len - 1], &mut y[..len]);
        for r in 0..len {
            t.data[i + (k + r) * p] = y[r];
        }
    }
}

#[allow(clippy::too_many_lines)] // steps FQ1–FQ5 in order, as in the chapter
/// Algorithms HR + FQ: `h` becomes the standardized real Schur form `T`,
/// `z ← z Q` with `h_in = Q T Qᵀ`.
pub(crate) fn real_schur<T: Real>(
    h: &mut DenseBlock<T>,
    z: &mut DenseBlock<T>,
    o: &SchurOptions<T>,
) -> Result<(), StatlaError> {
    let p = h.rows;
    hessenberg(h, z);
    if p == 0 {
        return Ok(());
    }
    let at = |i: usize, j: usize| i + j * p;
    let mu = max_abs(&h.data);
    let eps = T::EPSILON;
    let budget = o.qr_iters_per_row * p.max(10);
    let mut total = 0;
    let mut its = 0;
    let mut hi = p; // the window is [l, hi − 1]
    while hi > 0 {
        let i = hi - 1;
        // FQ1.
        let mut l = 0;
        for k in (1..=i).rev() {
            let s = h.data[at(k - 1, k - 1)].abs() + h.data[at(k, k)].abs();
            let s = if is_exact_zero(s) { mu } else { s };
            if le_exact(h.data[at(k, k - 1)].abs(), eps * s) {
                h.data[at(k, k - 1)] = T::ZERO;
                l = k;
                break;
            }
        }
        if l == i {
            // FQ2.
            hi -= 1;
            its = 0;
            continue;
        }
        if l + 1 == i {
            // FQ3.
            standardize(h, z, i - 1);
            hi -= 2;
            its = 0;
            continue;
        }
        // FQ4.
        total += 1;
        its += 1;
        if total > budget {
            return Err(StatlaError::NoConvergence {
                requested: p,
                converged: p - hi,
                restarts: total,
            });
        }
        let (h11, h12, h21, h22) = if its == 10 || its == 20 {
            let w = if its == 10 {
                h.data[at(l + 1, l)].abs() + h.data[at(l + 2, l + 1)].abs()
            } else {
                h.data[at(i, i - 1)].abs() + h.data[at(i - 1, i - 2)].abs()
            };
            let base = if its == 10 {
                h.data[at(l, l)]
            } else {
                h.data[at(i, i)]
            };
            let h11 = T::from_f64(0.75) * w + base;
            (h11, T::from_f64(-0.4375) * w, w, h11)
        } else {
            (
                h.data[at(i - 1, i - 1)],
                h.data[at(i - 1, i)],
                h.data[at(i, i - 1)],
                h.data[at(i, i)],
            )
        };
        let s = h11 + h22;
        let t = h11 * h22 - h12 * h21;
        // FQ5.
        let hll = h.data[at(l, l)];
        let h10 = h.data[at(l + 1, l)];
        let mut x = hll * hll + h.data[at(l, l + 1)] * h10 - s * hll + t;
        let mut y = h10 * (hll + h.data[at(l + 1, l + 1)] - s);
        let mut zz = h10 * h.data[at(l + 2, l + 1)];
        let sc = x.abs() + y.abs() + zz.abs();
        if !is_exact_zero(sc) {
            x = x / sc;
            y = y / sc;
            zz = zz / sc;
        }
        for k in l..i {
            let len = if k + 2 <= i { 3 } else { 2 };
            let (tau, v, beta) = if len == 3 {
                small_reflector(&[x, y, zz])
            } else {
                small_reflector(&[x, y])
            };
            if k > l {
                h.data[at(k, k - 1)] = beta;
                h.data[at(k + 1, k - 1)] = T::ZERO;
                if len == 3 {
                    h.data[at(k + 2, k - 1)] = T::ZERO;
                }
            }
            // Column k − 1 already holds the result (β, 0, 0) for k > l.
            let from = if k > l { k } else { l };
            reflect_rows(h, k, len, tau, &v, (from, p));
            reflect_cols(h, k, len, tau, &v, (0, (k + len + 1).min(i + 1)));
            let zr = z.rows;
            let mut yv = [T::ZERO; 3];
            for r in 0..zr {
                for q in 0..len {
                    yv[q] = z.data[r + (k + q) * zr];
                }
                apply_householder(tau, &v[..len - 1], &mut yv[..len]);
                for q in 0..len {
                    z.data[r + (k + q) * zr] = yv[q];
                }
            }
            if k + 1 < i {
                x = h.data[at(k + 1, k)];
                y = h.data[at(k + 2, k)];
                if k + 3 <= i {
                    zz = h.data[at(k + 3, k)];
                }
            }
        }
    }
    debug_assert!(check_quasi_triangular(h), "(E2) real Schur form");
    Ok(())
}

/// The diagonal blocks `(start, size)` of a quasi-triangular `t`.
pub(crate) fn blocks<T: Real>(t: &DenseBlock<T>) -> Vec<(usize, usize)> {
    let p = t.rows;
    let mut out = Vec::new();
    let mut i = 0;
    while i < p {
        if i + 1 < p && !is_exact_zero(t.data[(i + 1) + i * p]) {
            out.push((i, 2));
            i += 2;
        } else {
            out.push((i, 1));
            i += 1;
        }
    }
    out
}

/// The eigenvalue `(re, im)` of a block, `im ≥ 0`.
pub(crate) fn block_value<T: Real>(t: &DenseBlock<T>, (q, s): (usize, usize)) -> (T, T) {
    let p = t.rows;
    if s == 1 {
        (t.data[q + q * p], T::ZERO)
    } else {
        let b = t.data[q + (q + 1) * p];
        let c = t.data[(q + 1) + q * p];
        (t.data[q + q * p], b.abs().sqrt() * c.abs().sqrt())
    }
}

/// (E2)'s shape: exact zeros below the subdiagonal, no two consecutive
/// nonzero subdiagonal entries, and every `2 × 2` block standardized.
pub(crate) fn check_quasi_triangular<T: Real>(t: &DenseBlock<T>) -> bool {
    let p = t.rows;
    for j in 0..p {
        for i in j + 2..p {
            if !is_exact_zero(t.data[i + j * p]) {
                return false;
            }
        }
    }
    for (q, s) in blocks(t) {
        if s == 2 {
            let (a, b, c, d) = (
                t.data[q + q * p],
                t.data[q + (q + 1) * p],
                t.data[(q + 1) + q * p],
                t.data[(q + 1) + (q + 1) * p],
            );
            if !is_exact_zero(a - d) || !lt(b * c, T::ZERO) {
                return false;
            }
        }
    }
    true
}

/// Gaussian elimination with complete pivoting on an `n × n` system (`n ≤
/// 4`), pivots below `smin` replaced by `smin`.
fn solve_small<T: Real>(m: &mut [[T; 4]; 4], rhs: &mut [T; 4], n: usize, smin: T) -> [T; 4] {
    let mut perm_c = [0, 1, 2, 3];
    for k in 0..n {
        // Pivot search.
        let (mut pi, mut pj, mut best) = (k, k, T::ZERO);
        for i in k..n {
            for j in k..n {
                if lt(best, m[i][j].abs()) {
                    best = m[i][j].abs();
                    pi = i;
                    pj = j;
                }
            }
        }
        m.swap(k, pi);
        rhs.swap(k, pi);
        for row in m.iter_mut() {
            row.swap(k, pj);
        }
        perm_c.swap(k, pj);
        if lt(m[k][k].abs(), smin) {
            m[k][k] = smin;
        }
        for i in k + 1..n {
            let f = m[i][k] / m[k][k];
            for j in k..n {
                let v = m[k][j];
                m[i][j] -= f * v;
            }
            let r = rhs[k];
            rhs[i] -= f * r;
        }
    }
    let mut y = [T::ZERO; 4];
    for k in (0..n).rev() {
        let mut s = rhs[k];
        for j in k + 1..n {
            s -= m[k][j] * y[j];
        }
        y[k] = s / m[k][k];
    }
    let mut x = [T::ZERO; 4];
    for k in 0..n {
        x[perm_c[k]] = y[k];
    }
    x
}

#[allow(clippy::too_many_lines)] // steps SW1–SW4 in order, as in the chapter
/// Algorithm SW: swap the adjacent blocks at `j` of sizes `n1`, `n2`.
/// Returns `false` (nothing modified) when SW3 refuses.
pub(crate) fn swap_blocks<T: Real>(
    t: &mut DenseBlock<T>,
    z: &mut DenseBlock<T>,
    j: usize,
    n1: usize,
    n2: usize,
    o: &SchurOptions<T>,
) -> bool {
    let p = t.rows;
    let zr = z.rows;
    let at = |i: usize, c: usize| i + c * p;
    let nd = n1 + n2;
    if n1 == 1 && n2 == 1 {
        let (t11, t22) = (t.data[at(j, j)], t.data[at(j + 1, j + 1)]);
        let (g, _) = givens(t.data[at(j, j + 1)], t22 - t11);
        rot_rows(&mut t.data, p, j, j + 1, (j + 2, p), g.c, g.s);
        rot_cols(&mut t.data, p, j, j + 1, (0, j), g.c, g.s);
        t.data[at(j, j)] = t22;
        t.data[at(j + 1, j + 1)] = t11;
        rot_cols(&mut z.data, zr, j, j + 1, (0, zr), g.c, g.s);
        return true;
    }
    // SW1.
    let mut d = [[T::ZERO; 4]; 4];
    let mut mu = T::ZERO;
    for (r, row) in d.iter_mut().enumerate().take(nd) {
        for (c, v) in row.iter_mut().enumerate().take(nd) {
            *v = t.data[at(j + r, j + c)];
            mu = mu.max(v.abs());
        }
    }
    let eps = T::EPSILON;
    let smin = (eps * mu).max(T::from_f64(f64::MIN_POSITIVE));
    // vec(X) column-major (x_rc at r + c·n1): (I ⊗ T11 − T22ᵀ ⊗ I) vec X = vec T12.
    let nk = n1 * n2;
    let mut kmat = [[T::ZERO; 4]; 4];
    let mut rhs = [T::ZERO; 4];
    for c in 0..n2 {
        for r in 0..n1 {
            let row = r + c * n1;
            rhs[row] = d[r][n1 + c];
            for c2 in 0..n2 {
                for r2 in 0..n1 {
                    let col = r2 + c2 * n1;
                    let mut v = T::ZERO;
                    if c == c2 {
                        v += d[r][r2];
                    }
                    if r == r2 {
                        v -= d[n1 + c2][n1 + c];
                    }
                    kmat[row][col] = v;
                }
            }
        }
    }
    let xv = solve_small(&mut kmat, &mut rhs, nk, smin);
    // SW2: M = [−X; I] (nd × n2), Householder QR.
    let mut m = [[T::ZERO; 2]; 4];
    for c in 0..n2 {
        for r in 0..n1 {
            m[r][c] = -xv[r + c * n1];
        }
        m[n1 + c][c] = T::ONE;
    }
    let mut refl: Vec<(usize, T, [T; 3])> = Vec::with_capacity(n2);
    for c in 0..n2 {
        let len = nd - c;
        let mut x = [T::ZERO; 4];
        for r in 0..len {
            x[r] = m[c + r][c];
        }
        let tau = householder(&mut x[..len]);
        let mut v = [T::ZERO; 3];
        v[..len - 1].copy_from_slice(&x[1..len]);
        m[c][c] = x[0];
        for r in 1..len {
            m[c + r][c] = T::ZERO;
        }
        for c2 in c + 1..n2 {
            let mut y = [T::ZERO; 4];
            for r in 0..len {
                y[r] = m[c + r][c2];
            }
            apply_householder(tau, &v[..len - 1], &mut y[..len]);
            for r in 0..len {
                m[c + r][c2] = y[r];
            }
        }
        refl.push((c, tau, v));
    }
    // Apply Qᵀ · Q to a block given as rows/cols accessors.
    let apply_left = |data: &mut [T], rows: usize, cols: (usize, usize)| {
        for &(c, tau, v) in &refl {
            let len = nd - c;
            let mut y = [T::ZERO; 4];
            for col in cols.0..cols.1 {
                for r in 0..len {
                    y[r] = data[(j + c + r) + col * rows];
                }
                apply_householder(tau, &v[..len - 1], &mut y[..len]);
                for r in 0..len {
                    data[(j + c + r) + col * rows] = y[r];
                }
            }
        }
    };
    let apply_right = |data: &mut [T], rows: usize, rr: (usize, usize)| {
        for &(c, tau, v) in &refl {
            let len = nd - c;
            let mut y = [T::ZERO; 4];
            for row in rr.0..rr.1 {
                for r in 0..len {
                    y[r] = data[row + (j + c + r) * rows];
                }
                apply_householder(tau, &v[..len - 1], &mut y[..len]);
                for r in 0..len {
                    data[row + (j + c + r) * rows] = y[r];
                }
            }
        }
    };
    // SW3 on a copy of the block (embedded at offset j in a p-row array is
    // not needed: copy the nd × nd block into a small array with j = 0).
    {
        let mut dd = vec![T::ZERO; p * p];
        for r in 0..nd {
            for c in 0..nd {
                dd[(j + r) + (j + c) * p] = d[r][c];
            }
        }
        apply_left(&mut dd, p, (j, j + nd));
        apply_right(&mut dd, p, (j, j + nd));
        let mut e = T::ZERO;
        for r in n2..nd {
            for c in 0..n2 {
                e = e.max(dd[(j + r) + (j + c) * p].abs());
            }
        }
        if lt(
            (o.swap_factor * eps * mu).max(T::from_f64(f64::MIN_POSITIVE)),
            e,
        ) {
            return false;
        }
    }
    // SW4.
    apply_left(&mut t.data, p, (j, p));
    apply_right(&mut t.data, p, (0, (j + nd).min(p)));
    apply_right(&mut z.data, zr, (0, zr));
    for r in n2..nd {
        for c in 0..n2 {
            t.data[at(j + r, j + c)] = T::ZERO;
        }
    }
    if n2 == 2 {
        standardize(t, z, j);
    }
    if n1 == 2 {
        standardize(t, z, j + n2);
    }
    // Entries of the new leading block below its own subdiagonal stay zero.
    for c in j..j + nd {
        for r in c + 2..j + nd {
            t.data[at(r, c)] = T::ZERO;
        }
    }
    debug_assert!(check_quasi_triangular(t), "(E2) after a swap");
    true
}

/// Algorithm RO: reorder `t` (and `z`) so that the blocks of least rank
/// lead, until at least `need` rows are settled. `rank[b]` is the rank of
/// block `b` of [`blocks`]. Returns the final blocks with their ranks and
/// whether every swap was accepted.
pub(crate) fn reorder<T: Real>(
    t: &mut DenseBlock<T>,
    z: &mut DenseBlock<T>,
    rank: &[usize],
    need: usize,
    o: &SchurOptions<T>,
) -> (Vec<(usize, usize, usize)>, bool) {
    let mut bl: Vec<(usize, usize, usize)> = blocks(t)
        .iter()
        .zip(rank)
        .map(|(&(q, s), &r)| (q, s, r))
        .collect();
    let mut pos_b = 0;
    loop {
        if pos_b >= bl.len() || bl[pos_b].0 >= need {
            return (bl, true);
        }
        // The least rank at or after pos_b (first on ties).
        let mut best = pos_b;
        for b in pos_b..bl.len() {
            if bl[b].2 < bl[best].2 {
                best = b;
            }
        }
        let mut cur = best;
        while cur > pos_b {
            let (q1, s1, r1) = bl[cur - 1];
            let (_, s2, r2) = bl[cur];
            if !swap_blocks(t, z, q1, s1, s2, o) {
                return (rebuild(t, &bl), false);
            }
            // New layout: block of rank r2 at q1 (size s2, or split), then r1.
            let mut fresh = Vec::new();
            let p = t.rows;
            let push = |q: usize, s: usize, r: usize, fresh: &mut Vec<(usize, usize, usize)>| {
                if s == 2 && is_exact_zero(t.data[(q + 1) + q * p]) {
                    fresh.push((q, 1, r));
                    fresh.push((q + 1, 1, r));
                } else {
                    fresh.push((q, s, r));
                }
            };
            push(q1, s2, r2, &mut fresh);
            push(q1 + s2, s1, r1, &mut fresh);
            bl.splice(cur - 1..=cur, fresh);
            // Continue moving the block of rank r2, now at index cur − 1 (its
            // first half if it split: both halves keep rank r2).
            cur -= 1;
        }
        pos_b += 1;
    }
}

/// The blocks of `t` with the ranks of `old` carried over by start row.
fn rebuild<T: Real>(
    t: &DenseBlock<T>,
    old: &[(usize, usize, usize)],
) -> Vec<(usize, usize, usize)> {
    blocks(t)
        .into_iter()
        .map(|(q, s)| {
            let r = old
                .iter()
                .find(|b| b.0 <= q && q < b.0 + b.1)
                .map_or(usize::MAX, |b| b.2);
            (q, s, r)
        })
        .collect()
}

/// A complex value for Algorithm EV (no public arithmetic: internal).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Cx<T> {
    pub(crate) re: T,
    pub(crate) im: T,
}

impl<T: Real> Cx<T> {
    pub(crate) fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
    fn zero() -> Self {
        Self::new(T::ZERO, T::ZERO)
    }
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
    pub(crate) fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    fn scale(self, s: T) -> Self {
        Self::new(self.re * s, self.im * s)
    }
    /// `|z|` (as the larger component, for pivoting) .
    fn mag1(self) -> T {
        self.re.abs().max(self.im.abs())
    }
    /// Smith's division.
    pub(crate) fn div(self, o: Self) -> Self {
        if le_exact(o.im.abs(), o.re.abs()) {
            let r = o.im / o.re;
            let den = o.re + o.im * r;
            Self::new((self.re + self.im * r) / den, (self.im - self.re * r) / den)
        } else {
            let r = o.re / o.im;
            let den = o.re * r + o.im;
            Self::new((self.re * r + self.im) / den, (self.im * r - self.re) / den)
        }
    }
}

/// Algorithm EV: the eigenvector of the quasi-triangular `t` for the block
/// `(q, s)` (for `s = 2`, the eigenvalue with positive imaginary part), as
/// `p` complex entries (zero below the block).
pub(crate) fn schur_eigvec<T: Real>(
    t: &DenseBlock<T>,
    bl: &[(usize, usize)],
    bi: usize,
) -> Vec<Cx<T>> {
    let p = t.rows;
    let at = |i: usize, j: usize| t.data[i + j * p];
    let (q, s) = bl[bi];
    let mut x = vec![Cx::<T>::zero(); p];
    let mu = max_abs(&t.data);
    let smin = (T::EPSILON * mu).max(T::from_f64(f64::MIN_POSITIVE));
    // EV1.
    let lam = if s == 1 {
        x[q] = Cx::new(T::ONE, T::ZERO);
        Cx::new(at(q, q), T::ZERO)
    } else {
        let (a, b, c) = (at(q, q), at(q, q + 1), at(q + 1, q));
        let w = b.abs().sqrt() * c.abs().sqrt();
        if le_exact(c.abs(), b.abs()) {
            x[q] = Cx::new(T::ONE, T::ZERO);
            x[q + 1] = Cx::new(T::ZERO, w / b);
        } else {
            x[q] = Cx::new(T::ZERO, w / c);
            x[q + 1] = Cx::new(T::ONE, T::ZERO);
        }
        Cx::new(a, w)
    };
    let end = q + s;
    let big = T::from_f64(BIG);
    // EV2.
    for b in (0..bi).rev() {
        let (r0, rs) = bl[b];
        let mut rhs = [Cx::<T>::zero(); 2];
        for (k, rv) in rhs.iter_mut().enumerate().take(rs) {
            let mut acc = Cx::zero();
            for j in r0 + rs..end {
                acc = acc.add(x[j].scale(at(r0 + k, j)));
            }
            *rv = Cx::new(-acc.re, -acc.im);
        }
        if rs == 1 {
            let mut d = Cx::new(at(r0, r0), T::ZERO).sub(lam);
            if lt(d.mag1(), smin) {
                d = Cx::new(smin, T::ZERO);
            }
            x[r0] = rhs[0].div(d);
        } else {
            // [m11 m12; m21 m22] y = rhs, partial pivoting on the first column.
            let mut m = [
                [
                    Cx::new(at(r0, r0), T::ZERO).sub(lam),
                    Cx::new(at(r0, r0 + 1), T::ZERO),
                ],
                [
                    Cx::new(at(r0 + 1, r0), T::ZERO),
                    Cx::new(at(r0 + 1, r0 + 1), T::ZERO).sub(lam),
                ],
            ];
            let mut r = rhs;
            if lt(m[0][0].mag1(), m[1][0].mag1()) {
                m.swap(0, 1);
                r.swap(0, 1);
            }
            if lt(m[0][0].mag1(), smin) {
                m[0][0] = Cx::new(smin, T::ZERO);
            }
            let f = m[1][0].div(m[0][0]);
            let mut m22 = m[1][1].sub(f.mul(m[0][1]));
            let r1 = r[1].sub(f.mul(r[0]));
            if lt(m22.mag1(), smin) {
                m22 = Cx::new(smin, T::ZERO);
            }
            let y1 = r1.div(m22);
            let y0 = r[0].sub(m[0][1].mul(y1)).div(m[0][0]);
            x[r0] = y0;
            x[r0 + 1] = y1;
        }
        // Rescale when growing too large.
        let mut xm = T::ZERO;
        for v in &x[..end] {
            xm = xm.max(v.mag1());
        }
        if lt(big, xm) {
            let f = T::from_f64(1.0 / pow2_floor(xm.to_f64()));
            for v in &mut x[..end] {
                *v = v.scale(f);
            }
        }
    }
    x
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use crate::random::Rng;
    use xingo_extprec::DoubleDouble as Dd;

    fn rand_mat<T: Real>(p: usize, seed: u64, sym: bool) -> DenseBlock<T> {
        let mut rng = Rng::new(seed);
        let mut a = DenseBlock::<T>::zeros(p, p);
        for v in &mut a.data {
            *v = T::from_f64(rng.gaussian());
        }
        if sym {
            for j in 0..p {
                for i in 0..j {
                    a.data[i + j * p] = a.data[j + i * p];
                }
            }
        }
        a
    }

    fn mul<T: Real>(a: &DenseBlock<T>, b: &DenseBlock<T>) -> DenseBlock<T> {
        let (m, k, n) = (a.rows, a.cols, b.cols);
        let mut c = DenseBlock::zeros(m, n);
        for j in 0..n {
            for l in 0..k {
                let f = b.data[l + j * k];
                for i in 0..m {
                    c.data[i + j * m] += a.data[i + l * m] * f;
                }
            }
        }
        c
    }

    fn transpose<T: Real>(a: &DenseBlock<T>) -> DenseBlock<T> {
        let mut t = DenseBlock::zeros(a.cols, a.rows);
        for j in 0..a.cols {
            for i in 0..a.rows {
                t.data[j + i * a.cols] = a.data[i + j * a.rows];
            }
        }
        t
    }

    fn max_diff<T: Real>(a: &DenseBlock<T>, b: &DenseBlock<T>) -> f64 {
        a.data
            .iter()
            .zip(&b.data)
            .map(|(&x, &y)| (x - y).abs().to_f64())
            .fold(0.0, f64::max)
    }

    fn orth_err<T: Real>(z: &DenseBlock<T>) -> f64 {
        max_diff(&mul(&transpose(z), z), &identity(z.cols))
    }

    /// (E1) in both scalars, on random symmetric and graded matrices.
    #[test]
    fn jacobi_diagonalizes_symmetric_matrices() {
        fn run<T: Real>(u: f64) -> f64 {
            let mut worst = 0.0f64;
            for (p, seed) in [(1, 1), (2, 2), (7, 3), (20, 4), (41, 5)] {
                let a = rand_mat::<T>(p, seed, true);
                let (d, z) = sym_eig(&a, &SchurOptions::new()).unwrap();
                let mut dz = z.clone();
                for j in 0..p {
                    for i in 0..p {
                        dz.data[i + j * p] = z.data[i + j * p] * d[j];
                    }
                }
                let r = max_diff(&mul(&a, &z), &dz) / max_abs(&a.data).to_f64();
                worst = worst
                    .max(r / (p as f64 * u))
                    .max(orth_err(&z) / (p as f64 * u));
            }
            worst
        }
        let r64 = run::<f64>(f64::EPSILON / 2.0);
        let rdd = run::<Dd>(Dd::EPSILON.to_f64() / 2.0);
        eprintln!("(E1) / (p u): f64 {r64:.2}, dd {rdd:.2}");
        assert!(r64 < 8.0 && rdd < 8.0, "{r64} {rdd}");
        // Zero and diagonal inputs.
        let (d, z) = sym_eig(&DenseBlock::<f64>::zeros(3, 3), &SchurOptions::new()).unwrap();
        assert_eq!(d, vec![0.0; 3]);
        assert_eq!(z.data, identity::<f64>(3).data);
    }

    /// SV: the relation `A = R A′ Rᵀ` and the standardized shape, on every branch.
    #[test]
    fn lanv2_standardizes_every_case() {
        let cases = [
            (1.0, 2.0, 0.0, 3.0),
            (1.0, 0.0, 2.0, 3.0),
            (1.0, 2.0, -3.0, 1.0),
            (1.0, 2.0, 3.0, 4.0),
            (4.0, -5.0, 1.0, 2.0),
            (0.0, 1.0, -1e-20, 0.0),
            (2.0, 1e-8, 1e-8, 2.0),
            (1.0, -1.0, 1.0, 1.0 + 1e-15),
            (3.0, 7.0, -0.5, -1.0),
        ];
        for (a, b, c, d) in cases {
            let ([a2, b2, c2, d2], cs, sn) = lanv2(a, b, c, d);
            assert!((cs * cs + sn * sn - 1.0).abs() < 4e-16);
            // R A′ Rᵀ with R = [cs −sn; sn cs].
            let r = [[cs, -sn], [sn, cs]];
            let ap = [[a2, b2], [c2, d2]];
            let mut out = [[0.0; 2]; 2];
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            out[i][j] += r[i][k] * ap[k][l] * r[j][l];
                        }
                    }
                }
            }
            let scale = a.abs().max(b.abs()).max(c.abs()).max(d.abs());
            let err = (out[0][0] - a).abs()
                + (out[0][1] - b).abs()
                + (out[1][0] - c).abs()
                + (out[1][1] - d).abs();
            assert!(err <= 8.0 * f64::EPSILON * scale, "{a} {b} {c} {d}: {err}");
            assert!(
                c2 == 0.0 || (a2 == d2 && b2 * c2 < 0.0),
                "{a2} {b2} {c2} {d2}"
            );
        }
    }

    fn schur_of<T: Real>(a: &DenseBlock<T>) -> (DenseBlock<T>, DenseBlock<T>) {
        let mut t = a.clone();
        let mut z = identity::<T>(a.rows);
        real_schur(&mut t, &mut z, &SchurOptions::new()).unwrap();
        (t, z)
    }

    /// (E2) on random, Hessenberg-structured and defective-ish matrices.
    #[test]
    fn francis_reaches_real_schur_form() {
        fn run<T: Real>(u: f64) -> f64 {
            let mut worst = 0.0f64;
            for (p, seed) in [(1, 1), (2, 2), (3, 3), (6, 4), (20, 5), (45, 6)] {
                let a = rand_mat::<T>(p, seed, false);
                let (t, z) = schur_of(&a);
                assert!(check_quasi_triangular(&t));
                let r = max_diff(&mul(&a, &z), &mul(&z, &t)) / max_abs(&a.data).to_f64();
                worst = worst
                    .max(r / (p as f64 * u))
                    .max(orth_err(&z) / (p as f64 * u));
            }
            worst
        }
        let r64 = run::<f64>(f64::EPSILON / 2.0);
        let rdd = run::<Dd>(Dd::EPSILON.to_f64() / 2.0);
        eprintln!("(E2) / (p u): f64 {r64:.2}, dd {rdd:.2}");
        assert!(r64 < 16.0 && rdd < 16.0, "{r64} {rdd}");
        // A rotation matrix: eigenvalues ±i, a single standardized block.
        let mut a = DenseBlock::<f64>::zeros(2, 2);
        a.data = vec![0.0, 1.0, -1.0, 0.0];
        let (t, _) = schur_of(&a);
        let (re, im) = block_value(&t, (0, 2));
        assert!(re.abs() < 1e-15 && (im - 1.0).abs() < 1e-15);
        // Jordan block (defective): converges.
        let mut j = DenseBlock::<f64>::zeros(4, 4);
        for i in 0..4 {
            j.data[i + i * 4] = 2.0;
            if i + 1 < 4 {
                j.data[i + (i + 1) * 4] = 1.0;
            }
        }
        let (t, _) = schur_of(&j);
        assert!(check_quasi_triangular(&t));
        // The companion matrix of x⁵ − 1 (eigenvalues on the unit circle;
        // Francis needs the exceptional shifts there).
        let mut c = DenseBlock::<f64>::zeros(5, 5);
        for i in 1..5 {
            c.data[i + (i - 1) * 5] = 1.0;
        }
        c.data[4 * 5] = 1.0;
        let (t, z) = schur_of(&c);
        assert!(check_quasi_triangular(&t));
        assert!(max_diff(&mul(&c, &z), &mul(&z, &t)) < 1e-14);
        let mut mods: Vec<f64> = blocks(&t)
            .iter()
            .map(|&b| {
                let (re, im) = block_value(&t, b);
                hypot(re, im)
            })
            .collect();
        mods.sort_by(f64::total_cmp);
        for m in mods {
            assert!((m - 1.0).abs() < 1e-13, "{m}");
        }
    }

    /// SW and RO: after reordering by a rank, the leading blocks have the
    /// least ranks, the relation holds and the eigenvalues are preserved.
    #[test]
    fn reordering_moves_wanted_blocks_first() {
        for (p, seed) in [(6, 11), (12, 12), (25, 13), (40, 14)] {
            let a = rand_mat::<f64>(p, seed, false);
            let (mut t, mut z) = schur_of(&a);
            let bl = blocks(&t);
            let vals: Vec<(f64, f64)> = bl.iter().map(|&b| block_value(&t, b)).collect();
            // Rank by increasing real part (SR).
            let mut order: Vec<usize> = (0..bl.len()).collect();
            order.sort_by(|&i, &j| vals[i].0.total_cmp(&vals[j].0));
            let mut rank = vec![0; bl.len()];
            for (r, &i) in order.iter().enumerate() {
                rank[i] = r;
            }
            let (nb, ok) = reorder(&mut t, &mut z, &rank, p, &SchurOptions::new());
            assert!(ok);
            assert!(check_quasi_triangular(&t));
            assert!(max_diff(&mul(&a, &z), &mul(&z, &t)) < 1e-13 * p as f64);
            assert!(orth_err(&z) < 1e-14 * p as f64);
            let res: Vec<f64> = blocks(&t).iter().map(|&b| block_value(&t, b).0).collect();
            for w in res.windows(2) {
                assert!(w[0] <= w[1] + 1e-12, "{res:?}");
            }
            for w in nb.windows(2) {
                assert!(w[0].2 <= w[1].2);
            }
        }
        // Diagonal T: swaps are exact permutations.
        let mut t = DenseBlock::<f64>::zeros(3, 3);
        t.data[0] = 3.0;
        t.data[4] = 1.0;
        t.data[8] = 2.0;
        let mut z = identity::<f64>(3);
        reorder(&mut t, &mut z, &[2, 0, 1], 3, &SchurOptions::new());
        assert_eq!((t.data[0], t.data[4], t.data[8]), (1.0, 2.0, 3.0));
        for v in &z.data {
            assert!(*v == 0.0 || v.abs() == 1.0);
        }
    }

    /// (E3): `T x = λ x` for every block, real and complex.
    #[test]
    fn eigenvectors_of_the_schur_form() {
        for (p, seed) in [(5, 21), (17, 22), (33, 23)] {
            let a = rand_mat::<f64>(p, seed, false);
            let (t, _) = schur_of(&a);
            let bl = blocks(&t);
            for bi in 0..bl.len() {
                let x = schur_eigvec(&t, &bl, bi);
                let (re, im) = block_value(&t, bl[bi]);
                let lam = Cx::new(re, im);
                let xn = x
                    .iter()
                    .map(|v| v.re * v.re + v.im * v.im)
                    .sum::<f64>()
                    .sqrt();
                let mut err = 0.0f64;
                for i in 0..p {
                    let mut acc = Cx::new(0.0, 0.0);
                    for j in 0..p {
                        acc = acc.add(x[j].scale(t.data[i + j * p]));
                    }
                    let r = acc.sub(lam.mul(x[i]));
                    err = err.max(r.mag1());
                }
                let tn = max_abs(&t.data);
                assert!(err <= 1e-14 * tn * xn * p as f64, "{p} {bi}: {err}");
            }
        }
    }
}
