//! # Chapter 12.4 — Randomized SVD (Halko, Martinsson & Tropp 2011)
//!
//! ## Purpose
//! *What are the leading `k` singular triplets of a matrix I can only
//! multiply by?* — when a stated *expected* error is enough. [`rsvd()`] spends
//! `2(q + 1)` passes over `A` with a block of `l = k + p` columns, plus
//! `O((m + n) l²)` dense work, and its error is governed by the tail of the
//! spectrum beyond `k` (bound below). [`range_finder`] returns the
//! orthonormal basis alone (sketching, CUR, randomized PCA by the caller).
//! For an answer accurate to `ε σ_1` on every value, [`crate::svds()`].
//!
//! ## Notation
//! `A ∈ T^{m×n}`, `σ_1 ≥ σ_2 ≥ …` its singular values; `k` the rank wanted,
//! `p` the oversampling ([`RsvdOptions::oversample`]), `q` the power
//! iterations ([`RsvdOptions::power_iters`]), `l = min(k + p, m, n)`; `Ω ∈
//! T^{n×l}` the test matrix of Algorithm G (Chapter 12.1); `orth(Y)` the
//! `Q` factor of Algorithm Q (Chapter 12.0.4: Householder, any rank).
//!
//! ## Algorithm S (randomized SVD; HMT §4–5)
//! S1. \[Sketch.\] `l ← min(k + p, m, n)`; draw `Ω` column by column from the
//!     seed's stream (G5), in the distribution [`RsvdOptions::sketch`].
//! S2. \[Range.\] `Y ← A Ω`; `Q ← orth(Y)` (`m × l`).
//! S3. \[Power.\] Repeat `q` times: `Z ← orth(Aᵀ Q)`; `Q ← orth(A Z)` — the
//!     re-orthonormalization at every half step that HMT recommend against
//!     the loss of small singular directions in rounding. (Task L3.3 could
//!     not exhibit that loss with `q ≤ 6` on spectra down to `10⁻¹²σ_1`:
//!     without the half step the results agreed to `ε σ_1`; the step costs
//!     one `n × l` QR per iteration and is kept.)
//! S4. \[Project.\] `Bᵀ ← Aᵀ Q` (`n × l`; `B = QᵀA`).
//! S5. \[Small SVD.\] `Bᵀ = Q_b R` (Algorithm Q); `R = X Σ Yᵀ` by the
//!     one-sided Jacobi SVD (Algorithm J, Chapter 12.0.6). Then `B = Y Σ (Q_b
//!     X)ᵀ`.
//! S6. \[Lift.\] `U ← Q Y`, `V ← Q_b X`, `s ← Σ`; keep the leading `k`.
//!
//! `range_finder(op, l, opts)` is S1–S3 with the width `l` given directly
//! (the oversampling is not added: it returns `Q ∈ T^{m×l}`).
//! After every application the output is checked for NaN/∞
//! ([`StatlaError::NonFinite`] with `what = "operator output"`).
//!
//! **Expectation bound** (Gaussian `Ω`, `p ≥ 2`; HMT Thm. 10.6 and Cor.
//! 10.10, cited from memory):
//!
//! ```text
//! E‖A − QQᵀA‖ ≤ [ (1 + √(k/(p−1))) σ_{k+1}^{2q+1}
//!               + (e√(k+p)/p) (Σ_{j>k} σ_j^{2(2q+1)})^{1/2} ]^{1/(2q+1)}
//! ```
//!
//! (for `q = 0` the familiar `(1 + √(k/(p−1)))σ_{k+1} + (e√(k+p)/p)(Σ_{j>k}
//! σ_j²)^{1/2}`). The bound is on the *mean*: the pilot `s01` saw one
//! Gaussian run at 0.99 of it, so the acceptance (L3.3) is on the mean over
//! ≥ 40 seeds, for `q ∈ {0, 1, 2}`, never per run. With `U Σ Vᵀ` as above,
//! `‖A − U Σ Vᵀ‖ = ‖A − QQᵀA‖` up to rounding, since `U Σ Vᵀ = Q B` exactly
//! when `l = k`; for `l > k` truncation to `k` adds at most `σ_{k+1}(B) ≤
//! σ_{k+1}`.
//!
//! ## Invariants
//! (R1) `Q`, `U`, `V` have orthonormal columns (Algorithm Q, J2); `s` is
//!      nonincreasing and nonnegative (J1) — tested.
//! (R2) *Determinism.* Same operator, `k`, options and seed ⇒ same bits on
//!      every platform (Algorithm G; fixed loop orders) — tested by running
//!      twice and by the frozen sequence of 12.1.
//!
//! ## References
//! Halko, Martinsson & Tropp, "Finding structure with randomness:
//! probabilistic algorithms for constructing approximate matrix
//! decompositions", SIAM Review 53 (2011), §4.2–4.5, §5.1, §10 (theorem
//! numbers cited from memory); R package `rsvd` (Erichson et al.) for the
//! defaults `p = 10`, `q = 2` (from memory).

use xingo_sparse::{DenseBlock, Real};

use crate::error::op_err;
use crate::jacobi::{JacobiOptions, jacobi_svd};
use crate::operator::AdjointOperator;
use crate::random::{Rng, Sketch};
use crate::small::{check_finite, householder_qr};
use crate::{MatMut, MatRef, StatlaError};

/// Options of [`rsvd()`] and [`range_finder`]; the defaults follow R's
/// `rsvd` (`p = 10`, `q = 2`; cited from memory) and HMT's advice to
/// re-orthonormalize at every half step of the power scheme.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RsvdOptions {
    /// Oversampling `p` (default 10): the sketch has `l = k + p` columns
    /// (capped at `min(m, n)`).
    pub oversample: usize,
    /// Power (subspace) iterations `q` (default 2).
    pub power_iters: usize,
    /// Distribution of `Ω` (default Gaussian).
    pub sketch: Sketch,
    /// Seed of Algorithm G (default 0: deterministic by default, unlike
    /// scikit-learn's `random_state=None`).
    pub seed: u64,
}

impl Default for RsvdOptions {
    fn default() -> Self {
        Self {
            oversample: 10,
            power_iters: 2,
            sketch: Sketch::Gaussian,
            seed: 0,
        }
    }
}

/// A truncated SVD `A ≈ U diag(s) Vᵀ`: `U` is `m × k`, `V` is `n × k` (not
/// transposed: R's `svd` convention), `s` nonincreasing.
#[derive(Clone, Debug)]
pub struct Svd<T: Real = f64> {
    /// Left singular vectors.
    pub u: DenseBlock<T>,
    /// Singular values, nonincreasing.
    pub s: Vec<T>,
    /// Right singular vectors.
    pub v: DenseBlock<T>,
    /// Counters.
    pub stats: SvdStats,
}

/// Work counters of [`rsvd()`] and [`crate::svds()`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SvdStats {
    /// Products with `A` (columns).
    pub applies: usize,
    /// Products with `Aᵀ` (columns).
    pub applies_t: usize,
    /// Restarts ([`crate::svds()`]; 0 for [`rsvd()`]).
    pub restarts: usize,
    /// Second Gram–Schmidt passes (DGKS) triggered ([`crate::svds()`]).
    pub reorthogonalizations: usize,
    /// Triplets converged ([`crate::svds()`]; `= k` unless `allow_partial`;
    /// `k` for [`rsvd()`]).
    pub converged: usize,
}

/// `Y ← A X` (or `Aᵀ X`) into a new block, checked for non-finite output.
pub(crate) fn apply_new<T: Real, O: AdjointOperator<T> + ?Sized>(
    op: &mut O,
    x: &DenseBlock<T>,
    transpose: bool,
    stats: &mut SvdStats,
) -> Result<DenseBlock<T>, StatlaError> {
    let rows = if transpose { op.ncols() } else { op.nrows() };
    let mut y = DenseBlock::zeros(rows, x.cols);
    {
        let xv = MatRef::from_block(x);
        let yv = MatMut::from_block(&mut y);
        if transpose {
            op.apply_t_block(xv, yv).map_err(op_err)?;
            stats.applies_t += x.cols;
        } else {
            op.apply_block(xv, yv).map_err(op_err)?;
            stats.applies += x.cols;
        }
    }
    check_finite("operator output", &y.data)?;
    Ok(y)
}

/// Steps S1–S3 with width `l` (validated by the callers).
fn range<T: Real, O: AdjointOperator<T> + ?Sized>(
    op: &mut O,
    l: usize,
    opts: &RsvdOptions,
    stats: &mut SvdStats,
) -> Result<DenseBlock<T>, StatlaError> {
    // S1.
    let mut omega = DenseBlock::zeros(op.ncols(), l);
    Rng::new(opts.seed).fill(opts.sketch, &mut omega.data);
    // S2.
    let (mut q, _) = householder_qr(apply_new(op, &omega, false, stats)?);
    // S3.
    for _ in 0..opts.power_iters {
        let (z, _) = householder_qr(apply_new(op, &q, true, stats)?);
        q = householder_qr(apply_new(op, &z, false, stats)?).0;
    }
    Ok(q)
}

/// Randomized SVD of rank `k`.
///
/// Algorithm S of Chapter 12.4: `l = min(k + p, m, n)` columns of a sketch
/// from the seed's stream, `q` power iterations re-orthonormalized at
/// every half step, and the SVD of the projection by Householder QR and
/// one-sided Jacobi. **Guarantee** (Gaussian `Ω`, `p ≥ 2`): the expectation
/// bound of HMT stated in the chapter; the acceptance of L3.3 tests the
/// mean over 40 seeds against it.
///
/// # Errors
/// [`StatlaError::InvalidArgument`] (`k = 0` or `k > min(m, n)`),
/// [`StatlaError::Operator`], [`StatlaError::NonFinite`] (the operator
/// returned NaN or ∞), and [`StatlaError::NoConvergence`] if the one-sided
/// Jacobi SVD of the small projected matrix (Chapter 12.0.6) has not
/// converged after its limit of sweeps — never observed in the tests;
/// `converged` is then `0`.
pub fn rsvd<T, O>(op: &mut O, k: usize, opts: &RsvdOptions) -> Result<Svd<T>, StatlaError>
where
    T: Real,
    O: AdjointOperator<T> + ?Sized,
{
    let (m, n) = (op.nrows(), op.ncols());
    if k == 0 || k > m.min(n) {
        return Err(StatlaError::InvalidArgument {
            what: "rsvd: k must satisfy 1 <= k <= min(m, n)",
        });
    }
    let l = k.saturating_add(opts.oversample).min(m).min(n);
    let mut stats = SvdStats::default();
    let q = range(op, l, opts, &mut stats)?;
    // S4.
    let bt = apply_new(op, &q, true, &mut stats)?;
    // S5.
    let (qb, r) = householder_qr(bt);
    let small = jacobi_svd(&r, &JacobiOptions::for_rows(l))?;
    // S6: U = Q Y, V = Q_b X, leading k columns.
    let mut u = DenseBlock::zeros(m, k);
    let mut v = DenseBlock::zeros(n, k);
    for c in 0..k {
        for t in 0..l {
            let (yc, xc) = (small.y.get(t, c), small.x.get(t, c));
            for i in 0..m {
                u.data[i + c * m] += q.data[i + t * m] * yc;
            }
            for i in 0..n {
                v.data[i + c * n] += qb.data[i + t * n] * xc;
            }
        }
    }
    stats.converged = k;
    Ok(Svd {
        u,
        s: small.s[..k].to_vec(),
        v,
        stats,
    })
}

/// Steps S1–S3 only: an orthonormal `Q ∈ R^{m×l}` whose range
/// approximates that of `A` (for sketching, CUR, randomized PCA by the
/// caller). The width is `l` itself: [`RsvdOptions::oversample`] is not
/// added (it belongs to [`rsvd()`]'s `k + p`).
///
/// # Errors
/// As [`rsvd()`] (`l = 0` or `l > min(m, n)` is
/// [`StatlaError::InvalidArgument`]).
pub fn range_finder<T, O>(
    op: &mut O,
    l: usize,
    opts: &RsvdOptions,
) -> Result<DenseBlock<T>, StatlaError>
where
    T: Real,
    O: AdjointOperator<T> + ?Sized,
{
    let (m, n) = (op.nrows(), op.ncols());
    if l == 0 || l > m.min(n) {
        return Err(StatlaError::InvalidArgument {
            what: "range_finder: l must satisfy 1 <= l <= min(m, n)",
        });
    }
    range(op, l, opts, &mut SvdStats::default())
}
