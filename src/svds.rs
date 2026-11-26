//! # Chapter 12.5 — Truncated SVD by Golub–Kahan–Lanczos bidiagonalization with thick restart
//!
//! ## Purpose
//! *The leading `k` singular triplets of an operator, each value accurate
//! to `ε σ_1`.* [`svds()`] is a Krylov engine of its own (decision S2 of
//! D-0127), not an eigensolver on `AᵀA`: it never squares the condition, so
//! every singular value comes out with an *absolute* error of order `ε σ_1`
//! — where scipy's ARPACK-on-the-normal-operator route gives `ε σ_1²/σ_i`.
//!
//! ## Notation
//! `A ∈ T^{m×n}`; `P_j = [p_1 … p_j] ∈ T^{n×j}`, `U_j = [u_1 … u_j] ∈ T^{m×j}`
//! with orthonormal columns (right and left Lanczos vectors); `B_j ∈ T^{j×j}`
//! upper triangular. After `j` steps
//!
//! ```text
//! A  P_j = U_j B_j                                     (B-left)
//! Aᵀ U_j = P_j B_jᵀ + β_j p_{j+1} e_jᵀ,  p_{j+1} ⟂ P_j  (B-right)
//! ```
//!
//! Before the first restart `B_j` is upper bidiagonal (`α_i` on the
//! diagonal, `β_i` above it). After a restart that kept `k′` triplets it is
//! `diag(σ̃_1 … σ̃_{k′})` with a **spike** `ρ` in column `k′ + 1`, then
//! bidiagonal again (the "arrowhead" of Baglama & Reichel). The spike
//! follows from (B-right) of the kept triplets: `Aᵀ ũ_i = σ̃_i p̃_i + ρ_i
//! p_{ncv+1}`, so `U_{k′}ᵀ A p_{k′+1} = (Aᵀ U_{k′})ᵀ p_{k′+1} = ρ` and the next
//! left step is `w = A p_{k′+1} − U_{k′} ρ`. `τ` is the tolerance of step B5
//! (`τ = ε_T` when [`SvdsOptions::tol`] is `0`), `σ̃_1` the largest Ritz
//! value. **Locked** triplets (`n_L` of them, always the first columns)
//! have `ρ_i = 0`, so `B` is block diagonal: `diag(σ̃_L)` and the *active*
//! block, which is all that Steps B4–B6 touch.
//!
//! ## Algorithm B (thick-restart Lanczos bidiagonalization)
//! B1. \[Start.\] `p_1 ← v0/‖v0‖` (`v0` from the options, or `n` Gaussian
//!     deviates of the seed's stream, Algorithm G); `j ← 0`; `k′ ← 0`;
//!     `n_L ← 0`; `σ̃_est ← 0`.
//! B2. \[Left step.\] `j ← j + 1`; `w ← A p_j − Σ_{i<j} b_{ij} u_i` (the
//!     recurrence: `β_{j−1} u_{j−1}`, or `U_{k′} ρ` right after a restart);
//!     **reorthogonalize against all of `U_{j−1}`** and take `α_j ← ‖w‖`,
//!     `u_j ← w/α_j` by Algorithm E of Chapter 12.0.7 with `floor = ε_T
//!     σ̃_est` (breakdown: `α_j ← 0`, `u_j` from the seed's stream). `b_jj ←
//!     α_j`.
//! B3. \[Right step.\] `r ← Aᵀ u_j − α_j p_j`; **reorthogonalize against all
//!     of `P_j`**, `β_j`, `p_{j+1}` by Algorithm E as in B2. `b_{j,j+1} ←
//!     β_j` (or `β_ncv ← β_j` when `j = ncv`). `σ̃_est ← max(σ̃_est, √(α_j² +
//!     β_j²))`. If `j < ncv`, go to B2.
//! B4. \[Ritz.\] SVD of the active block `B_act = X Σ̃ Yᵀ` by one-sided Jacobi
//!     (Algorithm J, Chapter 12.0.6), values nonincreasing. Active Ritz
//!     triplets `(σ̃_i, U_act X e_i, P_act Y e_i)`; by (B-left)/(B-right) `A
//!     p̃_i = σ̃_i ũ_i` and `Aᵀ ũ_i − σ̃_i p̃_i = ρ_i p_{ncv+1}`, `ρ_i = β_ncv ·
//!     e_lastᵀ X e_i`. `σ̃_1 ←` the largest of the locked and active values.
//! B5. \[Test.\] The *wanted* are the `k` largest among the locked and the
//!     active Ritz values (ties: locked first). Locked ones are converged;
//!     an active one when `|ρ_i| ≤ τ σ̃_1` (**absolute**: a test relative to
//!     `σ̃_i` is unattainable for small `σ_i` and is not offered). If the `k`
//!     wanted are converged, go to B7. If the restart limit is reached,
//!     [`StatlaError::NoConvergence`] (or, with `allow_partial`, B7 with the
//!     converged ones).
//! B6. \[Thick restart.\] Lock the converged wanted active triplets
//!     (appended to the locked block in value order; their `ρ_i` becomes
//!     `0`, a perturbation of `A` of norm `≤ τ σ̃_1`, and their vectors are
//!     never rotated again). Choose `k′` by Algorithm K of Chapter 12.0.7
//!     (`n_conv` = the converged wanted) and keep the `k′ − n_L` largest
//!     unlocked active triplets: `P ← [P_L, P_act Y_kept, p_{ncv+1}]`, `U ←
//!     [U_L, U_act X_kept]`, `B ← diag(σ̃_L, σ̃_kept)` with the spike `(0,
//!     ρ_kept)` in column `k′ + 1`. Count a restart; `j ← k′`; go to B2.
//! B7. \[Result.\] The (converged) wanted, nonincreasing: locked ones as
//!     stored, active ones as `U_act X e_i`, `P_act Y e_i`.
//!
//! **Reorthogonalization policy:** full and two-sided (B2 against `U`, B3
//! against `P`, locked vectors included); at most two Gram–Schmidt passes
//! per step. Cost per step: two operator applications plus `≈ 4(m + n) j`
//! flops. One-sided (Simon & Zha) and partial (PROPACK, Larsen)
//! reorthogonalization are candidates for a measured later version.
//!
//! ## Invariants
//! (G1) `‖P_jᵀP_j − I‖, ‖U_jᵀU_j − I‖ ≤ c_G · j · u` — debug-asserted at
//!      every restart (with `c_G` = `C_G`); measured in the tests.
//! (G2) `‖A P_j − U_j B_j‖` and `‖Aᵀ U_j − P_j B_jᵀ − β_j p_{j+1} e_jᵀ‖ ≤ c_G
//!      · j · u · σ̃_est` after every expansion: computed by the unit tests
//!      (it costs `2j` extra applications, so it is not a `debug_assert!`:
//!      the operator may count its calls).
//! (G3) `B` is upper triangular: diagonal in the kept block, the spike in
//!      column `k′ + 1`, bidiagonal after it; locked rows are zero off the
//!      diagonal — by construction, debug-asserted.
//! (G4) `σ̃` nonincreasing (Algorithm J), nonnegative, finite — the operator
//!      output is checked for NaN/∞ after every application.
//! (G5) A locked triplet is never modified again and its `ρ_i` is zero — by
//!      construction (Steps B4–B6 touch only the active block).
//!
//! ## Guarantee
//! For every returned triplet let `r = (A v − σ̂ u; Aᵀ u − σ̂ v)`. The
//! symmetric Jordan–Wielandt matrix `M = [0 A; Aᵀ 0]` has eigenvalues `±σ_j`
//! and `|m − n|` zeros, and `w = (u; v)/√2` is a unit vector (to (G1)) with
//! `M w − σ̂ w = r/√2`. By the residual bound for symmetric matrices (for
//! unit `w`, `min_λ |λ − σ̂| ≤ ‖Mw − σ̂w‖`: expand `w` in the eigenvectors of
//! `M`), **some** `σ_j` (or `0`) satisfies `|σ̂ − σ_j| ≤ ‖r‖/√2`. The
//! algorithm makes `‖r‖ ≤ (τ + c_G · ncv · u) σ̃_1`: the first term is step
//! B5 (`|ρ_i| ≤ τ σ̃_1` is the right half of `r` in exact arithmetic, and
//! the left half is zero), the second the rounding of the relations and the
//! loss of orthogonality, bounded by (G1)–(G2) with a *measured* constant.
//! Two limits: the index `j` is the `i`-th only if no copy of a multiple
//! singular value was missed (a single-vector Krylov method can miss
//! multiplicities: in exact arithmetic it sees one copy only), and the
//! error is absolute — small singular values carry `ε σ_1`, not `ε σ_i`.
//!
//! **The constant** `c_G = C_G = 8` (measured in task L3.3, margin 3.5×):
//! the largest recomputed `(‖r‖/σ̂_1 − τ)/(ncv·u)` over the acceptance
//! cases was `2.3` in `f64` (`2000 × 200`; `≤ 1.4` for `m ≤ 300`) and `0.05`
//! in double-double; (G2) at most `0.85·ncv·u·σ̃_est`; (G1) at most
//! `1.1·ncv·u` (`l33-tables.md`).
//!
//! ## Acceptance (L3.3, S9; as ratified, with the oracle used)
//! (i) every returned triplet: the recomputed `‖r‖ ≤ (τ + c_G·ncv·u) σ̂_1`;
//! (ii) `|σ̂_i − σ_i| ≤ 10⁻¹³ σ_1` (`f64`) on matrices with prescribed
//! spectra down to `10⁻¹⁴ σ_1`, clusters included, where the gap to the
//! neighbours is `≥ 10⁻⁸ σ_1` — the reference SVD is the construction
//! itself, certified to `u‖A‖_F` by Weyl, and cross-checked against a
//! one-sided Jacobi SVD in double-double (tests/svds.rs); (iii) against
//! `scipy.sparse.linalg.svds` on Matrix Market matrices, `1e-10` relative
//! on the `σ_i ≥ 10⁻³σ_1` with gap `≥ 10⁻⁶σ_1` — moved to task L3.4
//! (D-0134), in CI on committed fixtures (tests/arpack.rs); (iv)
//! `DoubleDouble`: (i) with `u = 2⁻¹⁰⁵`. Added by issue xingo-statla#7:
//! (i) and (ii) under many restarts with heavy locking (`σ_j = 1/(1 +
//! j/50)`, `k = 30`, `ncv = 36` on `400 × 300`: 34 restarts; and `k = 40`,
//! `ncv = 48` on `2000 × 1900`, an ignored test of 21 s in release).
//!
//! ## References
//! G. Golub & W. Kahan, "Calculating the singular values and
//! pseudo-inverse of a matrix", J. SIAM Ser. B Numer. Anal. 2(2), 205–224
//! (1965), doi:10.1137/0702016; J. Baglama & L. Reichel, "Augmented
//! implicitly restarted Lanczos bidiagonalization methods", SIAM J. Sci.
//! Comput. 27(1), 19–42 (2005), doi:10.1137/04060593X (the arrowhead after
//! a thick restart; `irlba`); Wu & Simon, "Thick-restart Lanczos method
//! for large symmetric eigenvalue problems", SIAM J. Matrix Anal. Appl.
//! 22(2), 602–616 (2000); H. D. Simon & H. Zha, "Low-rank matrix
//! approximation using the Lanczos bidiagonalization process with
//! applications", SIAM J. Sci. Comput. 21(6), 2257–2274 (2000),
//! doi:10.1137/S1064827597327309; R. M. Larsen, "Lanczos bidiagonalization
//! with partial reorthogonalization", DAIMI PB-537, Aarhus (1998),
//! doi:10.7146/dpb.v27i537.7070; Daniel, Gragg, Kaufman & Stewart, Math.
//! Comp. 30(136), 772–795 (1976); Parlett, *The Symmetric Eigenvalue
//! Problem*, SIAM Classics 20 (1998), Thm. 4.5.1 (the residual bound; the
//! number not verified). Checked in task I03 (issue xingo-statla#4) against Crossref and the publishers' tables of contents, task I03.

use xingo_sparse::{DenseBlock, Real};

use crate::StatlaError;
use crate::error::op_err;
use crate::jacobi::{JacobiOptions, jacobi_svd};
use crate::krylov::{expand, kept_dimension};
use crate::operator::AdjointOperator;
use crate::random::Rng;
use crate::real::{is_exact_zero, le_exact, lt};
use crate::rsvd::{Svd, SvdStats};
use crate::small::{axpy_sub, check_finite, check_len, dot, norm2};

/// The constant `c_G` of invariants (G1)–(G2) and of the guarantee
/// `‖r‖ ≤ (τ + c_G·ncv·u)σ̂_1` (measured in task L3.3, with a margin).
pub(crate) const C_G: f64 = 8.0;

/// Which singular values [`svds()`] computes. v0.1 has only the largest;
/// the smallest (harmonic Ritz or refined restarts, Baglama & Reichel;
/// cited from memory) would be a new variant, hence `#[non_exhaustive]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SvdWhich {
    /// The `k` largest (scipy's `which='LM'`, the default everywhere).
    #[default]
    Largest,
}

/// Options of [`svds()`]. Defaults are scipy's `svds` where it has one
/// (cited from memory), with ARPACK's dimension rule.
#[derive(Clone, Debug, PartialEq)]
pub struct SvdsOptions<T: Real = f64> {
    /// Which values (default [`SvdWhich::Largest`]).
    pub which: SvdWhich,
    /// Krylov dimension `ncv` (number of right Lanczos vectors kept before
    /// a restart); `None` = `min(min(m, n), max(2k + 1, 20))`. Must satisfy
    /// `k < ncv ≤ min(m, n)`.
    pub ncv: Option<usize>,
    /// `τ` of the convergence test (step B5); `0` (default) means
    /// `T::EPSILON`. *Absolute* in units of `σ̃_1`, the largest Ritz value:
    /// a relative test `≤ τ σ̃_i` is unattainable for small `σ_i` and is
    /// not offered.
    pub tol: f64,
    /// Restart limit; `None` = `10 · min(m, n)`.
    pub max_restarts: Option<usize>,
    /// Starting right vector `p_1 ∈ R^n`; `None` = Gaussian from
    /// [`SvdsOptions::seed`] (Algorithm G).
    pub v0: Option<Vec<T>>,
    /// Seed of the starting vector and of the replacement vectors after a
    /// breakdown (default 0).
    pub seed: u64,
    /// DGKS threshold of both reorthogonalizations (default `1/√2`).
    pub reorth_eta: f64,
    /// Compute the singular vectors (default `true`); with `false`, `u`
    /// and `v` of the result are `0`-column blocks.
    pub vectors: bool,
    /// On non-convergence return the converged triplets (`Ok`, fewer than
    /// `k`) instead of [`StatlaError::NoConvergence`] (default `false`).
    pub allow_partial: bool,
}

impl<T: Real> Default for SvdsOptions<T> {
    fn default() -> Self {
        Self {
            which: SvdWhich::Largest,
            ncv: None,
            tol: 0.0,
            max_restarts: None,
            v0: None,
            seed: 0,
            reorth_eta: core::f64::consts::FRAC_1_SQRT_2,
            vectors: true,
            allow_partial: false,
        }
    }
}

/// The validated parameters of one run.
struct Params<T> {
    ncv: usize,
    tau: T,
    eta: T,
    max_restarts: usize,
}

fn invalid(what: &'static str) -> StatlaError {
    StatlaError::InvalidArgument { what }
}

/// Check the arguments (in the order of the error conventions, 12.0.1).
fn validate<T: Real>(
    m: usize,
    n: usize,
    k: usize,
    o: &SvdsOptions<T>,
) -> Result<Params<T>, StatlaError> {
    let mn = m.min(n);
    if k == 0 || k >= mn {
        return Err(invalid("svds: k must satisfy 1 <= k < min(m, n)"));
    }
    let ncv = o.ncv.unwrap_or_else(|| mn.min((2 * k + 1).max(20)));
    if ncv <= k || ncv > mn {
        return Err(invalid("svds: ncv must satisfy k < ncv <= min(m, n)"));
    }
    if !o.tol.is_finite() || lt(o.tol, 0.0) {
        return Err(invalid("svds: tol must be finite and >= 0"));
    }
    if !o.reorth_eta.is_finite() || !lt(0.0, o.reorth_eta) || !lt(o.reorth_eta, 1.0) {
        return Err(invalid("svds: reorth_eta must lie in (0, 1)"));
    }
    if let Some(v0) = &o.v0 {
        check_len("v0", n, v0.len())?;
        check_finite("v0", v0)?;
        if is_exact_zero(norm2(v0)) {
            return Err(invalid("svds: v0 is the zero vector"));
        }
    }
    let tau = if is_exact_zero(o.tol) {
        T::EPSILON
    } else {
        T::from_f64(o.tol)
    };
    Ok(Params {
        ncv,
        tau,
        eta: T::from_f64(o.reorth_eta),
        max_restarts: o.max_restarts.unwrap_or(10 * mn),
    })
}

/// Where a candidate of step B5 lives.
#[derive(Clone, Copy, Debug)]
enum Slot {
    /// Locked column `i`.
    Locked(usize),
    /// Active Ritz triplet `i` (column of `X`, `Y`).
    Active(usize),
}

/// One candidate of step B5.
#[derive(Clone, Copy, Debug)]
struct Cand<T> {
    value: T,
    slot: Slot,
    converged: bool,
}

/// The state of Algorithm B.
struct Engine<T: Real> {
    m: usize,
    n: usize,
    ncv: usize,
    /// `P`, `n × (ncv + 1)`.
    p: Vec<T>,
    /// `U`, `m × ncv`.
    u: Vec<T>,
    /// `B`, `ncv × ncv`.
    b: DenseBlock<T>,
    /// `β_ncv`.
    beta_last: T,
    /// Locked count `n_L`.
    nl: usize,
    /// `σ̃_est`.
    sigma_est: T,
    rng: Rng,
    stats: SvdStats,
    /// Largest (G2) residual over the expansions, in units of `σ̃_est`
    /// (unit tests only).
    #[cfg(test)]
    g2: Option<f64>,
}

/// `y ← A x` (or `Aᵀ x`) with the counters and the NaN check.
fn apply<T: Real, O: AdjointOperator<T> + ?Sized>(
    op: &mut O,
    x: &[T],
    y: &mut [T],
    transpose: bool,
    stats: &mut SvdStats,
) -> Result<(), StatlaError> {
    if transpose {
        op.apply_t(x, y).map_err(op_err)?;
        stats.applies_t += 1;
    } else {
        op.apply(x, y).map_err(op_err)?;
        stats.applies += 1;
    }
    check_finite("operator output", y)
}

impl<T: Real> Engine<T> {
    /// Steps B2–B3 for `j = from, …, ncv − 1` (0-based).
    fn expand_from<O: AdjointOperator<T> + ?Sized>(
        &mut self,
        op: &mut O,
        from: usize,
        eta: T,
    ) -> Result<(), StatlaError> {
        let (m, n, ncv) = (self.m, self.n, self.ncv);
        let mut w = vec![T::ZERO; m];
        let mut r = vec![T::ZERO; n];
        for j in from..ncv {
            // B2.
            apply(
                op,
                &self.p[j * n..(j + 1) * n],
                &mut w,
                false,
                &mut self.stats,
            )?;
            for i in 0..j {
                let bij = self.b.data[i + j * ncv];
                if !is_exact_zero(bij) {
                    axpy_sub(bij, &self.u[i * m..(i + 1) * m], &mut w);
                }
            }
            let floor = T::EPSILON * self.sigma_est;
            let alpha = expand(
                &self.u,
                m,
                j,
                &mut w,
                eta,
                floor,
                &mut self.rng,
                &mut self.stats.reorthogonalizations,
            );
            self.u[j * m..(j + 1) * m].copy_from_slice(&w);
            self.b.data[j + j * ncv] = alpha;
            // B3.
            apply(
                op,
                &self.u[j * m..(j + 1) * m],
                &mut r,
                true,
                &mut self.stats,
            )?;
            axpy_sub(alpha, &self.p[j * n..(j + 1) * n], &mut r);
            let floor = T::EPSILON * self.sigma_est.max(alpha);
            let beta = expand(
                &self.p,
                n,
                j + 1,
                &mut r,
                eta,
                floor,
                &mut self.rng,
                &mut self.stats.reorthogonalizations,
            );
            self.p[(j + 1) * n..(j + 2) * n].copy_from_slice(&r);
            if j + 1 < ncv {
                self.b.data[j + (j + 1) * ncv] = beta;
            } else {
                self.beta_last = beta;
            }
            self.sigma_est = self.sigma_est.max((alpha * alpha + beta * beta).sqrt());
        }
        Ok(())
    }

    /// (G2), unit tests only: the two relations after a full expansion.
    #[cfg(test)]
    fn relations<O: AdjointOperator<T> + ?Sized>(&mut self, op: &mut O) -> f64 {
        let (m, n, ncv) = (self.m, self.n, self.ncv);
        let mut worst = 0.0f64;
        let mut y = vec![T::ZERO; m];
        let mut z = vec![T::ZERO; n];
        for j in 0..ncv {
            op.apply(&self.p[j * n..(j + 1) * n], &mut y).unwrap();
            for i in 0..=j {
                axpy_sub(
                    self.b.data[i + j * ncv],
                    &self.u[i * m..(i + 1) * m],
                    &mut y,
                );
            }
            worst = worst.max(norm2(&y).to_f64());
            op.apply_t(&self.u[j * m..(j + 1) * m], &mut z).unwrap();
            for i in j..ncv {
                axpy_sub(
                    self.b.data[j + i * ncv],
                    &self.p[i * n..(i + 1) * n],
                    &mut z,
                );
            }
            if j + 1 == ncv {
                axpy_sub(self.beta_last, &self.p[ncv * n..(ncv + 1) * n], &mut z);
            }
            worst = worst.max(norm2(&z).to_f64());
        }
        worst / self.sigma_est.to_f64().max(f64::MIN_POSITIVE)
    }

    /// The active block `B[nl.., nl..]`.
    fn active_block(&self) -> DenseBlock<T> {
        let (ncv, nl) = (self.ncv, self.nl);
        let na = ncv - nl;
        let mut a = DenseBlock::zeros(na, na);
        for j in 0..na {
            for i in 0..=j {
                a.data[i + j * na] = self.b.data[nl + i + (nl + j) * ncv];
            }
        }
        a
    }

    /// The active Ritz vector `U_act X e_c` (left) or `P_act Y e_c`.
    fn ritz(&self, coef: &DenseBlock<T>, c: usize, left: bool) -> Vec<T> {
        let (rows, base) = if left {
            (self.m, &self.u)
        } else {
            (self.n, &self.p)
        };
        let na = coef.rows;
        let mut out = vec![T::ZERO; rows];
        for t in 0..na {
            let f = coef.data[t + c * na];
            let col = &base[(self.nl + t) * rows..(self.nl + t + 1) * rows];
            for (o, &v) in out.iter_mut().zip(col) {
                *o += v * f;
            }
        }
        out
    }

    /// (G3), debug builds, right after a restart to `k′ = kp`: the locked and
    /// kept rows are zero off the diagonal except the spike column `kp`
    /// (and the locked rows are zero there too), nothing beyond `kp`.
    fn arrowhead_shape(&self, kp: usize) -> bool {
        let ncv = self.ncv;
        for j in 0..ncv {
            for i in 0..ncv {
                let v = self.b.data[i + j * ncv];
                let allowed = (i == j && j < kp) || (j == kp && i >= self.nl && i < kp);
                if !allowed && !is_exact_zero(v) {
                    return false;
                }
            }
        }
        true
    }

    /// (G1), debug builds: orthonormality of `P_{j+1}` and `U_j`.
    fn orthogonality(&self, j: usize) -> f64 {
        let mut e = 0.0f64;
        for (base, rows, cols) in [(&self.p, self.n, j + 1), (&self.u, self.m, j)] {
            for a in 0..cols {
                for c in 0..=a {
                    let d = dot(
                        &base[a * rows..(a + 1) * rows],
                        &base[c * rows..(c + 1) * rows],
                    );
                    let t = if a == c { d - T::ONE } else { d };
                    // A zero column (the no-room case of Algorithm E) is exempt.
                    if a == c && is_exact_zero(d) {
                        continue;
                    }
                    e = e.max(t.abs().to_f64());
                }
            }
        }
        e
    }
}

/// Leading `k` singular triplets by Golub–Kahan–Lanczos bidiagonalization
/// with thick restart — a Krylov engine of its own, not an eigensolver on
/// `AᵀA` (S2): it never squares the condition, so every singular value
/// comes out with an *absolute* error of order `ε σ_1`.
///
/// Algorithm B of Chapter 12.5: full two-sided DGKS reorthogonalization,
/// one-sided Jacobi SVD of the projected matrix, thick restart in the
/// arrowhead form of Baglama & Reichel, locking of converged triplets,
/// absolute convergence test `|ρ_i| ≤ τ σ̃_1`.
///
/// **Guarantee.** Every returned triplet has `‖(A v − σ̂ u; Aᵀ u − σ̂ v)‖ ≤
/// (τ + c_G·ncv·u) σ̂_1` (`c_G = 8`, measured; Chapter 12.5), hence some singular value
/// `σ_j` (or 0) lies within `‖r‖/√2` of `σ̂` (Jordan–Wielandt; proved in the
/// chapter). The index `j` is the `i`-th unless a copy of a multiple value
/// was missed.
///
/// # Errors
/// [`StatlaError::InvalidArgument`] (`k = 0`; `k ≥ min(m, n)`; `ncv ∉ (k,
/// min(m, n)]`; `tol < 0` or not finite; `reorth_eta ∉ (0, 1)`; `v0 = 0`),
/// [`StatlaError::Dimension`] (`v0`), [`StatlaError::NonFinite`] (`v0`, or
/// the operator returned NaN/∞), [`StatlaError::Operator`],
/// [`StatlaError::NoConvergence`].
pub fn svds<T, O>(op: &mut O, k: usize, opts: &SvdsOptions<T>) -> Result<Svd<T>, StatlaError>
where
    T: Real,
    O: AdjointOperator<T> + ?Sized,
{
    run(op, k, opts, false).map(|(svd, _)| svd)
}

/// Diagnostics of a run (read by the unit tests).
#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(test), allow(dead_code))]
struct Diag {
    /// Largest (G2) residual / `σ̃_est` (with `paranoid`; else 0).
    g2: f64,
    /// Triplets locked at the end.
    locked: usize,
}

/// Algorithm B; with `paranoid` (unit tests) the largest (G2) residual is
/// returned in the diagnostics.
#[allow(clippy::too_many_lines)]
fn run<T, O>(
    op: &mut O,
    k: usize,
    opts: &SvdsOptions<T>,
    paranoid: bool,
) -> Result<(Svd<T>, Diag), StatlaError>
where
    T: Real,
    O: AdjointOperator<T> + ?Sized,
{
    let (m, n) = (op.nrows(), op.ncols());
    let par = validate(m, n, k, opts)?;
    let ncv = par.ncv;
    let _ = paranoid;
    // B1.
    let mut rng = Rng::new(opts.seed);
    let mut p = vec![T::ZERO; n * (ncv + 1)];
    match &opts.v0 {
        Some(v0) => p[..n].copy_from_slice(v0),
        None => rng.fill(crate::Sketch::Gaussian, &mut p[..n]),
    }
    let nv = norm2(&p[..n]);
    for v in &mut p[..n] {
        *v = *v / nv;
    }
    let mut e = Engine {
        m,
        n,
        ncv,
        p,
        u: vec![T::ZERO; m * ncv],
        b: DenseBlock::zeros(ncv, ncv),
        beta_last: T::ZERO,
        nl: 0,
        sigma_est: T::ZERO,
        rng,
        stats: SvdStats::default(),
        #[cfg(test)]
        g2: None,
    };
    let mut locked: Vec<T> = Vec::new();
    let mut from = 0;
    let jopts = JacobiOptions::for_rows(ncv);
    loop {
        // B2–B3.
        e.expand_from(op, from, par.eta)?;
        #[cfg(test)]
        if paranoid {
            let g = e.relations(op);
            e.g2 = Some(e.g2.map_or(g, |h: f64| h.max(g)));
        }
        // B4.
        let na = ncv - e.nl;
        let small = jacobi_svd(&e.active_block(), &jopts)?;
        let rho: Vec<T> = (0..na)
            .map(|i| e.beta_last * small.x.data[(na - 1) + i * na])
            .collect();
        let mut sigma1 = small.s[0];
        for &v in &locked {
            sigma1 = sigma1.max(v);
        }
        // B5.
        let thresh = par.tau * sigma1;
        let mut cands: Vec<Cand<T>> = Vec::with_capacity(ncv);
        for (i, &v) in locked.iter().enumerate() {
            cands.push(Cand {
                value: v,
                slot: Slot::Locked(i),
                converged: true,
            });
        }
        for i in 0..na {
            cands.push(Cand {
                value: small.s[i],
                slot: Slot::Active(i),
                converged: le_exact(rho[i].abs(), thresh),
            });
        }
        // Stable sort by value, locked first on ties (they come first).
        cands.sort_by(|a, b| {
            if lt(b.value, a.value) {
                core::cmp::Ordering::Less
            } else if lt(a.value, b.value) {
                core::cmp::Ordering::Greater
            } else {
                core::cmp::Ordering::Equal
            }
        });
        let wanted = &cands[..k];
        let n_conv = wanted.iter().filter(|c| c.converged).count();
        let limit = e.stats.restarts >= par.max_restarts;
        if n_conv == k || limit {
            if n_conv < k && !opts.allow_partial {
                return Err(StatlaError::NoConvergence {
                    requested: k,
                    converged: n_conv,
                    restarts: e.stats.restarts,
                });
            }
            // B7.
            let take: Vec<Cand<T>> = wanted.iter().copied().filter(|c| c.converged).collect();
            let cols = if opts.vectors { take.len() } else { 0 };
            let mut uo = DenseBlock::zeros(m, cols);
            let mut vo = DenseBlock::zeros(n, cols);
            let mut s = Vec::with_capacity(take.len());
            for (c, cand) in take.iter().enumerate() {
                s.push(cand.value);
                if !opts.vectors {
                    continue;
                }
                let (uc, vc) = match cand.slot {
                    Slot::Locked(i) => (
                        e.u[i * m..(i + 1) * m].to_vec(),
                        e.p[i * n..(i + 1) * n].to_vec(),
                    ),
                    Slot::Active(i) => (e.ritz(&small.x, i, true), e.ritz(&small.y, i, false)),
                };
                uo.data[c * m..(c + 1) * m].copy_from_slice(&uc);
                vo.data[c * n..(c + 1) * n].copy_from_slice(&vc);
            }
            let mut stats = e.stats;
            stats.converged = s.len();
            #[cfg(test)]
            let g2 = e.g2.unwrap_or(0.0);
            #[cfg(not(test))]
            let g2 = 0.0;
            let diag = Diag { g2, locked: e.nl };
            return Ok((
                Svd {
                    u: uo,
                    s,
                    v: vo,
                    stats,
                },
                diag,
            ));
        }
        // B6.
        e.stats.restarts += 1;
        let newly: Vec<usize> = wanted
            .iter()
            .filter_map(|c| match c.slot {
                Slot::Active(i) if c.converged => Some(i),
                _ => None,
            })
            .collect();
        let nl_new = e.nl + newly.len();
        let kp = kept_dimension(k, ncv, n_conv, nl_new);
        let kept: Vec<usize> = (0..na)
            .filter(|i| !newly.contains(i))
            .take(kp - nl_new)
            .collect();
        let (mut pn, mut un) = (e.p.clone(), e.u.clone());
        let mut bn = DenseBlock::zeros(ncv, ncv);
        for i in 0..e.nl {
            bn.data[i + i * ncv] = e.b.data[i + i * ncv];
        }
        for (slot, &i) in newly.iter().chain(&kept).enumerate() {
            let col = e.nl + slot;
            un[col * m..(col + 1) * m].copy_from_slice(&e.ritz(&small.x, i, true));
            pn[col * n..(col + 1) * n].copy_from_slice(&e.ritz(&small.y, i, false));
            bn.data[col + col * ncv] = small.s[i];
            if col >= nl_new {
                bn.data[col + kp * ncv] = rho[i];
            }
        }
        pn[kp * n..(kp + 1) * n].copy_from_slice(&e.p[ncv * n..(ncv + 1) * n]);
        for &i in &newly {
            locked.push(small.s[i]);
        }
        e.p = pn;
        e.u = un;
        e.b = bn;
        e.nl = nl_new;
        from = kp;
        debug_assert!(e.arrowhead_shape(kp), "(G3) arrowhead after a restart");
        #[allow(clippy::cast_precision_loss)]
        let g1 = C_G * (ncv as f64) * T::EPSILON.to_f64() / 2.0;
        debug_assert!(
            e.orthogonality(kp) <= g1,
            "(G1) after a restart: {}",
            e.orthogonality(kp)
        );
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use xingo_extprec::DoubleDouble as Dd;

    /// A diagonal `m × n` operator with prescribed values, rotated by a
    /// fixed reflector on each side so that the start vector is generic.
    fn test_matrix<T: Real>(m: usize, n: usize, s: &[f64], seed: u64) -> DenseBlock<T> {
        let mut a = DenseBlock::<T>::zeros(m, n);
        for (i, &v) in s.iter().enumerate() {
            a.data[i + i * m] = T::from_f64(v);
        }
        let mut rng = Rng::new(seed);
        for _ in 0..3 {
            let v: Vec<T> = (0..m).map(|_| T::from_f64(rng.gaussian())).collect();
            let vv = dot(&v, &v);
            for j in 0..n {
                let col = &mut a.data[j * m..(j + 1) * m];
                let f = T::from_f64(2.0) * dot(col, &v) / vv;
                axpy_sub(f, &v, col);
            }
            let w: Vec<T> = (0..n).map(|_| T::from_f64(rng.gaussian())).collect();
            let ww = dot(&w, &w);
            for i in 0..m {
                let d: T = (0..n).map(|j| a.data[i + j * m] * w[j]).sum();
                let f = T::from_f64(2.0) * d / ww;
                for j in 0..n {
                    a.data[i + j * m] -= f * w[j];
                }
            }
        }
        a
    }

    fn g2_of<T: Real>(u: f64) -> f64 {
        let s: Vec<f64> = (0..60).map(|j| 0.9f64.powi(j)).collect();
        let mut a = test_matrix::<T>(90, 60, &s, 5);
        let (f, d) = run(&mut a, 6, &SvdsOptions::default(), true).unwrap();
        assert!(f.stats.restarts > 0, "the test must restart");
        assert!(d.locked > 0, "the test must lock");
        d.g2 / (20.0 * u)
    }

    /// (G2): the relations hold to `c_G·ncv·u·σ̃_est` after every expansion,
    /// restarts included.
    #[test]
    fn relations_hold_after_every_expansion() {
        let r64 = g2_of::<f64>(f64::EPSILON / 2.0);
        let rdd = g2_of::<Dd>(Dd::EPSILON.to_f64() / 2.0);
        eprintln!("(G2) / (ncv u): f64 {r64:.3}, dd {rdd:.3}");
        assert!(r64 <= C_G && rdd <= C_G, "{r64} {rdd}");
    }

    /// (G5) and B6: on a slowly converging spectrum triplets are locked, and
    /// a locked triplet's right residual is its dropped `ρ` (`≤ τ σ̃_1`).
    #[test]
    fn converged_triplets_are_locked() {
        let s: Vec<f64> = (0..80).map(|j| 1.0 / (1.0 + 0.05 * f64::from(j))).collect();
        let mut a = test_matrix::<f64>(100, 80, &s, 2);
        let (f, d) = run(&mut a, 8, &SvdsOptions::default(), false).unwrap();
        assert!(d.locked > 0 && f.stats.restarts > 2, "{d:?} {:?}", f.stats);
        let mut y = vec![0.0; 80];
        for c in 0..f.s.len() {
            a.apply_t(&f.u.data[c * 100..(c + 1) * 100], &mut y)
                .unwrap();
            axpy_sub(f.s[c], &f.v.data[c * 80..(c + 1) * 80], &mut y);
            assert!(norm2(&y) <= 1e-14 * f.s[0], "{c}: {}", norm2(&y));
            assert!(
                (f.s[c] - s[c]).abs() <= 1e-14,
                "{c}: {} vs {}",
                f.s[c],
                s[c]
            );
        }
    }

    #[test]
    fn options_are_validated() {
        let mut a = test_matrix::<f64>(8, 6, &[3.0, 2.0, 1.0, 0.5, 0.25, 0.1], 1);
        let d = SvdsOptions::<f64>::default();
        let bad =
            |o: &SvdsOptions<f64>, k: usize, a: &mut DenseBlock<f64>| svds(a, k, o).unwrap_err();
        assert!(matches!(
            bad(&d, 0, &mut a),
            StatlaError::InvalidArgument { .. }
        ));
        assert!(matches!(
            bad(&d, 6, &mut a),
            StatlaError::InvalidArgument { .. }
        ));
        for o in [
            SvdsOptions {
                ncv: Some(2),
                ..d.clone()
            },
            SvdsOptions {
                ncv: Some(7),
                ..d.clone()
            },
            SvdsOptions {
                tol: -1.0,
                ..d.clone()
            },
            SvdsOptions {
                tol: f64::NAN,
                ..d.clone()
            },
            SvdsOptions {
                reorth_eta: 1.0,
                ..d.clone()
            },
            SvdsOptions {
                reorth_eta: 0.0,
                ..d.clone()
            },
            SvdsOptions {
                v0: Some(vec![0.0; 6]),
                ..d.clone()
            },
        ] {
            assert!(
                matches!(bad(&o, 2, &mut a), StatlaError::InvalidArgument { .. }),
                "{o:?}"
            );
        }
        let o = SvdsOptions {
            v0: Some(vec![1.0; 5]),
            ..d.clone()
        };
        assert!(matches!(
            bad(&o, 2, &mut a),
            StatlaError::Dimension { what: "v0", .. }
        ));
        let o = SvdsOptions {
            v0: Some(vec![1.0, f64::INFINITY, 0.0, 0.0, 0.0, 0.0]),
            ..d.clone()
        };
        assert!(matches!(
            bad(&o, 2, &mut a),
            StatlaError::NonFinite {
                what: "v0",
                index: 1
            }
        ));
        // ncv = min(m, n): the full space, converged in one pass.
        let f = svds(
            &mut a,
            3,
            &SvdsOptions {
                ncv: Some(6),
                ..d.clone()
            },
        )
        .unwrap();
        assert_eq!(f.stats.restarts, 0);
        assert!((f.s[0] - 3.0).abs() < 1e-14 && (f.s[2] - 1.0).abs() < 1e-14);
    }
}
