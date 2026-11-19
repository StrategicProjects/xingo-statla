//! # Chapter 12.0.7 — Krylov machinery shared by the restarted drivers
//!
//! ## Purpose
//! [`crate::svds()`] (Chapter 12.5) and, in L3.4, `eigsh`/`eigs` (12.6)
//! grow orthonormal bases one vector at a time, restart them keeping a few
//! Ritz vectors, and lock what has converged. The three pieces they share
//! live here, so that the L3.4 engines reuse them unchanged: the expansion
//! step with full DGKS reorthogonalization and its breakdown rule, the
//! choice of the restart dimension, and the counters.
//!
//! ## Notation
//! `V ∈ T^{r×j}` the current basis (the first `j` columns of a preallocated
//! column-major array with `r` rows), orthonormal; `w ∈ T^r` the candidate
//! after the recurrence terms were subtracted; `η` the DGKS threshold;
//! `floor` the breakdown threshold `ε_T·σ̃_est` (an estimate of `‖A‖`).
//!
//! ## Algorithm E (expansion step)
//! E1. \[Reorthogonalize.\] `w ← (I − VVᵀ)w` by Algorithm D of Chapter
//!     12.0.4 (classical Gram–Schmidt against **all** of `V`, a second pass
//!     when `‖w‖` fell below `η` times its value, never a third); count the
//!     second passes.
//! E2. \[Breakdown?\] `ν ← ‖w‖`. If `ν > floor`: `v_{j+1} ← w/ν`, return `ν`.
//! E3. \[Replace.\] Otherwise the Krylov space is (numerically) invariant:
//!     if `j = r` there is no room — `v_{j+1} ← 0`, return `0` (the callers
//!     only reach this at the last step, where a zero coupling makes every
//!     residual zero). Else draw `w` from the seed's stream (Algorithm G),
//!     orthogonalize it against `V` twice (both passes unconditionally),
//!     `v_{j+1} ← w/‖w‖`, return `0`: the coupling coefficient is set to
//!     zero, a perturbation of `A` of norm `≤ floor` (recorded as backward
//!     error), and the iteration continues in the orthogonal complement.
//!
//! ## Algorithm K (restart dimension)
//! Given `k` wanted, `ncv` the basis size, `n_conv` the wanted values that
//! have converged and `n_lock` the locked ones: `k′ ← k + min(⌊(ncv −
//! k)/2⌋, n_conv)` (ARPACK's rule, `dsaup2`: keep more when more has
//! converged, to protect the unconverged), then `k′ ← max(k′, n_lock + 1)`
//! (at least one active vector) and `k′ ← min(k′, ncv − 1)` (room for the
//! residual vector).
//!
//! ## Invariants
//! (E1) After Algorithm E, `v_{j+1}` is a unit vector (or zero in the
//!      no-room case) and `‖Vᵀv_{j+1}‖ ≤ c·j·u` — the (G1)/(K1) bound of
//!      the drivers, measured there.
//! (E2) The coefficient returned is `0` exactly when the step broke down.
//!
//! ## References
//! Daniel, Gragg, Kaufman & Stewart (1976) (DGKS); Lehoucq, Sorensen &
//! Yang, *ARPACK Users' Guide* (1998), `dsaup2`/`dnaup2` (the `nev′` rule);
//! Wu & Simon (2000) (thick restart). Cited from memory.

use xingo_sparse::Real;

use crate::random::Rng;
use crate::real::lt;
use crate::small::{norm2, orthogonalize_against};

/// Algorithm E on the first `ncols` columns of `basis` (`rows` rows);
/// on return `w` holds `v_{ncols+1}`. Returns the coupling coefficient
/// (`0` on breakdown). `reorth` counts the second DGKS passes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn expand<T: Real>(
    basis: &[T],
    rows: usize,
    ncols: usize,
    w: &mut [T],
    eta: T,
    floor: T,
    rng: &mut Rng,
    reorth: &mut usize,
) -> T {
    // E1.
    let (nu, passes) = orthogonalize_against(basis, rows, ncols, w, eta);
    if passes == 2 {
        *reorth += 1;
    }
    // E2.
    if lt(floor, nu) {
        for v in w.iter_mut() {
            *v = *v / nu;
        }
        return nu;
    }
    // E3.
    if ncols >= rows {
        w.fill(T::ZERO);
        return T::ZERO;
    }
    for v in w.iter_mut() {
        *v = T::from_f64(rng.gaussian());
    }
    orthogonalize_against(basis, rows, ncols, w, T::ONE);
    orthogonalize_against(basis, rows, ncols, w, T::ONE);
    let nr = norm2(w);
    for v in w.iter_mut() {
        *v = *v / nr;
    }
    T::ZERO
}

/// Algorithm K.
pub(crate) fn kept_dimension(k: usize, ncv: usize, n_conv: usize, n_lock: usize) -> usize {
    let kp = k + ((ncv - k) / 2).min(n_conv);
    kp.max(n_lock + 1).min(ncv - 1)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn expansion_normalizes_or_replaces() {
        // Basis e_0, e_1 in R^3.
        let basis = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut rng = Rng::new(0);
        let mut c = 0;
        let mut w = [0.3, -0.2, 2.0];
        let nu = expand(&basis, 3, 2, &mut w, 0.7, 1e-12, &mut rng, &mut c);
        assert_eq!((nu, w), (2.0, [0.0, 0.0, 1.0]));
        assert_eq!(c, 0, "no second pass: ‖w‖ fell only from 2.03 to 2");
        // In the span: breakdown, a unit vector orthogonal to the basis.
        let mut w = [0.3, -0.2, 1e-20];
        let nu = expand(&basis, 3, 2, &mut w, 0.7, 1e-12, &mut rng, &mut c);
        assert_eq!((nu, c), (0.0, 1));
        assert!((w[2].abs() - 1.0).abs() < 1e-15 && w[0].abs() < 1e-15 && w[1].abs() < 1e-15);
        // No room: zero vector.
        let full = [1.0, 0.0, 0.0, 1.0];
        let mut w = [1e-30, 0.0];
        assert_eq!(
            expand(&full, 2, 2, &mut w, 0.7, 1e-12, &mut rng, &mut c),
            0.0
        );
        assert_eq!(w, [0.0, 0.0]);
    }

    #[test]
    fn restart_dimension_follows_the_arpack_rule() {
        assert_eq!(kept_dimension(5, 20, 0, 0), 5);
        assert_eq!(kept_dimension(5, 20, 3, 0), 8);
        assert_eq!(kept_dimension(5, 20, 4, 4), 9);
        assert_eq!(kept_dimension(5, 20, 100, 0), 12);
        assert_eq!(kept_dimension(5, 6, 0, 5), 5);
        assert_eq!(kept_dimension(1, 2, 0, 0), 1);
    }
}
