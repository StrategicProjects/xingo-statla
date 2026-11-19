//! # Chapter 12.0.3 — Comparisons
//!
//! ## Purpose
//! The rule of Chapter 5.C of `xingo-sparse` holds here too: **every
//! comparison of scalar values goes through the helpers of this module**,
//! and every threshold lives in an options struct ([`crate::CholOptions`],
//! [`crate::QrOptions`]) — never a literal in a kernel (D-0088). The
//! helpers of Chapter 5.6 are `pub(crate)` in `xingo-sparse`, so this is a
//! private copy *with the same definitions* (D-0127, §3), restricted to the
//! ones this crate uses.
//!
//! ## Notation
//! For finite `a`, `b` and `ε ≥ 0`, `s(a, b) = max(1, |a|, |b|)`:
//! * `lt(a, b)` — `a < b` exactly, written `b − a > 0·s(a, b)` for finite
//!   values and `a < b` otherwise (`Tolerances::lt(a, b, 0.0)`);
//! * `le_exact(a, b)` — `a ≤ b` exactly, without the scale;
//! * `is_exact_zero(x)` — `x = 0` (or `−0`): a structural test;
//! * `is_nan(x)` — `x ≠ x`.
//!
//! ## Invariant
//! (C1) No helper is called with NaN except `is_nan` (debug builds assert
//!      it): the public functions reject non-finite input first.
//!
//! ## References
//! Chapter 5.6 of `xingo-sparse` (`real.rs`), whose definitions these are.

use xingo_sparse::Real;

/// The scale `s(a, b) = max(1, |a|, |b|)`.
#[inline]
fn scale<T: Real>(a: T, b: T) -> T {
    T::ONE.max(a.abs()).max(b.abs())
}

/// `a < b` exactly (`Tolerances::lt(a, b, 0.0)`).
#[inline]
pub(crate) fn lt<T: Real>(a: T, b: T) -> bool {
    debug_assert!(!is_nan(a) && !is_nan(b), "lt called with NaN");
    if a.is_finite() && b.is_finite() {
        b - a > T::ZERO * scale(a, b)
    } else {
        a < b
    }
}

/// `a ≤ b` exactly, without the scale.
#[inline]
pub(crate) fn le_exact<T: Real>(a: T, b: T) -> bool {
    debug_assert!(!is_nan(a) && !is_nan(b), "le_exact called with NaN");
    a <= b
}

/// `x = 0` exactly (or `−0`): a structural test.
#[inline]
pub(crate) fn is_exact_zero<T: Real>(x: T) -> bool {
    x == T::ZERO
}

/// `x` is NaN (it is not equal to itself).
#[inline]
#[allow(clippy::eq_op)] // the NaN test is x ≠ x
pub(crate) fn is_nan<T: Real>(x: T) -> bool {
    x != x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparisons_match_their_definitions() {
        assert!(lt(1.0, 2.0) && !lt(2.0, 2.0) && !lt(3.0, 2.0));
        assert!(lt(f64::NEG_INFINITY, 0.0) && !lt(f64::INFINITY, 1e308));
        assert!(le_exact(2.0, 2.0) && !le_exact(2.0 + f64::EPSILON * 2.0, 2.0));
        assert!(is_exact_zero(-0.0) && !is_exact_zero(f64::MIN_POSITIVE));
        assert!(is_nan(f64::NAN) && !is_nan(f64::INFINITY));
    }
}
