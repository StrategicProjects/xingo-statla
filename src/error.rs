//! # Chapter 12.0.1 — Errors
//!
//! ## Purpose
//! One error type for every function of the crate (decision S10 of
//! D-0127), so that `xingo-ffi` maps one enum to R conditions and Python
//! exceptions. **No function of the crate panics on bad input**: wrong
//! lengths, bad indices, non-finite data and invalid options come back as
//! a [`StatlaError`]; a panic is a bug.
//!
//! ## Variants of this edition
//! The enum is `#[non_exhaustive]`. Task L3.2 has the six variants its
//! operations can return, with the names of the contract: `Dimension`,
//! `InvalidArgument`, `NonFinite`, `NotPositiveDefinite`,
//! `DowndateIndefinite`, `RankDeficient`. The contract's `NoConvergence`,
//! `Operator`, `Lu` and `Kkt` arrive with the drivers that return them
//! (L3.3–L3.4); adding a variant to a non-exhaustive enum is not breaking.
//!
//! ## Conventions
//! 1. Lengths are checked before indices, indices before values, values
//!    (finiteness) before any arithmetic: the first violation found in that
//!    order is the one reported.
//! 2. `NonFinite::index` is the position in the argument, column-major for
//!    matrices (`i + j·rows`, independent of the leading dimension).
//! 3. Every error of an updating operation leaves the factorization bit for
//!    bit as it was (invariants H3, Q3).

use core::fmt;

/// Everything that can go wrong, for every function of the crate (decision
/// S10: one error type). No function of the crate panics on bad input;
/// panics are bugs.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum StatlaError {
    /// Two lengths that must agree do not.
    Dimension {
        /// Which argument has the wrong length.
        what: &'static str,
        /// Expected length.
        expected: usize,
        /// Actual length.
        found: usize,
    },
    /// An argument outside its domain: an index out of range, a view whose
    /// `ld < rows`, a matrix that is not square, an option out of its
    /// range, …
    InvalidArgument {
        /// What is wrong, in a few words.
        what: &'static str,
    },
    /// A NaN or an infinity in the input.
    NonFinite {
        /// Which argument.
        what: &'static str,
        /// Position (column-major for matrices).
        index: usize,
    },
    /// Cholesky factorization met a nonpositive pivot at `column` (the
    /// matrix is not positive definite within [`CholOptions::pivot_tol`]).
    ///
    /// [`CholOptions::pivot_tol`]: crate::CholOptions::pivot_tol
    NotPositiveDefinite {
        /// The column of the failing pivot.
        column: usize,
    },
    /// A downdate `A − x xᵀ` would not be positive definite: with `Rᵀa =
    /// x`, `1 − ‖a‖² ≤ CholOptions::downdate_tol` (step C2 of 12.2). The
    /// factor is unchanged (H3).
    DowndateIndefinite {
        /// `1 − ‖a‖²` as computed (≤ the tolerance; may be negative;
        /// `−∞` if it overflowed).
        margin: f64,
    },
    /// `Qr` met a column (or, in `delete_row`, a complement vector) whose
    /// norm fell below [`QrOptions::rank_tol`] after reorthogonalization.
    ///
    /// [`QrOptions::rank_tol`]: crate::QrOptions::rank_tol
    RankDeficient {
        /// The column concerned (for `delete_row`, the index of the row
        /// that could not be deleted).
        column: usize,
    },
}

impl fmt::Display for StatlaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dimension {
                what,
                expected,
                found,
            } => write!(f, "{what}: expected length {expected}, found {found}"),
            Self::InvalidArgument { what } => write!(f, "invalid argument: {what}"),
            Self::NonFinite { what, index } => {
                write!(f, "{what}: non-finite value at position {index}")
            }
            Self::NotPositiveDefinite { column } => {
                write!(f, "matrix not positive definite (pivot of column {column})")
            }
            Self::DowndateIndefinite { margin } => write!(
                f,
                "downdate would leave the matrix indefinite (1 - |a|^2 = {margin:e})"
            ),
            Self::RankDeficient { column } => {
                write!(f, "rank deficient at column {column}")
            }
        }
    }
}

impl std::error::Error for StatlaError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_name_the_argument() {
        let e = StatlaError::Dimension {
            what: "x",
            expected: 3,
            found: 2,
        };
        assert_eq!(e.to_string(), "x: expected length 3, found 2");
        assert!(
            StatlaError::DowndateIndefinite { margin: -0.5 }
                .to_string()
                .contains("-5e-1")
        );
        assert!(
            StatlaError::RankDeficient { column: 4 }
                .to_string()
                .contains('4')
        );
        assert!(
            StatlaError::NonFinite {
                what: "a",
                index: 7
            }
            .to_string()
            .contains("position 7")
        );
        assert!(
            StatlaError::InvalidArgument { what: "ld < rows" }
                .to_string()
                .contains("ld < rows")
        );
        assert!(
            StatlaError::NotPositiveDefinite { column: 1 }
                .to_string()
                .contains("column 1")
        );
    }
}
