//! # Chapter 12 — `xingo-statla`: updatable and matrix-free linear algebra for statistics
//!
//! **Status: complete for v0.1 (tasks L3.2, L3.3 and L3.4).** The contract
//! of the crate was ratified on 2026-11-19 (decision log of the Xingó
//! solver, D-0127) and amended by what the implementation measured (D-0132,
//! D-0134, D-0135); where they differ, this text and the code are
//! authoritative. Labels such as L3.2 are tasks of the solver's plan, D-xxxx
//! are entries of its decision log, and names such as `l34-tables.md` are
//! the working notes of a task in the monorepo (not part of the package;
//! the numbers that matter are quoted where they are used). L3.2
//! implemented the *updatable factorizations*: the dense views
//! [`MatRef`]/[`MatMut`], the error type [`StatlaError`] (Chapter 12.0),
//! [`Cholesky`] with rank-1 update and transactional downdate (Chapter
//! 12.2) and the thin [`Qr`] with explicit `Q` and five updating operations
//! (Chapter 12.3). L3.3 added the *matrix-free* part: the operator traits
//! [`LinearOperator`]/[`AdjointOperator`] (Chapter 12.0.5), the
//! reproducible generator (Chapter 12.1, [`Sketch`]), [`rsvd()`] and
//! [`range_finder`] (12.4), and [`svds()`] by Golub–Kahan–Lanczos
//! bidiagonalization with thick restart (12.5), on the shared Krylov
//! machinery (12.0.6–12.0.7). L3.4 completed the v0.1 surface:
//! [`eigsh()`] and [`eigs()`] by Krylov–Schur restarted Lanczos/Arnoldi
//! with ARPACK's `which` vocabulary (12.6), the small dense eigensolvers
//! they need (12.6.1), and shift-invert over the factorizations of
//! `xingo-sparse`, [`eigsh_near`] with the inertia count and [`eigs_near`]
//! (12.6.2). (The chapter number 12 is that of the Xingó book: 1–8, 10 and
//! 11 are other parts of the solver.)
//!
//! ## Purpose
//! Statistical computing keeps asking questions that a dense library
//! answers only by starting from scratch. The four this edition answers:
//!
//! 1. *A row arrived (or left); what is the new least-squares fit?* — a QR
//!    factorization **updated** in `O(mn)` instead of refactorized in
//!    `O(mn²)` (sequential and windowed regression, recursive least
//!    squares, leave-one-out): [`Qr`].
//! 2. *A covariance or information matrix changed by `± x xᵀ`; what is its
//!    new Cholesky factor?* — a rank-1 **update/downdate** in `O(n²)`
//!    (Kalman/RLS filters, Gaussian-process and mixed-model refits, the
//!    Q-less QR, which *is* the Cholesky factor of `AᵀA`): [`Cholesky`].
//!
//! 3. *What are the leading `k` singular triplets of a matrix I can only
//!    multiply by?* — the **randomized** SVD of Halko, Martinsson & Tropp,
//!    with a stated expected error ([`rsvd()`]), and a deterministic truncated
//!    SVD by Golub–Kahan–Lanczos bidiagonalization accurate to `ε σ_1` on
//!    every singular value ([`svds()`]).
//! 4. *What are `k` eigenvalues at one end of the spectrum (or nearest
//!    `σ`) of a large sparse or implicit operator?* — [`eigsh()`]/[`eigs()`]
//!    by Krylov–Schur restarted Lanczos/Arnoldi with ARPACK's `which`
//!    vocabulary, and [`eigsh_near`]/[`eigs_near`] by shift-invert.
//!
//! The crate does not depend on the solver; the R and Python packages will
//! reach it through `xingo-ffi` (L4).
//!
//! ## Notation
//! `T` is the scalar ([`Real`]), `ε_T` its machine epsilon and `u = ε_T/2`
//! its unit roundoff (`2⁻⁵³` for `f64`, `2⁻¹⁰⁵` for the double-double of
//! `xingo-extprec`). Matrices are column-major: `a_ij = data[i + j·ld]`,
//! `ld ≥ rows`. `A ∈ R^{m×n}`; `R` upper triangular with positive diagonal
//! (the convention of R's `chol` and MATLAB's `chol`/`cholupdate`; `RᵀR =
//! A`); `Q ∈ R^{m×n}` with orthonormal columns; `‖·‖` the 2-norm, `‖·‖_F`
//! Frobenius; `e_j` the `j`-th unit vector. `A_t` is the matrix a sequence
//! of operations defines (the exact result of applying them to the input).
//!
//! ## Layout of the crate
//! | chapter | module | content |
//! |---|---|---|
//! | 12 | this page | overview, invariants, decisions |
//! | 12.0 | `error`, `view`, `real`, `small` (private) | errors, views, comparisons, Givens rotations, Householder reflectors and QR, DGKS Gram–Schmidt |
//! | 12.0.5 | `operator` | [`LinearOperator`], [`AdjointOperator`], [`OperatorError`], closures ([`from_fn`], [`from_fns`]) |
//! | 12.0.6 | `jacobi` (private) | one-sided Jacobi SVD of a small matrix |
//! | 12.0.7 | `krylov` (private) | expansion with DGKS and breakdown, restart dimension |
//! | 12.1 | `random` | Algorithm G (xoshiro256++, polar method, own `ln`), [`Sketch`] |
//! | 12.2 | `cholesky` | [`Cholesky`], [`CholOptions`], [`DowndateInfo`] |
//! | 12.3 | `qr` | [`Qr`], [`QrOptions`] |
//! | 12.4 | `rsvd` | [`rsvd()`], [`range_finder`], [`RsvdOptions`], [`Svd`], [`SvdStats`] |
//! | 12.5 | `svds` | [`svds()`], [`SvdsOptions`], [`SvdWhich`] |
//! | 12.6 | `eigs` | [`eigsh()`], [`eigs()`], [`EigsOptions`], [`EigsStats`], [`SymEigen`], [`Eigen`], [`Complex`], [`Which`] |
//! | 12.6.1 | `schur` (private) | symmetric Jacobi, Hessenberg + Francis real Schur, block swaps and reordering, eigenvectors of the quasi-triangular form |
//! | 12.6.2 | `shift_invert` | [`ShiftInvert`], [`SymShiftInvert`], [`eigsh_near`], [`eigs_near`] |
//!
//! ## What is reused from `xingo-sparse`, and why (D-0080 layering)
//! | need | taken from `xingo-sparse` | new here |
//! |---|---|---|
//! | scalar | [`Real`] (generic everywhere; `f64` default; `DoubleDouble` of `xingo-extprec` instantiates unchanged) | — |
//! | owned dense | [`DenseBlock`] (column-major, `ld = rows`) | borrowed views with `ld` (S3) |
//! | the initial Cholesky factorization | [`DenseKernel::potrf`] via a type parameter `K = NaiveDense`, as `SupernodalLdl<T, K>` does | — |
//! | Householder, Givens, Gram–Schmidt, Jacobi, Francis/Schur | — (not in `DenseKernel`) | internal, generic in `T` (S1) |
//! | shift-invert solves | `Lu<T>` (general), `SupernodalLdl<T, K>` `LBLᵀ` + inertia (symmetric) | [`ShiftInvert`], [`SymShiftInvert`], one step of iterative refinement per application |
//! | comparisons | the helpers of Chapter 5.6 are `pub(crate)` there | a private copy with the same definitions (module `real`) |
//!
//! **`DoubleDouble`.** Every algorithm needs only `+ − · /`, `sqrt`, `abs`
//! and the order — exactly [`Real`] — so `Cholesky<DoubleDouble>` and
//! `Qr<DoubleDouble>` instantiate without code (S1); the tests run the
//! whole acceptance suite in both scalars. Random test matrices are
//! generated in `f64` and converted exactly (`Real::from_f64`).
//!
//! **Where the kernel `K` is used.** In this edition only by
//! [`Cholesky::factor_with`] (`potrf`). The Householder QR, the updates and
//! the solves are `O(n²)`/`O(mn)` loops written here: the contract's "large
//! products" of `Qr` (a blocked WY Householder through `gemm_sub`) are a
//! measured optimization for later, and `Qr` carries its `K` to
//! [`Qr::to_cholesky`].
//!
//! ## Divergences from the references
//! * **LINPACK `dchud`/`dchdd`** (Saunders 1972; Stewart 1979) for the
//!   Cholesky update/downdate, not hyperbolic rotations: the pilot `s01`
//!   measured both (same backward error `≈ 4·10⁻¹⁶` for `κ` up to
//!   `10¹²`), and LINPACK detects indefiniteness *before* touching `R` —
//!   the transactional guarantee for free (12.2). This edition also
//!   computes the new diagonal before writing, so that an underflow to a
//!   zero pivot is refused transactionally too (step C3′).
//! * **Hammarling & Lucas (2008)** for the QR updates, in the "economy"
//!   forms with an explicit thin `Q`; `delete_row` and `rank1_update`
//!   complete `Q` by one Gram–Schmidt vector with DGKS reorthogonalization
//!   (Daniel, Gragg, Kaufman & Stewart 1976) instead of keeping a full
//!   `m × m` factor (12.3).
//! * **faer 0.24.4** has a rank-`r` LLᵀ update with signed weights (D-0082
//!   addendum); it lacks QR updating, `T: Real` and errors instead of
//!   `assert!`s.
//!
//! ## Invariants (each becomes a `debug_assert!` or `check_invariants()`)
//! (H1) `Cholesky`: `R` is upper triangular (strict lower triangle exactly
//!      zero), `r_jj > 0` and every entry finite.
//!      [`Cholesky::check_invariants`]; `debug_assert!` after every
//!      operation.
//! (H2) `Cholesky`: after any sequence of operations, `‖RᵀR − A_t‖_F ≤
//!      c_H · n · u · ‖A_t‖_F`, checked by the tests against `A_t`
//!      accumulated in double-double; `c_H = 8` (measured: `4.0` at `n =
//!      1`, `≤ 0.06` for `n ≥ 50`, in `f64`; smaller in double-double).
//!      When downdates cancel more than half of the matrix the tests take
//!      the norm along the path, `max_s ‖A_s‖_F`, instead of `‖A_t‖_F`
//!      (question Q-L32-3 of task L3.2).
//! (H3) *Transactional.* A failed `update`/`downdate` leaves the factor
//!      bit for bit as it was.
//! (Q1) `Qr`: `‖QᵀQ − I‖_F ≤ c_Q · m · u` (reorthogonalization keeps it);
//!      [`Qr::check_invariants`] with `c_Q` given.
//! (Q2) `Qr`: `R` upper triangular with `r_jj > 0`; `‖QR − A_t‖_F ≤ c_Q · m
//!      · u · ‖A_t‖_F` (the tests, against `A_t` in double-double, with the
//!      path norm as in H2); `c_Q = 16` for (Q1) and (Q2) (measured: `8.0`
//!      and `5.4` on `2 × 1` and `3 × 2` after `delete_row`, `≤ 0.6` for `m
//!      ≥ 100`).
//! (Q3) As (H3): a failed operation leaves `Q` and `R` bit for bit as they
//!      were.
//!
//! (R1)–(R2), (G1)–(G5): [`rsvd()`] and [`svds()`], Chapters 12.4 and 12.5
//!      (`c_G = 8`, measured).
//! (K1)–(K4), (E1)–(E3), (I1)–(I3): [`eigsh()`]/[`eigs()`], the small dense
//!      eigensolvers and shift-invert, Chapters 12.6–12.6.2 (`c_K = 8`,
//!      measured in `f64` on runs of few restarts; long runs exceed it and
//!      no form in the number of restarts is guaranteed, Chapter 12.6).
//!
//! The constants `c_H`, `c_Q` come from the measurements of the example
//! `accuracy` with a margin, and live in the tests and in the argument of
//! [`Qr::check_invariants`], not as literals in kernels.
//!
//! ## Decisions of the review (2026-11-19; D-0127)
//! * **S1** — generic `T: Real`, with own small dense routines (internal).
//! * **S2** — [`svds()`] is Golub–Kahan–Lanczos bidiagonalization with thick
//!   restart, its own engine, full two-sided reorthogonalization.
//! * **S3** — `MatRef`/`MatMut` live here.
//! * **S4** — own xoshiro256++, Marsaglia polar method, own `ln` (12.1).
//! * **S5** — `apply(&mut self, …) -> Result`; separate [`AdjointOperator`].
//! * **S6** — names `eigs`, `eigsh`, `svds`, `rsvd`, `range_finder`.
//! * **S7** — Krylov–Schur restart (Stewart 2001) for [`eigs()`] and
//!   [`eigsh()`], one engine for both (Chapter 12.6), not ARPACK's
//!   implicit QR shifts; results compared with ARPACK.
//! * **S8** — thin QR with five operations; Cholesky update/downdate;
//!   Cholesky insert/delete of a variable in v0.2.
//! * **S9** — acceptance criteria rewritten on the pilot's numbers (each
//!   chapter states its own).
//! * **S10** — one `#[non_exhaustive]` [`StatlaError`]; [`Sketch`] and
//!   [`SvdWhich`] `#[non_exhaustive]`; options with public fields and
//!   `Default`; non-convergence is `Err(NoConvergence)` unless
//!   `allow_partial`.

//!
//! ## References
//! Golub & Van Loan, *Matrix Computations*, 4th ed. (2013), §5.1 (Givens,
//! Householder), §6.5 (updating). Gill, Golub, Murray & Saunders, "Methods
//! for modifying matrix factorizations", Math. Comp. 28 (1974). Saunders,
//! "Large-scale linear programming using the Cholesky factorization",
//! Stanford report (1972). Stewart, "The effects of rounding error on an
//! algorithm for downdating a Cholesky factorization", J. IMA 23 (1979).
//! Dongarra, Bunch, Moler & Stewart, *LINPACK Users' Guide*, SIAM (1979),
//! ch. 10 (`dchud`, `dchdd`). Daniel, Gragg, Kaufman & Stewart,
//! "Reorthogonalization and stable algorithms for updating the Gram–Schmidt
//! QR factorization", Math. Comp. 30 (1976). Hammarling & Lucas, "Updating
//! the QR factorization and the least squares problem", MIMS EPrint
//! 2008.111 (2008). Higham, *Accuracy and Stability of Numerical
//! Algorithms*, 2nd ed. (2002), ch. 10 and 19. Halko, Martinsson & Tropp,
//! "Finding structure with randomness", SIAM Review 53 (2011). Golub &
//! Kahan, SIAM J. Numer. Anal. 2 (1965). Baglama & Reichel, "Augmented
//! implicitly restarted Lanczos bidiagonalization methods", SISC 27
//! (2005). Hestenes, J. SIAM 6 (1958); Demmel & Veselić, SIMAX 13 (1992).
//! Blackman & Vigna, ACM TOMS 47 (2021). Marsaglia & Bray, SIAM Review 6
//! (1964). Stewart, "A Krylov–Schur algorithm for large eigenproblems",
//! SIMAX 23 (2001); Wu & Simon, SIMAX 22 (2000); Lehoucq, Sorensen & Yang,
//! *ARPACK Users' Guide* (1998); Bai & Demmel, Linear Algebra Appl. 186
//! (1993); LAPACK `dlahqr`, `dlanv2`, `dlaexc`, `dtrevc`. **All cited from
//! memory**; no source was reopened in tasks L3.2–L3.4 (the xoshiro256++
//! reference outputs were read in the tests of `rand` 0.9.4; ARPACK's tie
//! order in `dngets` is recalled, and checked only against scipy's
//! results on the fixtures).

#![forbid(unsafe_code)]

mod cholesky;
mod eigs;
mod error;
mod jacobi;
mod krylov;
mod operator;
mod qr;
mod random;
mod real;
mod rsvd;
mod schur;
mod shift_invert;
mod small;
mod svds;
mod view;

/// The code blocks of the crate's README, compiled and run as doc-tests
/// (`cargo test --doc`) so that the README never drifts from the API, as in
/// `xingo-sparse` and `xingo-extprec`. The README is not the root
/// documentation — this chapter is.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

pub use cholesky::{CholOptions, Cholesky, DowndateInfo};
pub use eigs::{Complex, Eigen, EigsOptions, EigsStats, SymEigen, Which, eigs, eigsh};
pub use error::StatlaError;
pub use operator::{
    AdjointOperator, FnAdjointOperator, FnOperator, LinearOperator, OperatorError, from_fn,
    from_fns,
};
pub use qr::{Qr, QrOptions};
pub use random::Sketch;
pub use rsvd::{RsvdOptions, Svd, SvdStats, range_finder, rsvd};
pub use shift_invert::{ShiftInvert, SymShiftInvert, eigs_near, eigsh_near};
pub use svds::{SvdWhich, SvdsOptions, svds};
pub use view::{MatMut, MatRef};
pub use xingo_sparse::{DenseBlock, DenseKernel, NaiveDense, Real};
