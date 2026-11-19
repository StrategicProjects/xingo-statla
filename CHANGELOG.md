# Changelog

All notable changes to `xingo-statla` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the crate
follows [Semantic Versioning](https://semver.org/); before 1.0, a minor
version may change the public API.

## [0.1.0] — 2026-11-19

The first release: updatable and matrix-free linear algebra for statistics,
generic in the scalar `Real` of `xingo-sparse` 0.5, its only runtime
dependency.

### Added

* The result and statistics types (`Svd`, `SvdStats`, `SymEigen`,
  `Eigen`, `EigsStats`, `DowndateInfo`) are `#[non_exhaustive]`, so that a
  later field is not a breaking change; the options structs and `Complex`
  are plain structs.
* `MatRef` and `MatMut`: borrowed column-major views with a leading
  dimension, checked at construction; `StatlaError`, the one error type of
  the crate (`#[non_exhaustive]`; Chapter 12.0).
* `Cholesky<T, K = NaiveDense>`: factorization through the `DenseKernel`
  `K` of `xingo-sparse`, rank-1 `update` (LINPACK `dchud`) and
  transactional `downdate` (LINPACK `dchdd`, Saunders 1972) that returns
  `DowndateInfo` and leaves the factor bit for bit unchanged when the
  result would not be positive definite (with one more check, step C3′,
  against a new diagonal that underflows to zero); `from_upper`, solves,
  `log2_det_parts` (the determinant as mantissa and exponent),
  `check_invariants`; `CholOptions` (Chapter 12.2).
* `Qr<T, K = NaiveDense>`: the thin Householder QR with an explicit `Q`
  and five updates — `insert_row`, `delete_row`, `insert_col`,
  `delete_col`, `rank1_update` (Hammarling & Lucas 2008, with DGKS
  reorthogonalization), each transactional; `solve_least_squares`, the
  Q-less `to_cholesky`, `check_invariants`; `QrOptions` (`rank_tol = 0`
  means `n·ε_T`; Chapter 12.3).
* `LinearOperator` and `AdjointOperator` (`apply(&mut self, …) -> Result`,
  with block versions), `OperatorError`, implementations for
  `DenseBlock`, `MatRef`, `CscMatrix`, `CsrMatrix` and `&mut O`, and
  closures through `from_fn`/`from_fns` (Chapter 12.0.5).
* A reproducible generator — SplitMix64, xoshiro256++, the polar method
  with a logarithm in `+ − · /` (proven to `6.4u` in `f64`) — and
  `Sketch` (`Gaussian`, `Rademacher`; `#[non_exhaustive]`): the same bits
  on every platform, the first 64 deviates of seed 0 pinned by a test
  (Chapter 12.1).
* `rsvd` and `range_finder` (Halko, Martinsson & Tropp 2011; `p = 10`, `q =
  2`, seed 0 by default), `RsvdOptions`, `Svd`, `SvdStats` (Chapter 12.4).
* `svds`: thick-restart Golub–Kahan–Lanczos bidiagonalization with full
  two-sided reorthogonalization and an absolute convergence test,
  `SvdsOptions`, `SvdWhich` (`Largest`; `#[non_exhaustive]`; Chapter 12.5).
* `eigsh` and `eigs`: one Krylov–Schur engine (Stewart 2001) with ARPACK's
  `which` (`Which`, `#[non_exhaustive]`, with `from_arpack` and
  `arpack_code`), convergence test and scipy's defaults; locking, purging,
  conjugate pairs never split, exactly `k` values returned; `EigsOptions`,
  `EigsStats`, `SymEigen`, `Eigen`, `Complex` (Chapter 12.6), on small
  dense eigensolvers written here (cyclic Jacobi, Hessenberg + Francis
  double-shift QR, Bai–Demmel reordering, quasi-triangular eigenvectors;
  Chapter 12.6.1).
* Shift-invert: `ShiftInvert` over the `Lu` and `SymShiftInvert` over the
  `LBLᵀ` of `SupernodalLdl` of `xingo-sparse`, with one step of iterative
  refinement per application; `SymShiftInvert::count_below` (the inertia);
  `eigsh_near` and `eigs_near` (Chapter 12.6.2).
* Feature `faer` (off): `FaerDense` of `xingo-sparse` as the kernel `K`.
* The documentation, on docs.rs too, includes the private items: every
  chapter but the root documents a private module.
* The examples `accuracy` (the measurements behind the constants of the
  updatable factorizations) and `windowed_regression`.
* Matrix Market fixtures with ARPACK's answers from scipy, and the script
  that generates them (`tests/data/make_fixtures.py`).

### Known limitations

See "Known limitations" in the README: shift-invert was not tested with a
shift very close to an eigenvalue; no matrix above `400 × 400` and no
strongly non-normal matrix was tested for the eigensolvers; interior
`which` without shift-invert may not converge, as with ARPACK; the
symmetric shift-invert may factorize twice; the suite ran on one macOS
aarch64 machine only; the literature of the chapters is cited from memory;
the logarithm of the generator in double-double has a measured, not
proven, bound.

[0.1.0]: https://github.com/StrategicProjects/xingo-statla/releases/tag/v0.1.0
