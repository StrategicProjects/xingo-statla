# Changelog

All notable changes to `xingo-statla` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the crate
follows [Semantic Versioning](https://semver.org/); before 1.0, a minor
version may change the public API.

## [Unreleased]

### Added

* Example `timing` (issue
  [#8](https://github.com/StrategicProjects/xingo-statla/issues/8)): the
  updates of `Cholesky` and `Qr` in `f64` and double-double, `eigsh`,
  `eigs`, `eigsh_near`, `eigs_near`, `svds` and `rsvd` in `f64` on the
  fixtures of the tests, and the same drivers in double-double next to
  their `f64` runs; every call is timed over 21 calibrated rounds with its
  setup outside the clock (median, minimum, maximum), with the load
  average in the header. No change to the library.

### Documentation

* README section "Measured time", the note on the page and limitation 7
  (issue [#8](https://github.com/StrategicProjects/xingo-statla/issues/8)):
  the tables are now the output of `timing` on an idle Apple M2 Pro, five
  runs, `rsvd` and the double-double drivers included; a shared Xeon
  E5-2699 v4 is kept as orders of magnitude, from the older harnesses.

## [0.2.0] — 2026-11-27

`xingo-sparse` 0.6 (whose types are in this crate's public API: hence a
new minor version), an a posteriori residual for long runs, a more
accurate small eigensolver in `eigsh` (the bits of almost every `eigsh`
result change), shift-invert with one factorization and less memory, and
the issues of the 0.1 limitations closed: every one of #1–#7 and #9–#11
(only [#8](https://github.com/StrategicProjects/xingo-statla/issues/8),
timings, stays open).

### Added

* `SymEigen::residuals(&mut op)` and `Eigen::residuals(&mut op)` (issue
  [#10](https://github.com/StrategicProjects/xingo-statla/issues/10)):
  the residuals `fl(‖A y − θ y‖₂)` of the returned pairs, recomputed with
  one application of `op` per real pair (two per complex one), not
  counted in the statistics. Lemma R of Chapter 12.6 bounds the true
  residual by the recomputed one, `‖A y − θ y‖ ≤ (1 + γ_{2n+1}) ρ̂ +
  γ_3 (‖ŵ‖ + √2 |θ|) + η_op`, with no dependence on the number of
  restarts; `tests/long_runs.rs` asserts it on runs of 983 to 1 358
  restarts (`eigs` 983 and 1 271, `eigsh` 987 and 1 358; release builds). This is the guarantee for long runs (see
  "Changed").
* Lemma L′ of Chapter 12.1 (issue
  [#5](https://github.com/StrategicProjects/xingo-statla/issues/5)):
  the bound of the logarithm of the generator in double-double is now
  proven, `13.62u` (under `m = 1` or `|m − 1| ≥ 2⁻²³⁸`, asserted by the
  tests), and the test asserts `13.7u` instead of the `16u` measured; in
  `f64` the lemma gives `6.30u` (Lemma L: `6.4u`). Each step of the proof
  is checked against `dashu-float` at 448 bits (a new dev-dependency, MIT
  OR Apache-2.0). A step of the earlier proof (`w ≤ w̄`), which failed by
  `3·10⁻¹⁶` relative, is corrected. No bit of the generator changes.
* Tests, no change of the code they test:
  shift-invert with `σ` from `2⁻¹` down to 1 ulp of an eigenvalue (issue
  [#1](https://github.com/StrategicProjects/xingo-statla/issues/1)): with
  the `LBLᵀ` the backward error stays at most `1.3u` and a second
  refinement step changes nothing; with the `Lu` the documented refusal
  near the eigenvalue is the right answer (without it, `η = 3.9·10³u`).
  Fixtures of order 3 000 and 4 000 (`make_fixtures.py --large`, only
  ARPACK's references committed, the matrices rebuilt and checked by
  checksum; `#[ignore]`d for their cost) and strongly non-normal families
  on which the coupling check of `eigs` fires (issue
  [#2](https://github.com/StrategicProjects/xingo-statla/issues/2)).
  Graded and badly scaled families (to `10⁴⁰`) for `Cholesky` and `Qr`,
  and a test that reaches the guard C3′ (issue
  [#6](https://github.com/StrategicProjects/xingo-statla/issues/6)).
  `eigs_near` in double-double, `svds` with many restarts under locking
  (`400 × 300`; `2000 × 1900` `#[ignore]`d), and acceptance of `Cholesky`
  and `eigsh_near` with `K = FaerDense` (issue
  [#7](https://github.com/StrategicProjects/xingo-statla/issues/7)).

### Changed

* Depends on `xingo-sparse` 0.6 (`"0.6.0"`), no longer 0.5: the public
  API takes and returns its types (`Real`, `DenseBlock`, `CscMatrix`,
  `LuOptions`, `KktError` inside `StatlaError::Kkt`). The dev-dependency
  is `xingo-extprec` 0.2.
* The small eigensolver of `eigsh` (step KS3 of Chapter 12.6, the cyclic
  Jacobi, Algorithm SJ of Chapter 12.6.1) updates the diagonal with
  compensated (Kahan) sums, for every `T: Real`, and runs with the
  threshold `ε·2⁻²⁰` (an internal constant) from the first restart (issue
  [#11](https://github.com/StrategicProjects/xingo-statla/issues/11)).
  The residual of `eigsh` grew almost linearly with the number of
  restarts because the relative stopping test of the Jacobi left an
  off-diagonal remainder of order `u‖A‖` that returned with the same sign
  at every restart; Lemma T proves that the truncation part now adds at
  most `R·tol·p·‖A‖`, negligible up to `R ≈ 4.8·10⁸`. **The bits of
  almost every `eigsh` result change** (3 of the 156 `eigsh` runs of the
  test suite keep them), and so may the number of restarts; `eigs`, whose
  small solver is Hessenberg + Francis QR, keeps its bits (120 runs,
  compared by hash). Measured (9 interleaved rounds, loaded machine), the
  small solver of 0.1.0 against this one, on the same code otherwise: worst residual ratio against ARPACK on the
  fixtures `0.88 / 2.76` → `0.48 / 0.45` (small / large), worst
  eigenvalue error `5.5·10⁻¹³` → `5.4·10⁻¹³`; long runs, 1-D Laplacian of order 2 000 `48.6` →
  `1.90`, indefinite tridiagonal of order 2 000 (the worst) `125.5` →
  `13.8`; the four long runs that stagnated now converge; the worst
  constant of `(K2) ≤ c·m·u·√R` `4.32` → `0.159` (asserted `≤ 0.5` in a
  release-only test). The growth `√R` is measured, not proven. Cost:
  time `1.03` / `1.04` of the old solver on the small / large cases,
  `1.12`–`1.14` on long runs. One interior run outside the ARPACK comparison
  (`symrand_300`, `SM`) finds 4 of 6 pairs instead of 5 of 6.
* The residual guarantee of `eigs` and `eigsh`, `(τ + c_K·ncv·u)·‖A‖`
  with `c_K = 8`, is stated for runs of few restarts (issue
  [#9](https://github.com/StrategicProjects/xingo-statla/issues/9)): on
  the Grcar matrix of order 200 `eigs` reaches `12.1`, the excess comes
  from the Krylov relation (K2), which grows with the restarts, and no
  closed form in the number `R` of restarts covered every measured long
  run (a table of 19 long runs is in Chapter 12.6, "Long runs"). For long
  runs the guarantee is a posteriori: recompute the residual with
  `residuals` and bound it by Lemma R. The tests of the Grcar runs keep
  `≤ 4·c_K` and add `≤ 8 + √R` as a regression check, not as a
  guarantee.
* `eigs_near` factorizes `A − σI` with `LuOptions { tol_scale:
  TolScale::Column, ..LuOptions::for_scalar::<T>() }` instead of the
  absolute tolerances (step SG3 of Chapter 12.6.2; issue
  [#1](https://github.com/StrategicProjects/xingo-statla/issues/1)): a
  matrix scaled by `2⁻⁴⁰` now gives the refusals and the values (times
  `2⁻⁴⁰`, exactly) of the unscaled one, which before refused every `σ`;
  on unscaled matrices the refusals measured are unchanged. `ShiftInvert`
  uses the `LuOptions` its caller passes, as before.
* `eigs_near` and `eigsh_near` no longer keep a copy of `A − σI` beside
  its factorization (issue
  [#3](https://github.com/StrategicProjects/xingo-statla/issues/3)): the
  residual of the refinement step is formed from the caller's `A` by the
  routine that builds `A − σI`, with the same bits (step R2′ and Lemma B
  of Chapter 12.6.2; checked bit for bit and by a hash of the drivers'
  results). Memory held during the iteration, 2-D Laplacian: `eigsh_near`
  `n = 62 500` 49.32 → 45.98 MiB, `eigs_near` `n = 22 500` 41.39 → 39.68
  MiB; the peak, set by the factorization, is unchanged. The public
  `ShiftInvert` and `SymShiftInvert`, which have no lifetime, still keep
  it.
* `SymShiftInvert` (and so `eigsh_near`) factorizes `A − σI` once, with
  `KktForm::Indefinite` of `xingo-sparse` 0.6
  ([xingo-sparse#8](https://github.com/StrategicProjects/xingo-sparse/issues/8)),
  instead of up to two analyses and two factorizations with a guessed
  inertia. Same factors bit for bit (974 measured, 929 of which needed the
  second attempt before) and the same errors; a `σ` that is numerically an
  eigenvalue is `StatlaError::Kkt(KktError::ZeroPivot)`.
* The literature of the chapters was checked (issue
  [#4](https://github.com/StrategicProjects/xingo-statla/issues/4))
  against Crossref, the publishers, the texts of Halko, Martinsson & Tropp
  and the Fortran sources of ARPACK and LAPACK; venue, title, parts,
  chapter, section, hypotheses and the order of authors were corrected
  where wrong, and each chapter says which of its references were checked.
  Not verified: subsection and theorem numbers inside the books (Golub &
  Van Loan, Higham, Parlett, Björck), marked so in the chapters.

Migration: depend on `xingo-sparse` 0.6 together with this version (see
its CHANGELOG: `KktForm::expected_inertia` returns `Option<Inertia>`,
`KktForm` and `KktError` are `#[non_exhaustive]`, so a `match` on the
`KktError` inside `StatlaError::Kkt` needs a wildcard arm). No signature
of this crate changes. Code that compares `eigsh` results bit for bit, or
counts its restarts, against 0.1.0 will see other values.

### Known limitations

See "Known limitations" in the README: interior `which` without
shift-invert may not converge, as with ARPACK; the residual guarantee in
`c_K` holds for runs of few restarts, and for long runs only a
posteriori through Lemma R; the public `ShiftInvert` and
`SymShiftInvert` keep a copy of `A − σI`; s390x runs only under
emulation; the subsection and theorem numbers inside the books cited are
not verified; no timing is claimed
([#8](https://github.com/StrategicProjects/xingo-statla/issues/8)).

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

[0.2.0]: https://github.com/StrategicProjects/xingo-statla/releases/tag/v0.2.0
[0.1.0]: https://github.com/StrategicProjects/xingo-statla/releases/tag/v0.1.0
