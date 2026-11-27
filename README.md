# xingo-statla

[![crates.io](https://img.shields.io/crates/v/xingo-statla.svg)](https://crates.io/crates/xingo-statla)
[![docs.rs](https://docs.rs/xingo-statla/badge.svg)](https://docs.rs/xingo-statla)
![MSRV 1.90](https://img.shields.io/badge/MSRV-1.90-blue.svg)
![license: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24382503.svg)](https://doi.org/10.5281/zenodo.24382503)

**Website:** <https://strategicprojects.github.io/xingo-statla/> (overview, measured accuracy, limitations, API reference).

Updatable and matrix-free linear algebra **for statistics**, in pure Rust:
a thin QR and a Cholesky factor that are **updated** when a row, a column
or a rank-1 term arrives or leaves instead of being recomputed; the
randomized SVD of Halko, Martinsson & Tropp and a truncated SVD by
Golub–Kahan–Lanczos bidiagonalization; and ARPACK-style `eigsh`/`eigs` by
Krylov–Schur restarted Lanczos/Arnoldi, with shift-invert over sparse
factorizations — all over a small linear-operator trait, so the matrix can
be a dense block, a sparse matrix or a closure. It belongs to
[Xingó](https://github.com/StrategicProjects/xingo), a linear and
mixed-integer programming solver, but depends on nothing in it except its
sparse linear algebra crate
[`xingo-sparse`](https://crates.io/crates/xingo-sparse) (pure Rust).

Statistical computing keeps asking questions that a dense library answers
only by starting from scratch:

1. *A row arrived (or left); what is the new least-squares fit?* —
   `Qr::insert_row`/`delete_row` in `O(mn)` instead of a refactorization in
   `O(mn²)` (sequential and windowed regression, recursive least squares,
   leave-one-out).
2. *A covariance or information matrix changed by `± x xᵀ`; what is its new
   Cholesky factor?* — `Cholesky::update`/`downdate` in `O(n²)`, and a
   downdate that would leave the matrix indefinite is refused **before**
   the factor is touched.
3. *What are the leading `k` singular triplets of a matrix I can only
   multiply by?* — `rsvd` (randomized, reproducible bit for bit from a
   seed) and `svds` (deterministic, accurate to `ε σ₁` on every value).
4. *What are `k` eigenvalues at one end of the spectrum, or nearest `σ`, of
   a large sparse or implicit operator?* — `eigsh`/`eigs` with ARPACK's
   `which` vocabulary, and `eigsh_near`/`eigs_near` by shift-invert, with
   the count of eigenvalues below `σ` from the inertia of the symmetric
   factorization.

Every algorithm needs only `+ − · /`, `sqrt`, `abs` and the order of the
scalar, so the whole crate is generic in the `Real` trait of `xingo-sparse`:
`f64` by default, and the double-double of
[`xingo-extprec`](https://crates.io/crates/xingo-extprec) (about 32
digits) without one line of extra code.

It is **not** a dense LAPACK (see [`faer`](https://crates.io/crates/faer)
for that: the dense eigensolvers here are small internal routines for the
projected problems), has no complex scalar type (`eigs` returns complex
eigenvalues as pairs of reals), and solves standard eigenproblems only, not
generalized ones (see "Scope" below).

## Four entry points

**1. `Qr` and `Cholesky` with updates** — the thin `Q` is kept explicitly
and the five updates of Hammarling & Lucas (insert or delete a row or a
column, rank-1 `A + u vᵀ`) keep it orthonormal by Givens rotations and
DGKS reorthogonalization; the Cholesky update and downdate are LINPACK's
`dchud`/`dchdd`. A failed operation leaves the factor **bit for bit** as it
was.

```rust
use xingo_statla::{MatRef, Qr, QrOptions};

// y = 1 + 2 s, observed at s = 0, 1, 2, 3; the columns are (1, s),
// stored column-major.
let x = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0];
let mut qr = Qr::factor(MatRef::new(4, 2, 4, &x)?, &QrOptions::default())?;

// The window moves: the oldest observation leaves, s = 4 arrives.
qr.delete_row(0)?;
qr.insert_row(3, &[1.0, 4.0])?;

let y = [3.0, 5.0, 7.0, 9.0]; // at s = 1, 2, 3, 4
let mut beta = [0.0; 2];
let residual = qr.solve_least_squares(&y, &mut beta)?;
assert!((beta[0] - 1.0).abs() < 1e-14 && (beta[1] - 2.0).abs() < 1e-14);
assert!(residual < 1e-14);
assert!(qr.check_invariants(16.0).is_ok()); // ‖QᵀQ − I‖ ≤ 16 m u
# Ok::<(), Box<dyn std::error::Error>>(())
```

```rust
use xingo_statla::{CholOptions, Cholesky, MatRef, StatlaError};

// A = [4 2 0; 2 5 1; 0 1 3], symmetric positive definite.
let a = [4.0, 2.0, 0.0, 2.0, 5.0, 1.0, 0.0, 1.0, 3.0];
let mut chol = Cholesky::factor(MatRef::new(3, 3, 3, &a)?, &CholOptions::default())?;
let before = chol.upper().to_block().data;

// A + x xᵀ, then back: the factor returns to within a few ulps.
let x = [1.0, -1.0, 0.5];
chol.update(&x)?;
let info = chol.downdate(&x)?;
assert!(info.margin > 0.0); // 1 − ‖R⁻ᵀx‖², the distance to indefiniteness
for (p, q) in before.iter().zip(&chol.upper().to_block().data) {
    assert!((p - q).abs() <= 1e-14 * p.abs().max(1.0));
}

// A − 10 x xᵀ is indefinite: refused, and the factor is untouched.
let snapshot = chol.upper().to_block().data;
let big = x.map(|v| v * 10.0_f64.sqrt());
assert!(matches!(chol.downdate(&big), Err(StatlaError::DowndateIndefinite { .. })));
assert_eq!(snapshot, chol.upper().to_block().data);
# Ok::<(), Box<dyn std::error::Error>>(())
```

**2. `rsvd` and `svds`** — the randomized SVD (Gaussian or Rademacher
sketch, oversampling `p = 10`, `q = 2` power iterations, re-orthonormalized
at every half step) has the expected error of Halko, Martinsson & Tropp,
and its random numbers come from the crate's own xoshiro256++, so a seed
gives the same bits on every platform. `svds` is a thick-restart
Golub–Kahan–Lanczos bidiagonalization with full reorthogonalization on both
sides: every returned value is within `(τ + 8·ncv·u)·σ₁` of a singular
value.

```rust
use xingo_statla::{RsvdOptions, SvdsOptions, rsvd, svds};
use xingo_sparse::DenseBlock;

// A 60 × 40 matrix with singular values 5, 4, 3, 2, 1 and the rest zero.
let mut a = DenseBlock::<f64>::zeros(60, 40);
for (i, s) in [5.0, 4.0, 3.0, 2.0, 1.0].into_iter().enumerate() {
    a.set(2 * i, 3 * i, s);
}

let t = svds(&mut a, 3, &SvdsOptions::default())?;
for (got, want) in t.s.iter().zip([5.0, 4.0, 3.0]) {
    assert!((got - want).abs() <= 1e-13 * 5.0);
}

// Exactly rank 5: the randomized SVD recovers it to rounding.
let r = rsvd(&mut a, 5, &RsvdOptions::default())?;
assert!((r.s[4] - 1.0).abs() <= 1e-13);
// Same seed, same bits.
let again = rsvd(&mut a, 5, &RsvdOptions::default())?;
assert_eq!(r.s, again.s);
# Ok::<(), Box<dyn std::error::Error>>(())
```

**3. `eigsh` and `eigs`** — one Krylov–Schur engine (Stewart 2001) for the
symmetric and the general problem, with ARPACK's `which` (`LM SM LA SA BE`
and `LM SM LR SR LI SI`), its convergence test `‖r‖ ≤ tol·max(ε^{2/3},
|θ|)`, scipy's defaults, locking of converged Schur blocks, and conjugate
pairs never split. The operator can be a closure: here the second
difference matrix of order 100, never formed.

```rust
use xingo_statla::{EigsOptions, OperatorError, Which, eigsh, from_fn};

let n = 100;
// y = T x with T = tridiag(−1, 2, −1); λ_j = 2 − 2 cos(jπ/(n + 1)).
let mut t = from_fn(n, n, |x: &[f64], y: &mut [f64]| -> Result<(), OperatorError> {
    for i in 0..x.len() {
        let left = if i > 0 { x[i - 1] } else { 0.0 };
        let right = if i + 1 < x.len() { x[i + 1] } else { 0.0 };
        y[i] = 2.0 * x[i] - left - right;
    }
    Ok(())
});
let top = eigsh(&mut t, 4, Which::LargestAlgebraic, &EigsOptions::default())?;
for (j, got) in top.values.iter().enumerate() {
    let k = (n - j) as f64; // the largest first
    let want = 2.0 - 2.0 * (k * std::f64::consts::PI / (n as f64 + 1.0)).cos();
    assert!((got - want).abs() <= 1e-12 * 4.0);
}
assert_eq!(top.vectors.as_ref().map(|v| (v.rows, v.cols)), Some((n, 4)));
# Ok::<(), Box<dyn std::error::Error>>(())
```

```rust
use xingo_statla::{EigsOptions, OperatorError, Which, eigs, from_fn};

// Block diagonal with blocks [a −1; 1 a], a = 0, 0.1, …, 4.9: the
// eigenvalues are a ± i; the largest real part is the pair 4.9 ± i.
let n = 100;
let mut a = from_fn(n, n, |x: &[f64], y: &mut [f64]| -> Result<(), OperatorError> {
    for b in 0..x.len() / 2 {
        let d = b as f64 / 10.0;
        y[2 * b] = d * x[2 * b] - x[2 * b + 1];
        y[2 * b + 1] = x[2 * b] + d * x[2 * b + 1];
    }
    Ok(())
});
let e = eigs(&mut a, 2, Which::LargestReal, &EigsOptions::default())?;
assert!((e.values[0].re - 4.9).abs() < 1e-12 && (e.values[0].im - 1.0).abs() < 1e-12);
assert!((e.values[1].re - 4.9).abs() < 1e-12 && (e.values[1].im + 1.0).abs() < 1e-12);
# Ok::<(), Box<dyn std::error::Error>>(())
```

**4. Shift-invert** — `eigsh_near` and `eigs_near` run the same engine on
`(A − σI)⁻¹` (ARPACK's mode 3), factorized by the `LBLᵀ` of
`SupernodalLdl` (symmetric) or the threshold `Lu` (general) of
`xingo-sparse`, with one step of iterative refinement per application so
that each solve is backward stable. The symmetric driver also returns how
many eigenvalues lie below `σ` — Sylvester's law of inertia, a completeness
check ARPACK does not offer.

```rust
use xingo_statla::{EigsOptions, eigsh_near};
use xingo_sparse::{Col, CscMatrix, Row};

// The same T = tridiag(−1, 2, −1), now stored (only the lower triangle is read).
let n = 100;
let mut entries = Vec::new();
for i in 0..n {
    entries.push((Row::from_usize(i), Col::from_usize(i), 2.0));
    if i + 1 < n {
        entries.push((Row::from_usize(i + 1), Col::from_usize(i), -1.0));
    }
}
let t = CscMatrix::from_triplets(n, n, &entries)?;

let (near, below) = eigsh_near(&t, 3, 1.0, &EigsOptions::default())?;
// λ_j < 1 ⇔ j < (n + 1)/3: 33 eigenvalues lie below σ = 1.
assert_eq!(below, 33);
let lambda = |j: usize| 2.0 - 2.0 * (j as f64 * std::f64::consts::PI / (n as f64 + 1.0)).cos();
// Nearest σ first: λ_34 ≈ 1.026, λ_33 ≈ 0.964, λ_35 ≈ 1.088.
for (got, j) in near.values.iter().zip([34, 33, 35]) {
    assert!((got - lambda(j)).abs() <= 1e-12);
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

Every matrix argument is a `LinearOperator` (and an `AdjointOperator` for
the SVDs): `DenseBlock`, the borrowed column-major view `MatRef`,
`CscMatrix` and `CsrMatrix` of `xingo-sparse`, a closure through
`from_fn`/`from_fns`, or your own type. The programs in
[`examples/`](examples/) are `accuracy` (the measurements behind the
constants of the updatable factorizations; Markdown tables) and
`windowed_regression` (2 000 steps of a sliding-window fit, checked
against refitting from scratch).

## Guarantees, and what was measured

Each guarantee is the stated criterion of a chapter of the sources, checked
by the test suite; the constants `c_H`, `c_Q`, `c_G`, `c_K` were
**measured** and then fixed with a margin, so they are not theorems. `u`
is the unit roundoff of the scalar (`2⁻⁵³` for `f64`), `n`, `m` the
dimensions, `ncv` the Krylov dimension.

| what | criterion the tests enforce | worst measured |
|---|---|---|
| `Cholesky`: backward error after any sequence of updates and downdates | `‖RᵀR − A_t‖_F ≤ 8·n·u·‖A_t‖_F` (`c_H = 8`) | `4.0·n·u` (at `n = 1`); `≤ 0.06·n·u` for `n ≥ 50` |
| `Cholesky`: `update` then `downdate` (and back) | within `10·n·u` of the original factor | `4.9·n·u` (`n = 3`); `≤ 0.05·n·u` for `n ≥ 50` |
| `Cholesky`: forward error against a refactorization | `≤ 8·κ(A_t)·u` per instance; `≤ 10⁻¹²` where `κ ≤ 10⁶` (maximum of 20 seeds, `n = 100, 200`) | — |
| `Qr`: orthogonality and backward error | `‖QᵀQ − I‖_F`, `‖QR − A_t‖_F/‖A_t‖_F ≤ 16·m·u` (`c_Q = 16`) | `8.0` and `5.4·m·u` (on `2 × 1`, `3 × 2`); `≤ 0.6·m·u` for `m ≥ 100` |
| `Qr`/`Cholesky`: a failed operation | the factor unchanged, bit for bit | — |
| `rsvd`: the HMT expected error | the mean over 40 seeds below the bound | 0.19–0.63 of the bound (worst single run 0.66) |
| `svds`: residual of every triplet | `‖(Av − σu; Aᵀu − σv)‖ ≤ (τ + 8·ncv·u)·σ₁` (`c_G = 8`) | `2.30·ncv·u·σ₁` |
| `svds`: singular values, against a certified construction | `≤ 10⁻¹³·σ₁` where the gap is `≥ 10⁻⁸σ₁` | `3.6·10⁻¹⁵·σ₁` |
| `eigs`: residual of every pair | `‖Ay − θy‖ ≤ (τ + 8·ncv·u)·‖A‖` (`c_K = 8`) on runs of few restarts | `2.44·ncv·u·‖A‖` on the fixtures; long runs (hundreds of restarts) exceed it, up to `102.9·ncv·u·‖A‖` after 3000 restarts (Chapter 12.6, "Long runs") |
| `eigsh`: residual of every pair | `‖Ay − θy‖ ≤ (τ + 8·ncv·u)·‖A‖` (`c_K = 8`) on runs of few restarts | `2.44·ncv·u·‖A‖` on the fixtures; long runs exceed it, up to `13.8·ncv·u·‖A‖` after 3548 restarts since its small eigensolver compensates its diagonal and runs with a tighter threshold (`125.5` after 2983 restarts before; Chapter 12.6, "Long runs") |
| `eigsh`/`eigs`/shift-invert: eigenvalues, against ARPACK | `≤ 10⁻¹⁰` relative where the gap is `≥ 10⁻⁶‖A‖` | `9.2·10⁻¹⁴` |
| shift-invert: residual on `A` | as `eigsh`/`eigs` | `2.23·ncv·u·‖A‖` (`114` without the refinement step) |
| shift-invert: backward error of each solve | `≤ 2u` | `0.73u` (up to `526u` without refinement) |
| `svds` against `scipy.sparse.linalg.svds` | `≤ 10⁻¹⁰` relative on the `σᵢ ≥ 10⁻³σ₁` | `2.0·10⁻¹⁵` |
| `count_below` of `eigsh_near` | equal to the dense count | equal on every fixture |

The Cholesky and QR rows are over random matrices with prescribed
condition numbers `κ` from `10` to `10¹²` (Haar-distributed singular
vectors), `n ≤ 200`, in `f64` and in double-double. The `svds` rows are
over eleven matrices with prescribed spectra (geometric down to `10⁻¹⁴`,
rank-deficient, clustered to `10⁻⁸`, multiple values, `2000 × 200`, wide,
sparse); the eigenvalue rows over six Matrix Market matrices of order 200
to 400 (symmetric random, 2-D Laplacian, graded tridiagonal,
convection–diffusion, nonsymmetric random, and a `300 × 200` rectangle for
`svds`), with ARPACK's answers from scipy committed as fixtures.

Two properties of the guarantees a user should know:

* **`svds` and the symmetric eigenvalues are accurate in absolute terms.**
  Small singular values carry an error of order `ε σ₁`, not `ε σᵢ` (a
  relative test is unattainable for them and is not offered); for a
  general matrix the eigenvalue error is the residual times the
  eigenvalue's condition number.
* **A single-vector Krylov method can miss a copy of a multiple value**:
  the value returned is always within the bound of *some* singular value
  or eigenvalue, but not necessarily of the `i`-th when a multiplicity is
  missed — as in ARPACK.

And of the implementation:

* **Errors, not panics.** One `#[non_exhaustive]` error type,
  `StatlaError`; no function panics on bad input. Non-convergence is
  `Err(NoConvergence)` unless `allow_partial` asks for the converged part.
* **Deterministic, the same bits everywhere.** The starting vectors and
  sketches come from the crate's own generator (SplitMix64 → xoshiro256++
  → the polar method of Marsaglia & Bray, with a logarithm written in
  `+ − · /`), never from `libm` or `rand`; loop orders are fixed; no
  threads. The first 64 deviates of seed 0 are pinned by a test, and
  changing them is a breaking change.
* **No `unsafe`** (`#![forbid(unsafe_code)]`) and no C.

## Features

* `faer` (off) — enables the feature of the same name of `xingo-sparse`,
  whose `FaerDense` (`f64` only) can then be the dense kernel `K` of
  `Cholesky<f64, FaerDense>` and `Qr<f64, FaerDense>`. It brings in
  [`faer`](https://crates.io/crates/faer) (pure Rust, MIT).
* Without it the only dependency is `xingo-sparse` with
  `default-features = false`, which has no dependency of its own.

## Scope

In: `MatRef`/`MatMut` (column-major views with a leading dimension);
`Cholesky` (factor, update, transactional downdate, solves, determinant
as mantissa and exponent); the thin `Qr` with explicit `Q` and five
updates, least squares, the Q-less `to_cholesky`; the operator traits and
their implementations for dense, sparse and closures; `rsvd` and
`range_finder`; `svds` (the largest values); `eigsh`, `eigs`,
`eigsh_near`, `eigs_near`, `ShiftInvert`, `SymShiftInvert`.

Not in 0.2, and additive when they come: inserting or deleting a variable
of a `Cholesky` factor; the smallest singular values in `svds`
(`SvdWhich` is `#[non_exhaustive]` for them); a blocked (WY) Householder
QR for large factorizations; complex shifts.

Out, by decision: generalized eigenproblems `A x = λ B x`; complex
matrices; dense eigensolvers or SVDs as a public API (the small ones are
internal); `rand` or `libm` as dependencies.

## How it was verified

* **Exact or certified references.** The matrices `A_t` that a sequence
  of updates defines are accumulated in double-double; the `svds`
  references are matrices built in double-double by Householder
  reflectors and rounded once, so their singular values are the
  prescribed ones to `u‖A‖_F` (Weyl), cross-checked against a one-sided
  Jacobi SVD in double-double; the eigenvalue references are ARPACK's
  answers through `scipy.sparse.linalg` (scipy 1.18.1, numpy 2.5.5) on six
  matrices that a committed script generates (`tests/data/make_fixtures.py`;
  nothing downloaded); the inertia count is compared with a dense count.
* **Two scalars.** The suite of the updatable factorizations runs in full
  in `f64` and in double-double; `rsvd`, `svds`, `eigsh`, `eigs` and
  `eigsh_near` have smoke tests in double-double (`svds` values to
  `4·10⁻³²`), and so has `eigs_near` since 0.2.
* **Larger, non-normal, graded and near-singular cases** (0.2). Fixtures
  of order 3 000 and 4 000 against ARPACK (run with `--ignored`);
  strongly non-normal families (Jordan-like blocks, a convection-dominated
  operator, Grcar) on which the coupling check of `eigs` fires; graded and
  badly scaled matrices (to `10⁴⁰`) for `Cholesky` and `Qr`, with a test
  that reaches the guard against an underflowing diagonal; shift-invert
  with `σ` down to 1 ulp of an eigenvalue; the `faer` kernel in the
  acceptance tests of `Cholesky` and `eigsh_near`.
* **Mutation testing.** Deliberate defects were planted in the algorithms
  and the suite was rerun: 7 of 8 caught for the factorizations, 9 of 11
  for the SVDs, 13 of 14 for the eigensolvers, plus three more planted by
  the maintainer, all caught. The survivors are listed as limitations
  below when they matter (two of the SVD ones change nothing measurable:
  the orthonormalization of the half step of the power scheme for `q ≤ 6`,
  and a relative instead of absolute convergence test).
* **Defects found by the tests and fixed before release.** In the
  one-sided Jacobi SVD a column whose squared norm underflowed was rotated
  forever, and the accumulated rotations drifted from orthogonality
  (`1.3·10⁻¹⁴`, now `8·10⁻¹⁶`); in Krylov–Schur the symmetric projected
  matrix had to be mirrored for `tol = 0` to converge near `θ = 0`, and
  each restart lost about `u` of orthogonality cumulatively (`85u` in 30
  restarts) until one Gram–Schmidt pass was added after it.
* **Platforms.** The suite ran on macOS aarch64, in debug and release, on
  stable and on Rust 1.90, and in release as `wasm32-wasip1` under
  wasmtime on the same machine; the CI of the stand-alone repository runs
  it on x86_64 Linux and Windows, aarch64 Linux and macOS, i686 Linux,
  s390x Linux (big-endian, under QEMU, release only) and `wasm32-wasip1`
  (under wasmtime).

## Known limitations

What is still open in 0.2, each in one sentence. The ones that can be
closed have an issue.

1. **Interior `which` without shift-invert** (`SM` on an indefinite
   matrix, `SI` on a complex spectrum) may not converge, as with ARPACK;
   use `eigsh_near`/`eigs_near`.
2. **Long runs:** the residual criterion `(τ + 8·ncv·u)·‖A‖` of `eigs`
   and `eigsh` holds on runs of few restarts; after hundreds of restarts
   it can be exceeded (measured up to `102.9` for `eigs` and `13.8` for
   `eigsh`, in units of `ncv·u·‖A‖`), no closed form in the number of
   restarts is guaranteed, and the guarantee is a posteriori: recompute
   the residuals with `SymEigen::residuals`/`Eigen::residuals` and bound
   the true ones by Lemma R (Chapter 12.6).
3. **Shift-invert next to an eigenvalue:** the symmetric driver keeps one
   refinement step to within 1 ulp of an eigenvalue (measured backward
   error `≤ 1.3u`); the general one refuses a `σ` very close to an
   eigenvalue (the `Lu` declares `A − σI` singular), which is the
   documented answer.
4. **Memory of the public shift-invert operators:** `ShiftInvert` and
   `SymShiftInvert` keep a copy of `A − σI` beside its factorization (as
   much memory as `A`); `eigs_near` and `eigsh_near` do not.
5. **Hardware:** s390x runs under emulation (QEMU), not on hardware, and
   no x86_64 CPU without SSE2 or with AVX-512 was tried.
6. **Book references:** the literature in the chapters was checked
   against Crossref, the publishers, the HMT texts and the ARPACK and
   LAPACK sources, but the subsection and theorem numbers *inside the
   books* (Golub & Van Loan, Higham, Parlett, Björck) could not be checked
   and are marked so in the chapters.
7. **No timing is claimed:** nothing was timed on an idle machine ([#8](https://github.com/StrategicProjects/xingo-statla/issues/8)).

## Design notes

* **Literate.** Every module is a chapter: notation, the algorithm in
  numbered steps, invariants (each a `debug_assert!` or a
  `check_invariants()`), the measured constants, divergences from the
  references, references. The crate documentation *is* the text: start at
  the crate root on docs.rs (Chapter 12). Every chapter but the root is
  the documentation of a private module (the public items are re-exported
  at the root), so docs.rs builds with `--document-private-items`;
  private items are marked as such. Labels such as L3.4 are tasks of the
  plan of the Xingó solver, and D-xxxx entries of its decision log.
* **Read, not copied.** ARPACK, LAPACK and scipy were the references and,
  through scipy, the test oracle; `faer` was read for what it already
  offers; the defaults of scipy, scikit-learn, RSpectra and `irlba` that
  the options follow are recalled, not checked. The code is written from
  the papers and includes no code from any of them.
* **Small and generic.** Givens rotations, Householder reflectors, DGKS
  Gram–Schmidt, one-sided and cyclic Jacobi, Hessenberg reduction, Francis
  double-shift QR, Bai–Demmel block swaps and quasi-triangular
  eigenvectors are written here, generic in `T: Real`; the dense kernel of
  `xingo-sparse` enters only the initial Cholesky factorization.

## Minimum supported Rust version

1.90 (edition 2024), tested together with stable.

## Repository, issues and citation

The crate is developed inside the Xingó monorepo
([StrategicProjects/xingo](https://github.com/StrategicProjects/xingo),
directory `crates/statla`), where every change is validated together with
the crates it depends on. The stand-alone repository,
[StrategicProjects/xingo-statla](https://github.com/StrategicProjects/xingo-statla),
is a read-only mirror of that directory (`git subtree split`, full
history), updated at every release; it exists so that the library can be
archived and cited on its own. Issues and pull requests about the crate are
welcome in either repository. Each release of the mirror is archived on
Zenodo: the concept DOI [10.5281/zenodo.24382503](https://doi.org/10.5281/zenodo.24382503)
always resolves to the latest version, each release has its own DOI on its
release page, and `CITATION.cff` has the citation metadata.

## Authorship, credits and disclaimer

`xingo-statla` is written and maintained by **André Leite** as part of the
[Xingó](https://github.com/StrategicProjects/xingo) project, developed at
[StrategicProjects](https://github.com/StrategicProjects).

The algorithms are those of their authors, implemented here from the
published descriptions: LINPACK's `dchud`/`dchdd` (Dongarra, Moler, Bunch
& Stewart) and Saunders (the Cholesky update and downdate); Hammarling &
Lucas (the QR updates); Daniel, Gragg, Kaufman & Stewart (Gram–Schmidt
with reorthogonalization); Halko, Martinsson & Tropp (the randomized SVD);
Golub & Kahan (bidiagonalization) and Baglama & Reichel (its thick
restart); Stewart (the Krylov–Schur algorithm); Francis (the double-shift
QR); Bai & Demmel (swapping blocks of the real Schur form); Hestenes and
Demmel & Veselić (Jacobi methods); Blackman & Vigna (xoshiro256++);
Marsaglia & Bray (the polar method). ARPACK (Lehoucq, Sorensen & Yang),
LAPACK and scipy were the references and, through scipy, the test oracle.
Each chapter of the sources lists its references.

This is an independent project: it is not affiliated with, maintained by
or endorsed by the authors of ARPACK, LAPACK, scipy, numpy, `faer`,
RSpectra, `irlba` or scikit-learn, and it includes no code from them
(`faer` is linked only through the optional `faer` feature). All names and
trademarks belong to their owners. The software is provided "as is",
without warranty of any kind, under the licenses below.

## License

Licensed under either of the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
or the [MIT license](https://opensource.org/licenses/MIT), at your option.
