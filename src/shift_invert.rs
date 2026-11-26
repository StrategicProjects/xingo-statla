//! # Chapter 12.6.2 — Shift-invert: `eigsh_near` over `SupernodalLdl`, `eigs_near` over `Lu`
//!
//! ## Purpose
//! *The `k` eigenvalues nearest a shift `σ`.* Krylov methods find the ends
//! of the spectrum quickly and the interior slowly; ARPACK's mode 3 for the
//! standard problem runs them on `(A − σI)⁻¹`, whose largest eigenvalues
//! `ν = 1/(λ − σ)` are the `λ` nearest `σ`. The factorization comes from
//! `xingo-sparse`: the threshold LU (`Lu<T>`) for a general `A`, and the
//! supernodal `LBLᵀ` (`SupernodalLdl<T, K>`, Bunch–Kaufman pivoting,
//! no regularization) for a symmetric one — whose **inertia** counts, by
//! Sylvester's law, the eigenvalues of `A` below `σ`: a check of
//! completeness that ARPACK does not offer ([`SymShiftInvert::count_below`]).
//!
//! ## Notation
//! `A ∈ R^{n×n}` sparse (CSC); `σ` the real shift; `M = A − σI`; `ν` the
//! eigenvalues of `M⁻¹`; `In(M) = (n₊, n₋, n₀)` its inertia.
//!
//! ## Algorithm SI (symmetric, [`eigsh_near`])
//! SI1. \[Check.\] `A` square and structurally valid, entries and `σ`
//!      finite; the arguments of [`crate::eigsh()`] are validated first.
//! SI2. \[Form.\] The lower triangle of `M` (entries `a_ij`, `i ≥ j`, of the
//!      input — only the lower triangle is read — with `σ` subtracted on the
//!      diagonal; an entry that becomes exactly zero is not stored, as the
//!      CSC invariant requires, and the analysis adds the structural
//!      diagonal).
//! SI3. \[Factorize.\] Analyze with the crate's own AMD and factorize by the
//!      `LBLᵀ` path of `SupernodalLdl` (no regularization: `Regularization::
//!      none()` on an augmented form). That path checks the inertia against
//!      the form's (`KktForm::Augmented { n_primal, n_dual }` expects
//!      `(n_dual, n_primal, 0)`), and the inertia of `M` is not known in
//!      advance: the first attempt declares `(n, 0, 0)` (right whenever
//!      `σ` is below the spectrum); on `WrongInertia { found }` with `found.zero
//!      = 0` the analysis and the factorization are repeated with
//!      `n_primal = found.negative` — at most two factorizations. A zero
//!      pivot (`found.zero > 0`, or `ZeroPivot`) means `σ` is numerically an
//!      eigenvalue: [`StatlaError::Kkt`].
//! SI4. \[Count.\] `count_below = n₋`, the eigenvalues of `A` strictly below
//!      `σ` (Sylvester; exact for the computed factorization, i.e. for a
//!      matrix within the backward error of `LBLᵀ` from `M`).
//! SI5. \[Iterate.\] [`crate::eigsh()`] with `LM` on `x ↦ M⁻¹x`, each
//!      application by Algorithm R below (two solves and one product with
//!      `M`).
//! SI6. \[Map back.\] `λ = σ + 1/ν`; the order (largest `|ν|` first) is
//!      nearest `σ` first; the eigenvectors are those of `M⁻¹`.
//!
//! ## Algorithm SG (general, [`eigs_near`])
//! SG1–SG2 as SI1–SI2 on the whole of `M`. SG3: `Lu<T>` with
//! `LuOptions { tol_scale: TolScale::Column, ..LuOptions::for_scalar::<T>() }`
//! (the defaults of `xingo-sparse` with the pivot and drop tolerances
//! measured against each column of `M`, `τ_j = τ_abs·max_i |m_ij|`; `M` is
//! the "basis" whose columns are those of `M`); singular is
//! [`StatlaError::Lu`]. *Why per column* (task I01, decision of Q2): with
//! the absolute defaults (`τ_abs = 10⁻¹⁰`, `ε_drop = 10⁻¹²`) every shift of
//! a matrix of small norm was refused — all of them for `A` scaled by
//! `2⁻⁴⁰` — and the refusals near an eigenvalue depended on the units of
//! `A`. Per column, a scaling of `A` by a power of two changes no pivot
//! decision: the refusals are those of the unscaled matrix, at `|σ − λ|`
//! below about `τ_abs‖M e_j‖` (measured below, item 2). SG5: [`crate::eigs()`] with `LM` on `x ↦ M⁻¹x`
//! (`ftran`: row-indexed in, column-indexed out; Algorithm R). SG6: `λ = σ + 1/ν`
//! computed by complex division. `1/ν` *conjugates the imaginary part's
//! sign*, so a pair `ν, ν̄` (`+i` first) maps to `λ̄, λ`: the two values,
//! and the imaginary parts of their vectors, are swapped back to keep the
//! `+i` member first; a lone `k`-th member is replaced by its conjugate.
//!
//! ## Algorithm R (one step of iterative refinement per application)
//! *Task L3.4, second round (answer to question Q-L34-4).* Both operators
//! apply `x ↦ M⁻¹x` as
//! R1. \[Solve.\] `y ← F⁻¹x` with the factorization `F ≈ M` (LU `ftran`, or
//!     the `LBLᵀ` solve).
//! R2. \[Residual.\] `r ← x − M y`, in the scalar `T`, with `M` (the whole
//!     `A − σI` for the LU; its lower triangle, applied symmetrically, for
//!     the `LBLᵀ`) stored, or generated from `A` by step R2′ below.
//! R3. \[Correct.\] `y ← y + F⁻¹r`.
//!
//! *What it guarantees.* Without R2–R3 the computed `y` solves `(M + δM) y
//! = x` with `‖δM‖` the backward error of the factorization, which for a
//! threshold LU (`rel_pivot_tol = 0.01`) or a Bunch–Kaufman `LBLᵀ` carries
//! the element growth, `γ·u·‖M‖` with `γ` up to hundreds — and that, not
//! the convergence tolerance, bounded the residual on `A` (measured `114·ncv
//! ·u‖A‖` for `eigs_near`, `4.75` for `eigsh_near`, task L3.4 first round).
//! One step of refinement **in the working precision** makes the solve
//! backward stable with `‖δM‖ = O(n_row · u · ‖M‖)` (componentwise, the row
//! counts of `M`), provided `γ·κ(M)·u` is well below one (Skeel 1980;
//! Higham, *Accuracy and Stability*, ch. 12: fixed-precision refinement
//! improves the *backward* error, not the forward one). That is what the
//! residual on `A` needs: for a converged pair, `‖A y − λ y‖ = ‖M y − y/ν‖ ≤
//! ‖M‖·ρ_ν/|ν| + ‖δM‖`, so a residual in `T` suffices; a residual in higher
//! precision would improve the forward error of each solve, which the
//! eigenvalue problem does not need (the error along the wanted eigenvector
//! is harmless, as in ARPACK's mode 3). It does **not** remove the
//! dependence on `κ(M)`: for `σ` within `γ·u·‖M‖` of an eigenvalue one step
//! may not suffice (and the factorization would usually have failed with
//! a zero pivot). *Measured* (`l34-tables.md`): with R2–R3 the residual on
//! `A` satisfies the criterion of `eigsh`/`eigs`, `≤ (τ + c_K·ncv·u)‖A‖`, on
//! every fixture (runs of few restarts). Long runs, measured in Chapter
//! 12.6 ("Long runs", up to 189 restarts here), stay below `1`.
//!
//! *σ near an eigenvalue* (issue xingo-statla#1; `tests/near_shift.rs`).
//! Measured on tridiagonal matrices of order 201 with the eigenvalue `λ⋆ =
//! 2` exact in `f64` (the 1-D Laplacian, and a nonnormal convection–
//! diffusion operator whose eigenvector matrix has condition `≈ 5·10²`),
//! permuted, with `σ = 2 ± δ`, `δ` from `2⁻¹` down to one ulp:
//!
//! 1. *`LBLᵀ` (`eigsh_near`).* Accepted down to one ulp, refused (zero
//!    pivot, [`StatlaError::Kkt`]) only at `σ = λ⋆`; `η ≤ 1.3u` at every
//!    accepted `σ` after Algorithm R. A second refinement step changes
//!    nothing (`η ≤ 1.2u` after it). The value nearest `σ` comes out
//!    *exact* and its pair meets the residual criterion of 12.6 at every
//!    distance.
//! 2. *Threshold LU (`eigs_near`).* With the pivot tolerance per column
//!    (`τ_j = 10⁻¹⁰·max_i |m_ij|`, columns of norm `≈ 2` here) `A − σI` is
//!    accepted down to `δ = 2⁻³⁶ ≈ 1.5·10⁻¹¹` (`η ≤ 1.4u`, `λ⋆` exact) and
//!    refused ([`StatlaError::Lu`], "singular") from `2⁻⁴⁰` on — the same
//!    refusals as the absolute tolerance of the earlier edition gave on
//!    this matrix. Below that point
//!    one step is *not* enough: with `τ_abs = 0` the LU accepts `σ` down
//!    to one ulp, and `η` reaches `3.9·10³ u` at `δ = 2⁻⁴⁰` (`2.4·10² u`
//!    at `2⁻⁴⁴`), and a second step does not lower it (its correction is
//!    `10⁴` times `y`): fixed-precision refinement converges only while
//!    `γ κ(M) u` is small. So the documented error of the defaults is the
//!    right answer there, not a second step nor a stopping test on the
//!    correction — which would only detect the failure the refusal
//!    already reports. (With the absolute tolerance of the earlier
//!    edition `eigs_near` refused *every* shift of the same matrix scaled
//!    by `2⁻⁴⁰`; per column it behaves as the unscaled one, with the same
//!    refusals and the same results. The `LBLᵀ` of `eigsh_near` already
//!    tests its pivots relatively, `SupernodalOptions::zero_pivot_tol`;
//!    measured: scaled by `2⁻⁴⁰` it gives the same refusals.)
//! 3. *The other pairs.* The `k − 1` pairs farther from `σ` lose accuracy
//!    as `σ` approaches `λ⋆`, whatever the solve: the Krylov process on
//!    `M⁻¹` rounds at the level `u·|ν₁|`, `ν₁ = 1/(λ⋆ − σ)`, so pair `i`
//!    has `‖A y − θ y‖ ≲ (τ + c_K·ncv·u·a_i)‖A‖` with the *amplification*
//!    `a_i = |θ_i − σ|/|θ_1 − σ|` (measured: `≤ 0.1·c_K·a_i` in the
//!    criterion's units, `a_i` up to `1.5·10¹⁴` at one ulp, where the
//!    other values of the Laplacian are still correct to `2·10⁻⁶‖A‖`).
//!    That is the price of shift-invert itself (ARPACK's mode 3 has it
//!    too); `σ` should be put near the wanted values, not on one of them,
//!    when several are wanted to full accuracy.
//!
//! *Where `M` comes from in R2* (issue xingo-statla#3, task I03). The
//! public operators [`ShiftInvert`] and [`SymShiftInvert`] own their data
//! (no lifetime), so they keep `M` (the whole of it, or its lower
//! triangle) beside the factorization. The drivers [`eigs_near`] and
//! [`eigsh_near`] hold the caller's `A` for the whole run, so they do not
//! keep `M`: their private operators drop it once it is factorized and
//! form the products of R2 from `A` and `σ`, entry by entry:
//!
//! R2′. \[Residual from `A`.\] For `j = 1, …, n`, the entries of column `j`
//!      of `M` are generated from column `j` of `A` by the rule of SI2/SG2
//!      — `a_ij` for `i ≠ j` (skipped for `i < j` in the symmetric case),
//!      `fl(a_jj − σ)` on the diagonal, `−σ` there when `A` stores no
//!      diagonal entry, an exact zero skipped — and consumed at once by the
//!      loop of R2.
//!
//! *Lemma B (same bits).* R2′ performs the same floating-point operations
//! in the same order as R2 on the stored `M`. *Proof.* Both build `M` and
//! visit its entries through one routine (`shifted_col`): SI2/SG2 push the
//! entries of column `j` in the order it generates them, and R2′ receives
//! the same sequence; the values are the same roundings `fl(a_jj − σ)` of
//! the same operands. R2 updates `r_i ← r_i − m_ij y_j` per entry, in
//! column order, and, in the symmetric case, adds `m_ij y_i` (`i ≠ j`) to
//! one accumulator per column in the entry order and subtracts it from
//! `r_j` after the column; R2′ executes that same code on that same
//! sequence. Every application therefore returns the bits the stored
//! operator returns, and so do [`eigs_near`]/[`eigsh_near`] (tested:
//! `near_drivers_keep_no_copy_and_the_same_bits`). ∎
//!
//! *Cost.* Per application one more solve and one sparse product with `M`
//! (`2·nnz(M)` flops; the solves dominate), and the `n`-vectors `r`, `d`
//! kept in the operator. The public operators keep `M`, `nnz(M)·(s_T + 4)
//! + 8(n + 1)` bytes (`s_T` the size of `T`, `u32` row indices) — as much
//! as `A` itself; the drivers keep none of it after the factorization.
//! The factorization itself still needs `M` while it runs, so the peak of
//! a driver is not lowered when the factorization is the peak. *Measured*
//! (task I03, counting allocator, 5-point 2-D Laplacian, `k = 6`,
//! `ncv = 20`, `σ = 0.05`): the memory held when the Krylov basis is
//! allocated drops by the bytes of `M` (with its spare capacity) —
//! `eigsh_near`, `n = 62 500`: `49.32 → 45.98` MiB; `eigs_near`, `n =
//! 22 500`: `41.39 → 39.68` MiB — and the peak (`59.6`, `59.8` MiB, the
//! factorization) does not change. The factor, not `M`, dominates: `M` is
//! `3.3` of those `49.3` MiB.
//!
//! ## Invariants
//! (I1) `count_below + n₀ + n₊ = n` with `n₀ = 0` for every operator built.
//! (I2) The applications solve `M y = x` with backward error `η = ‖x −
//!      My‖_∞/(‖M‖_∞‖y‖_∞) = O(u)` after Algorithm R (measured `≤ 0.73u`
//!      on the fixtures, against up to `526u` without it; tested `≤ 2u` by
//!      `tests/arpack.rs`).
//! (I4) Algorithm R runs on every application (one refinement step, never
//!      more, never skipped: the results are reproducible bit for bit).
//! (I3) `eigsh_near`/`eigs_near` return exactly the values `σ + 1/ν` of the
//!      pairs `eigsh`/`eigs` returned on `M⁻¹`, in that order (up to the swap
//!      of SG6 inside a pair).
//!
//! ## References
//! Lehoucq, Sorensen & Yang, *ARPACK Users' Guide* (1998), mode 3;
//! Parlett, *The Symmetric Eigenvalue Problem* (1998), §3.3 (Sylvester's
//! inertia and spectrum slicing); Ericsson & Ruhe, "The spectral
//! transformation Lanczos method", Math. Comp. 35 (1980); Chapters 5.15–5.16
//! of `xingo-sparse` (`LBLᵀ`, Bunch–Kaufman). Cited from memory.

use xingo_sparse::{
    Amd, Col, CscMatrix, DenseKernel, KktError, KktForm, KktSolver, LuError, LuFactor, LuOptions,
    NaiveDense, NumericLdl, Real, Regularization, SparseVec, SupernodalLdl, SupernodalOptions,
    TolScale,
};

use crate::StatlaError;
use crate::eigs::{Complex, Eigen, EigsOptions, SymEigen, Which, eigs, eigsh, validate_for};
use crate::operator::{LinearOperator, OperatorError};
use crate::real::{is_exact_zero, lt};
use crate::schur::Cx;

/// `j` as a `u32` index (a valid `CscMatrix` has dimensions `≤ u32::MAX`,
/// checked by `check_invariants` before any use).
fn idx32(j: usize) -> u32 {
    u32::try_from(j).unwrap_or(u32::MAX)
}

/// The entries of `M = A − σI` column by column, in row order: the stored
/// `M` of the public operators, or `M` generated from `A` (step R2′).
trait ShiftedColumns<T: Real> {
    /// Order of `M`.
    fn order(&self) -> usize;
    /// Call `f(i, m_ij)` for the stored entries of column `j`, in row order.
    fn for_col<F: FnMut(u32, T)>(&self, j: usize, f: F);
}

impl<T: Real> ShiftedColumns<T> for CscMatrix<T> {
    fn order(&self) -> usize {
        self.n_cols()
    }
    fn for_col<F: FnMut(u32, T)>(&self, j: usize, mut f: F) {
        let (rows, vals) = self.col(Col::from_usize(j));
        for (&i, &v) in rows.iter().zip(vals) {
            f(i, v);
        }
    }
}

/// `M = A − σI` generated from the caller's `A` (step R2′): the whole of
/// it, or (`lower_only`) its lower triangle.
#[derive(Debug, Clone, Copy)]
struct FromA<'a, T: Real> {
    a: &'a CscMatrix<T>,
    sigma: T,
    lower_only: bool,
}

impl<T: Real> ShiftedColumns<T> for FromA<'_, T> {
    fn order(&self) -> usize {
        self.a.n_cols()
    }
    fn for_col<F: FnMut(u32, T)>(&self, j: usize, f: F) {
        shifted_col(self.a, self.sigma, j, self.lower_only, f);
    }
}

/// Step R2: `r ← r − M y`, with `m` the whole `M` or (`symmetric`) its
/// lower triangle applied as the symmetric matrix it stands for. One body
/// for the stored and the generated `M` (Lemma B).
fn sub_product<T: Real, S: ShiftedColumns<T>>(m: &S, y: &[T], r: &mut [T], symmetric: bool) {
    for j in 0..m.order() {
        let yj = y[j];
        let mut acc = T::ZERO;
        m.for_col(j, |i, v| {
            let iu = i as usize;
            r[iu] -= v * yj;
            if symmetric && iu != j {
                acc += v * y[iu];
            }
        });
        if symmetric {
            r[j] -= acc;
        }
    }
}

/// The length check of an application of a shift-invert operator.
fn check_apply(n: usize, x: usize, y: usize) -> Result<(), OperatorError> {
    if x == n && y == n {
        Ok(())
    } else {
        Err(OperatorError {
            message: "shift-invert: x and y must have the order of A".to_owned(),
        })
    }
}

/// SI1/SG1 (after the structure check): square, finite entries and shift.
fn check_input<T: Real>(a: &CscMatrix<T>, sigma: T) -> Result<usize, StatlaError> {
    if a.n_rows() != a.n_cols() {
        return Err(StatlaError::InvalidArgument {
            what: "shift-invert: A is not square",
        });
    }
    if let Some(index) = a.val.iter().position(|v| !v.is_finite()) {
        return Err(StatlaError::NonFinite { what: "a", index });
    }
    if !sigma.is_finite() {
        return Err(StatlaError::NonFinite {
            what: "sigma",
            index: 0,
        });
    }
    Ok(a.n_cols())
}

/// SI2/SG2 for column `j`: `push(i, m_ij)` for the stored entries of column
/// `j` of `A − σI` (rows `i ≥ j` only if `lower_only`), in row order. The
/// one rule of both the stored `M` and step R2′ (Lemma B).
fn shifted_col<T: Real, F: FnMut(u32, T)>(
    a: &CscMatrix<T>,
    sigma: T,
    j: usize,
    lower_only: bool,
    mut push: F,
) {
    // An exact zero is not stored (invariant of `CscMatrix`): a diagonal
    // entry that cancels, or `−σ` for `σ = 0`.
    let mut put = |i: u32, v: T| {
        if !is_exact_zero(v) {
            push(i, v);
        }
    };
    let (rows, vals) = a.col(Col::from_usize(j));
    let mut diag_done = false;
    for (&i, &v) in rows.iter().zip(vals) {
        let iu = i as usize;
        if lower_only && iu < j {
            continue;
        }
        if iu > j && !diag_done {
            put(idx32(j), -sigma);
            diag_done = true;
        }
        if iu == j {
            put(i, v - sigma);
            diag_done = true;
        } else {
            put(i, v);
        }
    }
    if !diag_done {
        put(idx32(j), -sigma);
    }
}

/// SI2/SG2: `A − σI`, the whole matrix or its lower triangle.
fn shifted<T: Real>(a: &CscMatrix<T>, sigma: T, lower_only: bool) -> CscMatrix<T> {
    let n = a.n_cols();
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::with_capacity(a.nnz() + n);
    let mut val = Vec::with_capacity(a.nnz() + n);
    col_ptr.push(0);
    for j in 0..n {
        shifted_col(a, sigma, j, lower_only, |i, v| {
            row_idx.push(i);
            val.push(v);
        });
        col_ptr.push(row_idx.len());
    }
    CscMatrix {
        n_rows: n,
        col_ptr,
        row_idx,
        val,
    }
}

/// The threshold LU of `M` with the workspace of Algorithm R (no `M`).
#[derive(Debug)]
struct LuSolver<T: Real> {
    n: usize,
    lu: xingo_sparse::Lu<T>,
    work: SparseVec<T>,
    /// The residual `r` of step R2.
    r: Vec<T>,
    /// The correction `d = F⁻¹r` of step R3.
    d: Vec<T>,
}

impl<T: Real> LuSolver<T> {
    /// SG1–SG3: check, form `M` and factorize it; `M` is returned to the
    /// caller, which keeps it or drops it.
    fn new(
        a: &CscMatrix<T>,
        sigma: T,
        opts: LuOptions,
    ) -> Result<(Self, CscMatrix<T>), StatlaError> {
        let n = check_input(a, sigma)?;
        a.check_invariants()
            .map_err(|e| StatlaError::Lu(LuError::Structure(e)))?;
        let fin = |x: f64| x.is_finite() && !lt(x, 0.0);
        if opts.search_limit == 0
            || !lt(0.0, opts.rel_pivot_tol)
            || lt(1.0, opts.rel_pivot_tol)
            || !fin(opts.abs_pivot_tol)
            || !fin(opts.drop_tol)
            || !fin(opts.update_stability_tol)
            || !lt(0.0, opts.update_multiplier_limit)
        {
            return Err(StatlaError::InvalidArgument {
                what: "ShiftInvert: LuOptions outside their domain",
            });
        }
        let m = shifted(a, sigma, false);
        let mut lu = xingo_sparse::Lu::<T>::new(opts);
        lu.factorize(&m).map_err(StatlaError::Lu)?;
        Ok((
            Self {
                n,
                lu,
                work: SparseVec::new(n),
                r: vec![T::ZERO; n],
                d: vec![T::ZERO; n],
            },
            m,
        ))
    }

    /// `y ← F⁻¹x` by `ftran` (row-indexed in, column-indexed out).
    fn lu_solve(&mut self, x: &[T], y: &mut [T]) {
        self.work = SparseVec::from_dense(x);
        self.lu.ftran(&mut self.work);
        for (i, yi) in y.iter_mut().enumerate() {
            *yi = self.work.get(idx32(i));
        }
    }

    /// Algorithm R with `M` from `m`.
    fn apply<S: ShiftedColumns<T>>(
        &mut self,
        m: &S,
        x: &[T],
        y: &mut [T],
    ) -> Result<(), OperatorError> {
        check_apply(self.n, x.len(), y.len())?;
        // R1.
        self.lu_solve(x, y);
        // R2.
        self.r.copy_from_slice(x);
        sub_product(m, y, &mut self.r, false);
        // R3.
        let (r, mut d) = (core::mem::take(&mut self.r), core::mem::take(&mut self.d));
        self.lu_solve(&r, &mut d);
        for (yi, &di) in y.iter_mut().zip(&d) {
            *yi += di;
        }
        (self.r, self.d) = (r, d);
        Ok(())
    }
}

/// `x ↦ (A − σI)⁻¹ x` for a square sparse `A`, by the threshold LU of
/// `xingo-sparse` (`Lu<T>` over the columns of `A − σI`, factorized once),
/// with one step of iterative refinement per application (Algorithm R: two
/// solves and one product with `A − σI`, which the operator keeps). The
/// operator of [`eigs_near`] (Algorithm SG of Chapter 12.6.2).
#[derive(Debug)]
pub struct ShiftInvert<T: Real = f64> {
    sigma: T,
    solver: LuSolver<T>,
    /// `M = A − σI` (Algorithm R).
    m: CscMatrix<T>,
}

impl<T: Real> ShiftInvert<T> {
    /// Factorize `A − σI`.
    ///
    /// # Errors
    /// [`StatlaError::InvalidArgument`] (not square; options outside the
    /// domain `xingo-sparse` asserts), [`StatlaError::NonFinite`] (`a`,
    /// `sigma`), [`StatlaError::Lu`] (a structurally invalid `a`, or
    /// singular: `σ` is, numerically, an eigenvalue).
    pub fn new(a: &CscMatrix<T>, sigma: T, opts: LuOptions) -> Result<Self, StatlaError> {
        let (solver, m) = LuSolver::new(a, sigma, opts)?;
        Ok(Self { sigma, solver, m })
    }

    /// The shift.
    pub fn sigma(&self) -> T {
        self.sigma
    }
}

impl<T: Real> LinearOperator<T> for ShiftInvert<T> {
    fn nrows(&self) -> usize {
        self.solver.n
    }
    fn ncols(&self) -> usize {
        self.solver.n
    }
    fn apply(&mut self, x: &[T], y: &mut [T]) -> Result<(), OperatorError> {
        self.solver.apply(&self.m, x, y)
    }
}

/// The operator of [`eigs_near`]: [`ShiftInvert`] without the stored `M`,
/// which it generates from the borrowed `A` (step R2′).
#[derive(Debug)]
struct NearLu<'a, T: Real> {
    solver: LuSolver<T>,
    m: FromA<'a, T>,
}

impl<'a, T: Real> NearLu<'a, T> {
    fn new(a: &'a CscMatrix<T>, sigma: T, opts: LuOptions) -> Result<Self, StatlaError> {
        // `M` is dropped here, once factorized.
        let (solver, _) = LuSolver::new(a, sigma, opts)?;
        Ok(Self {
            solver,
            m: FromA {
                a,
                sigma,
                lower_only: false,
            },
        })
    }
}

impl<T: Real> LinearOperator<T> for NearLu<'_, T> {
    fn nrows(&self) -> usize {
        self.solver.n
    }
    fn ncols(&self) -> usize {
        self.solver.n
    }
    fn apply(&mut self, x: &[T], y: &mut [T]) -> Result<(), OperatorError> {
        self.solver.apply(&self.m, x, y)
    }
}

/// The `LBLᵀ` of `M` with its inertia and the workspace of Algorithm R (no
/// `M`).
#[derive(Debug)]
struct LdlSolver<T: Real, K: DenseKernel<T>> {
    n: usize,
    below: usize,
    ldl: SupernodalLdl<T, K>,
    num: NumericLdl<T>,
    /// The residual `r` of step R2.
    r: Vec<T>,
}

impl<T: Real, K: DenseKernel<T>> LdlSolver<T, K> {
    /// SI1–SI4: check, form the lower triangle of `M`, factorize it and
    /// count; the lower triangle is returned to the caller, which keeps it
    /// or drops it.
    fn new(a: &CscMatrix<T>, sigma: T, kernel: K) -> Result<(Self, CscMatrix<T>), StatlaError> {
        let n = check_input(a, sigma)?;
        a.check_invariants()
            .map_err(|e| StatlaError::Kkt(KktError::Structure(e)))?;
        let low = shifted(a, sigma, true);
        let ldl = SupernodalLdl::new(kernel, SupernodalOptions::default());
        let pattern = low.pattern();
        let attempt = |n_primal: usize| {
            let form = KktForm::Augmented {
                n_primal,
                n_dual: n - n_primal,
            };
            let sym = ldl.analyze(&pattern, form, &Amd::default())?;
            ldl.factorize(&sym, &low, Regularization::none())
        };
        // SI3.
        let (num, inertia) = match attempt(0) {
            Ok(r) => r,
            Err(KktError::WrongInertia { found, .. }) if found.zero == 0 => {
                attempt(found.negative).map_err(StatlaError::Kkt)?
            }
            Err(e) => return Err(StatlaError::Kkt(e)),
        };
        debug_assert_eq!(
            inertia.positive + inertia.negative + inertia.zero,
            n,
            "(I1) the inertia counts every pivot"
        );
        Ok((
            Self {
                n,
                below: inertia.negative,
                ldl,
                num,
                r: vec![T::ZERO; n],
            },
            low,
        ))
    }

    /// Algorithm R with the lower triangle of `M` from `low`.
    fn apply<S: ShiftedColumns<T>>(
        &mut self,
        low: &S,
        x: &[T],
        y: &mut [T],
    ) -> Result<(), OperatorError> {
        check_apply(self.n, x.len(), y.len())?;
        // R1.
        y.copy_from_slice(x);
        self.ldl.solve(&self.num, y);
        // R2.
        self.r.copy_from_slice(x);
        sub_product(low, y, &mut self.r, true);
        // R3.
        self.ldl.solve(&self.num, &mut self.r);
        for (yi, &di) in y.iter_mut().zip(&self.r) {
            *yi += di;
        }
        Ok(())
    }
}

/// `x ↦ (A − σI)⁻¹ x` for a symmetric sparse `A` (lower triangle read), by
/// `SupernodalLdl<T, K>` with no regularization; the inertia of `A − σI`
/// is kept: by Sylvester's law it counts the eigenvalues below `σ`
/// ([`SymShiftInvert::count_below`]) — a check ARPACK does not offer. Each
/// application refines once (Algorithm R: two solves and one product with
/// the kept lower triangle of `A − σI`). The operator of [`eigsh_near`]
/// (Algorithm SI of Chapter 12.6.2).
#[derive(Debug)]
pub struct SymShiftInvert<T: Real = f64, K: DenseKernel<T> = NaiveDense> {
    sigma: T,
    solver: LdlSolver<T, K>,
    /// The lower triangle of `M = A − σI` (Algorithm R).
    low: CscMatrix<T>,
}

impl<T: Real, K: DenseKernel<T>> SymShiftInvert<T, K> {
    /// Analyze (own AMD ordering) and factorize `A − σI`.
    ///
    /// # Errors
    /// [`StatlaError::InvalidArgument`] (not square), [`StatlaError::NonFinite`]
    /// (`a`, `sigma`), [`StatlaError::Kkt`] (a structurally invalid `a`; a
    /// zero pivot: `σ` is numerically an eigenvalue).
    pub fn new(a: &CscMatrix<T>, sigma: T, kernel: K) -> Result<Self, StatlaError> {
        let (solver, low) = LdlSolver::new(a, sigma, kernel)?;
        Ok(Self { sigma, solver, low })
    }

    /// The shift.
    pub fn sigma(&self) -> T {
        self.sigma
    }

    /// Number of eigenvalues of `A` strictly below `σ` (negative inertia of
    /// `A − σI`).
    #[must_use]
    pub fn count_below(&self) -> usize {
        self.solver.below
    }
}

impl<T: Real, K: DenseKernel<T>> LinearOperator<T> for SymShiftInvert<T, K> {
    fn nrows(&self) -> usize {
        self.solver.n
    }
    fn ncols(&self) -> usize {
        self.solver.n
    }
    fn apply(&mut self, x: &[T], y: &mut [T]) -> Result<(), OperatorError> {
        self.solver.apply(&self.low, x, y)
    }
}

/// The operator of [`eigsh_near`]: [`SymShiftInvert`] without the stored
/// lower triangle, which it generates from the borrowed `A` (step R2′).
#[derive(Debug)]
struct NearLdl<'a, T: Real, K: DenseKernel<T>> {
    solver: LdlSolver<T, K>,
    low: FromA<'a, T>,
}

impl<'a, T: Real, K: DenseKernel<T>> NearLdl<'a, T, K> {
    fn new(a: &'a CscMatrix<T>, sigma: T, kernel: K) -> Result<Self, StatlaError> {
        // The lower triangle is dropped here, once factorized.
        let (solver, _) = LdlSolver::new(a, sigma, kernel)?;
        Ok(Self {
            solver,
            low: FromA {
                a,
                sigma,
                lower_only: true,
            },
        })
    }
}

impl<T: Real, K: DenseKernel<T>> LinearOperator<T> for NearLdl<'_, T, K> {
    fn nrows(&self) -> usize {
        self.solver.n
    }
    fn ncols(&self) -> usize {
        self.solver.n
    }
    fn apply(&mut self, x: &[T], y: &mut [T]) -> Result<(), OperatorError> {
        self.solver.apply(&self.low, x, y)
    }
}

/// The `k` eigenvalues of the symmetric `A` nearest `σ`: [`eigsh()`] with
/// `LargestMagnitude` on [`SymShiftInvert`], mapped back by `λ = σ + 1/ν`
/// (ARPACK mode 3 for the standard problem), nearest first. Returns also
/// [`SymShiftInvert::count_below`]. Only the lower triangle of `a` is read.
///
/// # Errors
/// As [`eigsh()`] and [`SymShiftInvert::new`].
pub fn eigsh_near<T: Real>(
    a: &CscMatrix<T>,
    k: usize,
    sigma: T,
    opts: &EigsOptions<T>,
) -> Result<(SymEigen<T>, usize), StatlaError> {
    validate_for(
        a.n_rows(),
        a.n_cols(),
        k,
        Which::LargestMagnitude,
        opts,
        true,
    )?;
    let mut op = NearLdl::<T, NaiveDense>::new(a, sigma, NaiveDense)?;
    let below = op.solver.below;
    let mut f = eigsh(&mut op, k, Which::LargestMagnitude, opts)?;
    // SI6.
    for v in &mut f.values {
        *v = sigma + T::ONE / *v;
    }
    Ok((f, below))
}

/// The `k` eigenvalues of the general `A` nearest the real shift `σ`, by
/// [`eigs()`] on [`ShiftInvert`] (`LuOptions::for_scalar::<T>()` with
/// `TolScale::Column`: the singularity test is relative to the columns of
/// `A − σI`, so scaling `A` does not change which shifts are refused), mapped
/// back by `λ = σ + 1/ν`, nearest first; conjugate pairs keep the `+i`
/// member first.
///
/// # Errors
/// As [`eigs()`] and [`ShiftInvert::new`].
pub fn eigs_near<T: Real>(
    a: &CscMatrix<T>,
    k: usize,
    sigma: T,
    opts: &EigsOptions<T>,
) -> Result<Eigen<T>, StatlaError> {
    validate_for(
        a.n_rows(),
        a.n_cols(),
        k,
        Which::LargestMagnitude,
        opts,
        false,
    )?;
    // SG3: the defaults, with every tolerance measured per column.
    let lu_opts = LuOptions {
        tol_scale: TolScale::Column,
        ..LuOptions::for_scalar::<T>()
    };
    let mut op = NearLu::new(a, sigma, lu_opts)?;
    let mut f = eigs(&mut op, k, Which::LargestMagnitude, opts)?;
    // SG6.
    let n = a.n_rows();
    let len = f.values.len();
    let one = Cx::new(T::ONE, T::ZERO);
    let mapped: Vec<Complex<T>> = f
        .values
        .iter()
        .map(|nu| {
            let w = one.div(Cx::new(nu.re, nu.im));
            Complex {
                re: sigma + w.re,
                im: w.im,
            }
        })
        .collect();
    let mut i = 0;
    while i < len {
        let nu = f.values[i];
        if lt(T::ZERO, nu.im) {
            let paired = i + 1 < len && f.values[i + 1].re == nu.re && f.values[i + 1].im == -nu.im;
            if paired {
                f.values[i] = Complex {
                    re: mapped[i].re,
                    im: -mapped[i].im,
                };
                f.values[i + 1] = mapped[i];
                if let Some(vi) = &mut f.vectors_im {
                    for r in 0..n {
                        vi.data.swap(r + i * n, r + (i + 1) * n);
                    }
                }
                i += 2;
                continue;
            }
            // A lone `+i` member: its conjugate.
            f.values[i] = Complex {
                re: mapped[i].re,
                im: -mapped[i].im,
            };
            if let Some(vi) = &mut f.vectors_im {
                for v in &mut vi.data[i * n..(i + 1) * n] {
                    *v = -*v;
                }
            }
        } else {
            f.values[i] = mapped[i];
            if is_exact_zero(nu.im) {
                f.values[i].im = T::ZERO;
            }
        }
        i += 1;
    }
    Ok(f)
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::*;
    use xingo_sparse::Triplets;

    /// The tridiagonal `tridiag(−1, 2, −1)` of order `n`, full or lower.
    fn lap1d(n: usize, lower: bool) -> CscMatrix<f64> {
        let mut t = Triplets::new(n, n);
        for i in 0..n {
            t.push(i as u32, i as u32, 2.0);
            if i + 1 < n {
                t.push((i + 1) as u32, i as u32, -1.0);
                if !lower {
                    t.push(i as u32, (i + 1) as u32, -1.0);
                }
            }
        }
        t.to_csc().unwrap()
    }

    fn exact(n: usize) -> Vec<f64> {
        (1..=n)
            .map(|j| 2.0 - 2.0 * (j as f64 * core::f64::consts::PI / (n as f64 + 1.0)).cos())
            .collect()
    }

    #[test]
    fn count_below_and_nearest_values() {
        let n = 50;
        let lam = exact(n);
        for lower in [true, false] {
            let a = lap1d(n, lower);
            for sigma in [-0.5, 0.3, 1.01, 2.05, 3.9, 4.5] {
                let (f, below) = eigsh_near(&a, 4, sigma, &EigsOptions::default())
                    .unwrap_or_else(|e| panic!("lower {lower} sigma {sigma}: {e:?}"));
                assert_eq!(below, lam.iter().filter(|&&l| l < sigma).count(), "{sigma}");
                let mut near = lam.clone();
                near.sort_by(|x, y| (x - sigma).abs().total_cmp(&(y - sigma).abs()));
                for (g, e) in f.values.iter().zip(&near) {
                    assert!((g - e).abs() < 1e-12, "{sigma}: {:?} {near:?}", f.values);
                }
            }
        }
        // σ an eigenvalue: λ_17 = 2 − 2cos(π/3) = 1 for n = 50, and λ_3 = 2
        // for n = 5 — a zero pivot, an error.
        assert!(matches!(
            eigsh_near(&lap1d(50, true), 4, 1.0, &EigsOptions::default()),
            Err(StatlaError::Kkt(_))
        ));
        let a = lap1d(5, true);
        assert!(matches!(
            SymShiftInvert::<f64>::new(&a, 2.0, NaiveDense),
            Err(StatlaError::Kkt(_))
        ));
        assert!(matches!(
            ShiftInvert::new(&lap1d(5, false), 2.0, LuOptions::default()),
            Err(StatlaError::Lu(_))
        ));
    }

    #[test]
    fn general_shift_invert_maps_pairs_back() {
        // Block diagonal: rotation-scalings and reals; σ near a pair.
        let blocks = [(1.0, 0.5), (2.0, 0.25), (3.0, 0.0), (0.5, 0.0), (2.2, 1.0)];
        let mut t = Triplets::new(8, 8);
        let mut r = 0u32;
        for &(re, im) in &blocks {
            if im == 0.0 {
                t.push(r, r, re);
                r += 1;
            } else {
                t.push(r, r, re);
                t.push(r + 1, r + 1, re);
                t.push(r, r + 1, -im);
                t.push(r + 1, r, im);
                r += 2;
            }
        }
        let a = t.to_csc().unwrap();
        let f = eigs_near(&a, 3, 1.9, &EigsOptions::default()).unwrap();
        let want = [(2.0, 0.25), (2.0, -0.25), (1.0, 0.5)];
        for (g, w) in f.values.iter().zip(want) {
            assert!(
                (g.re - w.0).abs() < 1e-12 && (g.im - w.1).abs() < 1e-12,
                "{:?}",
                f.values
            );
        }
        // Residuals with the returned vectors (A dense here).
        let yr = f.vectors_re.as_ref().unwrap();
        let yi = f.vectors_im.as_ref().unwrap();
        let mut dense = [[0.0; 8]; 8];
        for j in 0..8 {
            let (rows, vals) = a.col(Col::from_usize(j));
            for (&i, &v) in rows.iter().zip(vals) {
                dense[i as usize][j] = v;
            }
        }
        for (c, th) in f.values.iter().enumerate() {
            let (y, z) = (&yr.data[c * 8..(c + 1) * 8], &yi.data[c * 8..(c + 1) * 8]);
            let mut e = 0.0f64;
            for i in 0..8 {
                let (mut sr, mut si) = (0.0, 0.0);
                for j in 0..8 {
                    sr += dense[i][j] * y[j];
                    si += dense[i][j] * z[j];
                }
                e = e
                    .max((sr - (th.re * y[i] - th.im * z[i])).abs())
                    .max((si - (th.re * z[i] + th.im * y[i])).abs());
            }
            assert!(e < 1e-12, "{c}: {e}");
        }
    }

    /// A nonsymmetric matrix with missing diagonal entries, a diagonal entry
    /// equal to the shift (it cancels) and both triangles.
    fn awkward(n: usize, sigma: f64) -> CscMatrix<f64> {
        let mut t = Triplets::new(n, n);
        for i in 0..n {
            match i % 3 {
                0 => {}                                 // no diagonal entry
                1 => t.push(i as u32, i as u32, sigma), // cancels in A − σI
                _ => t.push(i as u32, i as u32, 3.0 + i as f64 / 7.0),
            }
            if i + 1 < n {
                t.push((i + 1) as u32, i as u32, -1.25 + (i % 5) as f64 / 3.0);
                t.push(i as u32, (i + 1) as u32, 0.5 - (i % 4) as f64 / 9.0);
            }
            if i + 4 < n {
                t.push((i + 4) as u32, i as u32, 0.3);
            }
        }
        t.to_csc().unwrap()
    }

    /// Lemma B and issue xingo-statla#3: the private operators of the
    /// drivers keep no copy of `M` and apply bit for bit as the public ones,
    /// so the drivers return the same bits.
    #[test]
    fn near_drivers_keep_no_copy_and_the_same_bits() {
        let x: Vec<f64> = (0..40)
            .map(|i| ((i * 37 % 11) as f64 - 5.0) / 3.0)
            .collect();
        let bits = |v: &[f64]| v.iter().map(|z| z.to_bits()).collect::<Vec<_>>();
        // General: LU.
        for (a, sigma) in [
            (lap1d(40, false), 0.3),
            (lap1d(40, false), 0.0),
            (awkward(40, 0.7), 0.7),
        ] {
            let opts = LuOptions {
                tol_scale: TolScale::Column,
                ..LuOptions::for_scalar::<f64>()
            };
            let mut stored = ShiftInvert::new(&a, sigma, opts).unwrap();
            let mut near = NearLu::new(&a, sigma, opts).unwrap();
            let (mut y1, mut y2) = (vec![0.0; 40], vec![0.0; 40]);
            stored.apply(&x, &mut y1).unwrap();
            near.apply(&x, &mut y2).unwrap();
            assert_eq!(bits(&y1), bits(&y2), "LU σ = {sigma}");
            let o = EigsOptions::default();
            let f1 = eigs(&mut stored, 4, Which::LargestMagnitude, &o).unwrap();
            let f2 = eigs(&mut near, 4, Which::LargestMagnitude, &o).unwrap();
            assert_eq!(f1.stats, f2.stats);
            for (p, q) in f1.values.iter().zip(&f2.values) {
                assert_eq!(
                    (p.re.to_bits(), p.im.to_bits()),
                    (q.re.to_bits(), q.im.to_bits())
                );
            }
            let (v1, v2) = (f1.vectors_re.unwrap(), f2.vectors_re.unwrap());
            assert_eq!(bits(&v1.data), bits(&v2.data));
        }
        // Symmetric: LBLᵀ, from the lower triangle or from the whole matrix
        // (the upper part is skipped), and a shift on a diagonal entry.
        let mut sym = lap1d(40, false);
        for (j, v) in sym.val.iter_mut().enumerate() {
            if *v == 2.0 && j % 2 == 0 {
                *v = 1.5; // diagonal 1.5 at some columns: cancels for σ = 1.5
            }
        }
        for (a, sigma) in [
            (lap1d(40, true), 0.3),
            (lap1d(40, false), 1.01),
            (sym.clone(), 1.5),
            (sym, 0.0),
        ] {
            let mut stored = SymShiftInvert::<f64>::new(&a, sigma, NaiveDense).unwrap();
            let mut near = NearLdl::<f64, NaiveDense>::new(&a, sigma, NaiveDense).unwrap();
            assert_eq!(stored.count_below(), near.solver.below);
            let (mut y1, mut y2) = (vec![0.0; 40], vec![0.0; 40]);
            stored.apply(&x, &mut y1).unwrap();
            near.apply(&x, &mut y2).unwrap();
            assert_eq!(bits(&y1), bits(&y2), "LBLᵀ σ = {sigma}");
            let o = EigsOptions::default();
            let f1 = eigsh(&mut stored, 4, Which::LargestMagnitude, &o).unwrap();
            let f2 = eigsh(&mut near, 4, Which::LargestMagnitude, &o).unwrap();
            assert_eq!(f1.stats, f2.stats);
            assert_eq!(bits(&f1.values), bits(&f2.values));
            assert_eq!(
                bits(&f1.vectors.unwrap().data),
                bits(&f2.vectors.unwrap().data)
            );
        }
    }

    #[test]
    fn errors_are_returned() {
        let a = lap1d(6, true);
        let o = EigsOptions::<f64>::default();
        let rect = CscMatrix::<f64>::zeros(3, 4);
        assert!(matches!(
            eigsh_near(&rect, 1, 0.0, &o),
            Err(StatlaError::InvalidArgument { .. })
        ));
        assert!(matches!(
            eigs_near(&rect, 1, 0.0, &o),
            Err(StatlaError::InvalidArgument { .. })
        ));
        assert!(matches!(
            SymShiftInvert::<f64>::new(&rect, 0.0, NaiveDense),
            Err(StatlaError::InvalidArgument { .. })
        ));
        assert!(matches!(
            eigsh_near(&a, 0, 0.0, &o),
            Err(StatlaError::InvalidArgument { .. })
        ));
        assert!(matches!(
            eigsh_near(&a, 2, f64::NAN, &o),
            Err(StatlaError::NonFinite { what: "sigma", .. })
        ));
        let mut bad = a.clone();
        bad.val[0] = f64::INFINITY;
        assert!(matches!(
            eigsh_near(&bad, 2, 0.0, &o),
            Err(StatlaError::NonFinite {
                what: "a",
                index: 0
            })
        ));
        let mut broken = a.clone();
        broken.row_idx[0] = 99;
        assert!(matches!(
            eigsh_near(&broken, 2, 0.0, &o),
            Err(StatlaError::Kkt(KktError::Structure(_)))
        ));
        assert!(matches!(
            eigs_near(&broken, 2, 0.0, &o),
            Err(StatlaError::Lu(LuError::Structure(_)))
        ));
        let lo = LuOptions {
            rel_pivot_tol: 0.0,
            ..LuOptions::default()
        };
        assert!(matches!(
            ShiftInvert::new(&lap1d(6, false), 0.0, lo),
            Err(StatlaError::InvalidArgument { .. })
        ));
        // Wrong lengths in an application.
        let mut op = SymShiftInvert::<f64>::new(&a, 0.1, NaiveDense).unwrap();
        assert!(op.apply(&[1.0; 5], &mut [0.0; 6]).is_err());
        let mut op = ShiftInvert::new(&lap1d(6, false), 0.1, LuOptions::default()).unwrap();
        assert!(op.apply(&[1.0; 6], &mut [0.0; 7]).is_err());
        assert_eq!(op.sigma(), 0.1);
    }
}
