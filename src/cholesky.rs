//! # Chapter 12.2 — Cholesky with rank-1 update and downdate
//!
//! ## Purpose
//! A covariance, information or Gram matrix `A` changes by `± x xᵀ` — an
//! observation enters or leaves a recursive least-squares fit, a Kalman
//! filter absorbs a measurement, a Gaussian-process refit drops a point —
//! and its upper Cholesky factor `R` (`RᵀR = A`) must follow in `O(n²)`
//! instead of the `O(n³)` of a refactorization. [`Cholesky`] keeps `R`
//! under `update` (`A + x xᵀ`) and `downdate` (`A − x xᵀ`), with solves,
//! the whitening `R⁻ᵀ b` and the log-determinant.
//!
//! ## Notation
//! `A ∈ T^{n×n}` symmetric positive definite (only its upper triangle is
//! read); `R` upper triangular with `r_jj > 0`; `x ∈ T^n`; `u` the unit
//! roundoff of `T`; `A_t` the matrix the sequence of operations defines.
//! A rotation `G = (c, s)` acts as `(p, q) ↦ (c p + s q, c q − s p)`
//! (Chapter 12.0.4).
//!
//! ## Algorithm F (factor)
//! F1. \[Check.\] `A` square, its upper triangle finite, the options valid.
//! F2. \[Factor.\] Copy the upper triangle into the lower triangle of a
//!     block `W` (`w_ij = a_ji`, `i ≥ j`) and run `K::potrf` (Algorithm P of
//!     Chapter 5.16): `W = L Lᵀ`; a nonpositive pivot at `j` is
//!     [`StatlaError::NotPositiveDefinite`]` { column: j }`.
//! F3. \[Pivot test.\] Reject column `j` unless `l_jj² > pivot_tol · a_jj`
//!     (and `l_jj` is finite and positive, whatever the kernel).
//! F4. \[Transpose.\] `R ← Lᵀ`, strict lower triangle zero.
//!
//! ## Algorithm U (update, `A + x xᵀ`; LINPACK `dchud`)
//! U0. \[Check.\] `x.len() = n` and `x` finite — the only failures, checked
//!     before any write (H3).
//! U1. \[Loop.\] For `j = 0, …, n − 1`: `ξ ← x_j`; apply the rotations `G_0
//!     … G_{j−1}` already found to the pairs `(r_ij, ξ)`, `i = 0, …, j − 1`,
//!     in that order: `(r_ij, ξ) ← G_i (r_ij, ξ)`.
//! U2. \[Rotate.\] Find `G_j` with `G_j (r_jj, ξ) = (ρ, 0)`, `ρ ≥ 0`
//!     (Algorithm G of 12.0.4, the safe `hypot`); `r_jj ← ρ`.
//! Cost `2n²` flops (`≈ 3n²` with the scaling of G2). Since `ρ ≥ r_jj >
//! 0`, (H1) is kept.
//!
//! ## Algorithm C (downdate, `A − x xᵀ`; LINPACK `dchdd`, Saunders 1972)
//! C0. \[Check.\] `x.len() = n`, `x` finite.
//! C1. \[Solve.\] `Rᵀ a = x` (forward substitution).
//! C2. \[Test.\] `α² ← 1 − ‖a‖²` (‖a‖ by Algorithm N); if `α²` is not
//!     finite or `α² ≤ downdate_tol`, return
//!     [`StatlaError::DowndateIndefinite`] — **before any change to `R`**
//!     (H3 for free). (`RᵀR − xxᵀ = Rᵀ(I − aaᵀ)R` is positive definite iff
//!     `‖a‖ < 1`.)
//! C3. \[Rotations.\] `α ← √α²`; for `i = n − 1, …, 0`: `μ ← α + |a_i|`,
//!     `α̂ ← α/μ`, `b ← a_i/μ`, `ν ← √(α̂² + b²)`, `c_i ← α̂/ν`, `s_i ← b/ν`,
//!     `α ← μν`.
//! C3′. \[Diagonal first.\] The new diagonal is `r_jj ← c_j r_jj` (step C4
//!     at `i = j`, where `ξ = 0`). If one of them is not finite and
//!     positive (only an underflow can do that), return
//!     `DowndateIndefinite` with the margin of C2 — still before any
//!     write. *An addition of this edition to LINPACK's algorithm.* Since
//!     `c_j ≥ α/√(α² + 1) ≈ √margin` and `margin ≥ u` when it is positive,
//!     only a diagonal entry below about `2⁻¹⁰⁴⁷` (subnormal, or nearly)
//!     reaches it: the test `downdate_whose_diagonal_underflows_is_refused_
//!     by_c3_prime` uses `r_ll = 2⁻¹⁰⁶⁰`, `margin = 2⁻⁴⁰`, `c_l ≈ 2⁻¹⁹`
//!     (without C3′ the downdate returns a zero diagonal; issue
//!     xingo-statla#6).
//! C4. \[Apply.\] For each column `j`: `ξ ← 0`; for `i = j, …, 0`: `(r_ij,
//!     ξ) ← (c_i r_ij − s_i ξ, c_i ξ + s_i r_ij)`.
//! C5. \[Signs.\] Flip the rows with `r_ii < 0` (none, after C3′; kept as
//!     the contract states it, so that (H1) does not rest on C3′ alone).
//! Cost `≈ 3n²` flops plus the `n²` of C1. Backward stable in the mixed
//! sense of Stewart (1979). The forward error against a refactorization
//! grows like `κ₂(R̃) u` — the error of any algorithm, because two
//! refactorizations of `A_t` perturbed at the level `u` already differ by
//! that much (pilot s01, table A).
//!
//! ## Solves and the determinant
//! `solve`: `Rᵀy = b` forward, then `Rx = y` backward (both
//! column-oriented on `R`). `solve_upper_t`: the forward half only, `b ←
//! R⁻ᵀb`, so `‖R⁻ᵀb‖² = bᵀA⁻¹b`. `log2_det_parts`: `det R = Π r_jj = m ·
//! 2^e` with `m ∈ [1, 2)`, accumulated by splitting each `r_jj = f_j 2^{k_j}`
//! with powers of two read off the `f64` bits (exact scalings), so it
//! neither overflows nor underflows for any `n`; then `ln det A = 2(ln m +
//! e ln 2)` in the caller's precision.
//!
//! ## Invariants
//! (H1) `R` upper triangular (strict lower triangle exactly zero),
//!      `r_jj > 0`, every entry finite — [`Cholesky::check_invariants`],
//!      `debug_assert!` after each of F, U, C.
//! (H2) `‖RᵀR − A_t‖_F ≤ c_H · n · u · ‖A_t‖_F` after any sequence; `c_H =
//!      8` in the tests (`tests/cholesky.rs`, against `A_t` and `RᵀR` in
//!      double-double; the norm along the path when downdates cancel more
//!      than half of `A`).
//! (H3) A failed `update`/`downdate` leaves `R` bit for bit unchanged:
//!      U0 and C0–C3′ read `R` and write only local vectors.
//!
//! ## Acceptance (L3.2, S9 of D-0127) and what was measured
//! In `f64` and `DoubleDouble`, `n ≤ 200`, `κ(A_t)` from `10` to `10¹²`
//! (`tests/cholesky.rs`, proptests and fixed seeds):
//! (i) `update∘downdate` and `downdate∘update` within `10 n u` of the
//!     original factor (relative Frobenius);
//! (ii) backward error (H2) `≤ c_H n u` at every `κ`;
//! (iii) forward error against a refactorization of `A_t` `≤ 10⁻¹²` where
//!      `κ(A_t) ≤ 10⁸`, and elsewhere `≤ 10×` the distance between two
//!      refactorizations of `A_t` perturbed at the level `u`;
//! (iv) a failed operation leaves the factor bit for bit unchanged.
//!
//! **What holds.** (i), (ii), (iv) per instance with wide margins
//! (`upd∘dd`, `dd∘upd` `≤ 4.9 n u` at `n = 3`, `≤ 0.05 n u` for `n ≥ 50`).
//! (iii) as worded does **not** hold per instance: the forward error of
//! the downdate is governed by `1 − ‖a‖²`, not by `κ` alone — `n = 2`,
//! `κ = 10⁶` gave `9.4·10⁻¹²`; at `κ = 10⁸`, `n = 100`, 10 of 200 seeds
//! exceed `10⁻¹²`; the per-instance ratio to the distance between two
//! refactorizations reaches 40–150 for `n ≤ 3`. The tests enforce what was
//! measured to hold — per instance `fwd ≤ 8 κ u` (the first-order bound of
//! any algorithm working from `R`), and the pilot's aggregate form (the
//! maximum over 20 seeds at `n = 100, 200`: `≤ 10⁻¹²` where `κ ≤ 10⁶`, `≤
//! 10×` the maximum distance everywhere) — and keep the literal criterion
//! as an ignored test (question Q-L32-2 of task L3.2, settled this way by
//! D-0132). The measured
//! values are in `.claude/scratch/l32/l32-tables.md`; the example
//! `accuracy` regenerates them.
//!
//! With the dense kernel `FaerDense` (feature `faer`) as `K`, the
//! acceptance (i)–(iii-a) and the aggregate (iii) hold with the same
//! constants (issue xingo-statla#7).
//!
//! **Graded and badly scaled matrices** (issue xingo-statla#6). Algorithms
//! F, U and C only combine entries of one column of `R` (and `x_j`), which
//! share the scale `d_j` of `A = D A′ D`, so their errors are those of the
//! equilibrated `A′`. The tests check it for `D` graded or in random order
//! over up to `10⁴⁰` (`A` over `10⁸⁰`), `n ≤ 120`: (i), (ii) and (iii-a)
//! measured in the units of `D` (`‖D⁻¹(RᵀR − A_t)D⁻¹‖`, `‖(R̃ − R⋆)D⁻¹‖`),
//! against `κ(R⋆D⁻¹)²`. Largest ratios on the survey grid (f64): (ii)
//! `3.0 n u`, (i) `1.8 n u`, (iii-a) `0.31 κ u` — the constants of the
//! unscaled tests hold unchanged.
//!
//! ## References
//! LINPACK Users' Guide (1979), ch. 10 (`dchud`, `dchdd`); Saunders (1972);
//! Stewart, J. IMA 23 (1979); Gill, Golub, Murray & Saunders, Math. Comp.
//! 28 (1974); Higham (2002), Theorem 10.3 (backward error of Cholesky);
//! pilot s01 (`.claude/scratch/s01/s01-pilot`). All cited from memory.

use xingo_sparse::{DenseBlock, DenseError, DenseKernel, NaiveDense, Real};

use crate::real::{is_exact_zero, le_exact, lt};
use crate::small::{Rot, axpy_sub, check_finite, check_len, dot, givens, norm2};
use crate::{MatMut, MatRef, StatlaError};

/// Options of [`Cholesky`]: every threshold of chapter 12.2 lives here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CholOptions {
    /// Step F3: a pivot `d_j` is rejected unless `d_j > pivot_tol · a_jj`
    /// (default `0`: reject only nonpositive pivots, as LAPACK `potrf`).
    pub pivot_tol: f64,
    /// Step C2: the downdate is rejected unless `1 − ‖a‖² > downdate_tol`
    /// (default `0`, LINPACK's test). A positive value refuses downdates
    /// whose result is too ill conditioned to be trusted (forward error ≈
    /// `u / margin`; pilot table A).
    pub downdate_tol: f64,
}

impl Default for CholOptions {
    fn default() -> Self {
        Self {
            pivot_tol: 0.0,
            downdate_tol: 0.0,
        }
    }
}

impl CholOptions {
    /// Both thresholds finite and nonnegative (a negative `downdate_tol`
    /// would let C3 take the square root of a nonpositive `α²`).
    fn validate(&self) -> Result<(), StatlaError> {
        let ok = |v: f64| v.is_finite() && le_exact(0.0, v);
        if !ok(self.pivot_tol) {
            return Err(StatlaError::InvalidArgument {
                what: "CholOptions::pivot_tol must be finite and >= 0",
            });
        }
        if !ok(self.downdate_tol) {
            return Err(StatlaError::InvalidArgument {
                what: "CholOptions::downdate_tol must be finite and >= 0",
            });
        }
        Ok(())
    }
}

/// The upper Cholesky factor `R` of a symmetric positive definite `A =
/// RᵀR`, kept up to date under `A ← A ± x xᵀ` (Chapter 12.2: Algorithms
/// F, U, C).
///
/// ```
/// use xingo_statla::{CholOptions, Cholesky, MatRef};
/// // A = [[4, 2], [2, 3]] (upper triangle read).
/// let a = [4.0, 0.0, 2.0, 3.0];
/// let mut c = Cholesky::factor(MatRef::new(2, 2, 2, &a)?, &CholOptions::default())?;
/// c.update(&[1.0, 1.0])?; // A + x xᵀ = [[5, 3], [3, 4]]
/// let info = c.downdate(&[1.0, 1.0])?; // back to A
/// assert!(info.margin > 0.0 && info.margin <= 1.0);
/// let mut b = [6.0, 5.0]; // A · (1, 1)
/// c.solve(&mut b)?;
/// assert!((b[0] - 1.0).abs() < 1e-14 && (b[1] - 1.0).abs() < 1e-14);
/// assert!(c.downdate(&[3.0, 0.0]).is_err()); // A − 9 e₀e₀ᵀ is indefinite
/// # Ok::<(), xingo_statla::StatlaError>(())
/// ```
#[derive(Clone, Debug)]
pub struct Cholesky<T: Real = f64, K: DenseKernel<T> = NaiveDense> {
    r: DenseBlock<T>,
    // Kept with the factor (the kernel it was built with); read again by
    // the operations of v0.2 that refactor trailing blocks (S8).
    #[allow(dead_code)]
    kernel: K,
    opts: CholOptions,
}

/// What a successful downdate saw.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct DowndateInfo {
    /// `1 − ‖a‖²` of step C2, in `(downdate_tol, 1]`: near `0` means the
    /// result is nearly singular and its forward error is large.
    pub margin: f64,
}

impl<T: Real> Cholesky<T, NaiveDense> {
    /// Factor the symmetric positive definite `A` (upper triangle read)
    /// with the reference kernel.
    ///
    /// # Errors
    /// [`StatlaError::InvalidArgument`] (not square, invalid options),
    /// [`StatlaError::NonFinite`], [`StatlaError::NotPositiveDefinite`].
    pub fn factor(a: MatRef<'_, T>, opts: &CholOptions) -> Result<Self, StatlaError> {
        Self::factor_with(a, NaiveDense, opts)
    }
}

impl<T: Real, K: DenseKernel<T>> Cholesky<T, K> {
    /// As [`Cholesky::factor`] with the dense kernel `kernel`.
    ///
    /// # Errors
    /// As [`Cholesky::factor`].
    pub fn factor_with(
        a: MatRef<'_, T>,
        kernel: K,
        opts: &CholOptions,
    ) -> Result<Self, StatlaError> {
        // F1.
        opts.validate()?;
        let n = a.rows();
        if a.cols() != n {
            return Err(StatlaError::InvalidArgument {
                what: "Cholesky::factor: matrix not square",
            });
        }
        let mut w = DenseBlock::zeros(n, n);
        for j in 0..n {
            let col = a.col(j).unwrap_or(&[]);
            for (i, &v) in col.iter().enumerate().take(j + 1) {
                if !v.is_finite() {
                    return Err(StatlaError::NonFinite {
                        what: "a",
                        index: i + j * n,
                    });
                }
                // F2: w_ji = a_ij for i ≤ j (lower triangle of W).
                w.data[j + i * n] = v;
            }
        }
        // F2.
        kernel.potrf(&mut w).map_err(|e| match e {
            DenseError::NotPositiveDefinite { position } => {
                StatlaError::NotPositiveDefinite { column: position }
            }
            DenseError::Dimension => StatlaError::InvalidArgument {
                what: "Cholesky::factor: the kernel rejected the dimensions",
            },
        })?;
        // F3.
        let tol = T::from_f64(opts.pivot_tol);
        for j in 0..n {
            let l = w.data[j + j * n];
            let ajj = a.get(j, j).unwrap_or(T::ZERO);
            if !l.is_finite() || !lt(T::ZERO, l) || le_exact(l * l, tol * ajj) {
                return Err(StatlaError::NotPositiveDefinite { column: j });
            }
        }
        // F4.
        let mut r = DenseBlock::zeros(n, n);
        for j in 0..n {
            for i in 0..=j {
                r.data[i + j * n] = w.data[j + i * n];
            }
        }
        let c = Self {
            r,
            kernel,
            opts: *opts,
        };
        debug_assert_eq!(c.check_invariants(), Ok(()), "(H1) after factor");
        Ok(c)
    }

    /// Adopt an existing upper factor (for instance the `R` of a
    /// [`crate::Qr`]: the Q-less QR of streaming regression *is* this type).
    ///
    /// # Errors
    /// [`StatlaError::InvalidArgument`] if `r` is not square upper
    /// triangular with positive diagonal (or the options are invalid);
    /// [`StatlaError::NonFinite`].
    pub fn from_upper(
        r: DenseBlock<T>,
        kernel: K,
        opts: &CholOptions,
    ) -> Result<Self, StatlaError> {
        opts.validate()?;
        let n = r.rows;
        if r.cols != n || r.data.len() != n.saturating_mul(n) {
            return Err(StatlaError::InvalidArgument {
                what: "Cholesky::from_upper: not a square block",
            });
        }
        check_finite("r", &r.data)?;
        let c = Self {
            r,
            kernel,
            opts: *opts,
        };
        c.check_invariants()
            .map_err(|what| StatlaError::InvalidArgument { what })?;
        Ok(c)
    }

    /// Order `n`.
    #[must_use]
    pub fn n(&self) -> usize {
        self.r.rows
    }

    /// `R` (the strict lower triangle of the view is zero).
    #[must_use]
    pub fn upper(&self) -> MatRef<'_, T> {
        MatRef::from_block(&self.r)
    }

    /// `R`, owned.
    #[must_use]
    pub fn into_upper(self) -> DenseBlock<T> {
        self.r
    }

    /// `A ← A + x xᵀ` (Algorithm U).
    ///
    /// # Errors
    /// [`StatlaError::Dimension`], [`StatlaError::NonFinite`]; the factor
    /// is unchanged on error (H3).
    pub fn update(&mut self, x: &[T]) -> Result<(), StatlaError> {
        let n = self.n();
        // U0.
        check_len("x", n, x.len())?;
        check_finite("x", x)?;
        let mut rots = vec![Rot::IDENTITY; n];
        let r = &mut self.r.data;
        for j in 0..n {
            // U1.
            let mut xi = x[j];
            let col = &mut r[j * n..(j + 1) * n];
            for (rij, g) in col[..j].iter_mut().zip(&rots[..j]) {
                let (p, q) = g.apply(*rij, xi);
                *rij = p;
                xi = q;
            }
            // U2.
            let (g, rho) = givens(col[j], xi);
            rots[j] = g;
            col[j] = rho;
        }
        debug_assert_eq!(self.check_invariants(), Ok(()), "(H1) after update");
        Ok(())
    }

    /// `A ← A − x xᵀ` (Algorithm C).
    ///
    /// # Errors
    /// [`StatlaError::Dimension`], [`StatlaError::NonFinite`],
    /// [`StatlaError::DowndateIndefinite`]; unchanged on error (H3).
    pub fn downdate(&mut self, x: &[T]) -> Result<DowndateInfo, StatlaError> {
        let n = self.n();
        // C0.
        check_len("x", n, x.len())?;
        check_finite("x", x)?;
        // C1.
        let mut a = x.to_vec();
        self.forward(&mut a);
        // C2.
        let nu = norm2(&a);
        let alpha2 = T::ONE - nu * nu;
        if !alpha2.is_finite() {
            return Err(StatlaError::DowndateIndefinite {
                margin: f64::NEG_INFINITY,
            });
        }
        let margin = alpha2.to_f64();
        if le_exact(alpha2, T::from_f64(self.opts.downdate_tol)) {
            return Err(StatlaError::DowndateIndefinite { margin });
        }
        // C3.
        let mut alpha = alpha2.sqrt();
        let mut rots = vec![Rot::IDENTITY; n];
        for i in (0..n).rev() {
            let mu = alpha + a[i].abs();
            let ah = alpha / mu;
            let b = a[i] / mu;
            let nu = (ah * ah + b * b).sqrt();
            rots[i] = Rot {
                c: ah / nu,
                s: b / nu,
            };
            alpha = mu * nu;
        }
        // C3′.
        for (j, g) in rots.iter().enumerate() {
            let d = g.c * self.r.data[j + j * n];
            if !d.is_finite() || !lt(T::ZERO, d) {
                return Err(StatlaError::DowndateIndefinite { margin });
            }
        }
        // C4.
        let r = &mut self.r.data;
        for j in 0..n {
            let col = &mut r[j * n..(j + 1) * n];
            let mut xi = T::ZERO;
            for i in (0..=j).rev() {
                let g = rots[i];
                let rij = col[i];
                col[i] = g.c * rij - g.s * xi;
                xi = g.c * xi + g.s * rij;
            }
        }
        // C5.
        for i in 0..n {
            if lt(self.r.data[i + i * n], T::ZERO) {
                for j in i..n {
                    self.r.data[i + j * n] = -self.r.data[i + j * n];
                }
            }
        }
        debug_assert_eq!(self.check_invariants(), Ok(()), "(H1) after downdate");
        Ok(DowndateInfo { margin })
    }

    /// `b ← R⁻ᵀ b` without checks (`b.len() = n`).
    fn forward(&self, b: &mut [T]) {
        let n = self.n();
        for j in 0..n {
            let col = &self.r.data[(j * n)..=(j * n + j)];
            let s = b[j] - dot(&col[..j], &b[..j]);
            b[j] = s / col[j];
        }
    }

    /// `b ← R⁻¹ b` without checks.
    fn backward(&self, b: &mut [T]) {
        let n = self.n();
        for j in (0..n).rev() {
            let col = &self.r.data[(j * n)..=(j * n + j)];
            let xj = b[j] / col[j];
            b[j] = xj;
            axpy_sub(xj, &col[..j], &mut b[..j]);
        }
    }

    /// `b ← A⁻¹ b`.
    ///
    /// # Errors
    /// [`StatlaError::Dimension`].
    pub fn solve(&self, b: &mut [T]) -> Result<(), StatlaError> {
        check_len("b", self.n(), b.len())?;
        self.forward(b);
        self.backward(b);
        Ok(())
    }

    /// `B ← A⁻¹ B` for a block of right-hand sides.
    ///
    /// # Errors
    /// [`StatlaError::Dimension`].
    pub fn solve_block(&self, mut b: MatMut<'_, T>) -> Result<(), StatlaError> {
        let cols = b.as_ref().cols();
        check_len("b (rows)", self.n(), b.as_ref().rows())?;
        for j in 0..cols {
            if let Some(col) = b.col_mut(j) {
                self.forward(col);
                self.backward(col);
            }
        }
        Ok(())
    }

    /// `b ← R⁻ᵀ b` (whitening: `‖R⁻ᵀ b‖² = bᵀA⁻¹b`, a Mahalanobis
    /// distance).
    ///
    /// # Errors
    /// [`StatlaError::Dimension`].
    pub fn solve_upper_t(&self, b: &mut [T]) -> Result<(), StatlaError> {
        check_len("b", self.n(), b.len())?;
        self.forward(b);
        Ok(())
    }

    /// `ln det A = 2 Σ ln r_jj`, returned as `(m, e)` with `det R = Π r_jj
    /// = m · 2^e` exactly split, `m ∈ [1, 2)` (`(1, 0)` for `n = 0`): no `ln`
    /// in [`Real`], so the caller forms `ln det A = 2 (ln m + e ln 2)` in
    /// its own precision (`log₂ det R = log₂ m + e`).
    #[must_use]
    pub fn log2_det_parts(&self) -> (T, i64) {
        let n = self.n();
        let two = T::from_f64(2.0);
        let half = T::from_f64(0.5);
        let mut m = T::ONE;
        let mut e: i64 = 0;
        for j in 0..n {
            let d = self.r.data[j + j * n];
            let (f, k) = split_pow2(d);
            m *= f;
            e += k;
            while le_exact(two, m) {
                m *= half;
                e += 1;
            }
            while lt(m, T::ONE) && lt(T::ZERO, m) {
                m *= two;
                e -= 1;
            }
        }
        (m, e)
    }

    /// Invariant (H1).
    ///
    /// # Errors
    /// The text of the violated invariant.
    pub fn check_invariants(&self) -> Result<(), &'static str> {
        let n = self.r.rows;
        if self.r.cols != n || self.r.data.len() != n * n {
            return Err("(H1) R is not a consistent square block");
        }
        for j in 0..n {
            for i in 0..n {
                let v = self.r.data[i + j * n];
                if !v.is_finite() {
                    return Err("(H1) R has a non-finite entry");
                }
                if i > j && !is_exact_zero(v) {
                    return Err("(H1) R is not upper triangular");
                }
                if i == j && !lt(T::ZERO, v) {
                    return Err("(H1) R has a nonpositive diagonal entry");
                }
            }
        }
        Ok(())
    }
}

/// `d = f · 2^k` with `2^k` the power of two of `d`'s `f64` value (exact
/// scalings); `f ≈ [1, 2)` up to the rounding of `to_f64`, normalized by
/// the caller.
fn split_pow2<T: Real>(d: T) -> (T, i64) {
    let bits = d.to_f64().to_bits();
    let exp = i64::try_from((bits >> 52) & 0x7ff).unwrap_or(0);
    if exp == 0 || exp == 0x7ff {
        return (d, 0); // subnormal (normalized by the caller's loop) or not finite
    }
    let k = exp - 1023;
    // 2^-k as f64: exact for k in −1022..=1023 (2^-1023 is subnormal but exact).
    let inv = match u64::try_from(1023 - k) {
        Ok(0) | Err(_) => f64::from_bits(1 << 51), // 2^-1023, subnormal and exact
        Ok(b) => f64::from_bits(b << 52),
    };
    (d * T::from_f64(inv), k)
}
