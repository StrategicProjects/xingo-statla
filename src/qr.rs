//! # Chapter 12.3 — Thin QR with explicit `Q` and five updating operations
//!
//! ## Purpose
//! A least-squares fit `min ‖Ax − b‖` whose design `A` gains or loses an
//! observation (a row) or a regressor (a column), or changes by a rank-1
//! term, should cost `O(mn)`, not the `O(mn²)` of refactorizing. [`Qr`]
//! keeps the *thin* factorization `A = QR` — `Q ∈ T^{m×n}` with orthonormal
//! columns, explicit; `R ∈ T^{n×n}` upper triangular with positive diagonal
//! — through `insert_row`, `delete_row`, `insert_col`, `delete_col` and
//! `rank1_update` (decision S8 of D-0127). The `Q`-less variant is
//! [`Cholesky`] (`RᵀR = AᵀA`): [`Qr::to_cholesky`].
//!
//! ## Notation
//! `A ∈ T^{m×n}`, `m ≥ n`, full column rank; `q_j` the columns of `Q`,
//! `q_(i)ᵀ` its row `i`; `R̂`, `M` the bordered factors during an operation
//! (`A = M R̂` throughout). A rotation `G = (c, s)` (Chapter 12.0.4)
//! applied to rows `k, l` of `R̂` is always applied to columns `k, l` of `M`
//! with the same `(c, s)`: then `M R̂` is unchanged, because `c q_k + s q_l`,
//! `−s q_k + c q_l` against `c r_k + s r_l`, `−s r_k + c r_l` gives back
//! `q_k r_kᵀ + q_l r_lᵀ`. `τ = rank_tol` resolved (see [`QrOptions`]); `η =
//! reorth_eta`.
//!
//! ## Algorithm F (factor: Householder, GVL Algorithm 5.2.1)
//! F1. \[Check.\] `m ≥ n`, `A` finite, options valid.
//! F2. \[Reduce.\] For `k = 0, …, n − 1`: a reflector `H_k` (Algorithm H)
//!     maps `W[k.., k]` to `α_k e_0`; apply it to `W[k.., k+1..]`.
//! F3. \[Form `Q`.\] `Q ← H_0 ⋯ H_{n−1} [I_n; 0]`, applied backwards (the
//!     reflector `H_k` touches columns `k..n` only).
//! F4. \[Signs.\] Where `r_kk < 0`, negate row `k` of `R` and column `k` of
//!     `Q`.
//! F5. \[Rank.\] If `r_kk ≤ τ ‖a_k‖`, [`StatlaError::RankDeficient`] at `k`.
//!
//! ## Algorithm I (insert row `wᵀ` before row `i`)
//! I1. Check `|w| = n`, `i ≤ m`, `w` finite.
//! I2. Border: `M = [Q with a zero row at i, e_i]`, `R̂ = [R; wᵀ]`.
//! I3. For `j = 0, …, n − 1`: `G_j` with `G_j (r_jj, w_j) = (ρ, 0)`; rotate
//!     rows `j`, `n` of `R̂` (columns `j..`) and columns `j`, `n` of `M`.
//! I4. Drop the last row of `R̂` (now zero) and the last column of `M`.
//! Cost `3mn + 3n²/2` flops; nothing can fail after I1.
//!
//! ## Algorithm D (delete row `i`; requires `m > n`)
//! D1. `z ← e_i`, orthogonalized against `Q` by Algorithm D of 12.0.4
//!     (DGKS, at most two passes); `ν = ‖z‖` (in exact arithmetic `ν² = 1 −
//!     ‖q_(i)‖²`). If `ν ≤ τ`, the remaining rows are rank deficient:
//!     `RankDeficient` (reported with the row index), nothing written.
//! D2. Border: `M = [Q, z/ν]`, `R̂ = [R; 0]`, `w ← (q_(i); z_i/ν)` the row
//!     `i` of `M`.
//! D3. For `k = n − 1, …, 0`: `G_k (w_k, w_{k+1}) = (·, 0)`; rotate rows `k,
//!     k+1` of `R̂` (columns `k..`; this fills `(k+1, k)` only) and columns
//!     `k, k+1` of `M`. Now `w = ±e_0` (so column 0 of `M` is `±e_i`), `R̂` is
//!     upper Hessenberg, its row 0 is the deleted row of `A`.
//! D4. `R ← R̂[1.., ..]` (upper triangular), `Q ← M[rows ≠ i, 1..]`; signs as
//!     F4. Cost `≈ 5mn`.
//!
//! ## Algorithm C (insert column `c` before column `j`; requires `m > n`)
//! C1. Check `|c| = m`, `j ≤ n`, `c` finite.
//! C2. `h ← Qᵀc`, `z ← c − Qh` by Algorithm D of 12.0.4; `ρ = ‖z‖`; if `ρ ≤
//!     τ ‖c‖`, `RankDeficient` at `j`, nothing written.
//! C3. Border: `M = [Q, z/ρ]`; `R̂ ∈ T^{(n+1)×(n+1)}` with the columns of `R`
//!     (and a zero last row) and the spike `(h; ρ)` at position `j`.
//! C4. For `k = n, …, j + 1`: `G` with `G (r̂_{k−1,j}, r̂_{kj}) = (·, 0)`;
//!     rotate rows `k − 1, k` of `R̂` (columns `j+1..`) and columns `k − 1, k`
//!     of `M`. Signs as F4. Cost `≈ 4mn + 3n(n − j)`.
//!
//! ## Algorithm E (delete column `j`)
//! E1. Remove column `j` of `R`: an `n × (n−1)` matrix, upper Hessenberg
//!     from column `j` on.
//! E2. For `p = j, …, n − 2`: `G (r_pp, r_{p+1,p}) = (·, 0)`; rotate rows
//!     `p, p+1` (columns `p+1..`) and columns `p, p+1` of `Q`.
//! E3. Drop the last row of `R` (zero) and the last column of `Q`; signs.
//!     Cost `6m(n − j)`; nothing can fail after the index check.
//!
//! ## Algorithm R (rank-1 update `A + u vᵀ`; Hammarling & Lucas §2.5)
//! R1. Check `|u| = m`, `|v| = n`, finite.
//! R2. `w ← Qᵀu`, `z ← u − Qw` (Algorithm D of 12.0.4), `ρ = ‖z‖`. If `ρ ≤
//!     τ ‖u‖`, `u` is in the range of `Q` to working precision: `ρ ← 0`, `z
//!     ← 0` (the border column then never mixes in: every rotation that
//!     touches it has `s = 0` exactly).
//! R3. Work on copies: `M = [Q, z/ρ]`, `R̂ = [R; 0]`, `ŵ = (w; ρ)`.
//! R4. For `k = n − 1, …, 0`: `G_k (ŵ_k, ŵ_{k+1}) = (·, 0)` on rows `k, k+1`
//!     of `R̂` and columns `k, k+1` of `M`: `ŵ = ‖ŵ‖e_0`, `R̂` upper
//!     Hessenberg.
//! R5. `R̂_{0,:} ← R̂_{0,:} + ‖ŵ‖ vᵀ` (still Hessenberg).
//! R6. For `k = 0, …, n − 1`: `G (r̂_kk, r̂_{k+1,k}) = (·, 0)` on rows `k, k+1`
//!     (columns `k+1..`) and columns `k, k+1` of `M`. The last row of `R̂` is
//!     now zero.
//! R7. Signs; if some `r_kk ≤ τ (‖a_k‖ + |v_k| ‖u‖)` (the result lost
//!     rank; the scale is that of the column before any cancellation),
//!     `RankDeficient` at `k` and the copies are dropped; else commit `Q ←
//!     M[:, ..n]`, `R ← R̂[..n, :]`. Cost `≈ 10mn`.
//!
//! ## Least squares
//! `y ← Qᵀb`, `x ← R⁻¹y` (back substitution), and the residual norm `‖b −
//! Qy‖` (the residual is formed, not taken as `√(‖b‖² − ‖y‖²)`, which
//! cancels).
//!
//! ## Invariants
//! (Q1) `‖QᵀQ − I‖_F ≤ c_Q · m · u` — [`Qr::check_invariants`]`(c_Q)`;
//!      the DGKS step of D1, C2, R2 is what keeps it under deletions and
//!      nearly dependent insertions.
//! (Q2) `R` upper triangular with `r_jj > 0` and finite (checked by
//!      `check_invariants`, and as a `debug_assert!` after every
//!      operation); `‖QR − A_t‖_F ≤ c_Q · m · u · ‖A_t‖_F` (tests, against
//!      `A_t` in double-double).
//! (Q3) A failed operation leaves `Q` and `R` bit for bit unchanged: every
//!      failure is detected before the first write (I1, D1, C1–C2, E1) or
//!      on copies (R7).
//!
//! ## Acceptance (L3.2; the same as Chapter 12.2, plus Q1)
//! `κ` here is `κ(A_tᵀA_t) = κ₂(A_t)²`, so that the bands of the
//! criterion (iii) — `≤ 10⁻¹²` where `κ ≤ 10⁸`, `≤ 10×` the distance
//! between two refactorizations elsewhere — mean the same conditioning of
//! `R` as in Chapter 12.2. Tests in `tests/qr.rs`. Per instance: (Q1),
//! (Q2) with `c_Q = 16`, `fwd ≤ 8 κ u`; in aggregate (20 seeds, `150 ×
//! 100`): the ratified (iii) holds in full — `max fwd ≤ 4.6·10⁻¹⁵` where
//! `κ ≤ 10⁸` and `≤ 1.3×` the distance between refactorizations. The
//! literal per-instance (iii) fails on small matrices (ratio up to `3·10²`
//! on `5 × 3`, where the refactorization is nearly exact): question
//! Q-L32-2.
//!
//! ## References
//! Golub & Van Loan (2013), §5.2 (Householder QR), §6.5.1–6.5.3 (updating
//! the full QR); Hammarling & Lucas, MIMS EPrint 2008.111 (economy
//! updates); Daniel, Gragg, Kaufman & Stewart, Math. Comp. 30 (1976)
//! (Gram–Schmidt completion with reorthogonalization). Cited from memory.

use xingo_sparse::{DenseBlock, DenseKernel, NaiveDense, Real};

use crate::real::{is_exact_zero, le_exact, lt};
use crate::small::{
    apply_householder, axpy_sub, check_finite, check_len, dot, givens, householder, norm2,
    orthogonalize,
};
use crate::{CholOptions, Cholesky, MatRef, StatlaError};

/// Options of [`Qr`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QrOptions {
    /// DGKS reorthogonalization: a second Gram–Schmidt pass runs when the
    /// norm fell below `reorth_eta` times its value before the pass
    /// (default `1/√2`, Daniel–Gragg–Kaufman–Stewart). Must be in `(0, 1]`.
    pub reorth_eta: f64,
    /// A new column (or a complement vector in `delete_row`) whose norm
    /// after two passes is `≤ rank_tol · ‖column‖` is declared dependent.
    /// **`0` (the default) means `n · ε_T`**, resolved at each operation
    /// from the scalar `T` and the number of columns `n` of the result
    /// (the default cannot know either; see the question Q-L32-1 of task
    /// L3.2). Must be finite and `≥ 0`.
    pub rank_tol: f64,
}

impl Default for QrOptions {
    fn default() -> Self {
        Self {
            reorth_eta: core::f64::consts::FRAC_1_SQRT_2,
            rank_tol: 0.0,
        }
    }
}

impl QrOptions {
    fn validate(&self) -> Result<(), StatlaError> {
        if !(self.reorth_eta.is_finite()
            && lt(0.0, self.reorth_eta)
            && le_exact(self.reorth_eta, 1.0))
        {
            return Err(StatlaError::InvalidArgument {
                what: "QrOptions::reorth_eta must be in (0, 1]",
            });
        }
        if !(self.rank_tol.is_finite() && le_exact(0.0, self.rank_tol)) {
            return Err(StatlaError::InvalidArgument {
                what: "QrOptions::rank_tol must be finite and >= 0",
            });
        }
        Ok(())
    }

    /// `τ` for a result with `n` columns.
    fn tau<T: Real>(&self, n: usize) -> T {
        if is_exact_zero(self.rank_tol) {
            #[allow(clippy::cast_precision_loss)] // a column count, far below 2^53
            let nf = n.max(1) as f64;
            T::EPSILON * T::from_f64(nf)
        } else {
            T::from_f64(self.rank_tol)
        }
    }
}

/// Thin QR `A = QR`, `Q ∈ R^{m×n}` explicit with orthonormal columns, `R ∈
/// R^{n×n}` upper triangular with `r_jj > 0`, `m ≥ n`, full column rank
/// (Chapter 12.3: Algorithms F, I, D, C, E, R).
///
/// ```
/// use xingo_statla::{MatRef, Qr, QrOptions};
/// // A = [1 0; 1 1; 1 2] (column-major), b = (1, 2, 4).
/// let a = [1.0, 1.0, 1.0, 0.0, 1.0, 2.0];
/// let mut f = Qr::factor(MatRef::new(3, 2, 3, &a)?, &QrOptions::default())?;
/// let mut x = [0.0; 2];
/// let res = f.solve_least_squares(&[1.0, 2.0, 4.0], &mut x)?;
/// assert!((x[0] - 5.0 / 6.0).abs() < 1e-14 && (x[1] - 1.5).abs() < 1e-14);
/// assert!((res - (1.0f64 / 6.0).sqrt()).abs() < 1e-14);
/// f.insert_row(3, &[1.0, 3.0])?; // a fourth observation
/// f.delete_row(0)?; // and the first leaves
/// assert_eq!((f.nrows(), f.ncols()), (3, 2));
/// assert!(f.check_invariants(16.0).is_ok());
/// # Ok::<(), xingo_statla::StatlaError>(())
/// ```
#[derive(Clone, Debug)]
pub struct Qr<T: Real = f64, K: DenseKernel<T> = NaiveDense> {
    q: DenseBlock<T>,
    r: DenseBlock<T>,
    kernel: K,
    opts: QrOptions,
}

impl<T: Real> Qr<T, NaiveDense> {
    /// Householder QR of `A` (`m ≥ n`), `Q` formed explicitly.
    ///
    /// # Errors
    /// [`StatlaError::InvalidArgument`] (`m < n`, invalid options),
    /// [`StatlaError::NonFinite`], [`StatlaError::RankDeficient`].
    pub fn factor(a: MatRef<'_, T>, opts: &QrOptions) -> Result<Self, StatlaError> {
        Self::factor_with(a, NaiveDense, opts)
    }
}

/// Where `r_kk < 0` for `k ≥ from`, negate row `k` of `r` and column `k`
/// of `q` (step F4).
fn fix_signs<T: Real>(q: &mut DenseBlock<T>, r: &mut DenseBlock<T>, from: usize) {
    let (m, n) = (q.rows, r.cols);
    for k in from..n.min(r.rows) {
        if lt(r.data[k + k * r.rows], T::ZERO) {
            for j in k..n {
                r.data[k + j * r.rows] = -r.data[k + j * r.rows];
            }
            for v in &mut q.data[k * m..(k + 1) * m] {
                *v = -*v;
            }
        }
    }
}

/// `R̂[..n, ..]` of an `(n+1) × c` block, as an `n × c` block.
fn drop_last_row<T: Real>(r: &DenseBlock<T>) -> DenseBlock<T> {
    let (rows, cols) = (r.rows - 1, r.cols);
    let mut out = DenseBlock::zeros(rows, cols);
    for j in 0..cols {
        out.data[j * rows..(j + 1) * rows].copy_from_slice(&r.data[j * r.rows..j * r.rows + rows]);
    }
    out
}

impl<T: Real, K: DenseKernel<T>> Qr<T, K> {
    /// As [`Qr::factor`] with the dense kernel `kernel`.
    ///
    /// # Errors
    /// As [`Qr::factor`].
    pub fn factor_with(a: MatRef<'_, T>, kernel: K, opts: &QrOptions) -> Result<Self, StatlaError> {
        // F1.
        opts.validate()?;
        let (m, n) = (a.rows(), a.cols());
        if m < n {
            return Err(StatlaError::InvalidArgument {
                what: "Qr::factor: fewer rows than columns",
            });
        }
        let mut w = a.to_block();
        check_finite("a", &w.data)?;
        let col_norms: Vec<T> = (0..n).map(|j| norm2(&w.data[j * m..(j + 1) * m])).collect();
        // F2.
        let mut taus = vec![T::ZERO; n];
        for k in 0..n {
            let (left, right) = w.data.split_at_mut((k + 1) * m);
            let x = &mut left[k * m + k..];
            taus[k] = householder(x);
            let v = &x[1..];
            for j in 0..n - k - 1 {
                apply_householder(taus[k], v, &mut right[j * m + k..(j + 1) * m]);
            }
        }
        let mut r = DenseBlock::zeros(n, n);
        for j in 0..n {
            for i in 0..=j {
                r.data[i + j * n] = w.data[i + j * m];
            }
        }
        // F3.
        let mut q = DenseBlock::zeros(m, n);
        for k in 0..n {
            q.data[k + k * m] = T::ONE;
        }
        for k in (0..n).rev() {
            let v = &w.data[k * m + k + 1..(k + 1) * m];
            for j in k..n {
                apply_householder(taus[k], v, &mut q.data[j * m + k..(j + 1) * m]);
            }
        }
        // F4.
        fix_signs(&mut q, &mut r, 0);
        // F5.
        let tau = opts.tau::<T>(n);
        for k in 0..n {
            if le_exact(r.data[k + k * n], tau * col_norms[k]) {
                return Err(StatlaError::RankDeficient { column: k });
            }
        }
        let f = Self {
            q,
            r,
            kernel,
            opts: *opts,
        };
        debug_assert_eq!(f.check_shape(), Ok(()), "(Q2) after factor");
        Ok(f)
    }

    /// `m`.
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.q.rows
    }

    /// `n`.
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.r.cols
    }

    /// `Q`.
    #[must_use]
    pub fn q(&self) -> MatRef<'_, T> {
        MatRef::from_block(&self.q)
    }

    /// `R`.
    #[must_use]
    pub fn r(&self) -> MatRef<'_, T> {
        MatRef::from_block(&self.r)
    }

    /// Insert the row `w` (length `n`) before row `i` (`i = m` appends)
    /// (Algorithm I).
    ///
    /// # Errors
    /// [`StatlaError::Dimension`], [`StatlaError::InvalidArgument`],
    /// [`StatlaError::NonFinite`]; unchanged on error (Q3).
    pub fn insert_row(&mut self, i: usize, w: &[T]) -> Result<(), StatlaError> {
        let (m, n) = (self.nrows(), self.ncols());
        // I1.
        check_len("w", n, w.len())?;
        if i > m {
            return Err(StatlaError::InvalidArgument {
                what: "Qr::insert_row: row index > m",
            });
        }
        check_finite("w", w)?;
        // I2.
        let m1 = m + 1;
        let mut q = DenseBlock::zeros(m1, n);
        for j in 0..n {
            let src = &self.q.data[j * m..(j + 1) * m];
            let dst = &mut q.data[j * m1..(j + 1) * m1];
            dst[..i].copy_from_slice(&src[..i]);
            dst[i + 1..].copy_from_slice(&src[i..]);
        }
        let mut e = vec![T::ZERO; m1];
        e[i] = T::ONE;
        let mut wr = w.to_vec();
        // I3.
        for j in 0..n {
            let (g, rho) = givens(self.r.data[j + j * n], wr[j]);
            self.r.data[j + j * n] = rho;
            wr[j] = T::ZERO;
            for l in j + 1..n {
                let (p, s) = g.apply(self.r.data[j + l * n], wr[l]);
                self.r.data[j + l * n] = p;
                wr[l] = s;
            }
            for (x, y) in q.data[j * m1..(j + 1) * m1].iter_mut().zip(e.iter_mut()) {
                let (p, s) = g.apply(*x, *y);
                *x = p;
                *y = s;
            }
        }
        // I4.
        self.q = q;
        debug_assert_eq!(self.check_shape(), Ok(()), "(Q2) after insert_row");
        Ok(())
    }

    /// Delete row `i` (requires `m > n`) (Algorithm D).
    ///
    /// # Errors
    /// [`StatlaError::InvalidArgument`], [`StatlaError::RankDeficient`]
    /// (with `column = i`: the rows left are rank deficient); unchanged on
    /// error (Q3).
    pub fn delete_row(&mut self, i: usize) -> Result<(), StatlaError> {
        let (m, n) = (self.nrows(), self.ncols());
        if m <= n {
            return Err(StatlaError::InvalidArgument {
                what: "Qr::delete_row: requires m > n",
            });
        }
        if i >= m {
            return Err(StatlaError::InvalidArgument {
                what: "Qr::delete_row: row index >= m",
            });
        }
        // D1.
        let mut z = vec![T::ZERO; m];
        z[i] = T::ONE;
        let mut h = vec![T::ZERO; n];
        let (nu, _) = orthogonalize(&self.q, &mut z, &mut h, T::from_f64(self.opts.reorth_eta));
        if le_exact(nu, self.opts.tau::<T>(n)) {
            return Err(StatlaError::RankDeficient { column: i });
        }
        // D2.
        let n1 = n + 1;
        let mut mq = DenseBlock::zeros(m, n1);
        mq.data[..m * n].copy_from_slice(&self.q.data);
        for (d, &s) in mq.data[m * n..].iter_mut().zip(&z) {
            *d = s / nu;
        }
        let mut rh = DenseBlock::zeros(n1, n);
        for j in 0..n {
            rh.data[j * n1..j * n1 + n].copy_from_slice(&self.r.data[j * n..(j + 1) * n]);
        }
        let mut w: Vec<T> = (0..n1).map(|k| mq.data[i + k * m]).collect();
        // D3.
        for k in (0..n).rev() {
            let (g, rho) = givens(w[k], w[k + 1]);
            w[k] = rho;
            w[k + 1] = T::ZERO;
            g.rotate_rows(&mut rh, k, k + 1, k);
            g.rotate_cols(&mut mq.data, m, k, k + 1);
        }
        // D4.
        let mut r = DenseBlock::zeros(n, n);
        for j in 0..n {
            for p in 0..=j {
                r.data[p + j * n] = rh.data[p + 1 + j * n1];
            }
        }
        let m0 = m - 1;
        let mut q = DenseBlock::zeros(m0, n);
        for j in 0..n {
            let src = &mq.data[(j + 1) * m..(j + 2) * m];
            let dst = &mut q.data[j * m0..(j + 1) * m0];
            dst[..i].copy_from_slice(&src[..i]);
            dst[i..].copy_from_slice(&src[i + 1..]);
        }
        fix_signs(&mut q, &mut r, 0);
        self.q = q;
        self.r = r;
        debug_assert_eq!(self.check_shape(), Ok(()), "(Q2) after delete_row");
        Ok(())
    }

    /// Insert the column `c` (length `m`) before column `j` (requires `m >
    /// n`) (Algorithm C).
    ///
    /// # Errors
    /// [`StatlaError::Dimension`], [`StatlaError::InvalidArgument`],
    /// [`StatlaError::NonFinite`], [`StatlaError::RankDeficient`];
    /// unchanged on error.
    pub fn insert_col(&mut self, j: usize, c: &[T]) -> Result<(), StatlaError> {
        let (m, n) = (self.nrows(), self.ncols());
        // C1.
        check_len("c", m, c.len())?;
        if m <= n {
            return Err(StatlaError::InvalidArgument {
                what: "Qr::insert_col: requires m > n",
            });
        }
        if j > n {
            return Err(StatlaError::InvalidArgument {
                what: "Qr::insert_col: column index > n",
            });
        }
        check_finite("c", c)?;
        // C2.
        let cn = norm2(c);
        let mut z = c.to_vec();
        let mut h = vec![T::ZERO; n];
        let (rho, _) = orthogonalize(&self.q, &mut z, &mut h, T::from_f64(self.opts.reorth_eta));
        if le_exact(rho, self.opts.tau::<T>(n + 1) * cn) {
            return Err(StatlaError::RankDeficient { column: j });
        }
        // C3.
        let n1 = n + 1;
        let mut mq = DenseBlock::zeros(m, n1);
        mq.data[..m * n].copy_from_slice(&self.q.data);
        for (d, &s) in mq.data[m * n..].iter_mut().zip(&z) {
            *d = s / rho;
        }
        let mut rh = DenseBlock::zeros(n1, n1);
        for p in 0..n1 {
            let dst = &mut rh.data[p * n1..p * n1 + n];
            match p.cmp(&j) {
                core::cmp::Ordering::Less => {
                    dst.copy_from_slice(&self.r.data[p * n..(p + 1) * n]);
                }
                core::cmp::Ordering::Equal => {
                    dst.copy_from_slice(&h);
                    rh.data[n + p * n1] = rho;
                }
                core::cmp::Ordering::Greater => {
                    dst.copy_from_slice(&self.r.data[(p - 1) * n..p * n]);
                }
            }
        }
        // C4.
        for k in (j + 1..=n).rev() {
            let (g, top) = givens(rh.data[k - 1 + j * n1], rh.data[k + j * n1]);
            rh.data[k - 1 + j * n1] = top;
            rh.data[k + j * n1] = T::ZERO;
            g.rotate_rows(&mut rh, k - 1, k, j + 1);
            g.rotate_cols(&mut mq.data, m, k - 1, k);
        }
        fix_signs(&mut mq, &mut rh, j);
        self.q = mq;
        self.r = rh;
        debug_assert_eq!(self.check_shape(), Ok(()), "(Q2) after insert_col");
        Ok(())
    }

    /// Delete column `j` (Algorithm E).
    ///
    /// # Errors
    /// [`StatlaError::InvalidArgument`].
    pub fn delete_col(&mut self, j: usize) -> Result<(), StatlaError> {
        let (m, n) = (self.nrows(), self.ncols());
        if j >= n {
            return Err(StatlaError::InvalidArgument {
                what: "Qr::delete_col: column index >= n",
            });
        }
        // E1.
        let mut rh = DenseBlock::zeros(n, n - 1);
        for p in 0..n - 1 {
            let src = if p < j { p } else { p + 1 };
            rh.data[p * n..(p + 1) * n].copy_from_slice(&self.r.data[src * n..(src + 1) * n]);
        }
        // E2.
        for p in j..n - 1 {
            let (g, top) = givens(rh.data[p + p * n], rh.data[p + 1 + p * n]);
            rh.data[p + p * n] = top;
            rh.data[p + 1 + p * n] = T::ZERO;
            g.rotate_rows(&mut rh, p, p + 1, p + 1);
            g.rotate_cols(&mut self.q.data, m, p, p + 1);
        }
        // E3.
        let r = drop_last_row(&rh);
        self.q.data.truncate(m * (n - 1));
        self.q.cols = n - 1;
        self.r = r;
        fix_signs(&mut self.q, &mut self.r, j);
        debug_assert_eq!(self.check_shape(), Ok(()), "(Q2) after delete_col");
        Ok(())
    }

    /// `A ← A + u vᵀ` (`u` of length `m`, `v` of length `n`) (Algorithm
    /// R).
    ///
    /// # Errors
    /// [`StatlaError::Dimension`], [`StatlaError::NonFinite`],
    /// [`StatlaError::RankDeficient`] (the result lost rank); unchanged on
    /// error.
    pub fn rank1_update(&mut self, u: &[T], v: &[T]) -> Result<(), StatlaError> {
        let (m, n) = (self.nrows(), self.ncols());
        // R1.
        check_len("u", m, u.len())?;
        check_len("v", n, v.len())?;
        check_finite("u", u)?;
        check_finite("v", v)?;
        let tau = self.opts.tau::<T>(n);
        // R2.
        let un = norm2(u);
        let mut z = u.to_vec();
        let mut w = vec![T::ZERO; n];
        let (mut rho, _) =
            orthogonalize(&self.q, &mut z, &mut w, T::from_f64(self.opts.reorth_eta));
        if le_exact(rho, tau * un) {
            rho = T::ZERO;
        }
        // R3.
        let n1 = n + 1;
        let mut mq = DenseBlock::zeros(m, n1);
        mq.data[..m * n].copy_from_slice(&self.q.data);
        if !is_exact_zero(rho) {
            for (d, &s) in mq.data[m * n..].iter_mut().zip(&z) {
                *d = s / rho;
            }
        }
        let mut rh = DenseBlock::zeros(n1, n);
        for j in 0..n {
            rh.data[j * n1..j * n1 + n].copy_from_slice(&self.r.data[j * n..(j + 1) * n]);
        }
        let mut wh = w;
        wh.push(rho);
        // R4.
        for k in (0..n).rev() {
            let (g, top) = givens(wh[k], wh[k + 1]);
            wh[k] = top;
            wh[k + 1] = T::ZERO;
            g.rotate_rows(&mut rh, k, k + 1, k);
            g.rotate_cols(&mut mq.data, m, k, k + 1);
        }
        // R5.
        let t = wh.first().copied().unwrap_or(T::ZERO);
        for (j, &vj) in v.iter().enumerate() {
            rh.data[j * n1] += t * vj;
        }
        // R6.
        for k in 0..n {
            let (g, top) = givens(rh.data[k + k * n1], rh.data[k + 1 + k * n1]);
            rh.data[k + k * n1] = top;
            rh.data[k + 1 + k * n1] = T::ZERO;
            g.rotate_rows(&mut rh, k, k + 1, k + 1);
            g.rotate_cols(&mut mq.data, m, k, k + 1);
        }
        // R7.
        let mut r = drop_last_row(&rh);
        mq.data.truncate(m * n);
        mq.cols = n;
        fix_signs(&mut mq, &mut r, 0);
        for k in 0..n {
            // The scale of the new column a_k + v_k u, bounded before any
            // cancellation: ‖a_k‖ + |v_k| ‖u‖ (a relative test against the
            // new column's own norm would accept the rounding residue of a
            // cancelled column).
            let cn = norm2(&self.r.data[(k * n)..=(k * n + k)]) + v[k].abs() * un;
            if le_exact(r.data[k + k * n], tau * cn) {
                return Err(StatlaError::RankDeficient { column: k });
            }
        }
        self.q = mq;
        self.r = r;
        debug_assert_eq!(self.check_shape(), Ok(()), "(Q2) after rank1_update");
        Ok(())
    }

    /// `x = argmin ‖A x − b‖` (`x ← R⁻¹ Qᵀ b`); returns `‖b − Q Qᵀ b‖`,
    /// the residual norm.
    ///
    /// # Errors
    /// [`StatlaError::Dimension`].
    pub fn solve_least_squares(&self, b: &[T], x: &mut [T]) -> Result<T, StatlaError> {
        let (m, n) = (self.nrows(), self.ncols());
        check_len("b", m, b.len())?;
        check_len("x", n, x.len())?;
        let mut res = b.to_vec();
        for j in 0..n {
            let qj = &self.q.data[j * m..(j + 1) * m];
            x[j] = dot(qj, b);
        }
        for j in 0..n {
            axpy_sub(x[j], &self.q.data[j * m..(j + 1) * m], &mut res);
        }
        for j in (0..n).rev() {
            let col = &self.r.data[(j * n)..=(j * n + j)];
            let xj = x[j] / col[j];
            x[j] = xj;
            axpy_sub(xj, &col[..j], &mut x[..j]);
        }
        Ok(norm2(&res))
    }

    /// The Q-less view: a [`Cholesky`] of `AᵀA` holding a copy of `R`
    /// (with the default [`CholOptions`]).
    ///
    /// # Errors
    /// As [`Cholesky::from_upper`].
    pub fn to_cholesky(&self) -> Result<Cholesky<T, K>, StatlaError>
    where
        K: Clone,
    {
        Cholesky::from_upper(self.r.clone(), self.kernel.clone(), &CholOptions::default())
    }

    /// The shape part of (Q2): consistent blocks, `R` upper triangular with
    /// finite entries and positive diagonal, `Q` finite. `O(mn)`.
    fn check_shape(&self) -> Result<(), &'static str> {
        let (m, n) = (self.q.rows, self.r.cols);
        if self.r.rows != n
            || self.q.cols != n
            || self.q.data.len() != m * n
            || self.r.data.len() != n * n
        {
            return Err("(Q2) Q and R are not consistent blocks");
        }
        if m < n {
            return Err("(Q2) fewer rows than columns");
        }
        if self.q.data.iter().any(|v| !v.is_finite()) {
            return Err("(Q2) Q has a non-finite entry");
        }
        for j in 0..n {
            for i in 0..n {
                let v = self.r.data[i + j * n];
                if !v.is_finite() {
                    return Err("(Q2) R has a non-finite entry");
                }
                if i > j && !is_exact_zero(v) {
                    return Err("(Q2) R is not upper triangular");
                }
                if i == j && !lt(T::ZERO, v) {
                    return Err("(Q2) R has a nonpositive diagonal entry");
                }
            }
        }
        Ok(())
    }

    /// Invariants (Q1) and the shape part of (Q2), with `c_Q` given:
    /// `‖QᵀQ − I‖_F ≤ c_Q · m · u` (computed in `T`, `O(mn²)`).
    ///
    /// # Errors
    /// The text of the violated invariant.
    pub fn check_invariants(&self, c_q: f64) -> Result<(), &'static str> {
        self.check_shape()?;
        let (m, n) = (self.q.rows, self.q.cols);
        let mut s = T::ZERO;
        for j in 0..n {
            let qj = &self.q.data[j * m..(j + 1) * m];
            for i in 0..n {
                let mut d = dot(&self.q.data[i * m..(i + 1) * m], qj);
                if i == j {
                    d -= T::ONE;
                }
                s += d * d;
            }
        }
        #[allow(clippy::cast_precision_loss)] // a row count, far below 2^53
        let bound = T::from_f64(c_q * m.max(1) as f64) * T::EPSILON * T::from_f64(0.5);
        if !s.is_finite() || !le_exact(s.sqrt(), bound) {
            return Err("(Q1) Q has lost orthogonality");
        }
        Ok(())
    }
}
