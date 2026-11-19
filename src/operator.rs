//! # Chapter 12.0.5 — Linear operators
//!
//! ## Purpose
//! The randomized and Krylov drivers ([`crate::rsvd()`], [`crate::svds()`],
//! later `eigs`/`eigsh`) never look inside the matrix: they only ask for
//! `y ← A x` and `y ← Aᵀ x`. [`LinearOperator`] and [`AdjointOperator`] are
//! that interface (decision S5 of D-0127): `&mut self` because useful
//! operators keep state (a counter, a workspace, a factorization whose solve
//! needs `&mut`), fallible because a user's inner solve can fail and the
//! crate promises errors, not panics.
//!
//! ## Notation
//! `A ∈ T^{m×n}` (`m = nrows()`, `n = ncols()`); `x`, `y` vectors; `X ∈
//! T^{n×l}`, `Y ∈ T^{m×l}` blocks (column-major views, Chapter 12.0.2).
//! `y` is always *overwritten*, never accumulated.
//!
//! ## Algorithm A (the implementations of this crate)
//! A1. \[Check.\] `|x| = n`, `|y| = m` (for `Aᵀ`: `|x| = m`, `|y| = n`);
//!     otherwise an [`OperatorError`] (the drivers never trigger it; a direct
//!     caller may).
//! A2. \[Dense, `A x`.\] `y ← 0`; for `k = 0, …, n − 1`: `y ← y + a_k x_k`
//!     (column-oriented gaxpy, `a_k` the `k`-th column).
//! A3. \[Dense, `Aᵀ x`.\] `y_j ← a_jᵀ x`, the dot product in index order.
//! A4. \[CSC.\] `A x` as A2 over the stored entries (skipping `x_k = 0`, as
//!     `CscMatrix::gaxpy`); `Aᵀ x` as A3 over the stored entries of each
//!     column.
//! A5. \[CSR.\] `A x`: a dot product per row over its stored entries; `Aᵀ
//!     x`: `y ← 0`, then for every row `i` in order, `y_j ← y_j + a_ij x_i`
//!     (scatter).
//! A6. \[Blocks.\] `Y ← A X` column by column. The dense implementations
//!     override the default with the *same* operations in the same order,
//!     so `apply_block` and a loop of `apply` give the same bits
//!     (determinism R2 does not depend on which one a driver calls).
//!
//! A sparse matrix whose public fields were made inconsistent (an index out
//! of range) gives an [`OperatorError`], not a panic: every index is read
//! with a checked access.
//!
//! ## Invariants
//! (O1) `apply` writes every entry of `y` (no stale values survive).
//! (O2) For the implementations here, `apply_block` equals, bit for bit, the
//!      loop of `apply` over the columns (tested).
//!
//! ## References
//! Lehoucq, Sorensen & Yang, *ARPACK Users' Guide* (1998), reverse
//! communication (the interface this trait replaces); faer 0.24.4
//! `matrix_free::LinOp` (read, not copied). Cited from memory.

use xingo_sparse::{CscMatrix, CsrMatrix, DenseBlock, Real};

use crate::view::{MatMut, MatRef};

/// An error reported by a [`LinearOperator`] (for instance a closure whose
/// inner solve failed). Carries a message only, so that
/// [`crate::StatlaError`] stays `Clone + PartialEq`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperatorError {
    /// What went wrong.
    pub message: String,
}

impl OperatorError {
    fn new(message: &str) -> Self {
        Self {
            message: message.to_owned(),
        }
    }
}

/// Step A1: the length check of an application.
fn check(x: usize, xe: usize, y: usize, ye: usize) -> Result<(), OperatorError> {
    if x == xe && y == ye {
        Ok(())
    } else {
        Err(OperatorError::new(
            "operator applied to vectors of the wrong length",
        ))
    }
}

/// `y ← A x` for an `nrows × ncols` operator known only by its action.
///
/// `&mut self` because the useful operators keep state: a closure that
/// counts calls, a shift-invert operator whose `Lu::ftran` needs `&mut`,
/// a workspace. Fallible because a user's solve can fail and the crate
/// promises errors, not panics. The drivers call `apply` with `x.len() ==
/// ncols()` and `y.len() == nrows()` — implementors may rely on it.
/// `y` is overwritten (not accumulated).
pub trait LinearOperator<T: Real = f64> {
    /// Rows of `A` (length of `y`).
    fn nrows(&self) -> usize;
    /// Columns of `A` (length of `x`).
    fn ncols(&self) -> usize;
    /// `y ← A x`.
    ///
    /// # Errors
    /// Whatever the operator reports.
    fn apply(&mut self, x: &[T], y: &mut [T]) -> Result<(), OperatorError>;
    /// `Y ← A X` for a block of columns; the default loops over
    /// [`LinearOperator::apply`]. Dense and sparse implementations
    /// override it (one pass over `A` for `l` columns — the cost that
    /// matters in [`crate::rsvd()`]).
    ///
    /// # Errors
    /// Whatever the operator reports; an [`OperatorError`] if the shapes
    /// of `X` and `Y` do not match the operator.
    fn apply_block(&mut self, x: MatRef<'_, T>, mut y: MatMut<'_, T>) -> Result<(), OperatorError> {
        check_block(
            x.rows(),
            self.ncols(),
            y.rows(),
            self.nrows(),
            x.cols(),
            y.cols(),
        )?;
        for c in 0..x.cols() {
            let (Some(xc), Some(yc)) = (x.col(c), y.col_mut(c)) else {
                return Err(OperatorError::new("block column out of range"));
            };
            self.apply(xc, yc)?;
        }
        Ok(())
    }
}

/// An operator that also applies its transpose: `y ← Aᵀ x`. Needed by
/// [`crate::rsvd()`] and [`crate::svds()`].
pub trait AdjointOperator<T: Real = f64>: LinearOperator<T> {
    /// `y ← Aᵀ x` (`x.len() == nrows()`, `y.len() == ncols()`).
    ///
    /// # Errors
    /// Whatever the operator reports.
    fn apply_t(&mut self, x: &[T], y: &mut [T]) -> Result<(), OperatorError>;
    /// Block version; default loops over [`AdjointOperator::apply_t`].
    ///
    /// # Errors
    /// Whatever the operator reports; an [`OperatorError`] if the shapes
    /// do not match.
    fn apply_t_block(
        &mut self,
        x: MatRef<'_, T>,
        mut y: MatMut<'_, T>,
    ) -> Result<(), OperatorError> {
        check_block(
            x.rows(),
            self.nrows(),
            y.rows(),
            self.ncols(),
            x.cols(),
            y.cols(),
        )?;
        for c in 0..x.cols() {
            let (Some(xc), Some(yc)) = (x.col(c), y.col_mut(c)) else {
                return Err(OperatorError::new("block column out of range"));
            };
            self.apply_t(xc, yc)?;
        }
        Ok(())
    }
}

/// The shape check of a block application (step A1 for blocks).
fn check_block(
    xr: usize,
    xe: usize,
    yr: usize,
    ye: usize,
    xc: usize,
    yc: usize,
) -> Result<(), OperatorError> {
    if xr == xe && yr == ye && xc == yc {
        Ok(())
    } else {
        Err(OperatorError::new(
            "operator applied to blocks of the wrong shape",
        ))
    }
}

impl<T: Real, O: LinearOperator<T> + ?Sized> LinearOperator<T> for &mut O {
    fn nrows(&self) -> usize {
        (**self).nrows()
    }
    fn ncols(&self) -> usize {
        (**self).ncols()
    }
    fn apply(&mut self, x: &[T], y: &mut [T]) -> Result<(), OperatorError> {
        (**self).apply(x, y)
    }
    fn apply_block(&mut self, x: MatRef<'_, T>, y: MatMut<'_, T>) -> Result<(), OperatorError> {
        (**self).apply_block(x, y)
    }
}

impl<T: Real, O: AdjointOperator<T> + ?Sized> AdjointOperator<T> for &mut O {
    fn apply_t(&mut self, x: &[T], y: &mut [T]) -> Result<(), OperatorError> {
        (**self).apply_t(x, y)
    }
    fn apply_t_block(&mut self, x: MatRef<'_, T>, y: MatMut<'_, T>) -> Result<(), OperatorError> {
        (**self).apply_t_block(x, y)
    }
}

// ------------------------------------------------------------ dense (A2, A3)

/// Step A2 on a column-major `rows × cols` array with leading dimension
/// `ld`.
fn dense_apply<T: Real>(data: &[T], rows: usize, cols: usize, ld: usize, x: &[T], y: &mut [T]) {
    y.fill(T::ZERO);
    for k in 0..cols {
        let xk = x[k];
        let a = &data[k * ld..k * ld + rows];
        for (yi, &aik) in y.iter_mut().zip(a) {
            *yi += aik * xk;
        }
    }
}

/// Step A3.
fn dense_apply_t<T: Real>(data: &[T], rows: usize, cols: usize, ld: usize, x: &[T], y: &mut [T]) {
    for (j, yj) in y.iter_mut().enumerate().take(cols) {
        let a = &data[j * ld..j * ld + rows];
        *yj = a.iter().zip(x).map(|(&p, &q)| p * q).sum();
    }
}

/// Step A6 for dense storage: the columns of the block, each by A2 or A3.
fn dense_block<T: Real>(
    data: &[T],
    (rows, cols, ld): (usize, usize, usize),
    transpose: bool,
    x: MatRef<'_, T>,
    mut y: MatMut<'_, T>,
) -> Result<(), OperatorError> {
    let (xe, ye) = if transpose {
        (rows, cols)
    } else {
        (cols, rows)
    };
    check_block(x.rows(), xe, y.rows(), ye, x.cols(), y.cols())?;
    for c in 0..x.cols() {
        let (Some(xc), Some(yc)) = (x.col(c), y.col_mut(c)) else {
            return Err(OperatorError::new("block column out of range"));
        };
        if transpose {
            dense_apply_t(data, rows, cols, ld, xc, yc);
        } else {
            dense_apply(data, rows, cols, ld, xc, yc);
        }
    }
    Ok(())
}

/// The storage of a block, or an error if its public fields disagree.
fn block_parts<T: Real>(b: &DenseBlock<T>) -> Result<(usize, usize, usize), OperatorError> {
    if b.data.len() == b.rows.saturating_mul(b.cols) {
        Ok((b.rows, b.cols, b.rows))
    } else {
        Err(OperatorError::new(
            "DenseBlock: data.len() differs from rows * cols",
        ))
    }
}

/// Dense, with the reference loops (fixed summation order).
impl<T: Real> LinearOperator<T> for DenseBlock<T> {
    fn nrows(&self) -> usize {
        self.rows
    }
    fn ncols(&self) -> usize {
        self.cols
    }
    fn apply(&mut self, x: &[T], y: &mut [T]) -> Result<(), OperatorError> {
        let (r, c, ld) = block_parts(self)?;
        check(x.len(), c, y.len(), r)?;
        dense_apply(&self.data, r, c, ld, x, y);
        Ok(())
    }
    fn apply_block(&mut self, x: MatRef<'_, T>, y: MatMut<'_, T>) -> Result<(), OperatorError> {
        let parts = block_parts(self)?;
        dense_block(&self.data, parts, false, x, y)
    }
}
impl<T: Real> AdjointOperator<T> for DenseBlock<T> {
    fn apply_t(&mut self, x: &[T], y: &mut [T]) -> Result<(), OperatorError> {
        let (r, c, ld) = block_parts(self)?;
        check(x.len(), r, y.len(), c)?;
        dense_apply_t(&self.data, r, c, ld, x, y);
        Ok(())
    }
    fn apply_t_block(&mut self, x: MatRef<'_, T>, y: MatMut<'_, T>) -> Result<(), OperatorError> {
        let parts = block_parts(self)?;
        dense_block(&self.data, parts, true, x, y)
    }
}

/// A borrowed dense matrix (zero copy from R/NumPy through `xingo-ffi`).
impl<T: Real> LinearOperator<T> for MatRef<'_, T> {
    fn nrows(&self) -> usize {
        self.rows()
    }
    fn ncols(&self) -> usize {
        self.cols()
    }
    fn apply(&mut self, x: &[T], y: &mut [T]) -> Result<(), OperatorError> {
        check(x.len(), self.cols(), y.len(), self.rows())?;
        dense_apply(self.raw(), self.rows(), self.cols(), self.ld(), x, y);
        Ok(())
    }
    fn apply_block(&mut self, x: MatRef<'_, T>, y: MatMut<'_, T>) -> Result<(), OperatorError> {
        let parts = (self.rows(), self.cols(), self.ld());
        dense_block(self.raw(), parts, false, x, y)
    }
}
impl<T: Real> AdjointOperator<T> for MatRef<'_, T> {
    fn apply_t(&mut self, x: &[T], y: &mut [T]) -> Result<(), OperatorError> {
        check(x.len(), self.rows(), y.len(), self.cols())?;
        dense_apply_t(self.raw(), self.rows(), self.cols(), self.ld(), x, y);
        Ok(())
    }
    fn apply_t_block(&mut self, x: MatRef<'_, T>, y: MatMut<'_, T>) -> Result<(), OperatorError> {
        let parts = (self.rows(), self.cols(), self.ld());
        dense_block(self.raw(), parts, true, x, y)
    }
}

// ------------------------------------------------------------ sparse (A4, A5)

const BAD_SPARSE: &str = "sparse matrix with inconsistent fields (index out of range)";

/// The stored entries `ptr[j]..ptr[j+1]` of a compressed matrix, checked.
fn segment<'a, T>(
    ptr: &[usize],
    idx: &'a [u32],
    val: &'a [T],
    j: usize,
) -> Result<(&'a [u32], &'a [T]), OperatorError> {
    let (Some(&a), Some(&b)) = (ptr.get(j), ptr.get(j + 1)) else {
        return Err(OperatorError::new(BAD_SPARSE));
    };
    match (idx.get(a..b), val.get(a..b)) {
        (Some(i), Some(v)) => Ok((i, v)),
        _ => Err(OperatorError::new(BAD_SPARSE)),
    }
}

/// `y ← y + a x_k` over stored entries (A2 on a sparse column), checked.
fn scatter<T: Real>(rows: &[u32], vals: &[T], t: T, y: &mut [T]) -> Result<(), OperatorError> {
    for (&i, &v) in rows.iter().zip(vals) {
        let Some(yi) = y.get_mut(i as usize) else {
            return Err(OperatorError::new(BAD_SPARSE));
        };
        *yi += v * t;
    }
    Ok(())
}

/// `Σ a_i x_i` over stored entries (A3 on a sparse column), checked.
fn gather<T: Real>(rows: &[u32], vals: &[T], x: &[T]) -> Result<T, OperatorError> {
    let mut s = T::ZERO;
    for (&i, &v) in rows.iter().zip(vals) {
        let Some(&xi) = x.get(i as usize) else {
            return Err(OperatorError::new(BAD_SPARSE));
        };
        s += v * xi;
    }
    Ok(s)
}

/// `CscMatrix::gaxpy` columnwise; the transpose is a dot product per
/// column (no transposed copy).
impl<T: Real> LinearOperator<T> for CscMatrix<T> {
    fn nrows(&self) -> usize {
        self.n_rows()
    }
    fn ncols(&self) -> usize {
        self.n_cols()
    }
    fn apply(&mut self, x: &[T], y: &mut [T]) -> Result<(), OperatorError> {
        check(x.len(), self.n_cols(), y.len(), self.n_rows())?;
        y.fill(T::ZERO);
        for (j, &xj) in x.iter().enumerate() {
            if crate::real::is_exact_zero(xj) {
                continue;
            }
            let (r, v) = segment(&self.col_ptr, &self.row_idx, &self.val, j)?;
            scatter(r, v, xj, y)?;
        }
        Ok(())
    }
}
impl<T: Real> AdjointOperator<T> for CscMatrix<T> {
    fn apply_t(&mut self, x: &[T], y: &mut [T]) -> Result<(), OperatorError> {
        check(x.len(), self.n_rows(), y.len(), self.n_cols())?;
        for (j, yj) in y.iter_mut().enumerate() {
            let (r, v) = segment(&self.col_ptr, &self.row_idx, &self.val, j)?;
            *yj = gather(r, v, x)?;
        }
        Ok(())
    }
}
/// A dot product per row; the transpose scatters.
impl<T: Real> LinearOperator<T> for CsrMatrix<T> {
    fn nrows(&self) -> usize {
        self.n_rows()
    }
    fn ncols(&self) -> usize {
        self.n_cols()
    }
    fn apply(&mut self, x: &[T], y: &mut [T]) -> Result<(), OperatorError> {
        check(x.len(), self.n_cols(), y.len(), self.n_rows())?;
        for (i, yi) in y.iter_mut().enumerate() {
            let (c, v) = segment(&self.row_ptr, &self.col_idx, &self.val, i)?;
            *yi = gather(c, v, x)?;
        }
        Ok(())
    }
}
impl<T: Real> AdjointOperator<T> for CsrMatrix<T> {
    fn apply_t(&mut self, x: &[T], y: &mut [T]) -> Result<(), OperatorError> {
        check(x.len(), self.n_rows(), y.len(), self.n_cols())?;
        y.fill(T::ZERO);
        for (i, &xi) in x.iter().enumerate() {
            let (c, v) = segment(&self.row_ptr, &self.col_idx, &self.val, i)?;
            scatter(c, v, xi, y)?;
        }
        Ok(())
    }
}

// ------------------------------------------------------------ closures

/// An operator from a closure `f(x, y)` writing `y ← A x` (PLAN L3.1's
/// "operador-closure"). Build with [`from_fn`].
#[derive(Debug)]
pub struct FnOperator<F> {
    nrows: usize,
    ncols: usize,
    f: F,
}

/// An operator *and* its transpose from two closures. Build with
/// [`from_fns`].
#[derive(Debug)]
pub struct FnAdjointOperator<F, G> {
    nrows: usize,
    ncols: usize,
    f: F,
    g: G,
}

/// `FnOperator` of the closure `f`.
pub fn from_fn<T, F>(nrows: usize, ncols: usize, f: F) -> FnOperator<F>
where
    T: Real,
    F: FnMut(&[T], &mut [T]) -> Result<(), OperatorError>,
{
    FnOperator { nrows, ncols, f }
}

/// `FnAdjointOperator` of `f` (`y ← A x`) and `g` (`y ← Aᵀ x`).
pub fn from_fns<T, F, G>(nrows: usize, ncols: usize, f: F, g: G) -> FnAdjointOperator<F, G>
where
    T: Real,
    F: FnMut(&[T], &mut [T]) -> Result<(), OperatorError>,
    G: FnMut(&[T], &mut [T]) -> Result<(), OperatorError>,
{
    FnAdjointOperator { nrows, ncols, f, g }
}

impl<T: Real, F: FnMut(&[T], &mut [T]) -> Result<(), OperatorError>> LinearOperator<T>
    for FnOperator<F>
{
    fn nrows(&self) -> usize {
        self.nrows
    }
    fn ncols(&self) -> usize {
        self.ncols
    }
    fn apply(&mut self, x: &[T], y: &mut [T]) -> Result<(), OperatorError> {
        (self.f)(x, y)
    }
}

impl<T, F, G> LinearOperator<T> for FnAdjointOperator<F, G>
where
    T: Real,
    F: FnMut(&[T], &mut [T]) -> Result<(), OperatorError>,
{
    fn nrows(&self) -> usize {
        self.nrows
    }
    fn ncols(&self) -> usize {
        self.ncols
    }
    fn apply(&mut self, x: &[T], y: &mut [T]) -> Result<(), OperatorError> {
        (self.f)(x, y)
    }
}

impl<T, F, G> AdjointOperator<T> for FnAdjointOperator<F, G>
where
    T: Real,
    F: FnMut(&[T], &mut [T]) -> Result<(), OperatorError>,
    G: FnMut(&[T], &mut [T]) -> Result<(), OperatorError>,
{
    fn apply_t(&mut self, x: &[T], y: &mut [T]) -> Result<(), OperatorError> {
        (self.g)(x, y)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // the tests compare exact values
mod tests {
    use super::*;
    use xingo_sparse::{Col, Row};

    /// `A = [1 0 2; 0 3 0; 4 0 5; 0 6 0]` (4 × 3) in every storage.
    fn dense() -> DenseBlock<f64> {
        DenseBlock {
            rows: 4,
            cols: 3,
            data: vec![1.0, 0.0, 4.0, 0.0, 0.0, 3.0, 0.0, 6.0, 2.0, 0.0, 5.0, 0.0],
        }
    }
    fn csc() -> CscMatrix<f64> {
        let d = dense();
        let mut t = Vec::new();
        for j in 0..3 {
            for i in 0..4 {
                if d.get(i, j) != 0.0 {
                    t.push((Row::from_usize(i), Col::from_usize(j), d.get(i, j)));
                }
            }
        }
        CscMatrix::from_triplets(4, 3, &t).unwrap()
    }

    fn check_op<O: AdjointOperator<f64>>(op: &mut O) {
        let x = [1.0, -2.0, 0.5];
        let mut y = [9.0; 4];
        op.apply(&x, &mut y).unwrap();
        assert_eq!(y, [2.0, -6.0, 6.5, -12.0]);
        let z = [1.0, 1.0, -1.0, 2.0];
        let mut w = [9.0; 3];
        op.apply_t(&z, &mut w).unwrap();
        assert_eq!(w, [-3.0, 15.0, -3.0]);
        assert!(op.apply(&x[..2], &mut y).is_err());
        assert!(op.apply_t(&z, &mut y).is_err());
        // (O2) blocks equal the loop of single applications.
        let xb = [1.0, -2.0, 0.5, 0.3, 0.7, -1.1];
        let mut yb = [0.0; 8];
        op.apply_block(
            MatRef::new(3, 2, 3, &xb).unwrap(),
            MatMut::new(4, 2, 4, &mut yb).unwrap(),
        )
        .unwrap();
        let mut y2 = [0.0; 4];
        op.apply(&xb[3..], &mut y2).unwrap();
        assert_eq!(&yb[4..], &y2);
        let mut wb = [0.0; 6];
        let zb = [z, [0.5, -0.25, 2.0, 1.5]].concat();
        op.apply_t_block(
            MatRef::new(4, 2, 4, &zb).unwrap(),
            MatMut::new(3, 2, 3, &mut wb).unwrap(),
        )
        .unwrap();
        let mut w2 = [0.0; 3];
        op.apply_t(&zb[4..], &mut w2).unwrap();
        assert_eq!(&wb[3..], &w2);
        assert!(
            op.apply_block(
                MatRef::new(3, 2, 3, &xb).unwrap(),
                MatMut::new(3, 2, 3, &mut wb).unwrap()
            )
            .is_err()
        );
    }

    /// `check_op` on an operator passed by value (used with `&mut O`).
    fn by_value<O: AdjointOperator<f64>>(mut o: O) {
        check_op(&mut o);
    }

    #[test]
    fn every_storage_applies_the_same_matrix() {
        check_op(&mut dense());
        check_op(&mut csc());
        check_op(&mut csc().to_csr());
        let d = dense();
        // A view with ld = 5 inside a padded array.
        let mut padded = vec![f64::NAN; 15];
        for j in 0..3 {
            for i in 0..4 {
                padded[i + 5 * j] = d.get(i, j);
            }
        }
        check_op(&mut MatRef::new(4, 3, 5, &padded).unwrap());
        let (a, b) = (d.clone(), d.clone());
        let mut f = from_fns(
            4,
            3,
            move |x: &[f64], y: &mut [f64]| a.clone().apply(x, y),
            move |x: &[f64], y: &mut [f64]| b.clone().apply_t(x, y),
        );
        check_op(&mut f);
        // `&mut O` is itself an operator (the blanket impls).
        by_value(&mut f);
        let mut g = from_fn(1, 1, |_: &[f64], _: &mut [f64]| {
            Err(OperatorError::new("inner solve failed"))
        });
        assert_eq!(
            g.apply(&[1.0], &mut [0.0]).unwrap_err().message,
            "inner solve failed"
        );
        assert_eq!((g.nrows(), g.ncols()), (1, 1));
    }

    #[test]
    fn inconsistent_fields_are_errors_not_panics() {
        let mut a = csc();
        a.row_idx[0] = 99;
        assert!(a.apply(&[1.0, 1.0, 1.0], &mut [0.0; 4]).is_err());
        assert!(a.apply_t(&[1.0; 4], &mut [0.0; 3]).is_err());
        let mut c = csc().to_csr();
        c.row_ptr[1] = 1000;
        assert!(c.apply(&[1.0; 3], &mut [0.0; 4]).is_err());
        let mut d = dense();
        d.data.pop();
        assert!(d.apply(&[1.0; 3], &mut [0.0; 4]).is_err());
        assert!(d.apply_t(&[1.0; 4], &mut [0.0; 3]).is_err());
    }
}
