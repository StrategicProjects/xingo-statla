//! # Chapter 12.0.2 — Dense views
//!
//! ## Purpose
//! The caller's data — an R matrix, a NumPy array in Fortran order, a
//! block inside a larger array — is borrowed, not copied: [`MatRef`] and
//! [`MatMut`] are column-major views with a leading dimension (decision S3
//! of D-0127). The owned type of the crate's outputs is
//! [`DenseBlock`] of `xingo-sparse` (`ld = rows`).
//!
//! ## Notation
//! A view is `rows × cols` with leading dimension `ld ≥ rows`: `a_ij =
//! data[i + j·ld]`. Column `j` is the contiguous slice `data[j·ld .. j·ld
//! + rows]`. A C-order NumPy array is the view of its *transpose* (a rule
//! for `xingo-ffi`).
//!
//! ## Algorithm V (checked construction)
//! V1. \[Leading dimension.\] If `ld < rows`, `InvalidArgument`.
//! V2. \[Extent.\] If `cols = 0` the extent is `0`; otherwise `need = ld·(cols
//!     − 1) + rows`, computed with checked arithmetic (`InvalidArgument` on
//!     `usize` overflow).
//! V3. \[Length.\] If `data.len() < need`, `Dimension { what: "data",
//!     expected: need, found: data.len() }`. Longer slices are accepted
//!     (the tail is not part of the view).
//!
//! Finiteness is *not* checked here, but by the consumers (each reports
//! the column-major position `i + j·rows`).
//!
//! ## Invariant
//! (W1) `ld ≥ rows` and `data.len() ≥ need` for every live view, so every
//!      accessor below indexes in bounds; the accessors that take indices
//!      return `None` (or an error) out of range instead of panicking.

use xingo_sparse::{DenseBlock, Real};

use crate::StatlaError;

/// A borrowed column-major `rows × cols` matrix with leading dimension
/// `ld ≥ rows`: `a_ij = data[i + j·ld]` (the LAPACK/R layout; a C-order
/// NumPy array is the view of its *transpose*, a rule for `xingo-ffi`).
/// `data.len() ≥ ld·(cols − 1) + rows` when `cols > 0`.
#[derive(Clone, Copy, Debug)]
pub struct MatRef<'a, T: Real = f64> {
    rows: usize,
    cols: usize,
    ld: usize,
    data: &'a [T],
}

/// The mutable counterpart of [`MatRef`].
#[derive(Debug)]
pub struct MatMut<'a, T: Real = f64> {
    rows: usize,
    cols: usize,
    ld: usize,
    data: &'a mut [T],
}

/// Algorithm V: the extent `need` of a view, or the error.
fn extent(rows: usize, cols: usize, ld: usize, len: usize) -> Result<usize, StatlaError> {
    // V1.
    if ld < rows {
        return Err(StatlaError::InvalidArgument { what: "ld < rows" });
    }
    // V2.
    let need = if cols == 0 {
        0
    } else {
        ld.checked_mul(cols - 1)
            .and_then(|v| v.checked_add(rows))
            .ok_or(StatlaError::InvalidArgument {
                what: "view extent overflows usize",
            })?
    };
    // V3.
    if len < need {
        return Err(StatlaError::Dimension {
            what: "data",
            expected: need,
            found: len,
        });
    }
    Ok(need)
}

impl<'a, T: Real> MatRef<'a, T> {
    /// A view of `data` (checked: `ld ≥ rows`, length, no `usize`
    /// overflow; finiteness is *not* checked here, but by the consumers).
    ///
    /// # Errors
    /// [`StatlaError::InvalidArgument`], [`StatlaError::Dimension`].
    pub fn new(rows: usize, cols: usize, ld: usize, data: &'a [T]) -> Result<Self, StatlaError> {
        extent(rows, cols, ld, data.len())?;
        Ok(Self {
            rows,
            cols,
            ld,
            data,
        })
    }
    /// The view of a whole block (`ld = rows`).
    ///
    /// A block whose `data` is shorter than `rows · cols` (its public
    /// fields are inconsistent) gives a view with no columns: never a
    /// panic.
    #[must_use]
    pub fn from_block(b: &'a DenseBlock<T>) -> Self {
        let cols = if b.data.len() >= b.rows.saturating_mul(b.cols) {
            b.cols
        } else {
            0
        };
        Self {
            rows: b.rows,
            cols,
            ld: b.rows,
            data: &b.data,
        }
    }
    /// Number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Number of columns.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }
    /// Leading dimension.
    #[must_use]
    pub fn ld(&self) -> usize {
        self.ld
    }
    /// Entry `(i, j)`; `None` out of range.
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> Option<T> {
        if i < self.rows && j < self.cols {
            Some(self.data[i + j * self.ld])
        } else {
            None
        }
    }
    /// Column `j` (contiguous, length `rows`); `None` out of range.
    #[must_use]
    pub fn col(&self, j: usize) -> Option<&'a [T]> {
        if j < self.cols {
            let s = j * self.ld;
            Some(&self.data[s..s + self.rows])
        } else {
            None
        }
    }
    /// The columns `j0..j1` as a view (no copy).
    ///
    /// # Errors
    /// [`StatlaError::InvalidArgument`] if the range is not inside.
    pub fn cols_range(&self, j0: usize, j1: usize) -> Result<Self, StatlaError> {
        if j0 > j1 || j1 > self.cols {
            return Err(StatlaError::InvalidArgument {
                what: "column range outside the view",
            });
        }
        let start = (j0 * self.ld).min(self.data.len());
        Ok(Self {
            rows: self.rows,
            cols: j1 - j0,
            ld: self.ld,
            data: &self.data[start..],
        })
    }
    /// The underlying slice (crate-internal: the operator loops of Chapter
    /// 12.0.5 index it with `ld`; (W1) keeps every index in bounds).
    pub(crate) fn raw(&self) -> &'a [T] {
        self.data
    }
    /// An owned copy.
    #[must_use]
    pub fn to_block(&self) -> DenseBlock<T> {
        let mut b = DenseBlock::zeros(self.rows, self.cols);
        for j in 0..self.cols {
            let s = j * self.ld;
            b.data[j * self.rows..(j + 1) * self.rows]
                .copy_from_slice(&self.data[s..s + self.rows]);
        }
        b
    }
}

impl<'a, T: Real> MatMut<'a, T> {
    /// A mutable view (same checks as [`MatRef::new`]).
    ///
    /// # Errors
    /// [`StatlaError::InvalidArgument`], [`StatlaError::Dimension`].
    pub fn new(
        rows: usize,
        cols: usize,
        ld: usize,
        data: &'a mut [T],
    ) -> Result<Self, StatlaError> {
        extent(rows, cols, ld, data.len())?;
        Ok(Self {
            rows,
            cols,
            ld,
            data,
        })
    }
    /// The mutable view of a whole block (an inconsistent block gives a
    /// view with no columns, as [`MatRef::from_block`]).
    pub fn from_block(b: &'a mut DenseBlock<T>) -> Self {
        let cols = if b.data.len() >= b.rows.saturating_mul(b.cols) {
            b.cols
        } else {
            0
        };
        Self {
            rows: b.rows,
            cols,
            ld: b.rows,
            data: &mut b.data,
        }
    }
    /// Reborrow as a shared view.
    #[must_use]
    pub fn as_ref(&self) -> MatRef<'_, T> {
        MatRef {
            rows: self.rows,
            cols: self.cols,
            ld: self.ld,
            data: self.data,
        }
    }
    /// Number of rows.
    #[must_use]
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }
    /// Number of columns.
    #[must_use]
    pub(crate) fn cols(&self) -> usize {
        self.cols
    }
    /// Column `j`, mutable; `None` out of range.
    pub fn col_mut(&mut self, j: usize) -> Option<&mut [T]> {
        if j < self.cols {
            let s = j * self.ld;
            Some(&mut self.data[s..s + self.rows])
        } else {
            None
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // the tests compare exact values
mod tests {
    use super::*;

    #[test]
    fn views_index_with_the_leading_dimension() {
        // 2 × 3 inside a 3-row array: column j starts at 3j.
        let data = [1.0, 2.0, 99.0, 3.0, 4.0, 99.0, 5.0, 6.0];
        let a = MatRef::new(2, 3, 3, &data).unwrap();
        assert_eq!((a.rows(), a.cols(), a.ld()), (2, 3, 3));
        assert_eq!(a.get(1, 2), Some(6.0));
        assert_eq!(a.get(2, 0), None);
        assert_eq!(a.get(0, 3), None);
        assert_eq!(a.col(1), Some(&[3.0, 4.0][..]));
        assert_eq!(a.col(3), None);
        let b = a.to_block();
        assert_eq!(b.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let c = a.cols_range(1, 3).unwrap();
        assert_eq!(
            (c.cols(), c.get(0, 0), c.get(1, 1)),
            (2, Some(3.0), Some(6.0))
        );
        let e = a.cols_range(3, 3).unwrap();
        assert_eq!((e.cols(), e.col(0)), (0, None));
        assert!(a.cols_range(2, 1).is_err() && a.cols_range(0, 4).is_err());
        assert_eq!(MatRef::from_block(&b).get(1, 2), Some(6.0));
    }

    #[test]
    fn construction_is_checked() {
        let data = [0.0; 5];
        assert_eq!(
            MatRef::new(3, 2, 2, &data).unwrap_err(),
            StatlaError::InvalidArgument { what: "ld < rows" }
        );
        assert_eq!(
            MatRef::new(3, 2, 3, &data).unwrap_err(),
            StatlaError::Dimension {
                what: "data",
                expected: 6,
                found: 5
            }
        );
        assert!(MatRef::new(usize::MAX, 3, usize::MAX, &data).is_err());
        assert!(MatRef::<f64>::new(0, 0, 0, &[]).is_ok());
        assert!(MatRef::<f64>::new(4, 0, 4, &[]).is_ok());
        let inconsistent = DenseBlock {
            rows: 2,
            cols: 2,
            data: vec![1.0],
        };
        assert_eq!(MatRef::from_block(&inconsistent).cols(), 0);
    }

    #[test]
    fn mutable_views_write_through() {
        let mut data = vec![0.0; 8];
        {
            let mut m = MatMut::new(2, 3, 3, &mut data).unwrap();
            m.col_mut(2).unwrap()[1] = 8.0;
            assert!(m.col_mut(3).is_none());
            assert_eq!(m.as_ref().get(1, 2), Some(8.0));
        }
        assert_eq!(data[7], 8.0);
        let mut b = DenseBlock::<f64>::zeros(2, 2);
        MatMut::from_block(&mut b).col_mut(1).unwrap()[0] = 3.0;
        assert_eq!(b.get(0, 1), 3.0);
        assert!(MatMut::new(3, 1, 2, &mut data).is_err());
    }
}
