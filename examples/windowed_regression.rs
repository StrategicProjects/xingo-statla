//! Sliding-window least squares (Chapter 12.3): a window of `w`
//! observations moves along a stream; at each step the oldest row leaves
//! (`Qr::delete_row`) and a new one enters (`Qr::insert_row`), `O(wp)`
//! each instead of the `O(wp²)` of refitting. The coefficients are
//! compared with a refit from scratch at every step, and the Q-less
//! variant (`Cholesky` update/downdate of `XᵀX`) runs alongside.
//!
//! `cargo run -p xingo-statla --release --example windowed_regression`

use xingo_statla::{MatRef, Qr, QrOptions, StatlaError};

/// A deterministic stream: `y_t = 1 + 2 s_t − 0.5 s_t² + noise`, with the
/// regressors `(1, s_t, s_t²)`.
fn observation(t: usize) -> ([f64; 3], f64) {
    #[allow(clippy::cast_precision_loss)]
    let s = (t as f64 * 0.37).sin() * 3.0;
    #[allow(clippy::cast_precision_loss)]
    let noise = ((t * 7919 % 1000) as f64 / 1000.0 - 0.5) * 0.1;
    ([1.0, s, s * s], 1.0 + 2.0 * s - 0.5 * s * s + noise)
}

fn refit(rows: &[[f64; 3]], y: &[f64]) -> Result<Vec<f64>, StatlaError> {
    let m = rows.len();
    let mut a = vec![0.0; 3 * m];
    for (i, r) in rows.iter().enumerate() {
        for j in 0..3 {
            a[i + j * m] = r[j];
        }
    }
    let f = Qr::factor(MatRef::new(m, 3, m, &a)?, &QrOptions::default())?;
    let mut x = vec![0.0; 3];
    f.solve_least_squares(y, &mut x)?;
    Ok(x)
}

fn main() -> Result<(), StatlaError> {
    let w = 40;
    let steps = 2000;
    let mut rows: Vec<[f64; 3]> = (0..w).map(|t| observation(t).0).collect();
    let mut y: Vec<f64> = (0..w).map(|t| observation(t).1).collect();
    let mut a = vec![0.0; 3 * w];
    for (i, r) in rows.iter().enumerate() {
        for j in 0..3 {
            a[i + j * w] = r[j];
        }
    }
    let mut qr = Qr::factor(MatRef::new(w, 3, w, &a)?, &QrOptions::default())?;
    // Q-less: the Cholesky factor of XᵀX, updated by each row.
    let mut chol = qr.to_cholesky()?;
    let mut worst = 0.0f64;
    let mut worst_chol = 0.0f64;
    for t in w..w + steps {
        // The oldest observation (row 0) leaves, the new one is appended.
        qr.delete_row(0)?;
        chol.downdate(&rows[0])?;
        rows.remove(0);
        y.remove(0);
        let (r, yt) = observation(t);
        qr.insert_row(w - 1, &r)?;
        chol.update(&r)?;
        rows.push(r);
        y.push(yt);
        let mut x = vec![0.0; 3];
        qr.solve_least_squares(&y, &mut x)?;
        let x0 = refit(&rows, &y)?;
        // Normal equations through the Q-less factor: (XᵀX) x = Xᵀy.
        let mut xc: Vec<f64> = (0..3)
            .map(|j| rows.iter().zip(&y).map(|(r, v)| r[j] * v).sum())
            .collect();
        chol.solve(&mut xc)?;
        for j in 0..3 {
            worst = worst.max((x[j] - x0[j]).abs() / x0[j].abs().max(1.0));
            worst_chol = worst_chol.max((xc[j] - x0[j]).abs() / x0[j].abs().max(1.0));
        }
    }
    println!(
        "window {w}, {steps} steps: max coefficient error vs refit {worst:.2e} (QR), {worst_chol:.2e} (Q-less Cholesky)"
    );
    if let Err(e) = qr.check_invariants(16.0) {
        panic!("{e}");
    }
    assert!(worst < 1e-12, "QR window drifted: {worst:e}");
    assert!(worst_chol < 1e-9, "Cholesky window drifted: {worst_chol:e}");
    Ok(())
}
