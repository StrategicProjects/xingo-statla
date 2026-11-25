//! Acceptance tests of Chapter 12.3 (`Qr`), task L3.2, in `f64` and
//! `DoubleDouble`: after every operation (Q1) orthogonality, (Q2) `QR =
//! A_t` and the shape of `R` (through `check_invariants`), agreement with a
//! refactorization (iii), and (Q3) transactional failures; every error path
//! of the contract.
//!
//! Constant (measured by the example `accuracy`, with a margin): `C_Q =
//! 16` — the largest ratios seen are `‖QᵀQ − I‖_F = 8.0 m u` and `‖QR −
//! A_t‖_F = 5.4 m u ‖A_t‖_F` (`delete_row` on `2 × 1` and `3 × 2`); for `m
//! ≥ 100` both are `≤ 0.6 m u`.

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::if_not_else,
    clippy::items_after_statements
)] // tests: sizes as f64, exact checks

mod support;

use proptest::prelude::*;
use support::{
    Grading, QrOp, Rng, Scalar, apply_op, bits, conv, fro, orthogonality, qr_backward, qr_case,
    qr_graded_case, scalings, to_dd, with_cond,
};
use xingo_extprec::DoubleDouble as Dd;
use xingo_statla::{MatRef, Qr, QrOptions, StatlaError};

/// (Q1), (Q2) constant `c_Q`.
const C_Q: f64 = 16.0;
/// Criterion (iii-a), provisional (question Q-L32-2): `fwd ≤ C_F · κ(AᵀA) ·
/// u` per instance.
const C_F: f64 = 8.0;

fn log_kappa() -> impl Strategy<Value = f64> {
    (1.0f64..=12.0).prop_map(|e| 10f64.powf(e))
}

fn op() -> impl Strategy<Value = QrOp> {
    prop::sample::select(QrOp::ALL.to_vec())
}

fn check_case<T: Scalar>(
    m: usize,
    n: usize,
    kappa: f64,
    op: QrOp,
    seed: u64,
) -> Result<(), TestCaseError> {
    let c = qr_case::<T>(m, n, kappa, op, seed).map_err(TestCaseError::fail)?;
    let mu = c.m as f64 * T::u();
    prop_assert!(
        c.orth <= C_Q * mu,
        "{} {} (Q1) {:e} > {C_Q}·m·u",
        T::NAME,
        op.name(),
        c.orth
    );
    prop_assert!(
        c.bwd_path <= C_Q * mu,
        "{} {} (Q2, path) {:e}",
        T::NAME,
        op.name(),
        c.bwd_path
    );
    if c.shrink >= 0.5 {
        prop_assert!(
            c.bwd <= C_Q * mu,
            "{} {} (Q2) {:e} > {C_Q}·m·u",
            T::NAME,
            op.name(),
            c.bwd
        );
    }
    prop_assert!(
        c.fwd <= C_F * c.kappa * T::u(),
        "{} {} (iii-a) {:e} > {C_F}·κ·u",
        T::NAME,
        op.name(),
        c.fwd
    );
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(support::cases(64)))]

    /// One operation in `f64`: `n ≤ 200`, `m ≤ n + 40`.
    #[test]
    fn one_operation_f64(n in 1usize..=200, extra in 1usize..=40, kappa in log_kappa(), op in op(), seed in any::<u64>()) {
        check_case::<f64>(n + extra, n, kappa, op, seed)?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(support::cases(16)))]

    /// One operation in `DoubleDouble`.
    #[test]
    fn one_operation_dd(n in 1usize..=120, extra in 1usize..=20, kappa in log_kappa(), op in op(), seed in any::<u64>()) {
        check_case::<Dd>(n + extra, n, kappa, op, seed)?;
    }
}

// ------------------------------------------- graded and badly scaled

fn grading() -> impl Strategy<Value = Grading> {
    prop::sample::select(vec![Grading::Graded, Grading::Random])
}

/// The rank tolerance of the strongly row-scaled family (see
/// `check_graded`).
fn tiny_rank_tol() -> QrOptions {
    QrOptions {
        rank_tol: f64::MIN_POSITIVE,
        ..QrOptions::default()
    }
}

/// One graded case (issue xingo-statla#6): `A = D_r A' D_c` with row and
/// column scalings, one operation whose new data carries the same
/// scalings. Householder QR and the Givens updates only combine entries
/// of one column of `R` (rows of `R` are rotated, the scale `d_j` of
/// column `j` is common), so the backward error holds column by column:
/// (Q2) in the units of `D_c` — normalized by the norm along the path, as
/// in `sequence`, because a deleted row can carry most of a column when
/// the rows are badly scaled — and (Q1). (iii-a) in the units of `D_c`
/// against `κ(R⋆D_c⁻¹)²`, where the operation did not shrink `A D_c⁻¹`
/// below half (the forward error after a deletion that cancels is
/// measured against the norm before it, which (iii-a) does not model).
///
/// Two families. *Column-scaled* (any `D_c`, `D_r` of at most 8 decades,
/// the five operations, default options): a refusal `RankDeficient` is
/// allowed (the matrix is then numerically rank deficient by the
/// definition of step F5) and skipped. *Row-scaled* (`D_r` of up to 40
/// decades): then `r_kk / ‖a_k‖` is of the order of the spread of the
/// rows, below the default rank test `n ε`, and the complement of the
/// range of `Q` lies on the small rows — a new column or the complement
/// of a heavy row is dependent to working precision, and the default
/// options rightly refuse it. That family runs with the rank test at the
/// underflow threshold and only the operations that take no rank decision
/// on a complement vector, `insert_row` and `delete_col`.
fn check_graded<T: Scalar>(
    m: usize,
    n: usize,
    kappa: f64,
    op: QrOp,
    (gr, gc): (f64, f64),
    (kr, kc): (Grading, Grading),
    seed: u64,
) -> Result<Option<(support::QrCase, f64)>, TestCaseError> {
    let mut rng = Rng::new(seed ^ 0x0d5c);
    let dr = scalings(&mut rng, m, gr, kr);
    let dc = scalings(&mut rng, n, gc, kc);
    let row_scaled = gr > 8.0;
    let opts = if row_scaled {
        prop_assert!(matches!(op, QrOp::InsertRow | QrOp::DeleteCol));
        tiny_rank_tol()
    } else {
        QrOptions::default()
    };
    let (c, plain) = match qr_graded_case::<T>(m, n, kappa, op, &dr, &dc, &opts, seed) {
        Ok(v) => v,
        Err(msg) if !row_scaled && msg.contains("rank deficient") => return Ok(None),
        Err(msg) => return Err(TestCaseError::fail(msg)),
    };
    let mu = c.m as f64 * T::u();
    let tag = format!(
        "{} {} rows {kr:?} {gr:.1}, cols {kc:?} {gc:.1}",
        T::NAME,
        op.name()
    );
    prop_assert!(c.orth <= C_Q * mu, "{tag} (Q1) {:e}", c.orth);
    prop_assert!(plain <= C_Q * mu, "{tag} (Q2, normwise, path) {plain:e}");
    prop_assert!(
        c.bwd_path <= C_Q * mu,
        "{tag} (Q2, columns, path) {:e}",
        c.bwd_path
    );
    if c.shrink >= 0.5 {
        prop_assert!(c.bwd <= C_Q * mu, "{tag} (Q2, columns) {:e}", c.bwd);
        prop_assert!(
            c.fwd <= C_F * c.kappa * T::u(),
            "{tag} (iii-a) {:e} > {C_F}·κ·u, κ = {:e}",
            c.fwd,
            c.kappa
        );
    }
    Ok(Some((c, plain)))
}

fn row_op() -> impl Strategy<Value = QrOp> {
    prop::sample::select(vec![QrOp::InsertRow, QrOp::DeleteCol])
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(support::cases(64)))]

    /// The column-scaled family in `f64`: `n ≤ 100`, `m ≤ n + 30`.
    #[test]
    fn graded_columns_f64(
        n in 1usize..=100, extra in 1usize..=30, kappa in log_kappa(), op in op(),
        gr in 0.0f64..=8.0, gc in 0.0f64..=40.0, kr in grading(), kc in grading(), seed in any::<u64>(),
    ) {
        check_graded::<f64>(n + extra, n, kappa, op, (gr, gc), (kr, kc), seed)?;
    }

    /// The row-scaled family in `f64`.
    #[test]
    fn graded_rows_f64(
        n in 1usize..=100, extra in 1usize..=30, kappa in log_kappa(), op in row_op(),
        gr in 8.5f64..=40.0, gc in 0.0f64..=40.0, kr in grading(), kc in grading(), seed in any::<u64>(),
    ) {
        check_graded::<f64>(n + extra, n, kappa, op, (gr, gc), (kr, kc), seed)?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(support::cases(16)))]

    /// The column-scaled family in `DoubleDouble`.
    #[test]
    fn graded_columns_dd(
        n in 1usize..=60, extra in 1usize..=15, kappa in log_kappa(), op in op(),
        gr in 0.0f64..=8.0, gc in 0.0f64..=40.0, kr in grading(), kc in grading(), seed in any::<u64>(),
    ) {
        check_graded::<Dd>(n + extra, n, kappa, op, (gr, gc), (kr, kc), seed)?;
    }

    /// The row-scaled family in `DoubleDouble`.
    #[test]
    fn graded_rows_dd(
        n in 1usize..=60, extra in 1usize..=15, kappa in log_kappa(), op in row_op(),
        gr in 8.5f64..=40.0, gc in 0.0f64..=40.0, kr in grading(), kc in grading(), seed in any::<u64>(),
    ) {
        check_graded::<Dd>(n + extra, n, kappa, op, (gr, gc), (kr, kc), seed)?;
    }
}

/// The survey behind the graded tests (`cargo test --release --test qr --
/// --nocapture graded_survey`): the largest ratios over fixed seeds,
/// asserted as in `check_graded`, and the number of refusals.
#[test]
fn graded_survey() {
    let seeds = if cfg!(debug_assertions) { 1 } else { 4 };
    let (mut o, mut b, mut p, mut f) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut runs, mut refused) = (0, 0);
    let kinds = [
        (Grading::Graded, Grading::Graded),
        (Grading::Random, Grading::Random),
        (Grading::Graded, Grading::Random),
    ];
    for op in QrOp::ALL {
        let gs: &[(f64, f64)] = if matches!(op, QrOp::InsertRow | QrOp::DeleteCol) {
            &[(0.0, 20.0), (8.0, 40.0), (20.0, 0.0), (40.0, 40.0)]
        } else {
            &[(0.0, 20.0), (8.0, 40.0)]
        };
        for &(m, n) in &[(3usize, 2usize), (30, 20), (80, 50)] {
            for &kappa in &[1e2, 1e8] {
                for &g in gs {
                    for k in kinds {
                        for s in 0..seeds {
                            runs += 1;
                            let Some((c, plain)) =
                                check_graded::<f64>(m, n, kappa, op, g, k, 88_000 + s).unwrap()
                            else {
                                refused += 1;
                                continue;
                            };
                            let mu = c.m as f64 * f64::u();
                            o = o.max(c.orth / mu);
                            b = b.max(c.bwd_path / mu);
                            p = p.max(plain / mu);
                            if c.shrink >= 0.5 {
                                f = f.max(c.fwd / (c.kappa * f64::u()));
                            }
                        }
                    }
                }
            }
        }
    }
    println!(
        "graded QR, f64, {runs} runs, {refused} refused: (Q1) {o:.3} m·u, (Q2, columns) {b:.3} m·u, (Q2, normwise) {p:.3} m·u, (iii-a) {f:.3} κ·u"
    );
    assert!(refused * 4 <= runs, "{refused} of {runs} refused");
}

/// A random sequence of the five operations with `A_t` tracked in
/// double-double; failed or invalid operations must leave `Q` and `R` bit
/// for bit (Q3).
fn sequence<T: Scalar>(
    m: usize,
    n: usize,
    kappa: f64,
    seed: u64,
    ops: usize,
) -> Result<(), TestCaseError> {
    let a = with_cond(m, n, kappa.sqrt(), seed);
    let mut f = Qr::factor(
        MatRef::new(m, n, m, &conv::<T>(&a)).unwrap(),
        &QrOptions::default(),
    )
    .map_err(|e| TestCaseError::fail(format!("factor: {e}")))?;
    let mut at = to_dd(&a);
    let mut path = fro(&at);
    let mut rng = Rng::new(seed ^ 0xa11);
    for _ in 0..ops {
        let op = QrOp::ALL[rng.below(5)];
        let (q0, r0) = (bits(&f.q().to_block().data), bits(&f.r().to_block().data));
        let (m, n) = (f.nrows(), f.ncols());
        let legal = match op {
            QrOp::DeleteRow | QrOp::InsertCol => m > n,
            QrOp::DeleteCol => n > 0,
            _ => true,
        };
        if !legal {
            let e = match op {
                QrOp::DeleteRow => f.delete_row(0),
                QrOp::InsertCol => f.insert_col(0, &vec![T::ONE; m]),
                _ => f.delete_col(0),
            };
            prop_assert!(
                matches!(e, Err(StatlaError::InvalidArgument { .. })),
                "{e:?}"
            );
        } else {
            match apply_op(&mut f, &mut at, op, &mut rng) {
                Ok(()) => {
                    let now = fro(&at);
                    path = path.max(now);
                    let mu = f.nrows() as f64 * T::u();
                    prop_assert_eq!(f.check_invariants(C_Q), Ok(()), "{} {}", T::NAME, op.name());
                    let b = qr_backward(&f, &at);
                    prop_assert!(
                        b * now / path <= C_Q * mu,
                        "{} {} (Q2, path) {:e}",
                        T::NAME,
                        op.name(),
                        b
                    );
                    if now >= 0.5 * path {
                        prop_assert!(b <= C_Q * mu, "{} {} (Q2) {:e}", T::NAME, op.name(), b);
                    }
                    continue;
                }
                Err(msg) => prop_assert!(msg.contains("rank deficient"), "{msg}"),
            }
        }
        prop_assert_eq!(bits(&f.q().to_block().data), q0, "(Q3)");
        prop_assert_eq!(bits(&f.r().to_block().data), r0, "(Q3)");
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(support::cases(32)))]

    #[test]
    fn sequences_f64(n in 1usize..=40, extra in 0usize..=6, kappa in log_kappa(), seed in any::<u64>(), ops in 1usize..=12) {
        sequence::<f64>(n + extra, n, kappa, seed, ops)?;
    }

    #[test]
    fn sequences_dd(n in 1usize..=20, extra in 0usize..=4, kappa in log_kappa(), seed in any::<u64>(), ops in 1usize..=8) {
        sequence::<Dd>(n + extra, n, kappa, seed, ops)?;
    }
}

/// Criterion (iii) in aggregate (as in Chapter 12.2): per operation and
/// `κ(AᵀA)`, over 20 fixed seeds, the maximum forward error of `R` is `≤
/// 10⁻¹²` where `κ ≤ 10⁸` and `≤ 10×` the maximum distance between two
/// refactorizations of `A_t` perturbed at the level `u`.
fn aggregate<T: Scalar>(m: usize, n: usize, seeds: u64) {
    for op in QrOp::ALL {
        for &kappa in &[1e1, 1e3, 1e6, 1e8, 1e9, 1e12] {
            let (mut fwd, mut dist) = (0.0f64, 0.0f64);
            for s in 0..seeds {
                let c = qr_case::<T>(m, n, kappa, op, 41_000 + s).unwrap();
                fwd = fwd.max(c.fwd);
                dist = dist.max(c.dist);
            }
            if kappa <= 1e8 {
                assert!(
                    fwd <= 1e-12,
                    "{} {} κ = {kappa:e}: {fwd:e}",
                    T::NAME,
                    op.name()
                );
            }
            assert!(
                fwd <= 10.0 * dist,
                "{} {} κ = {kappa:e}: {fwd:e} > 10 × {dist:e}",
                T::NAME,
                op.name()
            );
        }
    }
}

#[test]
fn forward_error_in_aggregate_f64() {
    if cfg!(debug_assertions) {
        aggregate::<f64>(60, 40, 10);
    } else {
        aggregate::<f64>(150, 100, 20);
    }
}

#[test]
fn forward_error_in_aggregate_dd() {
    aggregate::<Dd>(60, 40, if cfg!(debug_assertions) { 4 } else { 20 });
}

// ------------------------------------------------------------ DGKS

/// `delete_row` of a row of leverage `1 − O(10⁻¹⁰)`: the complement `z = (I
/// − QQᵀ) e_i` has norm `≈ 10⁻⁵`, one Gram–Schmidt pass leaves it far from
/// orthogonal to `Q`, and only the DGKS second pass keeps (Q1).
#[test]
fn delete_row_of_high_leverage_needs_reorthogonalization() {
    fn run<T: Scalar>() {
        let (m, n) = (30, 6);
        let mut a = with_cond(m, n, 10.0, 5);
        for j in 0..n {
            a[j * m] *= 1e5; // row 0
        }
        let mut f = Qr::factor(
            MatRef::new(m, n, m, &conv::<T>(&a)).unwrap(),
            &QrOptions::default(),
        )
        .unwrap();
        f.delete_row(0).unwrap();
        let orth = orthogonality(f.q());
        assert!(
            orth <= C_Q * (m - 1) as f64 * T::u(),
            "{} (Q1) {orth:e}",
            T::NAME
        );
        assert_eq!(f.check_invariants(C_Q), Ok(()));
        let mut at = to_dd(&a);
        let mut b = Vec::new();
        for j in 0..n {
            b.extend_from_slice(&at[j * m + 1..(j + 1) * m]);
        }
        at = b;
        // Relative to the path (‖A_t‖ ≪ ‖A‖ here: the deletion cancels most of A).
        let back = qr_backward(&f, &at) * fro(&at) / fro(&to_dd(&a));
        assert!(
            back <= C_Q * (m - 1) as f64 * T::u(),
            "{} (Q2) {back:e}",
            T::NAME
        );
    }
    run::<f64>();
    run::<Dd>();
}

/// `insert_col` of a column at relative distance `10⁻⁹` from the range of
/// `Q` (accepted: above `n ε`), and `rank1_update` with `u` that close to
/// the range: the DGKS pass keeps (Q1).
#[test]
fn nearly_dependent_insertions_keep_orthogonality() {
    let (m, n) = (40, 8);
    let a = with_cond(m, n, 100.0, 6);
    let mut f = Qr::factor(MatRef::new(m, n, m, &a).unwrap(), &QrOptions::default()).unwrap();
    let mut rng = Rng::new(9);
    let h = rng.gauss_vec(n);
    let noise = support::unit(&mut rng, m);
    let mut c: Vec<f64> = (0..m)
        .map(|i| (0..n).map(|j| a[i + j * m] * h[j]).sum())
        .collect();
    let cn = c.iter().map(|v| v * v).sum::<f64>().sqrt();
    for (ci, e) in c.iter_mut().zip(&noise) {
        *ci += 1e-9 * cn * e;
    }
    let mut g = f.clone();
    g.insert_col(3, &c).unwrap();
    assert!(orthogonality(g.q()) <= C_Q * m as f64 * f64::EPSILON / 2.0);
    assert_eq!(g.check_invariants(C_Q), Ok(()));
    let v = rng.gauss_vec(n);
    f.rank1_update(&c, &v).unwrap();
    assert!(orthogonality(f.q()) <= C_Q * m as f64 * f64::EPSILON / 2.0);
}

// ------------------------------------------------------------ (Q3) and errors

type Bits = Vec<(u64, u64)>;

fn snapshot<T: Scalar>(f: &Qr<T>) -> (Bits, Bits) {
    (bits(&f.q().to_block().data), bits(&f.r().to_block().data))
}

/// Rank-deficient results are refused before any change: `insert_col` of
/// a column in the range, `delete_row` that leaves dependent columns,
/// `rank1_update` that zeroes a column.
#[allow(clippy::too_many_lines)] // one list of error paths
fn rank_failures<T: Scalar>() {
    // A = [I₂; 0; 0] scaled, plus a dependent structure.
    let (m, n) = (4, 2);
    let a: Vec<f64> = vec![2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0];
    let mut f = Qr::factor(
        MatRef::new(m, n, m, &conv::<T>(&a)).unwrap(),
        &QrOptions::default(),
    )
    .unwrap();
    let s = snapshot(&f);
    // c = a_0 + a_1: in the range.
    let c: Vec<T> = conv(&[2.0, 3.0, 0.0, 0.0]);
    assert_eq!(
        f.insert_col(1, &c),
        Err(StatlaError::RankDeficient { column: 1 })
    );
    assert_eq!(snapshot(&f), s);
    // Deleting row 1 leaves [2 0; 0 0; 0 0]: column 1 is gone.
    assert_eq!(
        f.delete_row(1),
        Err(StatlaError::RankDeficient { column: 1 })
    );
    assert_eq!(snapshot(&f), s);
    // A − a_0 e_0ᵀ zeroes column 0.
    let u: Vec<T> = conv(&[-2.0, 0.0, 0.0, 0.0]);
    let v: Vec<T> = conv(&[1.0, 0.0]);
    assert_eq!(
        f.rank1_update(&u, &v),
        Err(StatlaError::RankDeficient { column: 0 })
    );
    assert_eq!(snapshot(&f), s);
    // Bad input everywhere: unchanged.
    let nan = T::from_f64(f64::NAN);
    assert_eq!(
        f.insert_row(0, &[T::ONE, nan]),
        Err(StatlaError::NonFinite {
            what: "w",
            index: 1
        })
    );
    assert_eq!(
        f.insert_row(5, &[T::ONE, T::ONE]),
        Err(StatlaError::InvalidArgument {
            what: "Qr::insert_row: row index > m"
        })
    );
    assert_eq!(
        f.insert_row(0, &[T::ONE]),
        Err(StatlaError::Dimension {
            what: "w",
            expected: 2,
            found: 1
        })
    );
    assert!(matches!(
        f.delete_row(4),
        Err(StatlaError::InvalidArgument { .. })
    ));
    assert_eq!(
        f.insert_col(0, &[T::ONE; 3]),
        Err(StatlaError::Dimension {
            what: "c",
            expected: 4,
            found: 3
        })
    );
    assert!(matches!(
        f.insert_col(3, &[T::ONE; 4]),
        Err(StatlaError::InvalidArgument { .. })
    ));
    assert_eq!(
        f.insert_col(0, &[T::ONE, T::ONE, nan, T::ONE]),
        Err(StatlaError::NonFinite {
            what: "c",
            index: 2
        })
    );
    assert!(matches!(
        f.delete_col(2),
        Err(StatlaError::InvalidArgument { .. })
    ));
    assert_eq!(
        f.rank1_update(&[T::ONE; 3], &[T::ONE; 2]),
        Err(StatlaError::Dimension {
            what: "u",
            expected: 4,
            found: 3
        })
    );
    assert_eq!(
        f.rank1_update(&[T::ONE; 4], &[T::ONE; 3]),
        Err(StatlaError::Dimension {
            what: "v",
            expected: 2,
            found: 3
        })
    );
    assert_eq!(
        f.rank1_update(&[T::ONE; 4], &[nan, T::ONE]),
        Err(StatlaError::NonFinite {
            what: "v",
            index: 0
        })
    );
    assert_eq!(snapshot(&f), s);
    // m = n: delete_row and insert_col are invalid.
    let sq: Vec<f64> = vec![2.0, 1.0, 0.0, 3.0];
    let mut g = Qr::factor(
        MatRef::new(2, 2, 2, &conv::<T>(&sq)).unwrap(),
        &QrOptions::default(),
    )
    .unwrap();
    let s = snapshot(&g);
    assert!(matches!(
        g.delete_row(0),
        Err(StatlaError::InvalidArgument { .. })
    ));
    assert!(matches!(
        g.insert_col(0, &[T::ONE; 2]),
        Err(StatlaError::InvalidArgument { .. })
    ));
    assert_eq!(snapshot(&g), s);
}

#[test]
fn failed_operations_leave_q_and_r_bit_for_bit() {
    rank_failures::<f64>();
    rank_failures::<Dd>();
}

#[test]
fn factor_error_paths() {
    let o = QrOptions::default();
    let a = [1.0; 6];
    assert!(matches!(
        Qr::factor(MatRef::new(2, 3, 2, &a).unwrap(), &o),
        Err(StatlaError::InvalidArgument { .. })
    ));
    let mut a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0];
    a[4] = f64::INFINITY;
    assert_eq!(
        Qr::factor(MatRef::new(3, 2, 3, &a).unwrap(), &o).unwrap_err(),
        StatlaError::NonFinite {
            what: "a",
            index: 4
        }
    );
    // Duplicate column; zero column.
    let a = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    assert_eq!(
        Qr::factor(MatRef::new(3, 2, 3, &a).unwrap(), &o).unwrap_err(),
        StatlaError::RankDeficient { column: 1 }
    );
    let a = [0.0, 0.0, 1.0, 2.0];
    assert_eq!(
        Qr::factor(MatRef::new(2, 2, 2, &a).unwrap(), &o).unwrap_err(),
        StatlaError::RankDeficient { column: 0 }
    );
    // A user threshold: columns at an angle of 1e-6 pass the default, not
    // rank_tol = 1e-3.
    let a = [1.0, 0.0, 0.0, 1.0, 1e-6, 0.0];
    assert!(Qr::factor(MatRef::new(3, 2, 3, &a).unwrap(), &o).is_ok());
    let strict = QrOptions {
        rank_tol: 1e-3,
        ..o
    };
    assert_eq!(
        Qr::factor(MatRef::new(3, 2, 3, &a).unwrap(), &strict).unwrap_err(),
        StatlaError::RankDeficient { column: 1 }
    );
    for bad in [
        QrOptions {
            reorth_eta: 0.0,
            ..o
        },
        QrOptions {
            reorth_eta: 1.5,
            ..o
        },
        QrOptions {
            reorth_eta: f64::NAN,
            ..o
        },
        QrOptions {
            rank_tol: -1.0,
            ..o
        },
        QrOptions {
            rank_tol: f64::INFINITY,
            ..o
        },
    ] {
        assert!(matches!(
            Qr::factor(MatRef::new(3, 2, 3, &a).unwrap(), &bad),
            Err(StatlaError::InvalidArgument { .. })
        ));
    }
    // m × 0 is a valid factorization; columns can be inserted into it.
    let mut e = Qr::<f64>::factor(MatRef::new(3, 0, 3, &[]).unwrap(), &o).unwrap();
    e.insert_col(0, &[1.0, 2.0, 2.0]).unwrap();
    assert_eq!((e.nrows(), e.ncols(), e.r().get(0, 0)), (3, 1, Some(3.0)));
    e.delete_col(0).unwrap();
    assert_eq!(e.ncols(), 0);
}

#[test]
fn least_squares_and_views() {
    let (m, n) = (25, 6);
    let a = with_cond(m, n, 1e3, 7);
    // A view with ld > m.
    let ld = m + 2;
    let mut padded = vec![f64::NAN; ld * (n - 1) + m];
    for j in 0..n {
        padded[j * ld..j * ld + m].copy_from_slice(&a[j * m..(j + 1) * m]);
    }
    let f = Qr::factor(
        MatRef::new(m, n, ld, &padded).unwrap(),
        &QrOptions::default(),
    )
    .unwrap();
    let mut rng = Rng::new(8);
    let x0 = rng.gauss_vec(n);
    let mut b: Vec<f64> = (0..m)
        .map(|i| (0..n).map(|j| a[i + j * m] * x0[j]).sum())
        .collect();
    // Add a residual orthogonal to the range: r = (I − QQᵀ) g.
    let g = rng.gauss_vec(m);
    let q = f.q().to_block();
    let mut r = g.clone();
    for j in 0..n {
        let s: f64 = (0..m).map(|i| q.data[i + j * m] * g[i]).sum();
        for i in 0..m {
            r[i] -= s * q.data[i + j * m];
        }
    }
    let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    for (bi, ri) in b.iter_mut().zip(&r) {
        *bi += ri;
    }
    let mut x = vec![0.0; n];
    let res = f.solve_least_squares(&b, &mut x).unwrap();
    let err = x
        .iter()
        .zip(&x0)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max);
    assert!(err < 1e-11, "{err:e}");
    assert!((res - rn).abs() < 1e-12 * rn.max(1.0), "{res} vs {rn}");
    assert_eq!(
        f.solve_least_squares(&b[1..], &mut x),
        Err(StatlaError::Dimension {
            what: "b",
            expected: m,
            found: m - 1
        })
    );
    assert!(matches!(
        f.solve_least_squares(&b, &mut [0.0; 2]),
        Err(StatlaError::Dimension { .. })
    ));
    // The Q-less view: RᵀR = AᵀA, the same R.
    let c = f.to_cholesky().unwrap();
    assert_eq!(c.upper().to_block().data, f.r().to_block().data);
    let mut y: Vec<f64> = (0..n)
        .map(|j| (0..m).map(|i| a[i + j * m] * b[i]).sum())
        .collect();
    c.solve(&mut y).unwrap(); // normal equations
    let err = y
        .iter()
        .zip(&x)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max);
    assert!(err < 1e-9, "{err:e}");
    // check_invariants reports a lost orthogonality.
    assert_eq!(f.check_invariants(C_Q), Ok(()));
    assert_eq!(
        f.check_invariants(0.0),
        Err("(Q1) Q has lost orthogonality")
    );
    assert_eq!((f.q().rows(), f.q().cols(), f.r().rows()), (m, n, n));
}

#[test]
fn updates_agree_with_refactorization_exactly_in_simple_cases() {
    // A = [1 0; 0 1; 0 0]: appending a row (0, 2) gives R = diag(1, √5).
    let a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let mut f = Qr::factor(MatRef::new(3, 2, 3, &a).unwrap(), &QrOptions::default()).unwrap();
    f.insert_row(3, &[0.0, 2.0]).unwrap();
    let r = f.r().to_block();
    assert_eq!(r.get(0, 0), 1.0);
    assert!((r.get(1, 1) - 5f64.sqrt()).abs() < 1e-15 && r.get(0, 1) == 0.0);
    // Deleting it again gives back R = I.
    f.delete_row(3).unwrap();
    let r = f.r().to_block();
    assert!((r.get(0, 0) - 1.0).abs() < 1e-15 && (r.get(1, 1) - 1.0).abs() < 1e-15);
    assert_eq!(f.check_invariants(C_Q), Ok(()));
}

/// The ratified criterion (iii) literally, per instance (see
/// `tests/cholesky.rs`); fails at small sizes (`fwd/dist` up to `3·10²` on
/// `5 × 3`), question Q-L32-2.
#[test]
#[ignore = "the ratified per-instance criterion (iii) does not hold; question Q-L32-2"]
fn ratified_iii_per_instance() {
    let mut bad = 0;
    for (m, n) in [(3, 2), (5, 3), (10, 5)] {
        for op in QrOp::ALL {
            for s in 0..100 {
                let c = qr_case::<f64>(m, n, 1e9, op, 60_000 + s).unwrap();
                if !support::forward_ok(c.fwd, c.dist, c.kappa) {
                    bad += 1;
                }
            }
        }
    }
    assert_eq!(bad, 0);
}

/// `rank1_update` that cancels a column of a generic matrix: the new `r_00`
/// is rounding residue, not an exact zero, so the rank test must measure it
/// against the column before the cancellation (`‖a_0‖ + |v_0| ‖u‖`), not
/// against its own norm (step R7).
#[test]
fn rank1_cancellation_of_a_generic_column_is_refused() {
    fn run<T: Scalar>() {
        for seed in 0..20 {
            let (m, n) = (9, 4);
            let a = with_cond(m, n, 30.0, 100 + seed);
            let mut f = Qr::factor(
                MatRef::new(m, n, m, &conv::<T>(&a)).unwrap(),
                &QrOptions::default(),
            )
            .unwrap();
            let s = snapshot(&f);
            let j = usize::try_from(seed).unwrap() % n;
            let u: Vec<T> = conv(&a[j * m..(j + 1) * m].iter().map(|v| -v).collect::<Vec<_>>());
            let mut v = vec![T::ZERO; n];
            v[j] = T::ONE;
            assert_eq!(
                f.rank1_update(&u, &v),
                Err(StatlaError::RankDeficient { column: j }),
                "{} seed {seed}",
                T::NAME
            );
            assert_eq!(snapshot(&f), s);
        }
    }
    run::<f64>();
    run::<Dd>();
}
