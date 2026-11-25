//! Acceptance tests of Chapter 12.2 (`Cholesky`), task L3.2, in `f64` and
//! `DoubleDouble`: (i) round trips, (ii) backward error (H2), (iii)
//! forward error against refactorization, (iv) transactional failures
//! (H3), the invariants (H1) through `check_invariants`, and every error
//! path of the contract.
//!
//! Constants (measured by the example `accuracy`, with a margin):
//! `C_H = 8` — the largest ratio `‖RᵀR − A_t‖_F / (n u ‖A_t‖_F)` seen is
//! `4.0` (`n = 1`, f64), `≤ 0.06` for `n ≥ 50`.

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::if_not_else,
    clippy::items_after_statements
)] // tests: sizes as f64, exact checks

mod support;

use proptest::prelude::*;
use support::{
    Grading, Rng, Scalar, bits, chol_backward, chol_case, chol_graded_case, conv, rank1_dd,
    scalings, spd, to_dd, unit,
};
use xingo_extprec::DoubleDouble as Dd;
use xingo_statla::{CholOptions, Cholesky, DenseBlock, MatMut, MatRef, NaiveDense, StatlaError};

/// (H2) constant `c_H`.
const C_H: f64 = 8.0;
/// Criterion (iii-a), provisional (question Q-L32-2): `fwd ≤ C_F · κ · u`
/// per instance (the first-order bound of any algorithm that works from
/// `R`; measured `≤ 0.09` for `κ ≥ 10`, `2` at `n = 1`, where `κ = 1`).
const C_F: f64 = 8.0;

fn log_kappa() -> impl Strategy<Value = f64> {
    (1.0f64..=12.0).prop_map(|e| 10f64.powf(e))
}

/// One downdate/update case: (i), (ii), (iii-a), (H1) (inside `chol_case`).
fn check_case<T: Scalar>(n: usize, kappa: f64, seed: u64) -> Result<(), TestCaseError> {
    let c = chol_case::<T>(n, kappa, seed).map_err(TestCaseError::fail)?;
    let u = T::u();
    let nu = n as f64 * u;
    prop_assert!(
        c.bwd_down <= C_H * nu,
        "{} (ii) downdate: {:e} > {C_H}·n·u",
        T::NAME,
        c.bwd_down
    );
    prop_assert!(
        c.bwd_up <= C_H * nu,
        "{} (ii) update: {:e} > {C_H}·n·u",
        T::NAME,
        c.bwd_up
    );
    prop_assert!(
        c.ud <= 10.0 * nu,
        "{} (i) upd∘dd {:e} > 10·n·u",
        T::NAME,
        c.ud
    );
    prop_assert!(
        c.du <= 10.0 * nu,
        "{} (i) dd∘upd {:e} > 10·n·u",
        T::NAME,
        c.du
    );
    prop_assert!(
        c.fwd <= C_F * c.kappa * u,
        "{} (iii-a) fwd {:e} > κ·u = {:e}",
        T::NAME,
        c.fwd,
        c.kappa * u
    );
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(support::cases(48)))]

    /// (i), (ii), (iii-a) in `f64`, `n ≤ 200`, `κ(A_t)` from 10 to 10¹².
    #[test]
    fn downdate_update_f64(n in 1usize..=200, kappa in log_kappa(), seed in any::<u64>()) {
        check_case::<f64>(n, kappa, seed)?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(support::cases(12)))]

    /// (i), (ii), (iii-a) in `DoubleDouble`.
    #[test]
    fn downdate_update_dd(n in 1usize..=200, kappa in log_kappa(), seed in any::<u64>()) {
        check_case::<Dd>(n, kappa, seed)?;
    }
}

/// A random sequence of updates and downdates with `A_t` tracked in
/// double-double: (H1) and (H2) after every successful operation, (H3)
/// bit for bit after every failed one.
fn sequence<T: Scalar>(n: usize, kappa: f64, seed: u64, ops: usize) -> Result<f64, TestCaseError> {
    let a0 = spd(n, kappa, seed);
    let mut at = to_dd(&a0);
    let mut path = support::fro(&at);
    let mut c = Cholesky::factor(
        MatRef::new(n, n, n, &conv::<T>(&a0)).unwrap(),
        &CholOptions::default(),
    )
    .map_err(|e| TestCaseError::fail(format!("factor: {e}")))?;
    let mut rng = Rng::new(seed ^ 0x5e9);
    let mut stack: Vec<Vec<f64>> = Vec::new();
    let mut worst = 0.0f64;
    let nu = n as f64 * T::u();
    for _ in 0..ops {
        let before = bits(&c.upper().to_block().data);
        let (x, sigma) = match rng.below(3) {
            0 => {
                let x: Vec<f64> = unit(&mut rng, n).iter().map(|v| v * 0.7).collect();
                (x, 1.0)
            }
            1 if !stack.is_empty() => (stack.pop().unwrap(), -1.0),
            _ => {
                // Usually indefinite: ‖x‖ = 2 against λ_max(A) ≈ 1.
                let x: Vec<f64> = unit(&mut rng, n).iter().map(|v| v * 2.0).collect();
                (x, -1.0)
            }
        };
        let xt: Vec<T> = conv(&x);
        let res = if sigma > 0.0 {
            c.update(&xt).map(|()| None)
        } else {
            c.downdate(&xt).map(Some)
        };
        match res {
            Ok(info) => {
                if let Some(info) = info {
                    prop_assert!(info.margin > 0.0 && info.margin <= 1.0);
                } else {
                    stack.push(x.clone());
                }
                rank1_dd(&mut at, n, &to_dd(&x), sigma);
                let now = support::fro(&at);
                path = path.max(now);
                prop_assert_eq!(c.check_invariants(), Ok(()));
                let b = chol_backward(&c, &at); // relative to ‖A_t‖
                worst = worst.max(b / nu);
                // (H2) relative to the largest norm along the path; and
                // relative to ‖A_t‖ itself while no deletion cancelled
                // more than half of it (question Q-L32-3).
                prop_assert!(
                    b * now / path <= C_H * nu,
                    "{} (H2, path) {:e}",
                    T::NAME,
                    b * now / path
                );
                if now >= 0.5 * path {
                    prop_assert!(b <= C_H * nu, "{} (H2) {:e} > {C_H}·n·u", T::NAME, b);
                }
            }
            Err(StatlaError::DowndateIndefinite { margin }) => {
                prop_assert!(margin <= 0.0 || margin.is_nan());
                prop_assert_eq!(bits(&c.upper().to_block().data), before, "(H3)");
            }
            Err(e) => return Err(TestCaseError::fail(format!("unexpected error {e}"))),
        }
    }
    Ok(worst)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(support::cases(32)))]

    /// Sequences in `f64`.
    #[test]
    fn sequences_f64(n in 1usize..=60, kappa in log_kappa(), seed in any::<u64>(), ops in 1usize..=10) {
        sequence::<f64>(n, kappa, seed, ops)?;
    }

    /// Sequences in `DoubleDouble`.
    #[test]
    fn sequences_dd(n in 1usize..=30, kappa in log_kappa(), seed in any::<u64>(), ops in 1usize..=6) {
        sequence::<Dd>(n, kappa, seed, ops)?;
    }
}

// ------------------------------------------- graded and badly scaled

/// `g` decades of scaling for the graded families.
fn decades() -> impl Strategy<Value = f64> {
    0.0f64..=40.0
}

fn grading() -> impl Strategy<Value = Grading> {
    prop::sample::select(vec![Grading::Graded, Grading::Random])
}

/// One graded case (issue xingo-statla#6): `A = D (A' + x′x′ᵀ) D` with
/// `D` spanning up to `10⁴⁰` (so `A` spans `10⁸⁰` and `κ(A)` is far beyond
/// `1/u`), and every criterion of `check_case` measured in the units of
/// `D`: Algorithms F, U and C only ever combine entries of one column of
/// `R`, which share the scale `d_j`, so their errors must not depend on
/// `D` — (ii) and (i) as in `check_case`, (iii-a) against the conditioning
/// `κ(R⋆D⁻¹)²` of the equilibrated target.
fn check_graded<T: Scalar>(
    n: usize,
    kappa: f64,
    g: f64,
    kind: Grading,
    seed: u64,
) -> Result<support::CholCase, TestCaseError> {
    let d = scalings(&mut Rng::new(seed ^ 0x0d5c), n, g, kind);
    let c = chol_graded_case::<T>(n, kappa, &d, seed).map_err(TestCaseError::fail)?;
    let u = T::u();
    let nu = n as f64 * u;
    let tag = format!("{} {kind:?} g = {g:.1}", T::NAME);
    prop_assert!(
        c.bwd_down <= C_H * nu,
        "{tag} (ii) downdate {:e}",
        c.bwd_down
    );
    prop_assert!(c.bwd_up <= C_H * nu, "{tag} (ii) update {:e}", c.bwd_up);
    prop_assert!(c.ud <= 10.0 * nu, "{tag} (i) upd∘dd {:e}", c.ud);
    prop_assert!(c.du <= 10.0 * nu, "{tag} (i) dd∘upd {:e}", c.du);
    prop_assert!(
        c.fwd <= C_F * c.kappa * u,
        "{tag} (iii-a) fwd {:e} > κ·u = {:e}",
        c.fwd,
        c.kappa * u
    );
    Ok(c)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(support::cases(48)))]

    /// Graded and badly scaled families in `f64`, `n ≤ 120`.
    #[test]
    fn graded_downdate_update_f64(
        n in 1usize..=120, kappa in log_kappa(), g in decades(), kind in grading(), seed in any::<u64>(),
    ) {
        check_graded::<f64>(n, kappa, g, kind, seed)?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(support::cases(12)))]

    /// Graded and badly scaled families in `DoubleDouble`, `n ≤ 60`.
    #[test]
    fn graded_downdate_update_dd(
        n in 1usize..=60, kappa in log_kappa(), g in decades(), kind in grading(), seed in any::<u64>(),
    ) {
        check_graded::<Dd>(n, kappa, g, kind, seed)?;
    }
}

/// The survey behind the graded tests: the largest ratios over a grid of
/// fixed seeds (`cargo test --release --test cholesky -- --nocapture
/// graded_survey`), asserted as in `check_graded`.
#[test]
fn graded_survey() {
    let seeds = if cfg!(debug_assertions) { 2 } else { 8 };
    let (mut b, mut r, mut f) = (0.0f64, 0.0f64, 0.0f64);
    for &n in &[2usize, 10, 50] {
        for &kappa in &[1e2, 1e6, 1e10] {
            for &g in &[0.0, 8.0, 20.0, 40.0] {
                for kind in [Grading::Graded, Grading::Random] {
                    for s in 0..seeds {
                        let c = check_graded::<f64>(n, kappa, g, kind, 77_000 + s).unwrap();
                        let nu = n as f64 * f64::u();
                        b = b.max(c.bwd_down.max(c.bwd_up) / nu);
                        r = r.max(c.ud.max(c.du) / nu);
                        f = f.max(c.fwd / (c.kappa * f64::u()));
                    }
                }
            }
        }
    }
    println!("graded Cholesky, f64: (ii) {b:.3} n·u, (i) {r:.3} n·u, (iii-a) {f:.3} κ·u");
}

/// Step C3′ (the new diagonal `c_j r_jj` underflows to zero) is reached by
/// a graded factor whose last diagonal entry is subnormal, `r_ll = 2⁻¹⁰⁶⁰`,
/// and a downdate whose rotation at `l` has `c_l ≈ 2√margin ≈ 2⁻¹⁹`:
/// `c_l r_ll ≈ 2⁻¹⁰⁷⁹` rounds to zero although `1 − ‖a‖² = 2⁻⁴⁰ > 0`
/// passed step C2. The downdate must fail with the margin of C2 and leave
/// `R` bit for bit (H3); without C3′ it would return a factor with a zero
/// diagonal (and fail (H1) in debug builds).
/// `2ᵏ` exactly, subnormal included (`powi` may form `2^{−k}` first and
/// overflow to a zero result in debug builds).
fn pow2(k: i32) -> f64 {
    if k >= -1022 {
        f64::from_bits(u64::try_from(k + 1023).unwrap() << 52)
    } else {
        f64::from_bits(1u64 << u32::try_from(k + 1074).unwrap())
    }
}

fn c3_prime<T: Scalar>(n: usize, binades_per_col: i32) {
    // R = R′ D: R′ the factor of a well-conditioned SPD matrix with its
    // last column replaced by e_l, D = diag(2^{−k j}) for j < l and
    // r_ll = 2⁻¹⁰⁶⁰.
    let l = n - 1;
    let (_, base) = spd_factor(n, 10.0, 3);
    let rb = base.upper().to_block();
    let mut r = DenseBlock::zeros(n, n);
    for j in 0..l {
        let dj = pow2(-binades_per_col * i32::try_from(j).unwrap());
        for i in 0..=j {
            r.data[i + j * n] = T::from_f64(rb.data[i + j * n] * dj);
        }
    }
    r.data[l + l * n] = T::from_f64(pow2(-1060));
    let c = Cholesky::from_upper(r, NaiveDense, &CholOptions::default()).unwrap();
    // a = (a′, 1/2) with ‖a′‖² = 3/4 − 2⁻⁴⁰; x = Rᵀa. Column l of R is
    // r_ll e_l, so x_l = r_ll/2 = 2⁻¹⁰⁶¹ exactly and C1 returns a_l = 1/2.
    let mut rng = Rng::new(11);
    let s = (0.75f64 - 2f64.powi(-40)).sqrt();
    let ap: Vec<f64> = unit(&mut rng, l).iter().map(|v| v * s).collect();
    let mut x = vec![T::ZERO; n];
    let rt = c.upper().to_block();
    for j in 0..l {
        let mut v = T::ZERO;
        for i in 0..=j {
            v += rt.data[i + j * n] * T::from_f64(ap[i]);
        }
        x[j] = v;
    }
    x[l] = T::from_f64(pow2(-1061));
    let before = bits(&c.upper().to_block().data);
    let mut c1 = c.clone();
    match c1.downdate(&x) {
        Err(StatlaError::DowndateIndefinite { margin }) => {
            // The margin of C2, positive: it is C3′ that refused.
            assert!(
                margin > 0.0 && margin < 1e-9,
                "{} margin {margin:e}",
                T::NAME
            );
        }
        other => panic!("{} n = {n}: expected C3′ to refuse, got {other:?}", T::NAME),
    }
    assert_eq!(bits(&c1.upper().to_block().data), before, "(H3)");
    // The same downdate with r_ll = 2⁻¹⁰⁰⁰ (no underflow) succeeds.
    let mut r2 = c.upper().to_block();
    r2.data[l + l * n] = T::from_f64(pow2(-1000));
    let mut c2 = Cholesky::from_upper(r2, NaiveDense, &CholOptions::default()).unwrap();
    x[l] = T::from_f64(pow2(-1001));
    let info = c2.downdate(&x).unwrap();
    assert!(info.margin > 0.0 && info.margin < 1e-9);
    assert_eq!(c2.check_invariants(), Ok(()));
}

#[test]
fn downdate_whose_diagonal_underflows_is_refused_by_c3_prime() {
    c3_prime::<f64>(2, 0);
    c3_prime::<f64>(8, 60);
    c3_prime::<Dd>(2, 0);
    c3_prime::<Dd>(8, 60);
}

/// Criterion (iii) as the pilot s01 measured it, in aggregate and at the
/// pilot's size `n = 100` (and 200): per cell `(n, κ)`, the maximum forward
/// error over 20 fixed seeds (6 in double-double under debug) is `≤ 10⁻¹²`
/// where `κ ≤ 10⁶` and `≤ 10×` the maximum distance between two
/// refactorizations everywhere. The ratified per-instance wording fails
/// (see `ratified_iii_per_instance`), and so does this aggregate at `n =
/// 50` in double-double (ratio 11.6 at `κ = 10¹²`, 20 seeds): question
/// Q-L32-2.
fn aggregate<T: Scalar>(ns: &[usize], seeds: u64) {
    for &n in ns {
        for &kappa in &[1e1, 1e3, 1e6, 1e9, 1e12] {
            let (mut fwd, mut dist) = (0.0f64, 0.0f64);
            for s in 0..seeds {
                let c = chol_case::<T>(n, kappa, 31_000 + s).unwrap();
                fwd = fwd.max(c.fwd);
                dist = dist.max(c.dist);
            }
            if kappa <= 1e6 {
                assert!(
                    fwd <= 1e-12,
                    "{} n = {n}, κ = {kappa:e}: max fwd {fwd:e}",
                    T::NAME
                );
            }
            assert!(
                fwd <= 10.0 * dist,
                "{} n = {n}, κ = {kappa:e}: {fwd:e} > 10 × {dist:e}",
                T::NAME
            );
        }
    }
}

#[test]
fn forward_error_in_aggregate_f64() {
    let ns: &[usize] = if cfg!(debug_assertions) {
        &[100]
    } else {
        &[100, 200]
    };
    aggregate::<f64>(ns, 20);
}

#[test]
fn forward_error_in_aggregate_dd() {
    aggregate::<Dd>(&[100], if cfg!(debug_assertions) { 6 } else { 20 });
}

/// Criterion (iii) of the ratified L3.2 row, literally and per instance:
/// `fwd ≤ 10⁻¹²` where `κ ≤ 10⁸`, else `≤ 10 · dist`. **Fails** (measured
/// counterexamples in `l32-tables.md`: `n = 2`, `κ = 10⁶`, `fwd = 9.4·10⁻¹²`;
/// `n = 100`, `κ = 10⁸`, 10 of 200 seeds above `10⁻¹²`). Kept, ignored,
/// as the record of the counterexample (question Q-L32-2, settled by D-0132).
#[test]
#[ignore = "the ratified per-instance criterion (iii) does not hold; question Q-L32-2"]
fn ratified_iii_per_instance() {
    let mut bad = Vec::new();
    for &n in &[2usize, 10, 100] {
        for &kappa in &[1e6, 1e8, 1e12] {
            for s in 0..200 {
                let c = chol_case::<f64>(n, kappa, 50_000 + s).unwrap();
                if !support::forward_ok(c.fwd, c.dist, c.kappa) {
                    bad.push((n, kappa, s, c.fwd, c.dist));
                }
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{} failures, first {:?}",
        bad.len(),
        bad.first()
    );
}

// ------------------------------------------------------------ (iv), (H3)

fn spd_factor(n: usize, kappa: f64, seed: u64) -> (Vec<f64>, Cholesky) {
    let a = spd(n, kappa, seed);
    let c = Cholesky::factor(MatRef::new(n, n, n, &a).unwrap(), &CholOptions::default()).unwrap();
    (a, c)
}

/// (iv): a downdate that must fail leaves `R` bit for bit unchanged — C2
/// with `1 − ‖a‖² < 0`, `= 0`, below a positive `downdate_tol`, and after
/// an overflow of `‖a‖²`; in both scalars.
fn transactional<T: Scalar>() {
    for (n, kappa, seed) in [(1, 10.0, 1), (7, 1e3, 2), (40, 1e9, 3)] {
        let a = spd(n, kappa, seed);
        let mut c = Cholesky::factor(
            MatRef::new(n, n, n, &conv::<T>(&a)).unwrap(),
            &CholOptions::default(),
        )
        .unwrap();
        let r = c.upper().to_block();
        let before = bits(&r.data);
        // x = Rᵀ a with ‖a‖ = s: the margin is exactly 1 − s² in exact
        // arithmetic.
        let mut rng = Rng::new(seed);
        let dir = unit(&mut rng, n);
        for s in [1.5, 1.0 + 1e-6, 1e200] {
            let av: Vec<T> = dir.iter().map(|&v| T::from_f64(v * s)).collect();
            let x: Vec<T> = (0..n)
                .map(|j| (0..=j).map(|i| r.data[i + j * n] * av[i]).sum())
                .collect();
            let e = c.downdate(&x).unwrap_err();
            assert!(
                matches!(e, StatlaError::DowndateIndefinite { margin } if margin <= 0.0),
                "{e:?}"
            );
            assert_eq!(
                bits(&c.upper().to_block().data),
                before,
                "(H3) {} s = {s}",
                T::NAME
            );
        }
        // A tolerance refuses a legitimate downdate (margin ≈ 1 − 0.999²).
        let mut strict = Cholesky::from_upper(
            r.clone(),
            NaiveDense,
            &CholOptions {
                pivot_tol: 0.0,
                downdate_tol: 0.01,
            },
        )
        .unwrap();
        let av: Vec<T> = dir.iter().map(|&v| T::from_f64(v * 0.999)).collect();
        let x: Vec<T> = (0..n)
            .map(|j| (0..=j).map(|i| r.data[i + j * n] * av[i]).sum())
            .collect();
        let e = strict.downdate(&x).unwrap_err();
        assert!(
            matches!(e, StatlaError::DowndateIndefinite { margin } if margin > 0.0 && margin <= 0.01)
        );
        assert_eq!(bits(&strict.upper().to_block().data), before);
        // The same downdate passes with the default tolerance.
        let info = c.downdate(&x).unwrap();
        assert!(info.margin > 0.0 && info.margin < 0.01);
        // Bad input to update/downdate: unchanged as well.
        let snapshot = bits(&c.upper().to_block().data);
        let mut bad = vec![T::ONE; n];
        bad[n - 1] = T::from_f64(f64::NAN);
        assert_eq!(
            c.update(&bad),
            Err(StatlaError::NonFinite {
                what: "x",
                index: n - 1
            })
        );
        assert_eq!(
            c.downdate(&bad),
            Err(StatlaError::NonFinite {
                what: "x",
                index: n - 1
            })
        );
        bad[n - 1] = T::from_f64(f64::INFINITY);
        assert!(c.update(&bad).is_err());
        let short = vec![T::ONE; n + 1];
        assert_eq!(
            c.update(&short),
            Err(StatlaError::Dimension {
                what: "x",
                expected: n,
                found: n + 1
            })
        );
        assert!(matches!(
            c.downdate(&short),
            Err(StatlaError::Dimension { .. })
        ));
        assert_eq!(bits(&c.upper().to_block().data), snapshot);
    }
}

#[test]
fn failed_operations_leave_the_factor_bit_for_bit() {
    transactional::<f64>();
    transactional::<Dd>();
}

// ------------------------------------------------------------ errors

#[test]
fn factor_error_paths() {
    let o = CholOptions::default();
    // Not square.
    let a = [1.0; 6];
    assert!(matches!(
        Cholesky::factor(MatRef::new(2, 3, 2, &a).unwrap(), &o),
        Err(StatlaError::InvalidArgument { .. })
    ));
    // Non-finite in the upper triangle (column-major index), ignored below.
    let mut a = vec![4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0];
    a[1] = f64::NAN; // strict lower triangle: not read
    assert!(Cholesky::factor(MatRef::new(3, 3, 3, &a).unwrap(), &o).is_ok());
    a[3] = f64::INFINITY; // entry (0, 1)
    assert_eq!(
        Cholesky::factor(MatRef::new(3, 3, 3, &a).unwrap(), &o).unwrap_err(),
        StatlaError::NonFinite {
            what: "a",
            index: 3
        }
    );
    // Indefinite: the pivot of column 1 fails.
    let a = [1.0, 0.0, 2.0, 1.0];
    assert_eq!(
        Cholesky::factor(MatRef::new(2, 2, 2, &a).unwrap(), &o).unwrap_err(),
        StatlaError::NotPositiveDefinite { column: 1 }
    );
    let a = [-1.0];
    assert_eq!(
        Cholesky::factor(MatRef::new(1, 1, 1, &a).unwrap(), &o).unwrap_err(),
        StatlaError::NotPositiveDefinite { column: 0 }
    );
    // pivot_tol: d_1 = 1 − 0.99² = 0.0199 < 0.1 · a_11 = 0.1.
    let a = [1.0, 0.0, 0.99, 1.0];
    let strict = CholOptions {
        pivot_tol: 0.1,
        downdate_tol: 0.0,
    };
    assert_eq!(
        Cholesky::factor(MatRef::new(2, 2, 2, &a).unwrap(), &strict).unwrap_err(),
        StatlaError::NotPositiveDefinite { column: 1 }
    );
    assert!(Cholesky::factor(MatRef::new(2, 2, 2, &a).unwrap(), &o).is_ok());
    // Invalid options.
    for bad in [
        CholOptions {
            pivot_tol: -1.0,
            downdate_tol: 0.0,
        },
        CholOptions {
            pivot_tol: f64::NAN,
            downdate_tol: 0.0,
        },
        CholOptions {
            pivot_tol: 0.0,
            downdate_tol: -1e-3,
        },
        CholOptions {
            pivot_tol: 0.0,
            downdate_tol: f64::INFINITY,
        },
    ] {
        assert!(matches!(
            Cholesky::factor(MatRef::new(2, 2, 2, &a).unwrap(), &bad),
            Err(StatlaError::InvalidArgument { .. })
        ));
        assert!(matches!(
            Cholesky::from_upper(DenseBlock::<f64>::zeros(0, 0), NaiveDense, &bad),
            Err(StatlaError::InvalidArgument { .. })
        ));
    }
    // n = 0 is a valid (empty) factorization.
    let e = Cholesky::<f64>::factor(MatRef::new(0, 0, 0, &[]).unwrap(), &o).unwrap();
    assert_eq!((e.n(), e.log2_det_parts()), (0, (1.0, 0)));
}

#[test]
fn from_upper_error_paths() {
    let o = CholOptions::default();
    let good = DenseBlock {
        rows: 2,
        cols: 2,
        data: vec![2.0, 0.0, 1.0, 3.0],
    };
    assert!(Cholesky::from_upper(good.clone(), NaiveDense, &o).is_ok());
    let cases: [(DenseBlock<f64>, &str); 5] = [
        (
            DenseBlock {
                rows: 2,
                cols: 3,
                data: vec![1.0; 6],
            },
            "square",
        ),
        (
            DenseBlock {
                rows: 2,
                cols: 2,
                data: vec![1.0; 3],
            },
            "square",
        ),
        (
            DenseBlock {
                rows: 2,
                cols: 2,
                data: vec![2.0, 1.0, 1.0, 3.0],
            },
            "upper",
        ),
        (
            DenseBlock {
                rows: 2,
                cols: 2,
                data: vec![2.0, 0.0, 1.0, 0.0],
            },
            "diagonal",
        ),
        (
            DenseBlock {
                rows: 2,
                cols: 2,
                data: vec![-2.0, 0.0, 1.0, 3.0],
            },
            "diagonal",
        ),
    ];
    for (b, what) in cases {
        match Cholesky::from_upper(b, NaiveDense, &o) {
            Err(StatlaError::InvalidArgument { what: w }) => assert!(w.contains(what), "{w}"),
            other => panic!("{what}: {other:?}"),
        }
    }
    let nan = DenseBlock {
        rows: 2,
        cols: 2,
        data: vec![2.0, 0.0, f64::NAN, 3.0],
    };
    assert_eq!(
        Cholesky::from_upper(nan, NaiveDense, &o).unwrap_err(),
        StatlaError::NonFinite {
            what: "r",
            index: 2
        }
    );
}

#[test]
fn solve_error_paths() {
    let (_, c) = spd_factor(3, 10.0, 5);
    assert_eq!(
        c.solve(&mut [1.0; 2]),
        Err(StatlaError::Dimension {
            what: "b",
            expected: 3,
            found: 2
        })
    );
    assert!(c.solve_upper_t(&mut [1.0; 4]).is_err());
    let mut data = vec![1.0; 8];
    let m = MatMut::new(2, 4, 2, &mut data).unwrap();
    assert!(matches!(
        c.solve_block(m),
        Err(StatlaError::Dimension { .. })
    ));
}

// ------------------------------------------------------------ solves, determinant

#[test]
fn solves_whitening_and_determinant() {
    let n = 30;
    let (a, c) = spd_factor(n, 1e4, 9);
    let mut rng = Rng::new(3);
    let x0 = rng.gauss_vec(n);
    let b: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| a[i + j * n] * x0[j]).sum())
        .collect();
    let mut x = b.clone();
    c.solve(&mut x).unwrap();
    let err = x
        .iter()
        .zip(&x0)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max);
    assert!(err < 1e-10, "solve: {err:e}");
    // solve_block with ld > rows: two right-hand sides at stride n + 3.
    let ld = n + 3;
    let mut blk = vec![f64::NAN; ld + n];
    blk[..n].copy_from_slice(&b);
    blk[ld..ld + n].copy_from_slice(&b);
    blk[n..ld].fill(7.0);
    c.solve_block(MatMut::new(n, 2, ld, &mut blk).unwrap())
        .unwrap();
    assert_eq!(&blk[..n], &x[..]);
    assert_eq!(&blk[ld..ld + n], &x[..]);
    assert!(blk[n..ld].iter().all(|&v| v == 7.0), "padding untouched");
    // Whitening: ‖R⁻ᵀ b‖² = bᵀ A⁻¹ b = bᵀ x.
    let mut w = b.clone();
    c.solve_upper_t(&mut w).unwrap();
    let lhs: f64 = w.iter().map(|v| v * v).sum();
    let rhs: f64 = b.iter().zip(&x).map(|(p, q)| p * q).sum();
    assert!((lhs - rhs).abs() <= 1e-10 * rhs);
    // Determinant: Π r_jj = m 2^e against the product in double-double.
    let r = c.upper().to_block();
    let mut p = Dd::ONE;
    for j in 0..n {
        p *= Dd::from_f64(r.data[j + j * n]);
    }
    let (m, e) = c.log2_det_parts();
    assert!((1.0..2.0).contains(&m));
    let back = Dd::from_f64(m).ldexp(i32::try_from(e).unwrap());
    assert!(((back - p) / p).abs().to_f64() < 1e-14);
    // No overflow: R = 2^600 I (det = 2^(600 n)) and 2^-600 I.
    for (s, k) in [(2f64.powi(600), 600i64), (2f64.powi(-600), -600)] {
        let mut d = DenseBlock::zeros(4, 4);
        for j in 0..4 {
            d.data[j + j * 4] = s * 1.5;
        }
        let c = Cholesky::from_upper(d, NaiveDense, &CholOptions::default()).unwrap();
        let (m, e) = c.log2_det_parts();
        assert_eq!((m, e), (1.5f64.powi(4) / 4.0, 4 * k + 2));
        // and in double-double
        let mut d = DenseBlock::<Dd>::zeros(4, 4);
        for j in 0..4 {
            d.data[j + j * 4] = Dd::from_f64(s * 1.5);
        }
        let c = Cholesky::from_upper(d, NaiveDense, &CholOptions::default()).unwrap();
        let (m, e) = c.log2_det_parts();
        assert_eq!((m.to_f64(), e), (1.5f64.powi(4) / 4.0, 4 * k + 2));
    }
}

#[test]
fn a_dd_factor_downdates_where_f64_cannot() {
    // A' with κ = 10²⁰: A = A' + xxᵀ; in f64 the downdate back to A' is
    // indefinite or meaningless, in double-double it succeeds with the
    // backward error of (H2).
    let n = 12;
    let d = support::spectrum(n, 1e20);
    let mut ap = vec![0.0; n * n];
    for i in 0..n {
        ap[i + i * n] = d[i]; // exact in every scalar
    }
    let mut rng = Rng::new(78);
    let x = unit(&mut rng, n);
    let xd = to_dd(&x);
    let mut a = to_dd(&ap);
    rank1_dd(&mut a, n, &xd, 1.0);
    let mut c =
        Cholesky::factor(MatRef::new(n, n, n, &a).unwrap(), &CholOptions::default()).unwrap();
    let info = c.downdate(&xd).unwrap();
    assert!(info.margin > 0.0 && info.margin < 1e-10);
    let mut at = a.clone();
    rank1_dd(&mut at, n, &xd, -1.0);
    assert!(chol_backward(&c, &at) <= C_H * n as f64 * Dd::u());
}

#[test]
fn upper_view_and_into_upper() {
    let (_, c) = spd_factor(5, 10.0, 11);
    let v = c.upper();
    assert_eq!((v.rows(), v.cols(), v.ld()), (5, 5, 5));
    assert_eq!(v.get(3, 1), Some(0.0));
    let r = c.clone().into_upper();
    assert_eq!(r.data, v.to_block().data);
    assert_eq!(c.check_invariants(), Ok(()));
}

#[cfg(feature = "faer")]
#[test]
fn faer_kernel_factors_the_same_matrix() {
    let n = 40;
    let a = spd(n, 1e6, 12);
    let c1 = Cholesky::factor(MatRef::new(n, n, n, &a).unwrap(), &CholOptions::default()).unwrap();
    let c2 = Cholesky::factor_with(
        MatRef::new(n, n, n, &a).unwrap(),
        xingo_sparse_faer(),
        &CholOptions::default(),
    )
    .unwrap();
    let d = support::rel(&c2.upper().to_block().data, &c1.upper().to_block().data);
    assert!(d < 1e-12, "{d:e}");
}

#[cfg(feature = "faer")]
fn xingo_sparse_faer() -> impl xingo_statla::DenseKernel<f64> + Clone {
    // `FaerDense` is re-exported by `xingo-sparse` under its `faer` feature,
    // which the feature `faer` of this crate enables.
    xingo_sparse::FaerDense
}
