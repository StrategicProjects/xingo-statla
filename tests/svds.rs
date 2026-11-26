//! Acceptance tests of Chapter 12.5 (`svds`), task L3.3, as ratified
//! (D-0127, S9):
//!
//! (i) every returned triplet: the residual `‖(A v − σ̂ u; Aᵀ u − σ̂ v)‖`,
//!     *recomputed* in double-double, is `≤ (τ + c_G·ncv·u)·σ̂_1`;
//! (ii) `|σ̂_i − σ_i| ≤ 10⁻¹³ σ_1` on matrices with prescribed spectra down to
//!     `10⁻¹⁴ σ_1`, clusters included, where the gap to the neighbours is
//!     `≥ 10⁻⁸ σ_1`, `m, n ≤ 2000`. The reference SVD is the construction
//!     (support `prescribed`: formed in double-double, certified to `δ =
//!     u‖A‖_F` by Weyl), cross-checked by a one-sided Jacobi SVD in
//!     double-double on a small case, which is also the oracle of a sparse
//!     matrix without a prescribed spectrum;
//! (iv) a `DoubleDouble` smoke test of (i) with `u = 2⁻¹⁰⁵`.
//!
//! (iii) — against `scipy.sparse.linalg.svds` on Matrix Market matrices —
//! moved to task L3.4 (D-0134) and lives in `tests/arpack.rs`, on committed
//! fixtures. Every table line is printed with `--nocapture`.

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]

mod support;

use support::svd::{jacobi_values, orth_err, prescribed, prescribed_dd, residuals};
use support::{Rng, Scalar};
use xingo_extprec::DoubleDouble as Dd;
use xingo_sparse::{Col, CscMatrix, Row};
use xingo_statla::{
    AdjointOperator, DenseBlock, LinearOperator, OperatorError, StatlaError, Svd, SvdsOptions,
    from_fns, svds,
};

/// `c_G` of the guarantee (Chapter 12.5; measured, with a margin).
const C_G: f64 = 8.0;
/// Criterion (ii).
const VALUE_TOL: f64 = 1e-13;
/// Gap condition of criterion (ii).
const GAP: f64 = 1e-8;

/// The default `ncv` of the contract.
fn default_ncv(m: usize, n: usize, k: usize) -> usize {
    m.min(n).min((2 * k + 1).max(20))
}

/// Criterion (i) on a result: returns the largest `(‖r‖/σ̂_1 − τ)/(ncv·u)`.
fn check_residuals<T: Scalar>(
    name: &str,
    a: &[T],
    (m, n): (usize, usize),
    f: &Svd<T>,
    ncv: usize,
) -> f64 {
    let u = T::u();
    let tau = 2.0 * u; // τ = ε_T (tol = 0)
    let s1 = f.s[0].to_f64();
    let res = residuals(a, m, n, &f.u, &f.s, &f.v);
    let mut worst = 0.0f64;
    for (i, &r) in res.iter().enumerate() {
        let bound = (tau + C_G * ncv as f64 * u) * s1;
        assert!(
            r <= bound,
            "{name}: triplet {i}: residual {r:e} > {bound:e}"
        );
        worst = worst.max((r / s1.max(f64::MIN_POSITIVE) - tau) / (ncv as f64 * u));
    }
    worst
}

/// Criterion (ii) against the prescribed `sigma` (nonincreasing), with the
/// certified `delta`; returns `(max error on gapped values, max error on
/// all, how many were gapped)` in units of `σ_1`.
fn check_values(name: &str, s: &[f64], sigma: &[f64], delta: f64) -> (f64, f64, usize) {
    let s1 = sigma[0];
    let (mut gapped, mut all, mut count) = (0.0f64, 0.0f64, 0);
    for (i, &v) in s.iter().enumerate() {
        let err = (v - sigma[i]).abs();
        all = all.max(err / s1);
        let below = if i + 1 < sigma.len() {
            sigma[i] - sigma[i + 1]
        } else {
            sigma[i]
        };
        let above = if i == 0 {
            f64::INFINITY
        } else {
            sigma[i - 1] - sigma[i]
        };
        if below.min(above) >= GAP * s1 {
            count += 1;
            gapped = gapped.max(err / s1);
            assert!(
                err <= VALUE_TOL * s1 - delta,
                "{name}: sigma_{i}: {v:e} vs {:e} (error {:e} sigma_1)",
                sigma[i],
                err / s1
            );
        }
    }
    (gapped, all, count)
}

/// One acceptance case in `f64` on a prescribed spectrum.
fn case(name: &str, m: usize, n: usize, sigma: &[f64], k: usize, seed: u64) {
    case_ncv(name, m, n, sigma, k, seed, None);
}

/// As `case`, with a given `ncv`.
fn case_ncv(
    name: &str,
    m: usize,
    n: usize,
    sigma: &[f64],
    k: usize,
    seed: u64,
    ncv: Option<usize>,
) -> xingo_statla::SvdStats {
    let (mut a, delta) = prescribed(m, n, sigma, seed);
    let opts = SvdsOptions {
        ncv,
        ..SvdsOptions::default()
    };
    let f = svds(&mut a, k, &opts).unwrap();
    assert_eq!(f.s.len(), k);
    assert_eq!((f.u.rows, f.u.cols, f.v.rows, f.v.cols), (m, k, n, k));
    assert!(f.s.windows(2).all(|w| w[1] <= w[0]), "{name}: order");
    let ncv = ncv.unwrap_or_else(|| default_ncv(m, n, k));
    let r = check_residuals(name, &a.data, (m, n), &f, ncv);
    let (gapped, all, count) = check_values(name, &f.s, sigma, delta);
    let (ou, ov) = (orth_err(&f.u), orth_err(&f.v));
    let g1 = C_G * ncv as f64 * f64::EPSILON / 2.0;
    assert!(ou <= g1 && ov <= g1, "{name}: (G1) {ou:e} {ov:e}");
    eprintln!(
        "SVDS | {name} | {m}x{n} k={k} ncv={ncv} | resid {r:.3} ncv·u | value err (gapped, {count}/{k}) {gapped:.2e} | all {all:.2e} | delta {:.1e} | orth U {ou:.1e} V {ov:.1e} | restarts {} applies {}+{} reorth {}",
        delta / sigma[0],
        f.stats.restarts,
        f.stats.applies,
        f.stats.applies_t,
        f.stats.reorthogonalizations
    );
    assert_eq!(f.stats.converged, k);
    f.stats
}

fn geometric(len: usize, lo: f64) -> Vec<f64> {
    (0..len)
        .map(|j| lo.powf(j as f64 / (len - 1) as f64))
        .collect()
}

#[test]
fn geometric_spectrum_down_to_1e_14() {
    case(
        "geometric 1..1e-14",
        300,
        200,
        &geometric(200, 1e-14),
        10,
        1,
    );
}

#[test]
fn rank_deficient_spectrum_reaching_1e_14() {
    // 20 values from 1 to 1e-14, then exact zeros: the wanted include
    // 1e-14 (its gap to 0 is below 1e-8: reported, not asserted).
    let mut s = geometric(20, 1e-14);
    s.resize(200, 0.0);
    case("rank 20, 1..1e-14, zeros", 300, 200, &s, 20, 2);
}

/// Small *wanted* values (down to `σ_20 ≈ 3·10⁻¹⁰`) above a nonzero tail,
/// with the default and with a small `ncv`: the absolute test of B5
/// converges. (A test relative to `σ̃_i` happens to converge here too: the
/// projected `ρ_i` decays below `ε σ_i` — mutation M10 of task L3.3.)
#[test]
fn small_wanted_values_over_a_tail() {
    let mut s = geometric(25, 1e-12);
    s.extend((0..175).map(|j| 5e-13 * 0.99f64.powi(j)));
    case("small wanted, tail 5e-13", 300, 200, &s, 20, 12);
    case_ncv("small wanted, ncv = 24", 300, 200, &s, 20, 12, Some(24));
}

#[test]
fn clustered_spectrum() {
    let mut s = vec![1.0, 1.0 - 1e-8, 1.0 - 2e-8, 0.5, 0.5 - 1e-8, 0.5 - 2e-8];
    s.extend((0..194).map(|j| 0.3 * 0.9f64.powi(j)));
    case("clusters at 1e-8", 250, 200, &s, 8, 3);
}

#[test]
fn slowly_decaying_and_wide_spectra() {
    let s: Vec<f64> = (0..200)
        .map(|j: i32| 1.0 / f64::from(j + 1).powi(2))
        .collect();
    case("j^-2", 300, 200, &s, 10, 4);
    case("wide geometric", 150, 250, &geometric(150, 1e-14), 8, 5);
}

#[test]
fn large_matrix_2000_rows() {
    case(
        "2000x200 geometric",
        2000,
        200,
        &geometric(200, 1e-14),
        10,
        6,
    );
}

/// Many restarts under heavy locking (issue xingo-statla#7): a slowly
/// decaying spectrum `σ_j = 1/(1 + j/50)` (relative gaps `≈ 2%` at the
/// top, `0.7%` at `σ_30`), `k = 30` wanted with `ncv = 36`, so that each
/// restart keeps few unconverged vectors and the wanted lock a few at a
/// time; asserted as every `case`, and `> 4` restarts.
#[test]
fn many_restarts_with_locking() {
    let s: Vec<f64> = (0..300)
        .map(|j| 1.0 / (1.0 + f64::from(j) / 50.0))
        .collect();
    let stats = case_ncv("slow decay, ncv = k + 6", 400, 300, &s, 30, 21, Some(36));
    assert!(stats.restarts > 4, "{} restarts", stats.restarts);
}

/// The same at `2000 × 1900` (both dimensions near 2000), `k = 40`, `ncv
/// = 48`: ignored by default for its cost (`cargo test --release --test
/// svds -- --ignored`).
#[test]
#[ignore = "slow (a 2000 × 1900 dense operator formed in double-double): run with --ignored"]
fn many_restarts_with_locking_2000x1900() {
    let s: Vec<f64> = (0..1900)
        .map(|j| 1.0 / (1.0 + f64::from(j) / 50.0))
        .collect();
    let stats = case_ncv("2000x1900 slow decay", 2000, 1900, &s, 40, 22, Some(48));
    assert!(stats.restarts > 4, "{} restarts", stats.restarts);
}

/// Exact multiple values: criterion (i) and the guarantee (some `σ_j`
/// within `‖r‖/√2`); the index may not match (one-vector Krylov), so (ii)
/// does not apply (gap 0).
#[test]
fn multiple_values_satisfy_the_guarantee() {
    let mut s = vec![1.0, 1.0, 1.0, 0.8, 0.8];
    s.extend((0..195).map(|j| 0.5 * 0.95f64.powi(j)));
    let (m, n, k) = (250, 200, 6);
    let (mut a, delta) = prescribed(m, n, &s, 7);
    let f = svds(&mut a, k, &SvdsOptions::default()).unwrap();
    let ncv = default_ncv(m, n, k);
    let r = check_residuals("multiple", &a.data, (m, n), &f, ncv);
    let res = residuals(&a.data, m, n, &f.u, &f.s, &f.v);
    let mut matched = 0;
    for (i, (&v, &ri)) in f.s.iter().zip(&res).enumerate() {
        let near = s
            .iter()
            .chain(&[0.0])
            .map(|&x| (x - v).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(
            near <= ri / 2f64.sqrt() + delta + 1e-300,
            "{i}: {v} {near:e} {ri:e}"
        );
        if (v - s[i]).abs() <= 1e-13 {
            matched += 1;
        }
    }
    eprintln!(
        "SVDS | multiple (1 x3, 0.8 x2) | {m}x{n} k={k} | resid {r:.3} ncv·u | index matches {matched}/{k} | values {:?} | restarts {}",
        f.s, f.stats.restarts
    );
}

/// The construction oracle against an independent dense SVD (one-sided
/// Jacobi in double-double), and a sparse matrix whose oracle is that SVD.
#[test]
fn dense_double_double_oracle_agrees() {
    let sigma = geometric(40, 1e-14);
    let (a, delta) = prescribed(60, 40, &sigma, 8);
    let dd: Vec<Dd> = a.data.iter().map(|&x| Dd::from_f64(x)).collect();
    let jac = jacobi_values(&dd, 60, 40);
    let worst = jac
        .iter()
        .zip(&sigma)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(
        worst <= delta,
        "construction vs Jacobi: {worst:e} > {delta:e}"
    );
    // A sparse 100 × 60 matrix (8% of entries, Gaussian values), CSC.
    let mut rng = Rng::new(21);
    let mut trip = Vec::new();
    let mut dense = vec![0.0; 100 * 60];
    for j in 0..60 {
        for i in 0..100 {
            if rng.uniform() < 0.08 {
                let v = rng.gauss();
                trip.push((Row::from_usize(i), Col::from_usize(j), v));
                dense[i + j * 100] = v;
            }
        }
    }
    let mut csc = CscMatrix::from_triplets(100, 60, &trip).unwrap();
    let reference = jacobi_values(
        &dense.iter().map(|&x| Dd::from_f64(x)).collect::<Vec<_>>(),
        100,
        60,
    );
    let f = svds(&mut csc, 6, &SvdsOptions::default()).unwrap();
    let r = check_residuals(
        "sparse 100x60",
        &dense,
        (100, 60),
        &f,
        default_ncv(100, 60, 6),
    );
    let (gapped, all, count) = check_values("sparse 100x60", &f.s, &reference, 1e-28);
    eprintln!(
        "SVDS | sparse CSC 100x60 (8%) vs dd Jacobi | k=6 | resid {r:.3} ncv·u | value err (gapped {count}) {gapped:.2e} all {all:.2e} | construction vs dd Jacobi {:.1e} (delta {:.1e}) | restarts {}",
        worst / sigma[0],
        delta,
        f.stats.restarts
    );
}

/// (iv): `DoubleDouble`, criterion (i) with `u = 2⁻¹⁰⁵`, values to the
/// double-double level.
#[test]
fn double_double_smoke() {
    let sigma = geometric(60, 1e-20);
    let a = prescribed_dd(90, 60, &sigma, 3, 9);
    let mut op = DenseBlock {
        rows: 90,
        cols: 60,
        data: a.clone(),
    };
    let k = 5;
    let f = svds(&mut op, k, &SvdsOptions::default()).unwrap();
    let ncv = default_ncv(90, 60, k);
    let r = check_residuals("dd", &a, (90, 60), &f, ncv);
    let worst =
        f.s.iter()
            .zip(&sigma)
            .map(|(x, &y)| (*x - Dd::from_f64(y)).abs().to_f64())
            .fold(0.0, f64::max);
    assert!(worst < 1e-28, "dd values: {worst:e}");
    eprintln!(
        "SVDS | double-double 90x60 geometric 1..1e-20 | k={k} | resid {r:.3} ncv·u | value err {worst:.1e} | restarts {}",
        f.stats.restarts
    );
}

fn bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn determinism_storages_and_options() {
    let sigma: Vec<f64> = (0..60).map(|j| 0.9f64.powi(j)).collect();
    let (mut a, _) = prescribed(80, 60, &sigma, 10);
    let o = SvdsOptions::default();
    let f = svds(&mut a, 4, &o).unwrap();
    let g = svds(&mut a, 4, &o).unwrap();
    assert_eq!(bits(&f.s), bits(&g.s));
    assert_eq!(bits(&f.u.data), bits(&g.u.data));
    assert_eq!(f.stats, g.stats);
    // The same matrix in CSC (every entry stored) and as closures.
    let mut trip = Vec::new();
    for j in 0..60 {
        for i in 0..80 {
            trip.push((Row::from_usize(i), Col::from_usize(j), a.get(i, j)));
        }
    }
    let mut csc = CscMatrix::from_triplets(80, 60, &trip).unwrap();
    let h = svds(&mut csc, 4, &o).unwrap();
    assert_eq!(bits(&f.v.data), bits(&h.v.data));
    let (a1, a2) = (a.clone(), a.clone());
    let mut op = from_fns(
        80,
        60,
        move |x: &[f64], y: &mut [f64]| a1.clone().apply(x, y),
        move |x: &[f64], y: &mut [f64]| a2.clone().apply_t(x, y),
    );
    let c = svds(&mut op, 4, &o).unwrap();
    assert_eq!(bits(&f.s), bits(&c.s));
    // vectors = false: same values, empty blocks.
    let nv = svds(
        &mut a,
        4,
        &SvdsOptions {
            vectors: false,
            ..o.clone()
        },
    )
    .unwrap();
    assert_eq!(bits(&nv.s), bits(&f.s));
    assert_eq!((nv.u.cols, nv.v.cols, nv.u.rows), (0, 0, 80));
    // A given v0, another seed, another ncv, a looser tol: same values.
    let v0: Vec<f64> = (0..60u32).map(|i| 1.0 + f64::from(i % 7)).collect();
    for o2 in [
        SvdsOptions {
            v0: Some(v0.clone()),
            ..o.clone()
        },
        SvdsOptions {
            seed: 99,
            ..o.clone()
        },
        SvdsOptions {
            ncv: Some(9),
            ..o.clone()
        },
        SvdsOptions {
            tol: 1e-10,
            ..o.clone()
        },
    ] {
        let d = svds(&mut a, 4, &o2).unwrap();
        let tol = if o2.tol > 0.0 { 1e-9 } else { 1e-14 };
        for (x, y) in d.s.iter().zip(&sigma) {
            assert!((x - y).abs() <= tol, "{o2:?}: {x} {y}");
        }
    }
}

#[test]
fn non_convergence_partial_results_and_errors() {
    // A slowly decaying spectrum, few restarts allowed.
    let sigma: Vec<f64> = (0..150).map(|j: i32| 1.0 - 0.001 * f64::from(j)).collect();
    let (mut a, _) = prescribed(150, 150, &sigma, 11);
    let o = SvdsOptions {
        max_restarts: Some(1),
        ..SvdsOptions::default()
    };
    match svds(&mut a, 10, &o) {
        Err(StatlaError::NoConvergence {
            requested: 10,
            converged,
            restarts: 1,
        }) => assert!(converged < 10),
        other => panic!("expected NoConvergence, got {other:?}"),
    }
    let p = svds(
        &mut a,
        10,
        &SvdsOptions {
            allow_partial: true,
            ..o.clone()
        },
    )
    .unwrap();
    assert!(p.s.len() < 10 && p.stats.converged == p.s.len() && p.u.cols == p.s.len());
    let res = residuals(&a.data, 150, 150, &p.u, &p.s, &p.v);
    if let Some(&s1) = p.s.first() {
        assert!(
            res.iter()
                .all(|&r| r <= (2.0 + C_G * 21.0) * f64::EPSILON / 2.0 * s1 * 2.0)
        );
    }
    // Operator errors and NaN.
    let mut bad = from_fns(
        5,
        4,
        |_: &[f64], _: &mut [f64]| {
            Err(OperatorError {
                message: "inner".into(),
            })
        },
        |_: &[f64], _: &mut [f64]| Ok(()),
    );
    assert!(matches!(
        svds(&mut bad, 1, &SvdsOptions::default()),
        Err(StatlaError::Operator(_))
    ));
    let mut nan = from_fns(
        5,
        4,
        |_: &[f64], y: &mut [f64]| {
            y.fill(f64::NAN);
            Ok(())
        },
        |_: &[f64], y: &mut [f64]| {
            y.fill(0.0);
            Ok(())
        },
    );
    assert!(matches!(
        svds(&mut nan, 1, &SvdsOptions::default()),
        Err(StatlaError::NonFinite {
            what: "operator output",
            ..
        })
    ));
    // The zero matrix: breakdowns everywhere, zero values, orthonormal vectors.
    let mut z = DenseBlock::<f64>::zeros(30, 20);
    let f = svds(&mut z, 3, &SvdsOptions::default()).unwrap();
    assert_eq!(f.s, vec![0.0; 3]);
    assert!(orth_err(&f.u) < 1e-14 && orth_err(&f.v) < 1e-14);
}
