//! Acceptance tests of Chapter 12.4 (`rsvd`, `range_finder`), task L3.3:
//! the HMT expectation bound on the **mean over 40 seeds** for `q ∈ {0, 1,
//! 2}` on three prescribed spectra (the pilot's), invariants (R1)–(R2),
//! equality of the bits across operator storages, the error paths, and a
//! `DoubleDouble` smoke test.

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]

mod support;

use support::svd::{lambda_max_upper, orth_err, prescribed, residuals, spectral_norm};
use support::{Rng, haar};
use xingo_extprec::DoubleDouble as Dd;
use xingo_sparse::{Col, CscMatrix, Row};
use xingo_statla::{
    DenseBlock, OperatorError, RsvdOptions, Sketch, StatlaError, from_fns, range_finder, rsvd,
};

const SEEDS: u64 = 40;
const M: usize = 120;
const N: usize = 80;
const K: usize = 10;
const P: usize = 10;

/// `A = U Σ Vᵀ` with Haar `U` (`m × n`) and `V`.
fn haar_matrix(m: usize, n: usize, s: &[f64], seed: u64) -> DenseBlock<f64> {
    let mut rng = Rng::new(seed);
    let u = haar(&mut rng, m, n);
    let v = haar(&mut rng, n, n);
    let mut a = vec![0.0; m * n];
    for j in 0..n {
        for k in 0..n {
            let t = s[k] * v[j + k * n];
            for i in 0..m {
                a[i + j * m] += u[i + k * m] * t;
            }
        }
    }
    DenseBlock {
        rows: m,
        cols: n,
        data: a,
    }
}

/// The HMT expectation bound with `q` power iterations (chapter 12.4).
fn hmt_bound(s: &[f64], k: usize, p: usize, q: usize) -> f64 {
    let e = (2 * q + 1) as i32;
    let tail: f64 = s[k..].iter().map(|x| x.powi(2 * e)).sum::<f64>().sqrt();
    let inner = (1.0 + (k as f64 / (p as f64 - 1.0)).sqrt()) * s[k].powi(e)
        + std::f64::consts::E * ((k + p) as f64).sqrt() / p as f64 * tail;
    inner.powf(1.0 / f64::from(e))
}

/// `AᵀA` (`n × n`).
fn gram(a: &DenseBlock<f64>) -> Vec<f64> {
    let (m, n) = (a.rows, a.cols);
    let mut g = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            g[i + j * n] = (0..m).map(|r| a.get(r, i) * a.get(r, j)).sum();
        }
    }
    g
}

/// Squarings of the norm bound: the bound exceeds `‖E‖` by at most
/// `n^{1/2^{s+1}}` = 1.035 for `n = 80`, `s = 6`.
const SQUARINGS: u32 = 6;

/// A certified upper bound of `‖A − QQᵀA‖₂`: with `E = A − QQᵀA` and `B =
/// QᵀA`, `EᵀE = AᵀA − BᵀB` (`Q` orthonormal), whose largest eigenvalue is
/// bounded above by `lambda_max_upper`. Upper bounds only make the
/// acceptance stricter.
fn range_error(a: &DenseBlock<f64>, ata: &[f64], q: &DenseBlock<f64>) -> f64 {
    let (n, l) = (a.cols, q.cols);
    let dotv = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f64>();
    // B = QᵀA, stored by columns of length l.
    let mut b = vec![0.0; l * n];
    for j in 0..n {
        for t in 0..l {
            b[t + j * l] = dotv(
                &q.data[t * q.rows..(t + 1) * q.rows],
                &a.data[j * a.rows..(j + 1) * a.rows],
            );
        }
    }
    let mut g = ata.to_vec();
    for j in 0..n {
        for i in 0..=j {
            let d = dotv(&b[i * l..(i + 1) * l], &b[j * l..(j + 1) * l]);
            g[i + j * n] -= d;
            if i != j {
                g[j + i * n] -= d;
            }
        }
    }
    lambda_max_upper(&g, n, SQUARINGS).sqrt()
}

/// The bound of `range_error` against the exact norm (one-sided Jacobi of
/// `E`) on a few runs: within its stated slack, never below.
#[test]
fn norm_bound_is_an_upper_bound_within_its_slack() {
    let (_, s) = spectra().remove(1);
    let mut a = haar_matrix(M, N, &s, 7);
    let ata = gram(&a);
    for seed in 0..3 {
        let opts = RsvdOptions {
            power_iters: 0,
            seed,
            ..RsvdOptions::default()
        };
        let q = range_finder(&mut a, K + P, &opts).unwrap();
        let mut e = a.data.clone();
        for j in 0..N {
            for t in 0..K + P {
                let bt: f64 = (0..M).map(|i| q.get(i, t) * a.get(i, j)).sum();
                for i in 0..M {
                    e[i + j * M] -= q.get(i, t) * bt;
                }
            }
        }
        let exact = spectral_norm(&e, M, N);
        let up = range_error(&a, &ata, &q);
        let slack = (N as f64).powf(1.0 / f64::from(2u32.pow(SQUARINGS + 1)));
        assert!(
            up >= exact * (1.0 - 1e-10) && up <= exact * slack * (1.0 + 1e-10),
            "{up} {exact}"
        );
    }
}

fn spectra() -> Vec<(&'static str, Vec<f64>)> {
    vec![
        (
            "j^-2",
            (0..N).map(|j| 1.0 / ((j + 1) as f64).powi(2)).collect(),
        ),
        (
            "j^-1/2",
            (0..N).map(|j| 1.0 / ((j + 1) as f64).sqrt()).collect(),
        ),
        (
            "1 (j<=k), 1e-3",
            (0..N).map(|j| if j < K { 1.0 } else { 1e-3 }).collect(),
        ),
    ]
}

/// Acceptance (L3.3): the mean over 40 seeds stays below the HMT bound,
/// for every spectrum and `q ∈ {0, 1, 2}` (Gaussian sketch). Rademacher is
/// measured and printed, not asserted (the bound is not proven for it).
#[test]
fn mean_over_seeds_is_below_the_hmt_bound() {
    for (name, s) in spectra() {
        let mut a = haar_matrix(M, N, &s, 7);
        let ata = gram(&a);
        for q in 0..=2 {
            let bound = hmt_bound(&s, K, P, q);
            let mut stats = [(0.0f64, 0.0f64); 2];
            for (w, sketch) in [Sketch::Gaussian, Sketch::Rademacher]
                .into_iter()
                .enumerate()
            {
                for seed in 0..SEEDS {
                    let opts = RsvdOptions {
                        oversample: P,
                        power_iters: q,
                        sketch,
                        seed: 1000 + seed,
                    };
                    let qm = range_finder(&mut a, K + P, &opts).unwrap();
                    assert_eq!((qm.rows, qm.cols), (M, K + P));
                    let err = range_error(&a, &ata, &qm);
                    stats[w].0 += err / SEEDS as f64;
                    stats[w].1 = stats[w].1.max(err);
                }
            }
            let (mean, max) = stats[0];
            eprintln!(
                "HMT | {name} | q={q} | bound {bound:.3e} | gauss mean {mean:.3e} ({:.3}) max {max:.3e} ({:.3}) | rademacher mean {:.3e} ({:.3}) max {:.3e}",
                mean / bound,
                max / bound,
                stats[1].0,
                stats[1].0 / bound,
                stats[1].1
            );
            assert!(
                mean <= bound,
                "{name}, q = {q}: mean {mean:e} above the bound {bound:e}"
            );
        }
    }
}

/// (R1): orthonormal `U`, `V`, nonincreasing `s`, small residuals on a
/// low-rank matrix; (R2): the same bits twice and across storages.
#[test]
fn rsvd_invariants_determinism_and_storages() {
    // Exact rank 6 (prescribed, certified): rsvd recovers it to rounding.
    let mut sig = vec![0.0; 60];
    for (i, v) in sig.iter_mut().take(6).enumerate() {
        *v = 10f64.powi(-(i as i32));
    }
    let (mut a, delta) = prescribed(90, 60, &sig, 3);
    let opts = RsvdOptions::default();
    let f = rsvd(&mut a, 6, &opts).unwrap();
    assert_eq!((f.u.rows, f.u.cols, f.v.rows, f.v.cols), (90, 6, 60, 6));
    assert!(orth_err(&f.u) < 1e-14 && orth_err(&f.v) < 1e-14);
    assert!(f.s.windows(2).all(|w| w[1] <= w[0]));
    for (i, (&x, &y)) in f.s.iter().zip(&sig).enumerate() {
        assert!((x - y).abs() <= 1e-14 + delta, "sigma_{i}: {x} vs {y}");
    }
    let res = residuals(&a.data, 90, 60, &f.u, &f.s, &f.v);
    assert!(res.iter().all(|&r| r < 1e-14), "{res:?}");
    let st = f.stats;
    assert_eq!(
        (st.applies, st.applies_t, st.converged),
        (16 * 3, 16 * 3, 6)
    );
    // Determinism: the same bits again, and through CSC and closures.
    let g = rsvd(&mut a, 6, &opts).unwrap();
    assert_eq!(bits(&f.s), bits(&g.s));
    assert_eq!(bits(&f.u.data), bits(&g.u.data));
    let mut trip = Vec::new();
    for j in 0..60 {
        for i in 0..90 {
            trip.push((Row::from_usize(i), Col::from_usize(j), a.get(i, j)));
        }
    }
    let mut csc = CscMatrix::from_triplets(90, 60, &trip).unwrap();
    assert_eq!(csc.nnz(), 90 * 60, "a dense matrix stored in CSC");
    let h = rsvd(&mut csc, 6, &opts).unwrap();
    assert_eq!(bits(&f.v.data), bits(&h.v.data));
    let (a1, a2) = (a.clone(), a.clone());
    let mut op = from_fns(
        90,
        60,
        move |x: &[f64], y: &mut [f64]| {
            y.fill(0.0);
            for (j, &xj) in x.iter().enumerate() {
                for i in 0..90 {
                    y[i] += a1.data[i + j * 90] * xj;
                }
            }
            Ok(())
        },
        move |x: &[f64], y: &mut [f64]| {
            for (j, yj) in y.iter_mut().enumerate() {
                *yj = (0..90).map(|i| a2.data[i + j * 90] * x[i]).sum();
            }
            Ok(())
        },
    );
    let c = rsvd(&mut op, 6, &opts).unwrap();
    assert_eq!(bits(&f.s), bits(&c.s));
    // A different seed gives different bits, the same values.
    let d = rsvd(&mut a, 6, &RsvdOptions { seed: 5, ..opts }).unwrap();
    assert_ne!(bits(&f.u.data), bits(&d.u.data));
    assert!(f.s.iter().zip(&d.s).all(|(x, y)| (x - y).abs() < 1e-14));
    // q = 0 and k = min(m, n) (the sketch is capped at min(m, n)).
    let full = rsvd(
        &mut a,
        60,
        &RsvdOptions {
            power_iters: 0,
            ..opts
        },
    )
    .unwrap();
    assert_eq!(full.s.len(), 60);
    assert!(orth_err(&full.u) < 1e-13 && orth_err(&full.v) < 1e-13);
}

/// Steep spectra with power iterations: `q = 2` recovers `σ_1 … σ_k` to a
/// small relative error, and `q = 6` the values down to `10⁻¹²σ_1` to
/// `≈ ε σ_1`. (Task L3.3 measured the same accuracy *without* the half-step
/// re-orthonormalization of S3 on these cases — mutation M9 survives; the
/// step is kept as HMT recommend, not because a test here needs it.)
#[test]
fn power_iterations_keep_a_steep_spectrum() {
    let s: Vec<f64> = (0..N).map(|j| 10f64.powf(-(j as f64) / 3.0)).collect();
    let (mut a, delta) = prescribed(M, N, &s, 13);
    let f = rsvd(&mut a, K, &RsvdOptions::default()).unwrap();
    let worst =
        f.s.iter()
            .zip(&s)
            .map(|(x, y)| (x - y).abs() / y)
            .fold(0.0, f64::max);
    eprintln!(
        "rsvd steep spectrum 10^(-j/3), q = 2: max relative error {worst:.2e} (delta {delta:.1e})"
    );
    assert!(worst <= 1e-9, "{worst:e}");
    // Wanted values down to 1e-12 over a 1e-15 floor.
    let s: Vec<f64> = (0..N)
        .map(|j| 10f64.powf(-1.3 * j as f64).max(1e-15))
        .collect();
    let (mut a, delta) = prescribed(M, N, &s, 14);
    let f = rsvd(
        &mut a,
        K,
        &RsvdOptions {
            power_iters: 6,
            ..RsvdOptions::default()
        },
    )
    .unwrap();
    let worst =
        f.s.iter()
            .zip(&s)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
    eprintln!(
        "rsvd steep spectrum to 1e-12, q = 6: max absolute error {worst:.2e} sigma_1 (delta {delta:.1e})"
    );
    assert!(worst <= 1e-13, "{worst:e}");
}

fn bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn errors_are_reported_not_panicked() {
    let mut a = DenseBlock::<f64>::zeros(5, 4);
    a.set(0, 0, 1.0);
    let o = RsvdOptions::default();
    for k in [0, 5] {
        assert!(matches!(
            rsvd(&mut a, k, &o),
            Err(StatlaError::InvalidArgument { .. })
        ));
    }
    for l in [0, 5] {
        assert!(matches!(
            range_finder(&mut a, l, &o),
            Err(StatlaError::InvalidArgument { .. })
        ));
    }
    let mut nan = from_fns(
        3,
        3,
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
        rsvd(&mut nan, 1, &o),
        Err(StatlaError::NonFinite {
            what: "operator output",
            ..
        })
    ));
    let mut bad = from_fns(
        3,
        3,
        |_: &[f64], _: &mut [f64]| {
            Err(OperatorError {
                message: "solve failed".into(),
            })
        },
        |_: &[f64], _: &mut [f64]| Ok(()),
    );
    assert_eq!(
        rsvd(&mut bad, 1, &o).unwrap_err(),
        StatlaError::Operator(OperatorError {
            message: "solve failed".into()
        })
    );
    // The zero matrix: zero values, orthonormal vectors.
    let mut z = DenseBlock::<f64>::zeros(6, 4);
    let f = rsvd(&mut z, 2, &o).unwrap();
    assert_eq!(f.s, vec![0.0, 0.0]);
    assert!(orth_err(&f.u) < 1e-15 && orth_err(&f.v) < 1e-15);
}

/// `DoubleDouble` smoke test: the same sketch (generated in `f64`), the
/// same values as `f64` to `f64` accuracy, residuals at the double-double
/// level on an exactly low-rank matrix.
#[test]
fn double_double_smoke() {
    let mut sig = vec![0.0; 30];
    for (i, v) in sig.iter_mut().take(5).enumerate() {
        *v = 2f64.powi(-(i as i32));
    }
    let a_dd = support::svd::prescribed_dd(40, 30, &sig, 4, 11);
    let mut a = DenseBlock {
        rows: 40,
        cols: 30,
        data: a_dd.clone(),
    };
    let f = rsvd(&mut a, 5, &RsvdOptions::default()).unwrap();
    for (x, y) in f.s.iter().zip(&sig) {
        assert!((*x - Dd::from_f64(*y)).abs().to_f64() < 1e-28, "{x:?} {y}");
    }
    let res = residuals(&a_dd, 40, 30, &f.u, &f.s, &f.v);
    assert!(res.iter().all(|&r| r < 1e-28), "{res:?}");
    assert!(orth_err(&f.u) < 1e-29 && orth_err(&f.v) < 1e-29);
    let mut a64 = DenseBlock {
        rows: 40,
        cols: 30,
        data: a_dd.iter().map(|x| x.to_f64()).collect(),
    };
    let g = rsvd(&mut a64, 5, &RsvdOptions::default()).unwrap();
    assert!(
        f.s.iter()
            .zip(&g.s)
            .all(|(x, y)| (x.to_f64() - y).abs() < 1e-14)
    );
}
