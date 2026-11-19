//! The measurements behind the constants `c_H` and `c_Q` of Chapters 12.2
//! and 12.3 and behind criterion (iii) of L3.2: Markdown tables of the
//! maxima over seeds, per `κ` and per operation, in `f64` and
//! `DoubleDouble`, with indicative timings (the machine is shared).
//!
//! `cargo run -p xingo-statla --release --example accuracy [seeds]`
#![allow(clippy::cast_precision_loss, clippy::items_after_statements)]

#[path = "../tests/support/mod.rs"]
mod support;

use std::time::Instant;

use support::{QrOp, Scalar, chol_case, qr_case};
use xingo_extprec::DoubleDouble;

const KAPPAS: [f64; 6] = [1e1, 1e3, 1e6, 1e8, 1e9, 1e12];

fn chol_table<T: Scalar>(n: usize, seeds: u64) {
    let u = T::u();
    let nu = n as f64 * u;
    println!(
        "\n### Cholesky, {}, n = {n}, {seeds} seeds (maxima; bwd and round trips in units of n·u)\n",
        T::NAME
    );
    println!(
        "| κ(A_t) | bwd downdate /nu | bwd update /nu | upd∘dd /nu | dd∘upd /nu | fwd | dist (2 refact.) | max fwd/dist | min margin | failures |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for &kappa in &KAPPAS {
        let mut w = [0.0f64; 7];
        let mut min_margin = f64::INFINITY;
        let mut fails = 0;
        for s in 0..seeds {
            match chol_case::<T>(n, kappa, 1000 + s) {
                Ok(c) => {
                    w[0] = w[0].max(c.bwd_down / nu);
                    w[1] = w[1].max(c.bwd_up / nu);
                    w[2] = w[2].max(c.ud / nu);
                    w[3] = w[3].max(c.du / nu);
                    w[4] = w[4].max(c.fwd);
                    w[5] = w[5].max(c.dist);
                    w[6] = w[6].max(c.fwd / c.dist.max(f64::MIN_POSITIVE));
                    min_margin = min_margin.min(c.margin);
                }
                Err(e) => {
                    fails += 1;
                    eprintln!("κ = {kappa:e}, seed {s}: {e}");
                }
            }
        }
        println!(
            "| {kappa:.0e} | {:.3} | {:.3} | {:.3} | {:.3} | {:.1e} | {:.1e} | {:.2} | {:.1e} | {fails} |",
            w[0], w[1], w[2], w[3], w[4], w[5], w[6], min_margin
        );
    }
}

fn qr_table<T: Scalar>(m: usize, n: usize, seeds: u64) {
    let u = T::u();
    println!(
        "\n### QR, {}, m = {m}, n = {n}, {seeds} seeds (maxima; orth and bwd in units of m·u, m after the operation)\n",
        T::NAME
    );
    println!(
        "| operation | κ(AᵀA) | orth /mu | bwd /mu | fwd | dist (2 refact.) | max fwd/dist | failures |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|");
    for op in QrOp::ALL {
        for &kappa in &KAPPAS {
            let mut w = [0.0f64; 5];
            let mut fails = 0;
            for s in 0..seeds {
                match qr_case::<T>(m, n, kappa, op, 2000 + s) {
                    Ok(c) => {
                        let mu = c.m as f64 * u;
                        w[0] = w[0].max(c.orth / mu);
                        w[1] = w[1].max(c.bwd / mu);
                        w[2] = w[2].max(c.fwd);
                        w[3] = w[3].max(c.dist);
                        w[4] = w[4].max(c.fwd / c.dist.max(f64::MIN_POSITIVE));
                    }
                    Err(e) => {
                        fails += 1;
                        eprintln!("{} κ = {kappa:e}, seed {s}: {e}", op.name());
                    }
                }
            }
            println!(
                "| {} | {kappa:.0e} | {:.3} | {:.3} | {:.1e} | {:.1e} | {:.2} | {fails} |",
                op.name(),
                w[0],
                w[1],
                w[2],
                w[3],
                w[4]
            );
        }
    }
}

/// The tail of the forward error just below the band edge `κ = 10⁸` of
/// criterion (iii): maximum over many seeds and the count above `10⁻¹²`.
fn tail(seeds: u64) {
    println!("\n### Cholesky downdate, f64: forward error near κ = 1e8 ({seeds} seeds)\n");
    println!(
        "| n | κ(A') nominal | max κ est. | max fwd | count fwd > 1e-12 | max dist | count dist > 1e-12 | max fwd/dist |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|");
    for &n in &[10usize, 50, 100, 200] {
        for &kappa in &[1e6, 1e7, 3e7, 1e8] {
            let (mut mx, mut mk, mut cnt) = (0.0f64, 0.0f64, 0);
            let (mut md, mut cd, mut mr) = (0.0f64, 0, 0.0f64);
            for s in 0..seeds {
                if let Ok(c) = chol_case::<f64>(n, kappa, 50_000 + s) {
                    mx = mx.max(c.fwd);
                    mk = mk.max(c.kappa);
                    if c.fwd > 1e-12 && c.kappa <= 1e8 {
                        cnt += 1;
                    }
                    md = md.max(c.dist);
                    mr = mr.max(c.fwd / c.dist);
                    if c.dist > 1e-12 {
                        cd += 1;
                    }
                }
            }
            println!(
                "| {n} | {kappa:.0e} | {mk:.2e} | {mx:.2e} | {cnt} | {md:.2e} | {cd} | {mr:.2} |"
            );
        }
    }
}

/// Small sizes: the constants `c_H`, `c_Q` are fixed by the worst ratio
/// over all sizes, and small sizes have the largest ratios.
fn small(seeds: u64) {
    println!(
        "\n### Small sizes, maxima over {seeds} seeds and all κ of the table (ratios to n·u, m·u)\n"
    );
    println!(
        "| scalar | n (Cholesky) | bwd dd /nu | bwd upd /nu | upd∘dd /nu | dd∘upd /nu | fwd /(√κ u) |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|");
    fn run<T: Scalar>(seeds: u64) {
        for &n in &[1usize, 2, 3, 5, 10, 20] {
            let mut w = [0.0f64; 5];
            for &kappa in &KAPPAS {
                for s in 0..seeds {
                    if let Ok(c) = chol_case::<T>(n, kappa, 7000 + s) {
                        let nu = n as f64 * T::u();
                        w[0] = w[0].max(c.bwd_down / nu);
                        w[1] = w[1].max(c.bwd_up / nu);
                        w[2] = w[2].max(c.ud / nu);
                        w[3] = w[3].max(c.du / nu);
                        w[4] = w[4].max(c.fwd / (c.kappa.sqrt() * T::u()));
                    }
                }
            }
            println!(
                "| {} | {n} | {:.3} | {:.3} | {:.3} | {:.3} | {:.2} |",
                T::NAME,
                w[0],
                w[1],
                w[2],
                w[3],
                w[4]
            );
        }
    }
    run::<f64>(seeds);
    run::<DoubleDouble>(seeds / 4);
    println!(
        "\n| scalar | m × n (QR) | op | orth /mu | bwd (‖A_t‖) /mu | bwd (path) /mu | bwd (‖A_t‖ ≥ ½ path) /mu | fwd/dist (dist > 0) |"
    );
    println!("|---|---|---|---:|---:|---:|---:|---:|");
    fn runq<T: Scalar>(seeds: u64) {
        for &(m, n) in &[
            (2usize, 1usize),
            (3, 2),
            (5, 3),
            (10, 5),
            (20, 10),
            (12, 11),
        ] {
            for op in QrOp::ALL {
                let mut w = [0.0f64; 5];
                for &kappa in &KAPPAS {
                    for s in 0..seeds {
                        if let Ok(c) = qr_case::<T>(m, n, kappa, op, 9000 + s) {
                            let mu = c.m as f64 * T::u();
                            w[0] = w[0].max(c.orth / mu);
                            w[1] = w[1].max(c.bwd / mu);
                            w[2] = w[2].max(c.bwd_path / mu);
                            if c.shrink >= 0.5 {
                                w[3] = w[3].max(c.bwd / mu);
                            }
                            if c.dist > 0.0 {
                                w[4] = w[4].max(c.fwd / c.dist);
                            }
                        }
                    }
                }
                println!(
                    "| {} | {m} × {n} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.2} |",
                    T::NAME,
                    op.name(),
                    w[0],
                    w[1],
                    w[2],
                    w[3],
                    w[4]
                );
            }
        }
    }
    runq::<f64>(seeds);
    runq::<DoubleDouble>(seeds / 4);
}

/// Indicative timings (the machine is shared): median of 21 repetitions
/// of each operation, in microseconds.
fn timing() {
    use support::{conv, spd, unit, with_cond};
    use xingo_statla::{CholOptions, Cholesky, MatRef, Qr, QrOptions};
    fn med(mut v: Vec<f64>) -> f64 {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }
    fn run<T: Scalar>(n: usize, m: usize) {
        let a: Vec<T> = conv(&spd(n, 1e6, 1));
        let mut rng = support::Rng::new(2);
        let x: Vec<T> = conv(
            &unit(&mut rng, n)
                .iter()
                .map(|v| v * 0.5)
                .collect::<Vec<_>>(),
        );
        let t = Instant::now();
        let c =
            Cholesky::factor(MatRef::new(n, n, n, &a).unwrap(), &CholOptions::default()).unwrap();
        let tf = t.elapsed().as_secs_f64() * 1e6;
        let (mut tu, mut td) = (Vec::new(), Vec::new());
        for _ in 0..21 {
            let mut d = c.clone();
            let t = Instant::now();
            d.update(&x).unwrap();
            tu.push(t.elapsed().as_secs_f64() * 1e6);
            let t = Instant::now();
            d.downdate(&x).unwrap();
            td.push(t.elapsed().as_secs_f64() * 1e6);
        }
        println!(
            "| {} | Cholesky n = {n} | factor {tf:.0} | update {:.0} | downdate {:.0} | | | |",
            T::NAME,
            med(tu),
            med(td)
        );
        let a: Vec<T> = conv(&with_cond(m, n, 1e3, 3));
        let t = Instant::now();
        let f = Qr::factor(MatRef::new(m, n, m, &a).unwrap(), &QrOptions::default()).unwrap();
        let tf = t.elapsed().as_secs_f64() * 1e6;
        let w: Vec<T> = conv(&rng.gauss_vec(n));
        let c: Vec<T> = conv(&rng.gauss_vec(m));
        let mut ts = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for _ in 0..21 {
            let mut g = f.clone();
            let t = Instant::now();
            g.insert_row(m / 2, &w).unwrap();
            ts[0].push(t.elapsed().as_secs_f64() * 1e6);
            let t = Instant::now();
            g.delete_row(m / 2).unwrap();
            ts[1].push(t.elapsed().as_secs_f64() * 1e6);
            let t = Instant::now();
            g.insert_col(0, &c).unwrap();
            ts[2].push(t.elapsed().as_secs_f64() * 1e6);
            let t = Instant::now();
            g.delete_col(0).unwrap();
            ts[3].push(t.elapsed().as_secs_f64() * 1e6);
            let t = Instant::now();
            g.rank1_update(&c, &w).unwrap();
            ts[4].push(t.elapsed().as_secs_f64() * 1e6);
        }
        let [a0, a1, a2, a3, a4] = ts.map(med);
        println!(
            "| {} | QR {m} × {n} | factor {tf:.0} | insert_row {a0:.0} | delete_row {a1:.0} | insert_col (j = 0) {a2:.0} | delete_col (j = 0) {a3:.0} | rank1_update {a4:.0} |",
            T::NAME
        );
    }
    println!("\n### Indicative timings (µs; median of 21; shared machine, release build)\n");
    println!("| scalar | case | | | | | | |");
    println!("|---|---|---|---|---|---|---|---|");
    run::<f64>(200, 400);
    run::<DoubleDouble>(200, 400);
}

fn main() {
    if std::env::var_os("ACCURACY_TIMING").is_some() {
        timing();
        return;
    }
    if std::env::var_os("ACCURACY_SMALL").is_some() {
        small(
            std::env::args()
                .nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(100),
        );
        return;
    }
    if std::env::var_os("ACCURACY_TAIL").is_some() {
        tail(
            std::env::args()
                .nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(100),
        );
        return;
    }
    let seeds: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let t = Instant::now();
    chol_table::<f64>(100, seeds);
    chol_table::<f64>(200, seeds);
    chol_table::<DoubleDouble>(100, seeds.min(5));
    qr_table::<f64>(150, 100, seeds);
    qr_table::<DoubleDouble>(80, 50, seeds.min(5));
    eprintln!("total {:.1} s", t.elapsed().as_secs_f64());
}
