//! Shared oracle of the acceptance tests of Chapters 12.2 and 12.3.
//!
//! * Test matrices are generated in `f64` from a seed (xoshiro256++ and the
//!   Box–Muller transform — the libm of the tests is fine here: only the
//!   *matrices* depend on it, never a reference value) and converted
//!   exactly to the scalar under test.
//! * Prescribed conditioning: `A = H₃H₂H₁ D H₁H₂H₃` (SPD, eigenvalues
//!   `κ^{−i/(n−1)}`) and `A = (H₃H₂H₁ [Σ; 0]) (G₁G₂G₃)` (singular values
//!   `κ^{−i/(n−1)}`), with random Householder reflectors `H`, `G`.
//! * The exact matrix `A_t` of a sequence of operations is accumulated in
//!   double-double, and so are the residuals `RᵀR − A_t` and `QR − A_t`
//!   (for `T = f64` they are then exact to `≈ 2⁻¹⁰⁰`; for `T =
//!   DoubleDouble` their evaluation error is of the order of what they
//!   measure — stated in the tables).
#![allow(
    dead_code,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

pub mod svd;

use xingo_extprec::DoubleDouble as Dd;
use xingo_statla::{CholOptions, Cholesky, DenseBlock, MatRef, Qr, QrOptions, Real};

/// The scalars of the tests: `f64` and `DoubleDouble`.
pub trait Scalar: Real {
    /// Name for messages.
    const NAME: &'static str;
    /// Exact widening to double-double (identity for `Dd`).
    fn dd(self) -> Dd;
    /// Rounding from double-double.
    fn of_dd(x: Dd) -> Self;
    /// Unit roundoff `u = ε/2` as `f64`.
    fn u() -> f64 {
        Self::EPSILON.to_f64() / 2.0
    }
}

impl Scalar for f64 {
    const NAME: &'static str = "f64";
    fn dd(self) -> Dd {
        Dd::from_f64(self)
    }
    fn of_dd(x: Dd) -> Self {
        x.to_f64()
    }
}

impl Scalar for Dd {
    const NAME: &'static str = "dd";
    fn dd(self) -> Dd {
        self
    }
    fn of_dd(x: Dd) -> Self {
        x
    }
}

/// xoshiro256++ seeded by SplitMix64 (test generator only).
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [0u64; 4];
        for v in &mut s {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *v = x ^ (x >> 31);
        }
        Self { s }
    }
    pub fn next_u64(&mut self) -> u64 {
        let r = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    pub fn gauss(&mut self) -> f64 {
        let u1 = 1.0 - self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    pub fn gauss_vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.gauss()).collect()
    }
    pub fn sign(&mut self) -> f64 {
        if self.next_u64() >> 63 == 0 {
            1.0
        } else {
            -1.0
        }
    }
}

/// A random unit vector.
pub fn unit(rng: &mut Rng, n: usize) -> Vec<f64> {
    let mut v = rng.gauss_vec(n);
    let s = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v {
        *x /= s;
    }
    v
}

/// The spectrum `κ^{−i/(n−1)}`, `i = 0..n` (`[1]` for `n = 1`).
pub fn spectrum(n: usize, kappa: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            if n == 1 {
                1.0
            } else {
                kappa.powf(-(i as f64) / (n - 1) as f64)
            }
        })
        .collect()
}

/// A Haar-distributed `m × n` matrix with orthonormal columns: a Gaussian
/// matrix orthonormalized by classical Gram–Schmidt applied twice.
pub fn haar(rng: &mut Rng, m: usize, n: usize) -> Vec<f64> {
    let mut q = rng.gauss_vec(m * n);
    for j in 0..n {
        for _ in 0..2 {
            let s: Vec<f64> = (0..j)
                .map(|i| (0..m).map(|k| q[k + i * m] * q[k + j * m]).sum())
                .collect();
            for (i, si) in s.iter().enumerate() {
                for k in 0..m {
                    q[k + j * m] -= si * q[k + i * m];
                }
            }
        }
        let nrm = (0..m)
            .map(|k| q[k + j * m] * q[k + j * m])
            .sum::<f64>()
            .sqrt();
        for k in 0..m {
            q[k + j * m] /= nrm;
        }
    }
    q
}

/// Symmetric positive definite `n × n`, `A = Q D Qᵀ` with `Q` Haar and `D`
/// the spectrum `κ^{−i/(n−1)}` (full storage, exactly symmetric; the
/// pilot's family).
pub fn spd(n: usize, kappa: f64, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let d = spectrum(n, kappa);
    let q = haar(&mut rng, n, n);
    let mut a = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..=j {
            a[i + j * n] = (0..n).map(|k| q[i + k * n] * d[k] * q[j + k * n]).sum();
        }
    }
    symmetrize(&mut a, n);
    a
}

/// Copy the upper triangle onto the lower.
pub fn symmetrize(a: &mut [f64], n: usize) {
    for j in 0..n {
        for i in j + 1..n {
            a[i + j * n] = a[j + i * n];
        }
    }
}

/// `m × n` (`m ≥ n`), `A = U Σ Vᵀ` with `U`, `V` Haar and singular values
/// `κ^{−i/(n−1)}`.
pub fn with_cond(m: usize, n: usize, kappa: f64, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let s = spectrum(n, kappa);
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
    a
}

/// Convert a slice exactly.
pub fn conv<T: Scalar>(a: &[f64]) -> Vec<T> {
    a.iter().map(|&v| T::from_f64(v)).collect()
}

/// Widen to double-double.
pub fn to_dd<T: Scalar>(a: &[T]) -> Vec<Dd> {
    a.iter().map(|&v| v.dd()).collect()
}

/// Round from double-double.
pub fn of_dd<T: Scalar>(a: &[Dd]) -> Vec<T> {
    a.iter().map(|&v| T::of_dd(v)).collect()
}

/// `‖x‖_F` of double-double data.
pub fn fro(a: &[Dd]) -> f64 {
    a.iter().map(|&v| v * v).sum::<Dd>().sqrt().to_f64()
}

/// `‖a − b‖_F / ‖b‖_F` in double-double.
pub fn rel<T: Scalar>(a: &[T], b: &[T]) -> f64 {
    let d: Vec<Dd> = a.iter().zip(b).map(|(&x, &y)| x.dd() - y.dd()).collect();
    fro(&d) / fro(&to_dd(b)).max(f64::MIN_POSITIVE)
}

/// `RᵀR` (full `n × n`) in double-double.
pub fn gram<T: Scalar>(r: &DenseBlock<T>) -> Vec<Dd> {
    let n = r.rows;
    let rd = to_dd(&r.data);
    let mut g = vec![Dd::ZERO; n * n];
    for j in 0..n {
        for i in 0..=j {
            let mut s = Dd::ZERO;
            for k in 0..=i {
                s += rd[k + i * n] * rd[k + j * n];
            }
            g[i + j * n] = s;
            g[j + i * n] = s;
        }
    }
    g
}

/// `‖RᵀR − A_t‖_F / ‖A_t‖_F` (H2).
pub fn chol_backward<T: Scalar>(c: &Cholesky<T>, at: &[Dd]) -> f64 {
    let r = c.upper().to_block();
    let g = gram(&r);
    let d: Vec<Dd> = g.iter().zip(at).map(|(&x, &y)| x - y).collect();
    fro(&d) / fro(at)
}

/// `A + σ x xᵀ` in double-double (full storage).
pub fn rank1_dd(a: &mut [Dd], n: usize, x: &[Dd], sigma: f64) {
    for j in 0..n {
        for i in 0..n {
            a[i + j * n] += Dd::from_f64(sigma) * x[i] * x[j];
        }
    }
}

/// A lower estimate of `κ₂(RᵀR)` by 40 steps of the power method on `RᵀR`
/// and on its inverse (each estimate is a lower bound of its norm, so the
/// ratio is a lower bound of `κ`: classifying by it makes criterion (iii)
/// only stricter). Run in `f64`.
pub fn kappa_of_upper(r: &DenseBlock<f64>) -> f64 {
    let n = r.rows;
    if n == 0 {
        return 1.0;
    }
    let mul = |x: &[f64]| -> Vec<f64> {
        // y = RᵀR x
        let mut t = vec![0.0; n];
        for i in 0..n {
            for k in i..n {
                t[i] += r.data[i + k * n] * x[k];
            }
        }
        let mut y = vec![0.0; n];
        for j in 0..n {
            for k in 0..=j {
                y[j] += r.data[k + j * n] * t[k];
            }
        }
        y
    };
    let inv = |x: &[f64]| -> Vec<f64> {
        let mut y = x.to_vec();
        for j in 0..n {
            let s: f64 = (0..j).map(|i| r.data[i + j * n] * y[i]).sum();
            y[j] = (y[j] - s) / r.data[j + j * n];
        }
        for j in (0..n).rev() {
            y[j] /= r.data[j + j * n];
            for i in 0..j {
                y[i] -= r.data[i + j * n] * y[j];
            }
        }
        y
    };
    let power = |f: &dyn Fn(&[f64]) -> Vec<f64>| {
        let mut x: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64).sin()).collect();
        let mut lam = 0.0;
        for _ in 0..40 {
            let y = f(&x);
            let ny = y.iter().map(|v| v * v).sum::<f64>().sqrt();
            let nx = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            lam = ny / nx;
            x = y.iter().map(|v| v / ny).collect();
        }
        lam
    };
    power(&mul) * power(&inv)
}

/// The upper factor as an `f64` block (for the κ estimate).
pub fn upper_f64<T: Scalar>(r: &DenseBlock<T>) -> DenseBlock<f64> {
    DenseBlock {
        rows: r.rows,
        cols: r.cols,
        data: r.data.iter().map(|v| v.to_f64()).collect(),
    }
}

/// Bits of a factor, for the transactional checks (H3, Q3).
pub fn bits<T: Scalar>(a: &[T]) -> Vec<(u64, u64)> {
    a.iter()
        .map(|&v| {
            let d = v.dd();
            (d.hi().to_bits(), d.lo().to_bits())
        })
        .collect()
}

// --------------------------------------------------------------- QR oracle

/// `‖QᵀQ − I‖_F` in double-double.
pub fn orthogonality<T: Scalar>(q: MatRef<'_, T>) -> f64 {
    let (m, n) = (q.rows(), q.cols());
    let mut s = Dd::ZERO;
    for j in 0..n {
        let qj = to_dd(q.col(j).unwrap());
        for i in 0..n {
            let qi = to_dd(q.col(i).unwrap());
            let mut d: Dd = (0..m).map(|k| qi[k] * qj[k]).sum();
            if i == j {
                d -= Dd::ONE;
            }
            s += d * d;
        }
    }
    s.sqrt().to_f64()
}

/// `‖QR − A_t‖_F / ‖A_t‖_F` in double-double (`A_t` is `m × n`).
pub fn qr_backward<T: Scalar>(f: &Qr<T>, at: &[Dd]) -> f64 {
    let (m, n) = (f.nrows(), f.ncols());
    let q = to_dd(&f.q().to_block().data);
    let r = to_dd(&f.r().to_block().data);
    let mut d = vec![Dd::ZERO; m * n];
    for j in 0..n {
        for k in 0..=j {
            let rkj = r[k + j * n];
            for i in 0..m {
                d[i + j * m] += q[i + k * m] * rkj;
            }
        }
        for i in 0..m {
            d[i + j * m] -= at[i + j * m];
        }
    }
    fro(&d) / fro(at).max(f64::MIN_POSITIVE)
}

/// The Householder QR in `T` of `A_t` rounded to `T` and perturbed
/// componentwise by `±u` (signs from `seed`): a second refactorization of
/// the same matrix perturbed at the level of rounding (row and column
/// operations leave `A_t` exactly representable, so rounding alone would
/// not perturb it).
pub fn refactor_qr<T: Scalar>(at: &[Dd], m: usize, n: usize, seed: u64) -> Option<Qr<T>> {
    let mut a: Vec<T> = of_dd(at);
    let mut rng = Rng::new(seed);
    let u = T::from_f64(T::u());
    for v in &mut a {
        *v = *v + T::from_f64(rng.sign()) * u * *v;
    }
    Qr::factor(MatRef::new(m, n, m, &a).ok()?, &QrOptions::default()).ok()
}

// ------------------------------------------------------ acceptance cases

/// Criterion (iii) of L3.2: `fwd ≤ 10⁻¹²` where `κ ≤ 10⁸`, else `fwd ≤
/// 10 · dist`.
pub fn forward_ok(fwd: f64, dist: f64, kappa: f64) -> bool {
    if kappa <= 1e8 {
        fwd <= 1e-12
    } else {
        fwd <= 10.0 * dist
    }
}

/// One downdate case of Chapter 12.2 (the pilot's family): `A' ` with
/// eigenvalues `κ^{−i/(n−1)}`, a unit `x`, `A = fl(A' + xxᵀ)` factored,
/// then downdated by `x` (target `A_t = A − xxᵀ`, exact in double-double).
#[derive(Clone, Copy, Debug)]
pub struct CholCase {
    /// `κ(A')` (lower estimate from `R⋆`).
    pub kappa: f64,
    /// (H2) after the downdate, relative.
    pub bwd_down: f64,
    /// (H2) after an update of `R` by `x` (target `A + xxᵀ`).
    pub bwd_up: f64,
    /// `‖R̃ − R⋆‖/‖R⋆‖`, `R⋆` the factor of `A'` (computed in
    /// double-double).
    pub fwd: f64,
    /// `‖R_f − R⋆‖/‖R⋆‖`, `R_f` the refactorization in `T` of `A − xxᵀ`
    /// formed in `T`.
    pub dist: f64,
    /// `‖update(downdate(R)) − R‖/‖R‖`.
    pub ud: f64,
    /// `‖downdate(update(R)) − R‖/‖R‖`.
    pub du: f64,
    /// `1 − ‖a‖²` of the downdate.
    pub margin: f64,
}

/// Run one [`CholCase`]; `Err` with a message if a step that must succeed
/// fails.
pub fn chol_case<T: Scalar>(n: usize, kappa: f64, seed: u64) -> Result<CholCase, String> {
    let ap = spd(n, kappa, seed);
    let mut rng = Rng::new(seed ^ 0x5eed_0fc0_ffee);
    let x = unit(&mut rng, n);
    // A = A' + xxᵀ formed in T (exactly symmetric).
    let xt: Vec<T> = conv(&x);
    let xd = to_dd(&x);
    let mut at_view: Vec<T> = conv(&ap);
    for j in 0..n {
        for i in 0..n {
            at_view[i + j * n] += xt[i] * xt[j];
        }
    }
    let ad = to_dd(&at_view);
    let r0 = Cholesky::factor(
        MatRef::new(n, n, n, &at_view).unwrap(),
        &CholOptions::default(),
    )
    .map_err(|e| format!("factor: {e}"))?;
    let r0b = r0.upper().to_block();
    // Downdate.
    let mut c1 = r0.clone();
    let info = c1.downdate(&xt).map_err(|e| format!("downdate: {e}"))?;
    c1.check_invariants().map_err(str::to_string)?;
    let mut at = ad.clone();
    rank1_dd(&mut at, n, &xd, -1.0);
    let bwd_down = chol_backward(&c1, &at);
    // (iii), as the pilot s01 measured it: against R⋆, the factor of the
    // matrix A' the case was built from (computed in double-double, rounded
    // to T); the distance between two refactorizations of A' perturbed by
    // rounding is that of R_f, the refactorization in T of A − xxᵀ formed in
    // T (what a user would compute instead of downdating: the pilot's
    // "fresh"), from R⋆.
    let r1b = exact_chol::<T>(&to_dd(&ap), n).ok_or("exact refactorization failed")?;
    let mut af: Vec<T> = at_view.clone();
    for j in 0..n {
        for i in 0..n {
            af[i + j * n] -= xt[i] * xt[j];
        }
    }
    let rf = Cholesky::factor(MatRef::new(n, n, n, &af).unwrap(), &CholOptions::default())
        .map_err(|e| format!("refactorization in T: {e}"))?;
    let fwd = rel(&c1.upper().to_block().data, &r1b.data);
    let dist = rel(&rf.upper().to_block().data, &r1b.data);
    let kappa_est = kappa_of_upper(&upper_f64(&r1b));
    // Round trip update∘downdate.
    c1.update(&xt).map_err(|e| format!("update: {e}"))?;
    let ud = rel(&c1.upper().to_block().data, &r0b.data);
    // Update, then downdate∘update.
    let mut c2 = r0.clone();
    c2.update(&xt).map_err(|e| format!("update: {e}"))?;
    c2.check_invariants().map_err(str::to_string)?;
    let mut au = ad.clone();
    rank1_dd(&mut au, n, &xd, 1.0);
    let bwd_up = chol_backward(&c2, &au);
    c2.downdate(&xt)
        .map_err(|e| format!("downdate after update: {e}"))?;
    let du = rel(&c2.upper().to_block().data, &r0b.data);
    Ok(CholCase {
        kappa: kappa_est,
        bwd_down,
        bwd_up,
        fwd,
        dist,
        ud,
        du,
        margin: info.margin,
    })
}

/// The QR operations of Chapter 12.3.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QrOp {
    InsertRow,
    DeleteRow,
    InsertCol,
    DeleteCol,
    Rank1,
}

impl QrOp {
    pub const ALL: [QrOp; 5] = [
        QrOp::InsertRow,
        QrOp::DeleteRow,
        QrOp::InsertCol,
        QrOp::DeleteCol,
        QrOp::Rank1,
    ];
    pub fn name(self) -> &'static str {
        match self {
            QrOp::InsertRow => "insert_row",
            QrOp::DeleteRow => "delete_row",
            QrOp::InsertCol => "insert_col",
            QrOp::DeleteCol => "delete_col",
            QrOp::Rank1 => "rank1_update",
        }
    }
}

/// One QR case: `A` (`m × n`, `κ₂(A) = √κ`) factored, one operation applied.
#[derive(Clone, Copy, Debug)]
pub struct QrCase {
    /// `κ(A_tᵀA_t)` (lower estimate from the refactorization).
    pub kappa: f64,
    /// `‖QᵀQ − I‖_F` after the operation.
    pub orth: f64,
    /// `‖QR − A_t‖_F/‖A_t‖_F` after the operation.
    pub bwd: f64,
    /// `‖QR − A_t‖_F/max(‖A‖_F, ‖A_t‖_F)` (the norm along the path: a
    /// deletion can cancel most of `A`).
    pub bwd_path: f64,
    /// `‖A_t‖_F / max(‖A‖_F, ‖A_t‖_F)`.
    pub shrink: f64,
    /// `‖R̃ − R⋆‖/‖R⋆‖`, `R⋆` from the QR of `A_t` in double-double.
    pub fwd: f64,
    /// `‖R_f − R⋆‖/‖R⋆‖`, `R_f` from the QR in `T` of `A_t` perturbed by
    /// `±u` componentwise.
    pub dist: f64,
    /// Rows of the result.
    pub m: usize,
}

/// Apply `op` to `f` and `A_t` (both `m × n` before; `at` column-major in
/// double-double). Returns `Err` for an operation that must succeed but
/// failed.
pub fn apply_op<T: Scalar>(
    f: &mut Qr<T>,
    at: &mut Vec<Dd>,
    op: QrOp,
    rng: &mut Rng,
) -> Result<(), String> {
    let (m, n) = (f.nrows(), f.ncols());
    // The typical row norm (1 for an empty matrix: near the underflow
    // threshold the double-double has fewer than 106 bits).
    let scale = if n == 0 {
        1.0
    } else {
        fro(at) / (m.max(1) as f64).sqrt()
    };
    match op {
        QrOp::InsertRow => {
            let i = rng.below(m + 1);
            let w: Vec<f64> = rng
                .gauss_vec(n)
                .iter()
                .map(|v| v * scale / (n.max(1) as f64).sqrt())
                .collect();
            f.insert_row(i, &conv::<T>(&w))
                .map_err(|e| format!("insert_row: {e}"))?;
            let mut b = vec![Dd::ZERO; (m + 1) * n];
            for j in 0..n {
                for r in 0..=m {
                    b[r + j * (m + 1)] = match r.cmp(&i) {
                        std::cmp::Ordering::Less => at[r + j * m],
                        std::cmp::Ordering::Equal => Dd::from_f64(w[j]),
                        std::cmp::Ordering::Greater => at[r - 1 + j * m],
                    };
                }
            }
            *at = b;
        }
        QrOp::DeleteRow => {
            let i = rng.below(m);
            f.delete_row(i).map_err(|e| format!("delete_row: {e}"))?;
            let mut b = Vec::with_capacity((m - 1) * n);
            for j in 0..n {
                for r in 0..m {
                    if r != i {
                        b.push(at[r + j * m]);
                    }
                }
            }
            *at = b;
        }
        QrOp::InsertCol => {
            let j = rng.below(n + 1);
            let c: Vec<f64> = unit(rng, m).iter().map(|v| v * scale).collect();
            f.insert_col(j, &conv::<T>(&c))
                .map_err(|e| format!("insert_col: {e}"))?;
            let mut b = at[..j * m].to_vec();
            b.extend(c.iter().map(|&v| Dd::from_f64(v)));
            b.extend_from_slice(&at[j * m..]);
            *at = b;
        }
        QrOp::DeleteCol => {
            let j = rng.below(n);
            f.delete_col(j).map_err(|e| format!("delete_col: {e}"))?;
            at.drain(j * m..(j + 1) * m);
        }
        QrOp::Rank1 => {
            let u = unit(rng, m);
            let v: Vec<f64> = unit(rng, n).iter().map(|x| 0.5 * x * scale).collect();
            f.rank1_update(&conv::<T>(&u), &conv::<T>(&v))
                .map_err(|e| format!("rank1_update: {e}"))?;
            for j in 0..n {
                for i in 0..m {
                    at[i + j * m] += Dd::from_f64(u[i]) * Dd::from_f64(v[j]);
                }
            }
        }
    }
    Ok(())
}

/// The Cholesky factor of `A_t` computed in double-double, rounded to `T`.
pub fn exact_chol<T: Scalar>(at: &[Dd], n: usize) -> Option<DenseBlock<T>> {
    let c = Cholesky::<Dd>::factor(MatRef::new(n, n, n, at).ok()?, &CholOptions::default()).ok()?;
    let b = c.upper().to_block();
    Some(DenseBlock {
        rows: n,
        cols: n,
        data: of_dd(&b.data),
    })
}

/// The `R` of the Householder QR of `A_t` computed in double-double,
/// rounded to `T`.
pub fn exact_qr_r<T: Scalar>(at: &[Dd], m: usize, n: usize) -> Option<DenseBlock<T>> {
    let f = Qr::<Dd>::factor(MatRef::new(m, n, m, at).ok()?, &QrOptions::default()).ok()?;
    let b = f.r().to_block();
    Some(DenseBlock {
        rows: n,
        cols: n,
        data: of_dd(&b.data),
    })
}

/// Run one [`QrCase`]: `A` is `m × n`, `κ(AᵀA) = kappa`.
pub fn qr_case<T: Scalar>(
    m: usize,
    n: usize,
    kappa: f64,
    op: QrOp,
    seed: u64,
) -> Result<QrCase, String> {
    let a = with_cond(m, n, kappa.sqrt(), seed);
    let mut f = Qr::factor(
        MatRef::new(m, n, m, &conv::<T>(&a)).unwrap(),
        &QrOptions::default(),
    )
    .map_err(|e| format!("factor: {e}"))?;
    let mut at = to_dd(&a);
    let mut rng = Rng::new(seed ^ 0x00dd_ba11);
    let before = fro(&at);
    apply_op(&mut f, &mut at, op, &mut rng)?;
    let (m, n) = (f.nrows(), f.ncols());
    let orth = orthogonality(f.q());
    let bwd = qr_backward(&f, &at);
    let after = fro(&at);
    let path = before.max(after);
    let bwd_path = bwd * after / path;
    let shrink = after / path;
    // (iii): R⋆ from the Householder QR of A_t in double-double, rounded
    // to T; R_f from the QR in T of A_t perturbed by ±u.
    let r1b = exact_qr_r::<T>(&at, m, n).ok_or("exact refactorization failed")?;
    let rf =
        refactor_qr::<T>(&at, m, n, seed.wrapping_add(29)).ok_or("refactorization in T failed")?;
    let fwd = rel(&f.r().to_block().data, &r1b.data);
    let dist = rel(&rf.r().to_block().data, &r1b.data);
    let kappa = kappa_of_upper(&upper_f64(&r1b));
    Ok(QrCase {
        kappa,
        orth,
        bwd,
        bwd_path,
        shrink,
        fwd,
        dist,
        m,
    })
}

/// Proptest case counts: the default of each block, or the value of
/// `STATLA_PROPTEST_CASES` (for stress runs).
pub fn cases(default: u32) -> u32 {
    std::env::var("STATLA_PROPTEST_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
