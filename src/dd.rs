//! # Chapter 12.0.8 — Double-double arithmetic for the polish of KS3
//!
//! ## Purpose
//! Algorithm SP of Chapter 12.6.1 (the polish of the symmetric small
//! solver of `eigsh`) runs Algorithm SJ once more, on a nearly diagonal
//! matrix, in about twice the precision of `f64`, and rounds its result
//! once. This module is that arithmetic and nothing else: a private
//! [`Dd`], the unevaluated sum `hi + lo` of two `f64`, with the field
//! operations, the square root and the order — the operations Algorithm
//! SJ and the Gram–Schmidt of 12.0.4 use, through `xingo_sparse::Real`, so
//! that the polish *is* the same generic code in another scalar. It is a
//! private copy of a subset of Chapters 11.1 and 11.2 of `xingo-extprec`
//! (the same algorithms, operation for operation, hence the same bits;
//! checked by the tests below against `DoubleDouble`, which is a
//! dev-dependency only): this crate keeps `xingo-sparse` as its only
//! runtime dependency (D-0127, S4; decision Q2(b) of task I09).
//!
//! ## Notation
//! `RN` rounding to nearest even, `u = 2⁻⁵³`, `η = 2⁻¹⁰⁷⁴`. A pair `(h, ℓ)`
//! is *normalized* when `h = RN(h + ℓ)`, so `|ℓ| ≤ ulp(h)/2`.
//!
//! ## Algorithms
//! The error-free transformations (exact in the range stated):
//! * **2Sum** (Knuth, after Møller; 6 flops, no condition): `s ← RN(a +
//!   b)`, `a′ ← RN(s − b)`, `b′ ← RN(s − a′)`, `e ← RN(RN(a − a′) + RN(b −
//!   b′))`; then `s + e = a + b` exactly unless `s` overflows.
//! * **Fast2Sum** (Dekker; 3 flops) when the exponent of `a` is at least
//!   that of `b` (in particular `|a| ≥ |b|`): `s ← RN(a + b)`, `e ← RN(b −
//!   RN(s − a))`, `s + e = a + b` exactly.
//! * **2Prod** (one fused multiply-add): `p ← RN(a b)`, `e ← RN(a b − p)`;
//!   `p + e = a b` exactly when `a b = 0` or `|a b| ≥ 2⁻⁹⁶⁹` and `p` does
//!   not overflow, otherwise `|a b − (p + e)| ≤ η/2`.
//!
//! The operations, as Algorithms A2, M2, D2 and R of Chapter 11.2 of
//! `xingo-extprec` (Joldes, Muller & Popescu 2017, "JMP"; Muller & Rideau
//! 2022; Lefèvre, Louvet, Muller, Picot & Rideau 2023), with their proven
//! relative error bounds when no underflow or overflow occurs:
//!
//! | operation | algorithm | bound | flops |
//! |---|---|---|---:|
//! | `x + y` | AccurateDWPlusDW (JMP Alg. 6) | `3u²/(1 − 4u)` | 20 |
//! | `x · y` | DWTimesDW3 (JMP Alg. 12) | `4u²` (Muller & Rideau) | 9 |
//! | `x / y` | DWDivDW2 (JMP Alg. 17) | `15u² + 56u³` | 18 |
//! | `√x` | SQRTDWtoDW (LLMPR Alg. 8) | `25/8 u²` | 8 |
//!
//! Negation and `|x|` are exact; `RN(x)` to `f64` is `h` (normalization).
//! `x − y = x + (−y)`; `mul_add` is a product then a sum (two roundings);
//! sums of iterators add left to right from zero.
//!
//! **What is left out, and why it does not matter here.** Step E of
//! Chapter 11.2 (the IEEE results for infinite or NaN operands, and the
//! overflow of an intermediate quantity within a factor four of the
//! largest `f64`) is not needed: Algorithm SP feeds only finite values
//! scaled to `O(1)` by step SJ1 (entries of magnitude at most 2, the
//! entries of an orthogonal matrix at most 1), whose results stay far from
//! overflow; a debug assertion checks that every result is finite. A
//! product below `2⁻⁹⁶⁹` (tiny entries of a nearly diagonal matrix) is not
//! exact: its error is at most `η` absolute per operation, which is below
//! `u²·ε_dd` of the `O(1)` quantities the polish sums.
//!
//! **The fused multiply-add.** 2Prod and the product use `f64::mul_add`,
//! the correctly rounded `fma` of IEEE 754, as `xingo-extprec` does (its
//! condition C2): Rust provides it on every target, in hardware or by a
//! correct software routine where the CPU has none (slow, never wrong), so
//! the bits of every operation are the same on every platform (the mirror
//! checks `i686` and `wasm32-wasip1`); Rust never contracts a product and
//! a sum on its own (Chapter 12.1).
//!
//! ## Invariants
//! (DD1) Every finite result is normalized (`h = RN(h + ℓ)`) —
//!       debug-asserted after each operation.
//! (DD2) Each operation is a fixed sequence of `f64` operations, the same
//!       as `xingo_extprec::DoubleDouble`'s in the safe range: the tests
//!       compare them bit for bit on random and adversarial operands.
//!
//! ## References
//! As cited in Chapters 11.1 and 11.2 of `xingo-extprec` (not re-checked
//! in this task): Møller, "Quasi double-precision in floating point
//! addition", BIT 5 (1965); Knuth, TAOCP vol. 2, §4.2.2, Theorem B;
//! Dekker, "A floating-point technique for extending the available
//! precision", Numer. Math. 18 (1971); Joldes, Muller & Popescu, "Tight and
//! rigorous error bounds for basic building blocks of double-word
//! arithmetic", ACM TOMS 44(2) (2017); Muller & Rideau, "Formalization of
//! double-word arithmetic, and comments on 'Tight and rigorous error
//! bounds for basic building blocks of double-word arithmetic'", ACM TOMS
//! 48(1) (2022); Lefèvre, Louvet, Muller, Picot & Rideau, "Accurate
//! calculation of Euclidean norms using double-word arithmetic", ACM TOMS
//! 49(1) (2023). The bounds of the table are those proven there and tested
//! in exact rationals by `xingo-extprec`; they were not re-derived here.

use core::iter::Sum;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use xingo_sparse::Real;

use crate::real::{is_exact_zero, lt};

/// A double-double number `hi + lo`, normalized (DD1).
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub(crate) struct Dd {
    hi: f64,
    lo: f64,
}

/// 2Sum: `(s, e)` with `s = RN(a + b)` and `s + e = a + b`.
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let a1 = s - b;
    let b1 = s - a1;
    (s, (a - a1) + (b - b1))
}

/// Fast2Sum: as [`two_sum`] when the exponent of `a` is at least that of
/// `b` (the callers below guarantee it, as in JMP's proofs).
#[inline]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let z = s - a;
    (s, b - z)
}

/// 2Prod: `(p, e)` with `p = RN(a b)` and `p + e = a b` (one fused
/// multiply-add, correctly rounded on every target).
#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

impl Dd {
    /// The pair, debug-checked: finite and normalized (DD1).
    #[inline]
    fn new(hi: f64, lo: f64) -> Self {
        debug_assert!(
            hi.is_finite() && lo.is_finite() && fast_two_sum(hi, lo).0.to_bits() == hi.to_bits(),
            "(DD1) a finite, normalized result: ({hi:e}, {lo:e})"
        );
        Self { hi, lo }
    }

    /// The leading component (tests).
    #[cfg(test)]
    pub(crate) fn hi(self) -> f64 {
        self.hi
    }

    /// The trailing component (tests).
    #[cfg(test)]
    pub(crate) fn lo(self) -> f64 {
        self.lo
    }
}

impl Add for Dd {
    type Output = Self;
    #[inline]
    fn add(self, y: Self) -> Self {
        // AccurateDWPlusDW.
        let (sh, sl) = two_sum(self.hi, y.hi);
        let (th, tl) = two_sum(self.lo, y.lo);
        let c = sl + th;
        let (vh, vl) = fast_two_sum(sh, c);
        let w = tl + vl;
        let (zh, zl) = fast_two_sum(vh, w);
        Self::new(zh, zl)
    }
}

impl Neg for Dd {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl Sub for Dd {
    type Output = Self;
    #[inline]
    fn sub(self, y: Self) -> Self {
        self + (-y)
    }
}

impl Mul for Dd {
    type Output = Self;
    #[inline]
    fn mul(self, y: Self) -> Self {
        // DWTimesDW3.
        let (ch, cl1) = two_prod(self.hi, y.hi);
        let tl0 = self.lo * y.lo;
        let tl1 = self.hi.mul_add(y.lo, tl0);
        let cl2 = self.lo.mul_add(y.hi, tl1);
        let (zh, zl) = fast_two_sum(ch, cl1 + cl2);
        Self::new(zh, zl)
    }
}

impl Div for Dd {
    type Output = Self;
    #[inline]
    fn div(self, y: Self) -> Self {
        // DWDivDW2, with DWTimesFP1 for `y · t_h`.
        let th = self.hi / y.hi;
        let (ch, cl1) = two_prod(y.hi, th);
        let cl2 = y.lo * th;
        let (uh, ul1) = fast_two_sum(ch, cl2);
        let (rh, rl) = fast_two_sum(uh, ul1 + cl1);
        let pih = self.hi - rh;
        let dl = self.lo - rl;
        let tl = (pih + dl) / y.hi;
        let (zh, zl) = fast_two_sum(th, tl);
        Self::new(zh, zl)
    }
}

impl AddAssign for Dd {
    #[inline]
    fn add_assign(&mut self, y: Self) {
        *self = *self + y;
    }
}

impl SubAssign for Dd {
    #[inline]
    fn sub_assign(&mut self, y: Self) {
        *self = *self - y;
    }
}

impl MulAssign for Dd {
    #[inline]
    fn mul_assign(&mut self, y: Self) {
        *self = *self * y;
    }
}

impl Sum for Dd {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl Real for Dd {
    const ZERO: Self = Self { hi: 0.0, lo: 0.0 };
    const ONE: Self = Self { hi: 1.0, lo: 0.0 };
    /// `2⁻¹⁰⁴`, as `DoubleDouble::EPSILON` (the QD convention).
    const EPSILON: Self = Self {
        hi: 4.930_380_657_631_324e-32,
        lo: 0.0,
    };
    #[inline]
    fn from_f64(x: f64) -> Self {
        Self { hi: x, lo: 0.0 }
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self.hi
    }
    #[inline]
    fn abs(self) -> Self {
        if self.hi.is_sign_negative() {
            -self
        } else {
            self
        }
    }
    #[inline]
    fn sqrt(self) -> Self {
        debug_assert!(!lt(self.hi, 0.0), "√ of a negative double-double");
        if is_exact_zero(self.hi) {
            return self;
        }
        // SQRTDWtoDW.
        let sh = self.hi.sqrt();
        let rho1 = (-sh).mul_add(sh, self.hi);
        let sl = (self.lo + rho1) / (2.0 * sh);
        let (zh, zl) = fast_two_sum(sh, sl);
        Self::new(zh, zl)
    }
    #[inline]
    fn recip(self) -> Self {
        Self::ONE / self
    }
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }
    #[inline]
    fn is_finite(self) -> bool {
        self.hi.is_finite()
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        if self < other { other } else { self }
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        if other < self { other } else { self }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // the tests compare bits
mod tests {
    use super::*;
    use crate::random::Rng;
    use xingo_extprec::DoubleDouble as Ext;

    fn ext(x: Dd) -> Ext {
        Ext::from_parts(x.hi, x.lo)
    }

    fn same(x: Dd, y: Ext) -> bool {
        x.hi.to_bits() == y.hi().to_bits() && x.lo.to_bits() == y.lo().to_bits()
    }

    /// A normalized pair of magnitude about `2^e`, with a random tail.
    fn draw(r: &mut Rng, e: i32) -> Dd {
        let h = r.gaussian() * 2f64.powi(e);
        let l = r.gaussian() * h.abs() * f64::EPSILON / 4.0;
        let (s, t) = two_sum(h, l);
        Dd { hi: s, lo: t }
    }

    /// (DD2): every operation gives the bits of `xingo_extprec::DoubleDouble`,
    /// on operands of the magnitudes Algorithm SP meets (`2⁻⁶⁰ … 2⁴`) and on
    /// cancelling ones.
    #[test]
    fn operations_match_the_extprec_double_double_bit_for_bit() {
        let mut r = Rng::new(20_261_127);
        let mut n = 0;
        for _ in 0..20_000 {
            let ea = (r.next_u64() % 64) as i32 - 60;
            let eb = (r.next_u64() % 64) as i32 - 60;
            let (x, mut y) = (draw(&mut r, ea), draw(&mut r, eb));
            if r.next_u64().is_multiple_of(8) {
                // Cancellation: y ≈ −x.
                y = -x + draw(&mut r, ea - 40);
            }
            let (ex, ey) = (ext(x), ext(y));
            assert!(same(x + y, ex + ey), "{x:?} + {y:?}");
            assert!(same(x - y, ex - ey), "{x:?} - {y:?}");
            assert!(same(x * y, ex * ey), "{x:?} * {y:?}");
            if !is_exact_zero(y.hi) {
                assert!(same(x / y, ex / ey), "{x:?} / {y:?}");
            }
            let ax = x.abs();
            assert!(same(ax.sqrt(), ext(ax).sqrt()), "sqrt {ax:?}");
            assert!(same(x.abs(), ex.abs()));
            assert_eq!(x < y, ex < ey);
            assert!(same(Real::max(x, y), Real::max(ex, ey)));
            n += 1;
        }
        assert_eq!(n, 20_000);
        // Exact cases: 1 + u (a tie), squares and quotients of f64.
        let u = Dd::from_f64(f64::EPSILON / 2.0);
        assert_eq!((Dd::ONE + u).hi(), 1.0);
        assert_eq!((Dd::ONE + u).lo(), f64::EPSILON / 2.0);
        let three = Dd::from_f64(3.0);
        let q = Dd::ONE / three;
        assert!(same(q, Ext::ONE / Ext::from_f64(3.0)));
        assert!(same(
            q * three,
            Ext::ONE / Ext::from_f64(3.0) * Ext::from_f64(3.0)
        ));
        assert!(same(Dd::from_f64(4.0).sqrt(), Ext::from_f64(2.0)));
        assert!(same(Dd::ZERO.sqrt(), Ext::ZERO));
        // 1 + u + u = 1 + 2⁻⁵², an f64: the tail cancels exactly.
        let s: Dd = [Dd::ONE, u, u].into_iter().sum();
        assert_eq!((s.hi(), s.lo()), (1.0 + f64::EPSILON, 0.0));
    }

    /// The relative error bounds of the table, against the extprec type
    /// (whose own tests check them in exact rationals): the local results are
    /// those bits, so this is a smoke check of the constants quoted.
    #[test]
    fn relative_errors_are_within_the_quoted_bounds() {
        let u2 = (f64::EPSILON / 2.0) * (f64::EPSILON / 2.0);
        let mut r = Rng::new(7);
        for _ in 0..2000 {
            let (x, y) = (draw(&mut r, 0), draw(&mut r, 1));
            // (x·y)/y and √(x²) against x, through two operations each.
            let back = (x * y) / y;
            let e = ((back - x).abs() / x.abs()).to_f64();
            assert!(e <= (4.0 + 15.0 + 1.0) * u2, "{e:e}");
            let root = (x * x).sqrt();
            let e = ((root - x.abs()).abs() / x.abs()).to_f64();
            assert!(e <= (4.0 / 2.0 + 25.0 / 8.0 + 1.0) * u2, "{e:e}");
        }
    }
}
