//! # Chapter 12.1 — Reproducible randomness: Algorithm G and the logarithm
//!
//! ## Purpose
//! [`crate::rsvd()`] draws a test matrix `Ω`, [`crate::svds()`] a starting
//! vector and the replacement vectors after a breakdown. Decision S4 of
//! D-0127: the same seed gives **the same bits on every platform**, and the
//! sequence for a seed is part of the semver contract. Hence no `rand`
//! (its algorithms may change between versions) and no `libm` `ln` (its
//! last bit differs between platforms): an own xoshiro256++ seeded by
//! SplitMix64, Marsaglia's polar method, and an own logarithm written with
//! `+ − · /` only, all of which IEEE 754 rounds identically everywhere
//! (Rust never contracts `a·b + c` into a fused multiply-add on its own).
//!
//! ## Notation
//! `u = ε_T/2` the unit roundoff of the scalar; `fl(·)` a rounded
//! operation, `fl(a ∘ b) = (a ∘ b)(1 + δ)`, `|δ| ≤ u` (the IEEE model for
//! `f64`). `ln 2 = A + B + C` with three `f64` constants (those of
//! `xingo-extprec`, Chapter 11.2): `A` has 42 significant bits (so `e·A` is
//! exact for `|e| < 2¹¹`), `|B| < 2⁻⁴²`, `|C| < 2⁻⁹⁷`, `|ln 2 − (A + B + C)| <
//! 2⁻¹⁵⁰`.
//!
//! ## Algorithm G (Gaussian deviates, bit-reproducible)
//! G1. \[Seed.\] The state `s₀ … s₃` of xoshiro256++ (Blackman & Vigna) is
//!     four consecutive outputs of SplitMix64 started at the user's `seed`
//!     (`x ← x + 0x9e3779b97f4a7c15`; `z ← (z ⊕ z≫30)·0xbf58476d1ce4e5b9`;
//!     `z ← (z ⊕ z≫27)·0x94d049bb133111eb`; output `z ⊕ z≫31`). SplitMix64
//!     is a bijection of its counter, so at most one of the four words is
//!     zero: every seed, `0` included, gives a valid (nonzero) state.
//! G2. \[Uniform.\] `U = (next() ≫ 11)·2⁻⁵³ ∈ [0, 1)`, with `next()` the
//!     xoshiro256++ step: output `rotl(s₀ + s₃, 23) + s₀`; then `t ← s₁ ≪
//!     17`, `s₂ ⊕= s₀`, `s₃ ⊕= s₁`, `s₁ ⊕= s₂`, `s₀ ⊕= s₃`, `s₂ ⊕= t`, `s₃ ←
//!     rotl(s₃, 45)`.
//! G3. \[Polar.\] `u ← 2U − 1`, `v ← 2U′ − 1` (two uniforms, in that order;
//!     both exact); `s ← u² + v²`; start again unless `0 < s < 1`
//!     (Marsaglia & Bray). The smallest nonzero `s` is `2⁻¹⁰⁴`: normal.
//! G4. \[Transform.\] `g ← u · √(fl(−2·ln(s)) / s)` with this chapter's `ln`
//!     (Algorithm L) in `f64`. Only `u`'s deviate is used; `v`'s is
//!     discarded (one rule, no hidden state between calls).
//! G5. \[Fill.\] A test matrix is filled column by column, entry by entry,
//!     from one stream: `Ω_ij` is the `(i + j·n)`-th deviate. The deviates
//!     are `f64` and converted exactly (`Real::from_f64`), so the sketch is
//!     the same in every scalar.
//!
//! [`Sketch::Rademacher`] draws `±1` from the
//! top bit of one `next()` per entry (bit set: `−1`).
//!
//! ## Algorithm L (natural logarithm of a positive normal `x`, `T: Real`)
//! L1. \[Exponent.\] `h ← x.to_f64()` (for the double-double, its high
//!     part); if `h` is subnormal, `x ← x·2⁵⁴` first and remember `−54`.
//!     `e ←` the unbiased exponent of `h`, read off its bits.
//! L2. \[Reduce.\] `m ← x·2⁻ᵉ` (by at most two exact power-of-two products),
//!     so `m ∈ [1, 2)` (for the double-double `m ≥ 1 − 2⁻⁵³`). If `m > √2`
//!     (the `f64` nearest to it): `m ← m/2`, `e ← e + 1`. Now `m ∈ [√½,
//!     √2]` up to the rounding of the constant.
//! L3. \[Atanh argument.\] `z ← (m − 1)/(m + 1)`, `w ← z²`; `|z| ≤ (√2 −
//!     1)/(√2 + 1) = 0.1716`, `w ≤ w̄ = 0.02944`.
//! L4. \[Series.\] `P ← c_N`; for `i = N − 1, …, 0`: `P ← P·w + c_i`, with `c_i
//!     = fl(1/(2i + 1))` (Horner on `ln m = 2 atanh z = 2z Σ w^i/(2i + 1)`).
//!     `N` is the least integer with `w̄^{N+1}/((2N + 3)(1 − w̄)) ≤ ε_T/8`,
//!     a function of `ε_T` only: `N = 9` (ten terms) for `f64`, `N = 19`
//!     for the double-double. `L ← 2·z·P`.
//! L5. \[Combine.\] If `e = 0`, return `L`. Otherwise return
//!     `e·A + (L + (e·B + e·C))`, evaluated in that order.
//!
//! **Lemma L (accuracy in `f64`).** For every positive normal `x`, the
//! result of Algorithm L in `f64` has relative error at most `6.4u` (to
//! first order in `u`), and `ln 1 = 0` exactly.
//!
//! *Proof.* L1–L2 are exact (power-of-two scalings of a normal number).
//! L3: `m − 1` is exact (Sterbenz: `m ∈ [½, 2]`), so `z̃ = z(1 + θ)`, `|θ| ≤
//! 2u` (the rounding of `m + 1` and of the quotient), and `w̃ = w(1 + θ′)`,
//! `|θ′| ≤ 5u`. L4: every term is positive and `w P₁ ≤ w̄/(3(1 − w̄)) <
//! 0.0102` where `P = 1 + w P₁`; the last step `fl(1 + fl(w̃ P̃₁))` errs by
//! at most `u·P` plus `0.0102·(5u + u + 2u)` from `w̃`, the product and
//! `P̃₁` (whose own relative error is `≤ 2u` by the same argument one level
//! down), plus the truncation `≤ u/4` and the rounded `c_i` (inside the
//! `8u`): `P̃ = P(1 + θ_P)`, `|θ_P| ≤ 1.35u`. So `L̃ = L(1 + θ_L)`, `|θ_L| ≤
//! 2u + 1.35u + u = 4.35u` (the factor 2 is exact). L5, `e ≠ 0`: `e·A` is
//! exact; `t = fl(e·B + e·C)` and the missing tail of `ln 2` weigh `≤ |e|·
//! 2⁻⁹⁴`, negligible; the two remaining additions err by `u|L + t|` and
//! `u|r|`. Since `|L| ≤ ln √2` and `|ln x| ≥ ln 2 − ln √2 = ln √2` when `e ≠
//! 0`, `|L| ≤ |ln x|`, hence `|r − ln x| ≤ (4.35 + 1 + 1)u|ln x| + O(u²)`.
//! For `x = 1`: `z = 0`, `L = 0`, `e = 0`. ∎
//!
//! For the double-double the same derivation holds with the operation
//! bounds of Chapter 11 in place of `(1 + δ)` (each a small multiple of
//! `u = 2⁻¹⁰⁵`); the bound is *measured* there, not proved (tests below and
//! `l33-tables.md`). Measured in `f64` on 1.8·10⁵ arguments (all binades,
//! the polar range `(0, 1)`, and `1 ± 2⁻ᵏ`): at most `3.5u` against
//! `DoubleDouble::ln` (proved to `8u²`, an exact oracle at this level); in
//! the double-double, on 4.5·10⁴ full-precision arguments, at most `2.7u`
//! against the same oracle (whose own error, `≤ 8·2⁻¹⁰⁶`, is of the same
//! order: the measurement bounds the sum). The tests assert `6.4u` and
//! `16u`.
//!
//! ## Invariants
//! (R1) *Frozen sequence.* For seed 0 the first 64 Gaussian deviates, the
//!      first raw words and the first Rademacher signs are pinned bit for
//!      bit by a test; changing them is a breaking change.
//! (R2) *Platform independence.* Only integer operations, IEEE `+ − · /`
//!      and `sqrt` are used (Algorithm L avoids `libm`).
//! (R3) `ln` is exact at `1` and monotone enough for the polar method (not
//!      needed for correctness: any fixed function gives a fixed stream).
//!
//! ## References
//! Blackman & Vigna, "Scrambled linear pseudorandom number generators",
//! ACM TOMS 47 (2021), and the reference `xoshiro256plusplus.c` (whose
//! first ten outputs from the state `(1, 2, 3, 4)`, as reproduced in the
//! tests of `rand` 0.9.4, are pinned below); Steele, Lea & Flood,
//! "Fast splittable pseudorandom number generators", OOPSLA 2014
//! (SplitMix64); Marsaglia & Bray, "A convenient method for generating
//! normal variables", SIAM Review 6 (1964); Muller, *Elementary Functions*
//! (2016), ch. 6 (argument reduction; the atanh series); Higham, *Accuracy
//! and Stability of Numerical Algorithms* (2002), ch. 2–3. Cited from
//! memory, except the reference outputs.

use xingo_sparse::Real;

use crate::real::{le_exact, lt};

/// Distribution of the test matrix `Ω` of [`crate::rsvd()`] and
/// [`crate::range_finder`] (Algorithm G).
///
/// The sequence for a given seed is part of the semver contract (a
/// frozen-sequence test pins the first 64 deviates): changing it is a
/// breaking change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Sketch {
    /// Standard Gaussian entries (Algorithm G) — the distribution of the
    /// HMT bounds. Default.
    #[default]
    Gaussian,
    /// `±1` with probability ½ each (one bit of `next()` per entry). No
    /// transcendental function; HMT's expectation bound is *not* proven
    /// for it, though the pilot found it inside the bound on three
    /// spectra.
    Rademacher,
}

/// SplitMix64 increment (the golden ratio, `⌊2⁶⁴/φ⌋` odd).
const SPLITMIX_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
/// The two SplitMix64 multipliers.
const SPLITMIX_M1: u64 = 0xbf58_476d_1ce4_e5b9;
const SPLITMIX_M2: u64 = 0x94d0_49bb_1331_11eb;

/// `ln 2 = A + B + C` (the split of `xingo-extprec`, Chapter 11.2; `A` has
/// 42 significant bits).
const LN2_A: f64 = f64::from_bits(0x3fe6_2e42_fefa_3800);
const LN2_B: f64 = f64::from_bits(0x3d2e_f357_93c7_6730);
const LN2_C: f64 = f64::from_bits(0x398f_97b5_7a07_9a19);
/// `w̄ = ((√2 − 1)/(√2 + 1))² = 17 − 12√2` rounded up (checked at 200
/// bits): the bound on `w` in step L3.
const W_BAR: f64 = 0.029_437_251_522_859_417;
/// The exponent field of an `f64`.
const EXPONENT_MASK: u64 = 0x7ff0_0000_0000_0000;
/// `2⁵⁴`, the pre-scaling of a subnormal high part (step L1).
const TWO_54: f64 = 18_014_398_509_481_984.0;

/// The generator of Algorithm G: xoshiro256++ seeded by SplitMix64.
#[derive(Clone, Debug)]
pub(crate) struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// G1.
    pub(crate) fn new(seed: u64) -> Self {
        let mut x = seed;
        let mut s = [0; 4];
        for w in &mut s {
            x = x.wrapping_add(SPLITMIX_GAMMA);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(SPLITMIX_M1);
            z = (z ^ (z >> 27)).wrapping_mul(SPLITMIX_M2);
            *w = z ^ (z >> 31);
        }
        debug_assert_ne!(s, [0; 4], "(G1) SplitMix64 never gives four zeros");
        Self { s }
    }

    /// The xoshiro256++ step (G2).
    pub(crate) fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let out = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        out
    }

    /// G2: `U ∈ [0, 1)`, a multiple of `2⁻⁵³`.
    #[allow(clippy::cast_precision_loss)] // 53-bit integers are exact in f64
    pub(crate) fn uniform(&mut self) -> f64 {
        const TWO_M53: f64 = 1.0 / 9_007_199_254_740_992.0;
        (self.next_u64() >> 11) as f64 * TWO_M53
    }

    /// G3–G4: one standard Gaussian deviate.
    pub(crate) fn gaussian(&mut self) -> f64 {
        loop {
            // G3.
            let u = 2.0 * self.uniform() - 1.0;
            let v = 2.0 * self.uniform() - 1.0;
            let s = u * u + v * v;
            if lt(0.0, s) && lt(s, 1.0) {
                // G4.
                return u * ((-2.0 * ln(s)) / s).sqrt();
            }
        }
    }

    /// `±1` from the top bit of one word.
    pub(crate) fn rademacher(&mut self) -> f64 {
        if self.next_u64() >> 63 == 1 {
            -1.0
        } else {
            1.0
        }
    }

    /// One deviate of the distribution `sketch`.
    pub(crate) fn deviate(&mut self, sketch: Sketch) -> f64 {
        match sketch {
            Sketch::Rademacher => self.rademacher(),
            Sketch::Gaussian => self.gaussian(),
        }
    }

    /// G5: fill `data` (column-major) entry by entry from the stream.
    pub(crate) fn fill<T: Real>(&mut self, sketch: Sketch, data: &mut [T]) {
        for v in data {
            *v = T::from_f64(self.deviate(sketch));
        }
    }
}

/// `2ᵏ` for `−1022 ≤ k ≤ 1023` (a normal `f64`, built from its bits).
#[allow(clippy::cast_sign_loss)] // k + 1023 ∈ [1, 2046]
fn pow2(k: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&k));
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// The number `N` of step L4 for the scalar `T`.
fn series_degree<T: Real>() -> usize {
    let target = T::EPSILON.to_f64() / 8.0;
    let mut wp = W_BAR; // w̄^{N+1}
    let mut n = 0usize;
    loop {
        #[allow(clippy::cast_precision_loss)]
        let den = (2 * n + 3) as f64 * (1.0 - W_BAR);
        if le_exact(wp / den, target) {
            return n;
        }
        wp *= W_BAR;
        n += 1;
    }
}

/// Algorithm L: `ln x` for a positive normal finite `x` (the callers
/// guarantee it; anything else returns an unspecified value, never a
/// panic).
pub(crate) fn ln<T: Real>(x: T) -> T {
    debug_assert!(x.is_finite() && lt(T::ZERO, x), "Algorithm L needs x > 0");
    // L1.
    let mut x = x;
    let mut e_adj = 0i32;
    let mut bits = x.to_f64().to_bits();
    if bits & EXPONENT_MASK == 0 {
        x *= T::from_f64(TWO_54);
        e_adj = -54;
        bits = x.to_f64().to_bits();
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let mut e = ((bits & EXPONENT_MASK) >> 52) as i32 - 1023;
    // L2. Two exact scalings keep every factor normal.
    let half = -e / 2;
    let mut m = x * T::from_f64(pow2(half)) * T::from_f64(pow2(-e - half));
    if lt(T::from_f64(core::f64::consts::SQRT_2), m) {
        m *= T::from_f64(0.5);
        e += 1;
    }
    e += e_adj;
    // L3.
    let z = (m - T::ONE) / (m + T::ONE);
    let w = z * z;
    // L4.
    let n = series_degree::<T>();
    let coef = |i: usize| {
        #[allow(clippy::cast_precision_loss)]
        let d = (2 * i + 1) as f64;
        T::ONE / T::from_f64(d)
    };
    let mut p = coef(n);
    for i in (0..n).rev() {
        p = p * w + coef(i);
    }
    let l = T::from_f64(2.0) * z * p;
    // L5.
    if e == 0 {
        return l;
    }
    let ef = T::from_f64(f64::from(e));
    let t = ef * T::from_f64(LN2_B) + ef * T::from_f64(LN2_C);
    ef * T::from_f64(LN2_A) + (l + t)
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::unreadable_literal // frozen bit patterns
)]
mod tests {
    use super::*;
    use xingo_extprec::DoubleDouble as Dd;

    #[test]
    fn xoshiro_matches_the_reference_outputs() {
        // State (1, 2, 3, 4); outputs of the reference C implementation as
        // reproduced in the tests of `rand` 0.9.4.
        let mut r = Rng { s: [1, 2, 3, 4] };
        let expected: [u64; 10] = [
            41_943_041,
            58_720_359,
            3_588_806_011_781_223,
            3_591_011_842_654_386,
            9_228_616_714_210_784_205,
            9_973_669_472_204_895_162,
            14_011_001_112_246_962_877,
            12_406_186_145_184_390_807,
            15_849_039_046_786_891_736,
            10_450_023_813_501_588_000,
        ];
        for e in expected {
            assert_eq!(r.next_u64(), e);
        }
    }

    /// (R1): the frozen sequence of seed 0 (and the seeding of seed 1).
    #[test]
    fn sequence_of_seed_zero_is_frozen() {
        let mut r = Rng::new(0);
        assert_eq!(r.s, FROZEN_STATE_0);
        let mut r1 = Rng::new(1);
        let raw: Vec<u64> = (0..4).map(|_| r1.next_u64()).collect();
        assert_eq!(raw, FROZEN_RAW_1);
        let g: Vec<u64> = (0..64).map(|_| r.gaussian().to_bits()).collect();
        assert_eq!(g, FROZEN_GAUSS_0);
        let mut rr = Rng::new(0);
        let signs: Vec<f64> = (0..16).map(|_| rr.rademacher()).collect();
        assert_eq!(signs, FROZEN_RADEMACHER_0);
    }

    const FROZEN_STATE_0: [u64; 4] = [
        16294208416658607535,
        7960286522194355700,
        487617019471545679,
        17909611376780542444,
    ];
    const FROZEN_RAW_1: [u64; 4] = [
        14971601782005023387,
        13781649495232077965,
        1847458086238483744,
        13765271635752736470,
    ];
    /// Bits of the first 64 Gaussian deviates of seed 0 (the first is
    /// `-1.5411826072230725`).
    const FROZEN_GAUSS_0: [u64; 64] = [
        0xbff8a8af17f2fe67,
        0xbf708d7cb6a3955d,
        0x3fbc958a80b0b8e6,
        0xbfca60c5aa96ee22,
        0xbfc8321083be657e,
        0xbfe53100de8d3b86,
        0x3fd233ab345f0647,
        0xbffada9d88a0b370,
        0xbfd9624e017442d8,
        0xbff12f999b7f0784,
        0xbfed1aaaa24ca640,
        0xbff6a4ba59fafe1d,
        0xbfd81899a654bd69,
        0x3fd5e96343423eec,
        0x3fa70d7e06e38361,
        0x3fe258602afb1018,
        0x3fe1c8af537e33dd,
        0xbfc534af346ec048,
        0xbfdde34fc2b742fa,
        0xbff88db52ac5488e,
        0x40046727b91817bf,
        0x3fd57ba2329c55b9,
        0x3ffc30ea6e625005,
        0xbff651eca0888b18,
        0x3fe399d771a6dede,
        0xbfc8ae207ae66069,
        0xbfe511bbf0d3868f,
        0x3fd1e2d26a847ade,
        0x3fe67e6ce7ea2ca3,
        0x3ff3fd13aa3ae25f,
        0x3ff9b1584aa24b3e,
        0x3fe83534195628a4,
        0x3fefa20547b88b13,
        0x3ffdb0421ff79616,
        0x3fdc9c2a580e3862,
        0x3fdaa737134ed9c0,
        0xbfe7774cdecdb232,
        0x3fbcb2cf25eb1719,
        0xbff3ae5d8e9d3e80,
        0xbf59230c30a61bfa,
        0x3ff1a35b827f0a06,
        0x3fe33bceec22e0af,
        0xbff62602652954ec,
        0x3fee65bff8eaacf1,
        0xbff1604ce70bb525,
        0xbff2a3b0ab926856,
        0xbfce454bf4ff4b01,
        0x3fec727727057ace,
        0x3ff08305cdbc9f80,
        0x3ff501fe9e282575,
        0x3fc786802bf04146,
        0xbff77ee7a0537b14,
        0x3fe1ae9378cd2d2a,
        0xbfeec1282ba1af9d,
        0x3ff34afa4b3870ba,
        0x3ff205d8dc9f3a5f,
        0xbff08d8b20cb5b2e,
        0xbfede4e70a407787,
        0xbfe7399116c7218a,
        0x3fe6b38b58b12210,
        0x3fe39f58ac2a7984,
        0x3fe10b5b38bf6349,
        0x3ffa0c12c6765728,
        0x3ff5b042b55ceec0,
    ];
    const FROZEN_RADEMACHER_0: [f64; 16] = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];

    #[test]
    fn deviates_have_the_moments_of_a_standard_gaussian() {
        let mut r = Rng::new(42);
        let n = 200_000;
        let (mut s1, mut s2, mut s4) = (0.0, 0.0, 0.0);
        for _ in 0..n {
            let g = r.gaussian();
            s1 += g;
            s2 += g * g;
            s4 += g * g * g * g;
        }
        let n = f64::from(n);
        assert!((s1 / n).abs() < 0.01, "mean {}", s1 / n);
        assert!((s2 / n - 1.0).abs() < 0.015, "variance {}", s2 / n);
        assert!((s4 / n - 3.0).abs() < 0.08, "fourth moment {}", s4 / n);
        let mut rr = Rng::new(42);
        let plus = (0..10_000).filter(|_| rr.rademacher() > 0.0).count();
        assert!((4800..5200).contains(&plus));
        let u: f64 = (0..10_000).map(|_| rr.uniform()).sum::<f64>() / 10_000.0;
        assert!((u - 0.5).abs() < 0.01);
    }

    /// The arguments of the `ln` tests: every binade, the polar range, and
    /// `1 ± 2⁻ᵏ`.
    fn ln_arguments() -> Vec<f64> {
        let mut r = Rng::new(7);
        let mut xs = vec![1.0, 2.0, 0.5, core::f64::consts::SQRT_2, f64::MIN_POSITIVE];
        xs.push(f64::MAX);
        for k in 1..53 {
            let d = pow2(-k);
            xs.extend([1.0 + d, 1.0 - d / 2.0]);
        }
        for e in -1022..1024 {
            for _ in 0..40 {
                xs.push((1.0 + r.uniform()) * pow2(e.min(1023)));
            }
        }
        for _ in 0..100_000 {
            let s = r.uniform();
            if s > 0.0 {
                xs.push(s);
            }
        }
        xs
    }

    /// `|ln(x) − oracle| / |oracle|` in units of `u`.
    fn rel_err_u(own: Dd, oracle: Dd, u: f64) -> f64 {
        if oracle == Dd::ZERO {
            return if own == Dd::ZERO { 0.0 } else { f64::INFINITY };
        }
        ((own - oracle) / oracle).abs().to_f64() / u
    }

    /// Lemma L, measured: `f64` within `6.4u` of `DoubleDouble::ln`.
    #[test]
    fn ln_in_f64_is_within_the_proven_bound() {
        let u = f64::EPSILON / 2.0;
        let mut worst = 0.0f64;
        for x in ln_arguments() {
            let e = rel_err_u(Dd::from_f64(ln(x)), Dd::from_f64(x).ln(), u);
            assert!(e <= 6.4, "ln({x:e}): {e} u");
            worst = worst.max(e);
        }
        assert_eq!(ln(1.0), 0.0);
        assert_eq!(series_degree::<f64>(), 9);
        eprintln!("ln f64: max relative error {worst:.3} u");
    }

    /// Algorithm L in the double-double, against `DoubleDouble::ln`.
    #[test]
    fn ln_in_double_double_is_within_a_few_units() {
        let u = Dd::EPSILON.to_f64() / 2.0;
        let mut worst = 0.0f64;
        let mut r = Rng::new(9);
        for x in ln_arguments().into_iter().step_by(4) {
            // A full-precision argument: x (1 + small) with a nonzero low part.
            let xd = Dd::from_f64(x) * (Dd::ONE + Dd::from_f64(r.uniform() * f64::EPSILON));
            if !(xd.to_f64() > 0.0 && xd.is_finite()) {
                continue;
            }
            let e = rel_err_u(ln(xd), xd.ln(), u);
            assert!(e <= 16.0, "ln({x:e}) in dd: {e} u");
            worst = worst.max(e);
        }
        assert_eq!(series_degree::<Dd>(), 19);
        eprintln!("ln dd: max relative error {worst:.3} u (oracle error included)");
    }

    #[test]
    fn ln2_split_and_subnormal_reduction() {
        let ln2 = Dd::from_f64(2.0).ln();
        let split = Dd::from_f64(LN2_A) + Dd::from_f64(LN2_B) + Dd::from_f64(LN2_C);
        assert!(((split - ln2) / ln2).abs().to_f64() < 1e-31);
        assert_eq!(LN2_A.to_bits() & 0x7ff, 0, "A has 42 significant bits");
        let w = ((2f64.sqrt() - 1.0) / (2f64.sqrt() + 1.0)).powi(2);
        // `W_BAR ≥ w̄` exactly was checked at 200 bits (mpmath, task L3.3);
        // here only that the constant is the right one.
        assert!((W_BAR - w).abs() < 1e-16);
        // Subnormal arguments go through the pre-scaling of L1.
        for x in [f64::MIN_POSITIVE / 3.0, 5e-324, 1e-310] {
            let e = rel_err_u(
                Dd::from_f64(ln(x)),
                Dd::from_f64(x).ln(),
                f64::EPSILON / 2.0,
            );
            assert!(e <= 6.4, "{x:e}: {e}");
        }
    }
}
