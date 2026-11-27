//! # Chapter 12.6 — `eigsh` and `eigs`: Krylov–Schur restarted Lanczos and Arnoldi with ARPACK's contract
//!
//! ## Purpose
//! *`k` eigenpairs at one end of the spectrum of a large sparse or implicit
//! operator.* [`eigsh()`] (symmetric) and [`eigs()`] (general real) take
//! ARPACK's vocabulary — the two-letter `which` codes, `ncv`, the relative
//! convergence test, scipy's defaults — and restart by the Krylov–Schur
//! method of Stewart (2001) instead of ARPACK's implicit QR shifts
//! (decision S7 of D-0127): with exact shifts both keep the same subspace,
//! and in Schur form locking and purging are deletions of rows and columns.
//!
//! ## Notation
//! `A ∈ R^{n×n}` the operator; `m = ncv`; `V = [v_1 … v_{m+1}] ∈ T^{n×(m+1)}`
//! orthonormal; `H̄ ∈ T^{(m+1)×m}` with rows `0..m` the projected `H` and row
//! `m` the *coupling row* `bᵀ`. The **Krylov decomposition** kept throughout
//! is
//!
//! ```text
//! A V_j = V_j H_j + v_{j+1} b_jᵀ          (KS)
//! ```
//!
//! with `b_j = β_j e_j` after an Arnoldi step and a full vector after a
//! restart. The first `n_L` columns are **locked**: their part of `b` is
//! zero and `H = [S_LL H_LA; 0 H_AA]` is block upper triangular, so the
//! eigenvalues of `H` are those of `S_LL` (quasi-triangular, fixed) and of
//! the *active* block `H_AA`. In the symmetric case `H_LA = 0` and `S_LL`
//! is diagonal. `τ` is the tolerance ([`EigsOptions::tol`], `0` meaning
//! `ε_T`), `ε^{2/3}` ARPACK's floor of the test, `‖A‖_est = max_j ‖A v_j‖`.
//! A Ritz pair `(θ, y = V z)` of `H` has residual `‖A y − θ y‖ = |bᵀ z|` by
//! (KS) (`‖z‖ = 1`).
//!
//! ## Algorithm KS (Krylov–Schur, one engine for both drivers)
//! KS1. \[Start.\] `v_1 ← v0/‖v0‖` (`v0` from the options or `n` Gaussian
//!      deviates of the seed's stream, Algorithm G); `j ← 0`; `n_L ← 0`.
//! KS2. \[Expand.\] For `j, …, m − 1`: `w ← A v_{j+1}` (checked finite);
//!      `‖A‖_est ← max(‖A‖_est, ‖w‖)`; Gram–Schmidt against **all** of
//!      `V_{j+1}` with a DGKS second pass, the coefficients of both passes
//!      forming column `j` of `H`; `β ← ‖w‖`; if `β ≤ ε_T‖A‖_est`, an
//!      invariant subspace: `v_{j+2}` is drawn from the seed's stream and
//!      orthogonalized, `β ← 0` (Algorithm E of Chapter 12.0.7). `h_{j+1,j} ←
//!      β`. In `eigsh` (KS2′) the strict upper part of the new column is
//!      then overwritten by the mirror of its row (`β_{j−1}`, or the
//!      arrowhead `b` after a restart), so that `H` is exactly symmetric and
//!      the tiny couplings of converged Ritz vectors stay tiny: the
//!      Gram–Schmidt coefficients there are those couplings plus rounding
//!      `O(ε‖A‖)`, which only orthogonalize `w` (as in Lanczos, where they
//!      are not stored).
//! KS3. \[Schur.\] Decompose the active block `H_AA = Z_A T_A Z_Aᵀ`: in
//!      `eigsh`, `T_A` diagonal by the Jacobi method of the symmetrized
//!      block (Algorithm SJ of 12.6.1); from restart `R₀ =`
//!      [`POLISH_FROM_RESTART`] `= 100` on, polished in double-double and
//!      rounded once (Algorithm SP of 12.6.1: SJ in `f64`, then `Z_A`
//!      reorthonormalized and SJ run again on `Z_AᵀH_AAZ_A` in
//!      double-double; for a scalar other than `f64`, SJ alone), so that a
//!      run of fewer than `R₀` restarts has the bits of SJ alone (the
//!      reasons, measured, under "The polish"); in `eigs`, the standardized real Schur
//!      form by Hessenberg reduction and Francis double-shift QR (HR +
//!      FQ). Then `b_A ← b_Aᵀ Z_A`, `H_LA ← H_LA Z_A`.
//! KS4. \[Ritz.\] The eigenvalues `θ` of the locked blocks and of `T_A`;
//!      for an active block the eigenvector `x` of `T_A` (Algorithm EV) gives
//!      the estimate `ρ = |b_Aᵀ x|/‖x‖` (≥ the true one: the locked
//!      components of the eigenvector of `H` are left out). The candidates
//!      are ordered by `which` (a conjugate pair counts as two values,
//!      adjacent, `+i` first; `BE` alternates the high and the low end,
//!      high first; ties are broken by a secondary key — `LM`↔`LR`,
//!      `SM`↔`SR`, `LI`→`LM`, `SI`→`SM` as ARPACK's `dngets`, and
//!      `LA`/`BE`→`LM`, `SA`→`SM`, which are this crate's own (`dsgets` has
//!      no secondary key) — then by position); the
//!      **wanted** are the first `k`.
//! KS5. \[Test.\] A wanted value is converged when it is locked, or when
//!      `ρ ≤ τ·max(ε^{2/3}, |θ|)` (ARPACK's test, `|θ|` the modulus). If all
//!      `k` are converged, go to KS8; if the restart limit is reached,
//!      [`StatlaError::NoConvergence`] (or KS8 with the converged ones when
//!      `allow_partial`).
//! KS6. \[Reorder and lock.\] `k′ ←` Algorithm K of Chapter 12.0.7 (`n_conv` the
//!      converged wanted). Reorder `T_A` (Algorithm RO, the rotations
//!      accumulated into `Z_A`, `b_A` recomputed) so that the converged
//!      wanted lead, then the other wanted, then the rest, each group in
//!      `which` order. **Lock** the leading active blocks, in order, while
//!      they are converged wanted and their entries of `b_A` satisfy the
//!      test of KS5 (for a symmetric `T_A` that is the estimate itself):
//!      their entries of `b` become `0` — a perturbation of `A` of norm `≤
//!      τ·max(ε^{2/3}, |θ|)` — and their vectors are never rotated again.
//!      *Why the couplings and not the estimate:* for a nonnormal `T_A` the
//!      Schur vector of a block after the first is not its eigenvector,
//!      and its coupling can exceed the estimate `ρ` of KS4 by the
//!      departure from normality. The check refuses such blocks a few
//!      times per run on Jordan-like and convection-dominated matrices;
//!      without it the residual ratio of criterion (b) reaches `7.4·10³`
//!      (`tests/nonnormal.rs`, issue xingo-statla#2).
//! KS7. \[Truncate (purge).\] Round `k′` so that no `2 × 2` block is split
//!      (up if there is room, else down), keep `k′ ≥ n_L`. `V ← [V_L, V_A
//!      Z_A[:, :k′−n_L], v_{m+1}]`, `H ← [S_LL, H_LA Z_A; 0, T_A]` restricted
//!      to `k′`, coupling row `(0, b_A[:k′−n_L])` with the locked entries
//!      zeroed: the unwanted Ritz values beyond `k′` (converged or not) are
//!      purged with their vectors. Count a restart; `j ← k′`; go to KS2.
//! KS7′. \[Re-orthonormalize.\] One Gram–Schmidt sweep (DGKS) of the new
//!      active columns and of the residual vector against the columns before
//!      them, locked ones untouched. `V ← V Z_A` in floating point loses
//!      `O(u)` of orthogonality per restart, and the loss accumulates
//!      (measured: `1.9·10⁻¹⁴ = 85u` after 30 restarts at `m = 20`, against
//!      `≤ 4u` with the sweep); the sweep perturbs (KS) by the loss it
//!      removes, `O(m u ‖A‖)`.
//! KS8. \[Vectors.\] For each wanted value the eigenvector `x` of the whole
//!      quasi-triangular `T = [S_LL, H_LA Z_A; 0, T_A]` (Algorithm EV), `y ←
//!      V blockdiag(I, Z_A) x`, normalized to unit 2-norm in `Cⁿ` with its
//!      largest entry real and positive (the conjugate for the `−i` member).
//!
//! **Algorithm L** of the contract (thick-restart Lanczos, Wu & Simon 2000)
//! *is* Algorithm KS with a symmetric `T_A`: after a restart `H` is
//! `diag(θ)` plus the arrowhead `b` (row `k′`) and its mirror (column `k′`,
//! the Gram–Schmidt coefficients of the first new step), and the reordering
//! is a permutation. Both drivers therefore share one engine (two Krylov
//! engines in the crate instead of three: this and the bidiagonalization
//! of 12.5).
//!
//! **Reorthogonalization policy:** full (KS2 against all of `V`, locked
//! vectors included), at most two classical Gram–Schmidt passes — the
//! policy of 12.5. Cost per step: one application plus `≈ 4nj` flops; per
//! restart `O(n m k′ + m³)`.
//!
//! ## Invariants
//! (K1) `‖V_jᵀV_j − I‖ ≤ c_K · j · u` — debug-asserted after every restart
//!      (`c_K` = [`C_K`]).
//! (K2) `‖A V_j − V_j H_j − v_{j+1} b_jᵀ‖ ≤ c_K · j · u · ‖A‖_est` after every
//!      expansion **on runs of few restarts** — computed by the unit tests
//!      (`paranoid` runs; it costs `j` extra applications). It is *not* a
//!      bound for long runs: every restart adds the backward error of KS3
//!      and RO, and the relation error grows with the number `R` of
//!      restarts, like `√R` on most measured runs and faster on others (a
//!      cluster of spacing `10⁻¹⁰` that never converges: about linearly,
//!      `33 648u` after 10 000 restarts at `j = 24`; task I02, before the
//!      polish of KS3). No form in `R` bounds every measured run (issue
//!      xingo-statla#9, "Guarantee"). In `eigsh`, since the polish of KS3
//!      from restart 100 on (Algorithm SP, issue xingo-statla#11), the
//!      largest (K2) of a run is `≤ 0.314·m·u·√R·‖A‖_est` on the five
//!      operators of the table "The polish" below (`0.164` polishing from
//!      the first restart), asserted with `0.5` by a release-only unit test
//!      — a measurement, not a bound.
//! (K3) After a restart `H̄` is zero outside `[S_LL H_LA; 0 T_A]` (top `k′`
//!      rows) and the coupling row `k′`, whose locked entries are zero;
//!      `T_A` is quasi-triangular (diagonal in `eigsh`) — debug-asserted.
//! (K4) Every returned pair passed the test of KS5 at the moment it was
//!      locked or returned; locked columns of `V` are never modified again.
//!
//! ## Guarantee
//! A returned pair `(θ, y)` has, in exact arithmetic on the computed
//! decomposition, `‖A y − θ y‖ ≤ ρ + Σ_locked |b_dropped|·|x_L|`, each term
//! `≤ τ·max(ε^{2/3}, |θ|)`; rounding adds `c_K·m·u·‖A‖_est` by (K1)–(K2).
//! Hence the acceptance form `‖A y − θ y‖ ≤ (τ + c_K·m·u)·‖A‖` (when `‖A‖ ≥
//! ε^{2/3}`), **measured on runs of few restarts** — every Matrix Market
//! fixture and every acceptance test. For symmetric `A` this bounds the
//! distance of `θ` to the spectrum (Parlett, Thm. 4.5.1, number not
//! verified); for a general
//! `A` the eigenvalue error is that times its condition number.
//!
//! **Long runs** (issue xingo-statla#9, task I02). The rounding term grows
//! with the number `R` of restarts ([`EigsStats::restarts`]), because (K2)
//! does, and **no closed form in `R` is guaranteed**: two candidates fitted
//! to the first measurements were each exceeded by a later one —
//! `c_K(1 + √R/8) = 8 + √R` by `eigs` on a clustered spectrum (`102.9`
//! against `62.8`), `c_K(1 + √R/4) = 8 + 2√R` by `eigsh` on an indefinite
//! tridiagonal (`125.5` against `117.2`, with Algorithm SJ alone in KS3;
//! `13.6` against `127.1` since the polish, task I10). After hundreds of
//! restarts a caller that needs the residual should recompute it, and
//! Lemma R below bounds the recomputed value whatever `R`. Measured in
//! `f64`, ratio `(‖r‖/‖A‖ − τ)/(m u)` against `√(‖A‖₁‖A‖_∞)`, residual in
//! double-double, the worst returned pair (`allow_partial` where fewer
//! than `k` converged, marked `*`). The rows of `eigsh` and `eigsh_near`
//! were measured again with the polish of KS3 from restart 100 on (this
//! edition; task I10, issue xingo-statla#11); the column "SP always" polishes
//! from the first restart, and "SJ alone" is the edition before the polish
//! (task I02; reproduced in task I10 with the same restart counts and
//! ratios within `0.01`); the rows of `eigs` and `eigs_near` did not
//! change (the polish is not in their KS3):
//!
//! | driver | matrix | which | `k`, `m` | `R` | ratio | `8 + √R` | `8 + 2√R` | SP always: `R` / ratio | SJ alone: `R` / ratio |
//! |---|---|---|---|---:|---:|---:|---:|---:|---:|
//! | `eigs` | Grcar(100, 3) | LM | 16, 40 | 108 | 3.78 | 18.4 | 28.8 | — | — |
//! | `eigs` | Grcar(200, 3) | LM | 16, 40 | 988 | 12.10 | 39.4 | 70.9 | — | — |
//! | `eigs` | Grcar(200, 3) | LI | 16, 40 | 142 | 3.81 | 19.9 | 31.8 | — | — |
//! | `eigs` | Grcar(400, 3) | LR | 10, 24 | 1490 | 18.86 | 46.6 | 85.2 | — | — |
//! | `eigs` | Grcar(100, 3) | LM | 10, 24 `*` | 2000 | 22.14 | 52.7 | 97.4 | — | — |
//! | `eigs` | 1-D Laplacian, 2000 | LM | 10, 24 | 1271 | 13.82 | 43.7 | 79.3 | — | — |
//! | `eigs` | clusters of 50, 1000 | LM | 10, 24 `*` | 3000 | **102.86** | 62.8 | 117.5 | — | — |
//! | `eigsh` | 1-D Laplacian, 1000 | BE | 10, 24 | 1063 | 3.31 | 40.6 | 73.2 | 1043 / 2.07 | 10000 `*` / 31.64 |
//! | `eigsh` | 1-D Laplacian, 2000 | LM | 10, 24 | 1358 | 3.60 | 44.9 | 81.7 | 1360 / 1.81 | 1221 / 48.59 |
//! | `eigsh` | 1-D Laplacian, 3000 | LA | 10, 24 | 3035 | 4.58 | 63.1 | 118.2 | 3035 / 3.18 | 2659 / 93.50 |
//! | `eigsh` | 2-D Laplacian, 45² | LM | 10, 14 | 2050 | 4.11 | 53.3 | 98.6 | 2043 / 3.82 | 1984 / 18.37 |
//! | `eigsh` | 2-D Laplacian, 60² | SA | 10, 16 | 1473 | 2.63 | 46.4 | 84.8 | 1387 / 2.36 | 3000 `*` / 1.42 |
//! | `eigsh` | clusters of 50, 1000 `†` | LM | 10, 24 | 5380 | 3.46 | 81.3 | 154.7 | 2010 / 0.29 | 6127 / 52.61 |
//! | `eigsh` | indefinite tridiagonal, 1000 | LM | 10, 24 | 987 | 5.05 | 39.4 | 70.8 | 987 / 4.35 | 880 / 36.70 |
//! | `eigsh` | indefinite tridiagonal, 2000 | BE | 10, 24 | 3548 | **13.59** | 67.6 | 127.1 | 3548 / 13.16 | 2983 / **125.53** |
//! | `eigsh` | graded tridiagonal, 1000 | LM | 10, 14 | 131 | 1.40 | 19.4 | 30.9 | 131 / 0.27 | 128 / 1.40 |
//! | `eigsh_near` | 1-D Laplacian, 2000, `σ = 1.0001` | — | 10, 12 | 189 | 0.75 | 21.7 | 35.5 | 189 / 0.66 | 189 / 0.71 |
//! | `eigsh_near` | 2-D Laplacian, 45², `σ = 2.0001` | — | 10, 12 | 141 | 0.20 | 19.9 | 31.7 | 139 / 0.41 | 141 / 0.23 |
//! | `eigs_near` | 1-D Laplacian, 2000, `σ = 1.0001` | — | 10, 12 | 189 | 0.74 | 21.7 | 35.5 | — | — |
//!
//! (The 1-D Laplacian is `tridiag(−1, 2, −1)`; the indefinite tridiagonal
//! is that minus `2I`; the graded one has diagonal `(1 − i/n)⁴` and
//! off-diagonal `0.1√(d_i d_{i+1})`; the clusters are 20 groups of 50
//! eigenvalues `1 − 0.04c + 10⁻⁶j`, rotated by three reflectors. `†`: the
//! reflectors of the earlier edition of this row (`R = 2485`, ratio
//! `21.56`) were not recorded; the three columns of the row use new ones,
//! so the row compares the solvers on the same matrix, not with the
//! earlier number.) The shift-invert drivers converge fast on `(A −
//! σI)⁻¹` and stay below `1`. With the polish every `eigsh` row is below
//! `8 + √R`, and the two runs that did not converge within their limits
//! do; polishing costs restarts on most of the others (`+3 %` to `+19 %`,
//! the indefinite tridiagonal of order 2000 the most) and saves them where
//! SJ alone stagnated; starting it at restart 100 leaves those counts
//! almost as they are and multiplies the ratio by up to `2` (the error
//! accumulated before restart 100 stays). Being measurements, these rows
//! restore no guarantee in `R`.
//! The fixture tests, with few restarts, assert `(τ + c_K·m·u)·‖A‖`;
//! `tests/nonnormal.rs` asserts the four Grcar runs of its test against
//! `8 + √R` and `4 c_K` as a regression check of those runs, not as a
//! guarantee.
//!
//! **The constant** `c_K = C_K = 8` (measured in task L3.4, `l34-tables.md`):
//! the largest recomputed `(‖r‖/‖A‖ − τ)/(m u)` over the Matrix Market
//! fixtures was `2.44` in `f64` (margin 3.3), (K2) at most `1.8`, (K1)
//! debug-asserted with `c_K` in every test run. In double-double, whose
//! operations round to a few `u` (Chapter 11 of `xingo-extprec`), (K2)
//! reached `7.5` and the residual `19.4`: its smoke tests use `64`
//! (question Q-L34-7). *Strongly nonnormal matrices* (issue
//! xingo-statla#2): `≤ 5.7` on Jordan-like blocks and on a
//! convection-dominated operator, but `12.1` on the Grcar matrix of order
//! 200 (`LM`, `k = 16`, `ncv = 40`, against `√(‖A‖₁‖A‖_∞) ≥ ‖A‖₂`): the
//! constant was measured on near-normal fixtures with few restarts, and
//! does not bound long runs (see "Long runs" above).
//!
//! *Where the excess comes from* (issue xingo-statla#9, task I02, measured
//! along the runs in `f64`): not from the orthogonality of `V` — (K1) stays
//! at most `14u` in these runs, 988 restarts included — but from the Krylov
//! relation (K2), which grows with the number `R` of restarts, roughly as
//! `27√R · u‖A‖_est` at `m = 40` (`85u` after 10 restarts, `275u` after
//! 100, `817u` at the end, against `c_K m u = 320u`), in every Grcar run
//! measured (orders 100 to 400; the final ratio of criterion (b) is `0.3√R`
//! to `0.5√R`). Every restart replaces `Z_Aᵀ H_AA Z_A` by the computed
//! `T_A`, whose backward error is about `20u‖H_AA‖_F` after KS3 and `22u`
//! after RO (RO adds `2u`), and the kept columns carry the relation error
//! of the previous restarts, rotated; the growth like `√R` is that of
//! errors adding up as a random walk.
//! Neither candidate of the issue is the source: forming `V Z_A` in
//! double-double, or skipping the sweep of KS7′ (then (K1) reaches `526u`),
//! leaves the growth of (K2) unchanged; re-orthonormalizing `Z_A` and
//! recomputing `T_A = Z_Aᵀ H_AA Z_A` in double-double after RO lowers it by
//! about a tenth. Near-normal matrices converge in few restarts, which
//! is why the fixtures stay below `2.44`.
//!
//! *Why `eigsh` grew faster than `eigs`* (issue xingo-statla#10, task I03;
//! measured along the runs in `f64`, `‖A V − V H̄‖_F/‖A‖_est` over the kept
//! columns after each restart, in units of `u`; 1-D Laplacian of order
//! 2000 and the indefinite tridiagonal of order 1000, `LM`, `k = 10`,
//! `m = 24`). In `eigs` the relation error grows like `√R`; in `eigsh` with
//! Algorithm SJ alone in KS3 it grew about **linearly**, and the final
//! ratio of criterion (b) was three times larger on the same matrix (the
//! rows "with SP" are this edition, task I10):
//!
//! | run | (K2) at `R` = 100 / 300 / 600 / 900 | `R` | ratio |
//! |---|---|---:|---:|
//! | `eigs`, Laplacian | 499 / 743 / 967 / 1168 | 1271 | 13.8 |
//! | `eigsh`, Laplacian | 268 / 778 / 1937 / 3240 | 1221 | 48.6 |
//! | `eigsh` with HR + FQ in KS3, Laplacian | 504 / 975 / 1111 / 1128 | 1275 | 17.6 |
//! | `eigs`, indefinite | 339 / 707 / 1020 / 1084 | 983 | 21.9 |
//! | `eigsh`, indefinite | 226 / 864 / 2058 / — | 880 | 36.7 |
//! | `eigsh` with HR + FQ in KS3, indefinite | 348 / 603 / 847 / 1086 | 926 | 15.0 |
//! | `eigsh` with SP in KS3, Laplacian | 56 / 97 / 140 / 169 | 1360 | 1.8 |
//! | `eigsh` with SP in KS3, indefinite | 86 / 153 / 240 / 335 | 987 | 4.3 |
//!
//! The source is the symmetric small solver of KS3 (Algorithm SJ, `T_A`
//! diagonal, the reordering a permutation): with the Hessenberg reduction
//! and Francis QR of `eigs` on the *same* symmetric `H` (the rest of the
//! symmetric engine unchanged) the growth is that of `eigs`. It is not the
//! size of the error SJ commits — its backward error per restart,
//! `‖H_AA Z_A − Z_A T_A‖_F`, is `≈ 5u‖H_AA‖_F` against `≈ 9u` for FQ, and
//! `‖Z_AᵀZ_A − I‖_F ≈ 50u` in both — but its direction: on the nearly
//! diagonal arrowhead of a restart SJ makes few, small rotations, `Z_A`
//! stays close to a permutation, and the error of each restart is added
//! to the same kept columns, coherently, where the Householder reflectors
//! of HR mix them as a random walk. These do *not* change the growth:
//! keeping the Gram–Schmidt coefficients instead of the mirror of KS2′
//! (`48.2`; SJ symmetrizes them), a Jacobi threshold `2⁻¹⁰` or `2⁻²⁶`
//! times smaller (`42.0`, with more restarts), replacing the values of SJ
//! by the Rayleigh quotients `z_iᵀ H_AA z_i` (`51.9`). HR + FQ in `eigsh`
//! was a proposal of task I03; the edition took the polish instead
//! (below).
//!
//! *The polish* (issue xingo-statla#11, tasks I09 and I10). The coherent
//! error of SJ has two components, described with Algorithm SP in Chapter
//! 12.6.1: (C1), the rounding bias of SJ's diagonal updates, and (C2), the
//! rest, whose origin inside SJ was not located. HR + FQ does not remove
//! them on a clustered spectrum, and keeping only the diagonal of its `T`
//! breaks a test of this chapter (a triple eigenvalue off by `2.7·10⁻¹⁴`);
//! Kahan summation of SJ's increments removes (C1) only (ratio `48.6 →
//! 23.9` on the Laplacian, growth still about linear). SP — SJ in `f64`,
//! then the polish in double-double rounded once — keeps `T_A` diagonal
//! (K3) and Algorithm L as it is, and on the five operators of task I09
//! (`k = 10`, `LM` unless noted; ratio as in "Long runs"; the exponent is
//! the log–log slope of `‖A V − V H̄‖_F/(u‖A‖_est)` over the kept columns
//! after each restart against `R`, for `R ≥ 50`, the relation computed in
//! double-double; "coherence" is the sum over the restarts of twice the
//! inner product of the inherited and the new error of a restart over the
//! sum of the squared new errors — near 0 for a random walk):
//!
//! | operator | `m` | SJ: `R` / ratio / exponent / coherence | SP: `R` / ratio / exponent / coherence |
//! |---|---:|---|---|
//! | 1-D Laplacian, 2000 | 24 | 1221 / 48.6 / 1.15 / 58.4 | 1360 / 1.8 / 0.50 / 0.05 |
//! | indefinite tridiagonal, 1000 | 24 | 880 / 36.7 / 1.16 / 40.3 | 987 / 4.3 / 0.63 / 0.71 |
//! | 2-D Laplacian, 45² | 14 | 1980 / 19.1 / 0.42 / 21.7 | 2025 / 3.6 / 0.44 / 1.99 |
//! | clusters of 50, 1000 `‡` | 24 | 4378 / 128.7 / 0.91 / 491.6 | 7856 / 2.0 / 0.40 / 2.49 |
//! | geometric `0.995^i`, 1000, `SA`, `R ≤ 3000` | 24 | 3000 `*` / 0.3 / 0.88 / 59.2 | 495 / 0.2 / 0.10 / 0.65 |
//!
//! (`‡`: three reflectors of another seed than the row of "Long runs"; `*`:
//! 8 of 10 converged. The SJ columns are task I09's; the SP ones, task
//! I10's, are also those of SJ run wholly in double-double in task I09,
//! value for value.) The growth of (K2) is about `√R` on all five, where
//! SJ alone was about linear on three. *What it costs* (counts, not
//! times; dd operations at 9 to 20 flops, Chapter 12.0.8): on the
//! Laplacian, per restart (`p ≈ 24`), SJ makes about 1 450 rotations in 8
//! sweeps (`≈ 0.67` Mflop) and SP adds about 390 dd rotations in 3 sweeps
//! plus the projections (`≈ 3.9` Mflop: SP2 0.8, SP3 0.5, SP4 2.3, SP5 0.4),
//! independent of `n`, against `≈ 5.7` Mflop for the rest of a restart at
//! `n = 2000`. Polishing from the first restart, over the test suite of
//! this crate (every Krylov run of `cargo test --release --features
//! faer`, 156 runs of `eigsh` and `eigsh_near`), the restarts of `eigsh`
//! went from 6 012 to 6 271 (`+4.3 %`) and its applications from 82 796
//! to 87 459 (`+5.6 %`), with other bits in 152 of the 156 runs; the large
//! acceptance runs against ARPACK needed one or two restarts more, and the
//! worst residual ratio of the `eigsh` fixtures went from `0.88` to `0.35`
//! (large ones: `2.76` to `0.42`); measured incidentally, the small
//! fixtures took about `2.5×` the time.
//!
//! *Why the polish starts at restart 100* ([`POLISH_FROM_RESTART`], task
//! I10). Runs of few restarts — every fixture, every acceptance test — do
//! not need it (their (K2) is within `c_K` without it), and they pay its
//! cost in full. Polishing from restart `R₀` keeps the bits of SJ alone in
//! every run shorter than `R₀`; from `R₀` on the error accumulated so far
//! is carried as a constant: on the five operators above it neither grows
//! nor goes away (`‖A V − V H̄‖_F/u` at the end of the run against its value
//! at `R₀`: Laplacian `268 → 283` for `R₀ = 100`, `778 → 677` for `R₀ =
//! 300`), and what the restarts after `R₀` add is the small `√R` growth of
//! the polish. Measured over `R₀ ∈ {10, 30, 100, 300}` (ratio at the end;
//! restarts in parentheses where they differ from "always" by more than
//! 5 %):
//!
//! | operator | always | `R₀ = 10` | `30` | `100` | `300` |
//! |---|---:|---:|---:|---:|---:|
//! | 1-D Laplacian, 2000 | 1.81 | 1.83 | 1.91 | 3.60 | 10.69 |
//! | indefinite tridiagonal, 1000 | 4.35 | 4.64 | 4.57 | 5.05 | 16.50 |
//! | 2-D Laplacian, 45² (`R`: 2025 always) | 3.61 | 3.31 | 3.74 (2138) | 4.06 | 8.01 |
//! | clusters of 50, 1000 `‡` (`R`: 7856 always) | 2.04 | 2.85 (8778) | 3.99 (8652) | 14.94 (8286) | 15.34 (9131) |
//! | geometric, `SA` (`R`: 495 always) | 0.16 | 0.19 (605) | 0.23 | 0.30 | 0.31 (578) |
//!
//! `R₀ = 100` keeps 153 of the 156 `eigsh` runs of the suite bit for bit
//! those of SJ alone (the three others run 880 to 3000 restarts; `R₀ = 10`:
//! 132, `R₀ = 30`: 144, `R₀ = 300`: 153), so the fixtures and the
//! acceptance against ARPACK are unchanged; suite totals 6 012 → 6 256
//! restarts and 82 796 → 87 300 applications. The runs where SJ alone
//! stagnated converge with every `R₀` measured. No cheap signal in the
//! loop measures (K2): the estimates `ρ` of KS4 come from the coupling
//! row, which is exactly what the drift of the relation does not show, so
//! the trigger counts restarts. The unit test
//! `long_runs_relation_grows_like_the_square_root_of_restarts` asserts
//! the largest (K2) of the five runs against `0.5·m·u·√R` (measured
//! `0.314`, the clustered one, stopped at 3000 restarts; `0.164` polishing
//! from the first restart).
//!
//! *Lemma R (a posteriori residual; no dependence on `R`).* Let `(θ, y)` be
//! a returned pair, `‖y‖ = 1`, `θ = a + ib` and `y = y_r + i y_i` (`b = 0`,
//! `y_i = 0` for a real value). Suppose each application returns `ŵ =
//! fl(A x) = A x + δ` with `‖δ‖ ≤ η_op‖x‖` (for a sparse matrix with at
//! most `c` entries per row, summed in order, `|δ| ≤ γ_c |A||x|`, so `η_op
//! ≤ γ_c √(‖A‖₁‖A‖_∞)`; `γ_c = c u/(1 − c u)`). Compute `r̂_r =
//! fl(ŵ_r − a y_r + b y_i)` and `r̂_i = fl(ŵ_i − a y_i − b y_r)` entry by
//! entry (`ŵ_r = fl(A y_r)`, `ŵ_i = fl(A y_i)`; two products and two
//! sums each) and `ρ̂ = fl(‖(r̂_r, r̂_i)‖₂)`. Then
//!
//! ```text
//! ‖A y − θ y‖ ≤ (1 + γ_{2n+1}) ρ̂ + γ_3 (‖ŵ‖ + √2 |θ|) + η_op .
//! ```
//!
//! *Proof.* Entrywise, `r̂ = (ŵ − θ y) + e` with
//! `|e| ≤ γ_3(|ŵ| + |a||y| + |b||y|)` (each entry is a sum of three terms, two of them products,
//! each operation rounded once: the standard model and the bound for
//! sums of products of Higham, *Accuracy and Stability*, ch. 3), so `‖e‖ ≤
//! γ_3(‖ŵ‖ + (|a| + |b|)‖y‖) ≤ γ_3(‖ŵ‖ + √2|θ|)`. `A y − θ y = (ŵ − θ y)
//! − δ_c`, `δ_c = (δ_r, δ_i)` with `‖δ_c‖² ≤ η_op²(‖y_r‖² + ‖y_i‖²) =
//! η_op²`. Finally, for `ρ̂` computed as the rounded square root of a sum
//! of `2n` squares accumulated in order, `‖r̂‖ ≤ (1 + γ_{2n+1})ρ̂`. The
//! triangle inequality gives the bound. ∎
//!
//! The bound costs one application per real pair and two per complex one,
//! and its rounding terms are `O((n + c)u‖A‖)` whatever the number of
//! restarts: it certifies a returned pair after any run, where (K2) gives
//! no closed form. The drivers do not compute it (it would change the
//! number of applications, [`EigsStats::applies`]): [`SymEigen::residuals`]
//! and [`Eigen::residuals`] compute `ρ̂` on request, outside the counters;
//! `tests/long_runs.rs` asserts the lemma on long runs of both drivers.
//!
//! ## Acceptance (L3.4, S9; tests/eigs.rs)
//! On Matrix Market matrices the `k` wanted eigenvalues agree with
//! `scipy.sparse.linalg.eigsh`/`eigs` (ARPACK; fixtures committed) to
//! `1e-10` relative where the gap to the rest of the spectrum is `≥
//! 10⁻⁶‖A‖`, and every pair has `‖Ay − θy‖ ≤ (τ + c_K·m·u)‖A‖` (runs of
//! few restarts; with `τ =
//! 0` the literal `τ‖A‖` is below the rounding of the residual itself; see
//! question Q-L34-2). The double-double smoke tests cover `eigsh`, `eigs`,
//! `eigsh_near` and `eigs_near`, and `eigsh_near`'s criteria are also
//! checked with the dense kernel `FaerDense` (feature `faer`) inside its
//! `LBLᵀ` (issue xingo-statla#7). Larger matrices (issue xingo-statla#2):
//! three formula fixtures of 3000 and 4000 rows (`make_fixtures.py
//! --large`, references only, no `.mtx`) pass (a), (b), (c) and the `svds`
//! criterion against ARPACK (worst value error `5.5·10⁻¹³`, worst residual
//! ratio `2.8`; 3 s in release, ignored in debug builds; all of them
//! restart fewer than 100 times, so the polish of KS3 does not reach
//! them).
//!
//! ## Divergences from the references
//! * Restart by Krylov–Schur, not implicit QR (S7). Results are compared
//!   with ARPACK, not traces.
//! * The symmetric small solver is cyclic Jacobi, not tridiagonal QL (the
//!   projected matrix after a restart is an arrowhead, not tridiagonal; and
//!   Jacobi gives orthogonal eigenvectors to `O(ε)` in every case),
//!   polished in double-double and rounded once (Algorithm SP, this
//!   crate's addition: the references solve the small problem in the
//!   working precision — recalled, not re-checked).
//! * Exactly `k` values are returned, as scipy does: when the `k`-th value
//!   is the first member of a conjugate pair, its conjugate is omitted
//!   (question Q-L34-1).
//! * `BE` returns the values in its own wanted order (high, low, high, …).
//!
//! ## References
//! G. W. Stewart, "A Krylov–Schur algorithm for large eigenproblems",
//! SIAM J. Matrix Anal. Appl. 23(3), 601–614, doi:10.1137/S0895479800371529
//! (cited as 2001; Crossref dates the issue 2002); K. Wu & H. Simon,
//! "Thick-restart Lanczos method for large symmetric eigenvalue problems",
//! SIAM J. Matrix Anal. Appl. 22(2), 602–616 (2000),
//! doi:10.1137/S0895479898334605; R. B. Lehoucq, D. C. Sorensen & C. Yang,
//! *ARPACK Users' Guide*, SIAM (1998), doi:10.1137/1.9780898719628
//! (`dsaupd`/`dnaupd`: `which`, `ncv`, `nev′`; the test is `bounds(i) ≤
//! tol·max(eps23, |θ_i|)`, `eps23 = ε^{2/3}`, in `dsconv.f`/`dnconv.f` of
//! arpack-ng, `|θ|` the modulus in `dnconv`, `ε = dlamch('E') = 2⁻⁵³`);
//! D. C. Sorensen, "Implicit application of polynomial filters in a k-step
//! Arnoldi method", SIAM J. Matrix Anal. Appl. 13(1), 357–385 (1992),
//! doi:10.1137/0613025; B. N. Parlett, *The Symmetric Eigenvalue Problem*,
//! SIAM Classics 20 (1998), doi:10.1137/1.9781611971163; Daniel, Gragg,
//! Kaufman & Stewart, Math. Comp. 30(136), 772–795 (1976),
//! doi:10.1090/S0025-5718-1976-0431641-8. Checked in task I03 (issue xingo-statla#4) against Crossref and the publishers' tables of contents (and, for
//! ARPACK, its Fortran sources), task I03. *Not verified:* Parlett's
//! Thm. 4.5.1 (only the chapter titles were reachable: ch. 4, "Simple
//! vector iterations"), the online date of Stewart's article.

use xingo_sparse::{DenseBlock, Real};

use crate::StatlaError;
use crate::error::op_err;
use crate::krylov::{expand_arnoldi, kept_dimension};
use crate::operator::LinearOperator;
use crate::random::Rng;
use crate::real::{is_exact_zero, le_exact, lt};
use crate::schur::{
    Cx, SchurOptions, block_value, blocks, hypot, identity, real_schur, reorder, schur_eigvec,
    sym_eig, sym_eig_polished,
};
use crate::small::{check_finite, check_len, dot, norm2, orthogonalize_against};

/// The constant `c_K` of invariants (K1)–(K2) and of the residual guarantee
/// (measured in task L3.4, with a margin).
pub(crate) const C_K: f64 = 8.0;

/// KS3 of `eigsh` polishes (Algorithm SP) from this restart on and runs
/// Algorithm SJ alone before it, so that every run shorter than this keeps
/// the bits of the edition without the polish (task I10, "gated" variant).
pub(crate) const POLISH_FROM_RESTART: usize = 100;

/// A pair `re + i·im` (eigenvalues of a real nonsymmetric operator). No
/// arithmetic: a value type for results, not a complex scalar (complex
/// matrices are out of scope).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex<T: Real = f64> {
    /// Real part.
    pub re: T,
    /// Imaginary part.
    pub im: T,
}

/// Which part of the spectrum, with ARPACK's two-letter codes
/// ([`Which::from_arpack`]) so that R and Python pass their strings
/// through `xingo-ffi` unchanged. [`eigsh()`] accepts `LM SM LA SA BE`;
/// [`eigs()`] accepts `LM SM LR SR LI SI`; any other combination is
/// [`StatlaError::InvalidArgument`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Which {
    /// `LM`: largest `|λ|` (default of ARPACK, scipy, RSpectra).
    LargestMagnitude,
    /// `SM`: smallest `|λ|` (slow without shift-invert: prefer
    /// [`crate::eigsh_near`] with `σ = 0`).
    SmallestMagnitude,
    /// `LA`: largest algebraic (symmetric).
    LargestAlgebraic,
    /// `SA`: smallest algebraic (symmetric).
    SmallestAlgebraic,
    /// `BE`: half from each end, the extra one from the high end, as
    /// ARPACK (symmetric).
    BothEnds,
    /// `LR`: largest real part (general).
    LargestReal,
    /// `SR`: smallest real part (general).
    SmallestReal,
    /// `LI`: largest imaginary part in magnitude (general).
    LargestImag,
    /// `SI`: smallest imaginary part in magnitude (general).
    SmallestImag,
}

impl Which {
    /// The value of an ARPACK code (`"LM"`, …), case-sensitive as ARPACK.
    #[must_use]
    pub fn from_arpack(code: &str) -> Option<Self> {
        Some(match code {
            "LM" => Self::LargestMagnitude,
            "SM" => Self::SmallestMagnitude,
            "LA" => Self::LargestAlgebraic,
            "SA" => Self::SmallestAlgebraic,
            "BE" => Self::BothEnds,
            "LR" => Self::LargestReal,
            "SR" => Self::SmallestReal,
            "LI" => Self::LargestImag,
            "SI" => Self::SmallestImag,
            _ => return None,
        })
    }

    /// The ARPACK code.
    #[must_use]
    pub fn arpack_code(self) -> &'static str {
        match self {
            Self::LargestMagnitude => "LM",
            Self::SmallestMagnitude => "SM",
            Self::LargestAlgebraic => "LA",
            Self::SmallestAlgebraic => "SA",
            Self::BothEnds => "BE",
            Self::LargestReal => "LR",
            Self::SmallestReal => "SR",
            Self::LargestImag => "LI",
            Self::SmallestImag => "SI",
        }
    }

    fn symmetric_ok(self) -> bool {
        matches!(
            self,
            Self::LargestMagnitude
                | Self::SmallestMagnitude
                | Self::LargestAlgebraic
                | Self::SmallestAlgebraic
                | Self::BothEnds
        )
    }

    fn general_ok(self) -> bool {
        matches!(
            self,
            Self::LargestMagnitude
                | Self::SmallestMagnitude
                | Self::LargestReal
                | Self::SmallestReal
                | Self::LargestImag
                | Self::SmallestImag
        )
    }
}

/// Options of [`eigs()`] and [`eigsh()`]. Defaults are scipy's
/// (`scipy.sparse.linalg.eigs`/`eigsh`) where it has one.
#[derive(Clone, Debug, PartialEq)]
pub struct EigsOptions<T: Real = f64> {
    /// Krylov dimension; `None` = `min(n, max(2k + 1, 20))`. Must satisfy
    /// `k < ncv ≤ n`.
    pub ncv: Option<usize>,
    /// Relative tolerance of the convergence test (ARPACK's): a Ritz pair
    /// `(θ, y)` is converged when `‖f‖ |e_jᵀ s| ≤ tol · max(ε^{2/3}, |θ|)`;
    /// `0` (default) means `T::EPSILON`.
    pub tol: f64,
    /// Restart limit; `None` = `10 n` (scipy's `maxiter`).
    pub max_restarts: Option<usize>,
    /// Starting vector; `None` = Gaussian from [`EigsOptions::seed`]
    /// (Algorithm G), so runs are reproducible.
    pub v0: Option<Vec<T>>,
    /// Seed of the starting vector and of restarts after an invariant
    /// subspace is found (default 0).
    pub seed: u64,
    /// DGKS threshold (default `1/√2`).
    pub reorth_eta: f64,
    /// Compute eigenvectors (default `true`; scipy's
    /// `return_eigenvectors`).
    pub vectors: bool,
    /// On non-convergence return the converged pairs (`Ok`, fewer than
    /// `k`) instead of [`StatlaError::NoConvergence`] (default `false`;
    /// S10).
    pub allow_partial: bool,
}

impl<T: Real> Default for EigsOptions<T> {
    fn default() -> Self {
        Self {
            ncv: None,
            tol: 0.0,
            max_restarts: None,
            v0: None,
            seed: 0,
            reorth_eta: core::f64::consts::FRAC_1_SQRT_2,
            vectors: true,
            allow_partial: false,
        }
    }
}

/// Work counters of the Krylov drivers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct EigsStats {
    /// Restarts performed.
    pub restarts: usize,
    /// Operator applications.
    pub applies: usize,
    /// Second Gram–Schmidt passes (DGKS) triggered.
    pub reorthogonalizations: usize,
    /// Pairs converged (`= k` unless `allow_partial`).
    pub converged: usize,
}

/// Symmetric eigenpairs, most wanted first (in the order of `which`).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SymEigen<T: Real = f64> {
    /// Eigenvalues.
    pub values: Vec<T>,
    /// Eigenvectors as columns (`n × values.len()`), orthonormal; `None`
    /// when [`EigsOptions::vectors`] is off.
    pub vectors: Option<DenseBlock<T>>,
    /// Counters.
    pub stats: EigsStats,
}

/// Eigenpairs of a real nonsymmetric operator, most wanted first; complex
/// conjugate pairs are adjacent, positive imaginary part first (exactly
/// `k` values: the conjugate of a `k`-th value with positive imaginary
/// part is omitted, as scipy does).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Eigen<T: Real = f64> {
    /// Eigenvalues.
    pub values: Vec<Complex<T>>,
    /// Real parts of the eigenvectors (`n × values.len()`), unit 2-norm
    /// in `C^n` per column; `None` when vectors are off.
    pub vectors_re: Option<DenseBlock<T>>,
    /// Imaginary parts (zero columns for real eigenvalues).
    pub vectors_im: Option<DenseBlock<T>>,
    /// Counters.
    pub stats: EigsStats,
}

/// The validated parameters of one run.
struct Params<T> {
    ncv: usize,
    tol: T,
    eta: T,
    eps23: T,
    max_restarts: usize,
}

fn invalid(what: &'static str) -> StatlaError {
    StatlaError::InvalidArgument { what }
}

/// `ε_T^{2/3}` by Newton's iteration on `y³ = ε²` (only `+ − · /`, a fixed
/// number of steps: the same bits everywhere).
fn eps23<T: Real>() -> T {
    let a = T::EPSILON * T::EPSILON;
    let (two, three) = (T::from_f64(2.0), T::from_f64(3.0));
    let mut y = T::ONE;
    for _ in 0..300 {
        y = (two * y + a / (y * y)) / three;
    }
    y
}

/// Check the arguments (in the order of the error conventions, 12.0.1).
fn validate<T: Real>(
    rows: usize,
    cols: usize,
    k: usize,
    which: Which,
    o: &EigsOptions<T>,
    symmetric: bool,
) -> Result<Params<T>, StatlaError> {
    if rows != cols {
        return Err(invalid("eigs/eigsh: the operator is not square"));
    }
    let n = rows;
    if symmetric && !which.symmetric_ok() {
        return Err(invalid("eigsh: which must be one of LM SM LA SA BE"));
    }
    if !symmetric && !which.general_ok() {
        return Err(invalid("eigs: which must be one of LM SM LR SR LI SI"));
    }
    if symmetric && (k == 0 || k >= n) {
        return Err(invalid("eigsh: k must satisfy 1 <= k < n"));
    }
    if !symmetric && (k == 0 || k + 1 >= n) {
        return Err(invalid("eigs: k must satisfy 1 <= k < n - 1"));
    }
    let ncv = o.ncv.unwrap_or_else(|| n.min((2 * k + 1).max(20)));
    if ncv <= k || ncv > n {
        return Err(invalid("eigs/eigsh: ncv must satisfy k < ncv <= n"));
    }
    if !o.tol.is_finite() || lt(o.tol, 0.0) {
        return Err(invalid("eigs/eigsh: tol must be finite and >= 0"));
    }
    if !o.reorth_eta.is_finite() || !lt(0.0, o.reorth_eta) || !lt(o.reorth_eta, 1.0) {
        return Err(invalid("eigs/eigsh: reorth_eta must lie in (0, 1)"));
    }
    if let Some(v0) = &o.v0 {
        check_len("v0", n, v0.len())?;
        check_finite("v0", v0)?;
        if is_exact_zero(norm2(v0)) {
            return Err(invalid("eigs/eigsh: v0 is the zero vector"));
        }
    }
    let tol = if is_exact_zero(o.tol) {
        T::EPSILON
    } else {
        T::from_f64(o.tol)
    };
    Ok(Params {
        ncv,
        tol,
        eta: T::from_f64(o.reorth_eta),
        eps23: eps23::<T>(),
        max_restarts: o.max_restarts.unwrap_or_else(|| n.saturating_mul(10)),
    })
}

/// The argument checks of [`eigsh()`]/[`eigs()`], for the shift-invert
/// drivers (which validate before factorizing).
pub(crate) fn validate_for<T: Real>(
    rows: usize,
    cols: usize,
    k: usize,
    which: Which,
    o: &EigsOptions<T>,
    symmetric: bool,
) -> Result<(), StatlaError> {
    validate(rows, cols, k, which, o, symmetric).map(|_| ())
}

/// `a` before `b` in the order of `which` (strict), on representatives with
/// `im ≥ 0`.
fn before<T: Real>(which: Which, a: (T, T), b: (T, T)) -> bool {
    match which {
        Which::LargestMagnitude => lt(hypot(b.0, b.1), hypot(a.0, a.1)),
        Which::SmallestMagnitude => lt(hypot(a.0, a.1), hypot(b.0, b.1)),
        Which::LargestAlgebraic | Which::LargestReal | Which::BothEnds => lt(b.0, a.0),
        Which::SmallestAlgebraic | Which::SmallestReal => lt(a.0, b.0),
        Which::LargestImag => lt(b.1.abs(), a.1.abs()),
        Which::SmallestImag => lt(a.1.abs(), b.1.abs()),
    }
}

/// The tie-break of `which`: ARPACK's `dngets` pre-sorts by this key
/// before the sort by `which` (`LM`↔`LR`, `SM`↔`SR`, `LI`→`LM`,
/// `SI`→`SM`; checked against `dngets.f` of arpack-ng, task I03), so that
/// equal keys (a spectrum of real values under `LI`, `±λ` under `LM`) are
/// resolved the same way at every restart. The keys of `LA`, `SA` and `BE`
/// are this crate's: ARPACK's `dsgets` sorts by `which` alone (`BE` by `LA`,
/// then a block swap of the two ends).
fn tie_break(which: Which) -> Which {
    match which {
        Which::LargestMagnitude => Which::LargestReal,
        Which::SmallestMagnitude => Which::SmallestReal,
        Which::LargestReal | Which::LargestImag | Which::LargestAlgebraic | Which::BothEnds => {
            Which::LargestMagnitude
        }
        Which::SmallestReal | Which::SmallestImag | Which::SmallestAlgebraic => {
            Which::SmallestMagnitude
        }
    }
}

/// KS4: the order of the blocks with values `vals` (representatives), most
/// wanted first; ties by [`tie_break`], then the input order (locked
/// blocks come first).
fn rank_blocks<T: Real>(which: Which, vals: &[(T, T)]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..vals.len()).collect();
    let key = if which == Which::BothEnds {
        Which::LargestAlgebraic
    } else {
        which
    };
    let second = tie_break(which);
    let cmp = |w: Which, a: (T, T), b: (T, T)| {
        if before(w, a, b) {
            core::cmp::Ordering::Less
        } else if before(w, b, a) {
            core::cmp::Ordering::Greater
        } else {
            core::cmp::Ordering::Equal
        }
    };
    idx.sort_by(|&i, &j| cmp(key, vals[i], vals[j]).then_with(|| cmp(second, vals[i], vals[j])));
    if which != Which::BothEnds {
        return idx;
    }
    // BE: alternate the high end (front) and the low end (back), high first.
    let mut out = Vec::with_capacity(idx.len());
    let (mut lo, mut hi) = (0usize, idx.len());
    let mut high = true;
    while lo < hi {
        if high {
            out.push(idx[lo]);
            lo += 1;
        } else {
            hi -= 1;
            out.push(idx[hi]);
        }
        high = !high;
    }
    out
}

/// One block of `T` at the time of a test.
#[derive(Clone, Copy, Debug)]
struct Blk<T> {
    /// Start row in `T` (full coordinates).
    start: usize,
    size: usize,
    re: T,
    im: T,
    locked: bool,
    converged: bool,
}

/// The state of Algorithm KS.
struct Engine<T: Real> {
    n: usize,
    m: usize,
    sym: bool,
    /// `V`, `n × (m + 1)`.
    v: Vec<T>,
    /// `H̄`, `(m + 1) × m`.
    h: DenseBlock<T>,
    /// Locked count `n_L`.
    nl: usize,
    /// `‖A‖_est`.
    anorm: T,
    rng: Rng,
    stats: EigsStats,
    #[cfg(test)]
    k2: Option<f64>,
}

/// The result of KS3 on the active block.
struct ActiveSchur<T: Real> {
    /// `T_A`, `na × na`.
    t: DenseBlock<T>,
    /// `Z_A`.
    z: DenseBlock<T>,
    /// The coupling row before KS3 (`H̄[m, n_L..m]`).
    c0: Vec<T>,
    /// `H_LA` before KS3 (`n_L × na`).
    hla0: DenseBlock<T>,
}

impl<T: Real> ActiveSchur<T> {
    /// `b_A = c0ᵀ Z_A`.
    fn coupling(&self) -> Vec<T> {
        let na = self.z.rows;
        (0..na)
            .map(|c| dot(&self.c0, &self.z.data[c * na..(c + 1) * na]))
            .collect()
    }
    /// `H_LA Z_A`.
    fn hla(&self) -> DenseBlock<T> {
        let (nl, na) = (self.hla0.rows, self.z.rows);
        let mut out = DenseBlock::zeros(nl, na);
        for c in 0..na {
            for t in 0..na {
                let f = self.z.data[t + c * na];
                if is_exact_zero(f) {
                    continue;
                }
                for r in 0..nl {
                    out.data[r + c * nl] += self.hla0.data[r + t * nl] * f;
                }
            }
        }
        out
    }
}

impl<T: Real> Engine<T> {
    #[inline]
    fn hi(&self, i: usize, j: usize) -> usize {
        i + j * (self.m + 1)
    }

    /// KS2 for `j = from, …, m − 1`.
    fn expand_from<O: LinearOperator<T> + ?Sized>(
        &mut self,
        op: &mut O,
        from: usize,
        eta: T,
    ) -> Result<(), StatlaError> {
        let (n, m) = (self.n, self.m);
        let mut w = vec![T::ZERO; n];
        let mut col = vec![T::ZERO; m + 1];
        for j in from..m {
            op.apply(&self.v[j * n..(j + 1) * n], &mut w)
                .map_err(op_err)?;
            self.stats.applies += 1;
            check_finite("operator output", &w)?;
            self.anorm = self.anorm.max(norm2(&w));
            col.fill(T::ZERO);
            let floor = T::EPSILON * self.anorm;
            let beta = expand_arnoldi(
                &self.v,
                n,
                j + 1,
                &mut w,
                &mut col[..=j],
                eta,
                floor,
                &mut self.rng,
                &mut self.stats.reorthogonalizations,
            );
            for i in 0..=j {
                let at = self.hi(i, j);
                self.h.data[at] = col[i];
            }
            for i in j + 1..=m {
                let at = self.hi(i, j);
                self.h.data[at] = T::ZERO;
            }
            let at = self.hi(j + 1, j);
            self.h.data[at] = beta;
            if self.sym {
                // KS2′: the strict upper part of column j is the mirror of
                // row j (`β_{j−1}`, or the arrowhead `b` right after a
                // restart); the Gram–Schmidt coefficients there only
                // orthogonalized `w` (Lanczos: they are `b_i` plus rounding).
                for i in 0..j {
                    let (up, low) = (self.hi(i, j), self.hi(j, i));
                    self.h.data[up] = self.h.data[low];
                }
            }
            self.v[(j + 1) * n..(j + 2) * n].copy_from_slice(&w);
        }
        Ok(())
    }

    /// (K2), unit tests only: `max ‖A v_j − V H̄ e_j‖ / ‖A‖_est`.
    #[cfg(test)]
    fn relations<O: LinearOperator<T> + ?Sized>(&self, op: &mut O) -> f64 {
        let (n, m) = (self.n, self.m);
        let mut worst = 0.0f64;
        let mut y = vec![T::ZERO; n];
        for j in 0..m {
            op.apply(&self.v[j * n..(j + 1) * n], &mut y).unwrap();
            for i in 0..=m {
                let f = self.h.data[self.hi(i, j)];
                crate::small::axpy_sub(f, &self.v[i * n..(i + 1) * n], &mut y);
            }
            worst = worst.max(norm2(&y).to_f64());
        }
        worst / self.anorm.to_f64().max(f64::MIN_POSITIVE)
    }

    /// KS3.
    fn active_schur(&self, so: &SchurOptions<T>) -> Result<ActiveSchur<T>, StatlaError> {
        let (m, nl) = (self.m, self.nl);
        let na = m - nl;
        let mut ha = DenseBlock::zeros(na, na);
        for j in 0..na {
            for i in 0..na {
                ha.data[i + j * na] = self.h.data[self.hi(nl + i, nl + j)];
            }
        }
        let c0: Vec<T> = (0..na).map(|j| self.h.data[self.hi(m, nl + j)]).collect();
        let mut hla0 = DenseBlock::zeros(nl, na);
        if !self.sym {
            for j in 0..na {
                for i in 0..nl {
                    hla0.data[i + j * nl] = self.h.data[self.hi(i, nl + j)];
                }
            }
        }
        let (t, z) = if self.sym {
            let (d, z) = if self.stats.restarts >= POLISH_FROM_RESTART {
                sym_eig_polished(&ha, so)?
            } else {
                sym_eig(&ha, so)?
            };
            let mut t = DenseBlock::zeros(na, na);
            for (i, &di) in d.iter().enumerate() {
                t.data[i + i * na] = di;
            }
            (t, z)
        } else {
            let mut z = identity::<T>(na);
            real_schur(&mut ha, &mut z, so)?;
            (ha, z)
        };
        Ok(ActiveSchur { t, z, c0, hla0 })
    }

    /// The locked block `S_LL`.
    fn s_ll(&self) -> DenseBlock<T> {
        let nl = self.nl;
        let mut s = DenseBlock::zeros(nl, nl);
        for j in 0..nl {
            for i in 0..nl {
                s.data[i + j * nl] = self.h.data[self.hi(i, j)];
            }
        }
        s
    }

    /// The whole quasi-triangular `T = [S_LL, H_LA Z_A; 0, T_A]` (`m × m`).
    fn full_t(&self, a: &ActiveSchur<T>) -> DenseBlock<T> {
        let (m, nl) = (self.m, self.nl);
        let na = m - nl;
        let mut t = DenseBlock::zeros(m, m);
        let s = self.s_ll();
        for j in 0..nl {
            for i in 0..nl {
                t.data[i + j * m] = s.data[i + j * nl];
            }
        }
        if !self.sym {
            let hla = a.hla();
            for j in 0..na {
                for i in 0..nl {
                    t.data[i + (nl + j) * m] = hla.data[i + j * nl];
                }
            }
        }
        for j in 0..na {
            for i in 0..na {
                t.data[nl + i + (nl + j) * m] = a.t.data[i + j * na];
            }
        }
        t
    }

    /// (K1), debug builds: `max |V_jᵀV_j − I|` over the first `j` columns.
    fn orthogonality(&self, j: usize) -> f64 {
        let n = self.n;
        let mut e = 0.0f64;
        for a in 0..j {
            for c in 0..=a {
                let d = dot(&self.v[a * n..(a + 1) * n], &self.v[c * n..(c + 1) * n]);
                if a == c && is_exact_zero(d) {
                    continue; // the no-room zero vector of Algorithm E
                }
                let t = if a == c { d - T::ONE } else { d };
                e = e.max(t.abs().to_f64());
            }
        }
        e
    }

    /// (K3), debug builds, right after a restart to `kp`.
    fn restart_shape(&self, kp: usize) -> bool {
        let (m, nl) = (self.m, self.nl);
        for j in 0..m {
            for i in 0..=m {
                let v = self.h.data[self.hi(i, j)];
                if is_exact_zero(v) {
                    continue;
                }
                let ok = if j >= kp || (i > kp) || (i != kp && i > j + 1) {
                    false
                } else if i == kp {
                    j >= nl
                } else if i == j + 1 {
                    !self.sym
                } else {
                    !self.sym || i == j
                };
                if !ok {
                    return false;
                }
            }
        }
        true
    }
}

/// The test of KS5 for a value of modulus `abs`.
fn passes<T: Real>(rho: T, abs: T, p: &Params<T>) -> bool {
    le_exact(rho, p.tol * p.eps23.max(abs))
}

/// The raw result of Algorithm KS: values (in wanted order, `im` with its
/// sign) and the complex vectors.
struct Raw<T: Real> {
    values: Vec<Complex<T>>,
    vectors: Option<(DenseBlock<T>, DenseBlock<T>)>,
    stats: EigsStats,
    #[cfg_attr(not(test), allow(dead_code))]
    k2: f64,
    #[cfg_attr(not(test), allow(dead_code))]
    locked: usize,
}

/// Algorithm KS (`sym` selects the symmetric small solver of KS3).
#[allow(clippy::too_many_lines)] // steps KS1–KS8 in order, as in the chapter
fn run<T, O>(
    op: &mut O,
    k: usize,
    which: Which,
    opts: &EigsOptions<T>,
    sym: bool,
    paranoid: bool,
) -> Result<Raw<T>, StatlaError>
where
    T: Real,
    O: LinearOperator<T> + ?Sized,
{
    let par = validate(op.nrows(), op.ncols(), k, which, opts, sym)?;
    let n = op.nrows();
    let m = par.ncv;
    let so = SchurOptions::<T>::new();
    let _ = paranoid;
    // KS1.
    let mut rng = Rng::new(opts.seed);
    let mut v = vec![T::ZERO; n * (m + 1)];
    match &opts.v0 {
        Some(v0) => v[..n].copy_from_slice(v0),
        None => rng.fill(crate::Sketch::Gaussian, &mut v[..n]),
    }
    let nv = norm2(&v[..n]);
    for x in &mut v[..n] {
        *x = *x / nv;
    }
    let mut e = Engine {
        n,
        m,
        sym,
        v,
        h: DenseBlock::zeros(m + 1, m),
        nl: 0,
        anorm: T::ZERO,
        rng,
        stats: EigsStats::default(),
        #[cfg(test)]
        k2: None,
    };
    let mut from = 0;
    loop {
        // KS2.
        e.expand_from(op, from, par.eta)?;
        #[cfg(test)]
        if paranoid {
            let g = e.relations(op);
            e.k2 = Some(e.k2.map_or(g, |h: f64| h.max(g)));
        }
        // KS3.
        let mut act = e.active_schur(&so)?;
        let nl = e.nl;
        let na = m - nl;
        // KS4.
        let s_ll = e.s_ll();
        let mut cands: Vec<Blk<T>> = Vec::new();
        for (q, s) in blocks(&s_ll) {
            let (re, im) = block_value(&s_ll, (q, s));
            cands.push(Blk {
                start: q,
                size: s,
                re,
                im,
                locked: true,
                converged: true,
            });
        }
        let bl_a = blocks(&act.t);
        let b_a = act.coupling();
        for (bi, &(q, s)) in bl_a.iter().enumerate() {
            let (re, im) = block_value(&act.t, (q, s));
            let x = schur_eigvec(&act.t, &bl_a, bi);
            let mut acc = Cx::new(T::ZERO, T::ZERO);
            let mut xn2 = T::ZERO;
            for (i, xi) in x.iter().enumerate() {
                acc = Cx::new(acc.re + b_a[i] * xi.re, acc.im + b_a[i] * xi.im);
                xn2 += xi.re * xi.re + xi.im * xi.im;
            }
            let rho = hypot(acc.re, acc.im) / xn2.sqrt();
            cands.push(Blk {
                start: nl + q,
                size: s,
                re,
                im,
                locked: false,
                converged: passes(rho, hypot(re, im), &par),
            });
        }
        let vals: Vec<(T, T)> = cands.iter().map(|c| (c.re, c.im)).collect();
        let order = rank_blocks(which, &vals);
        // The wanted: the first k values (a pair counts two).
        let mut wanted: Vec<(usize, bool)> = Vec::with_capacity(k); // (cand, conj)
        for &ci in &order {
            if wanted.len() == k {
                break;
            }
            wanted.push((ci, false));
            if cands[ci].size == 2 && wanted.len() < k {
                wanted.push((ci, true));
            }
        }
        // KS5.
        let n_conv = wanted.iter().filter(|w| cands[w.0].converged).count();
        let limit = e.stats.restarts >= par.max_restarts;
        if n_conv == k || limit {
            if n_conv < k && !opts.allow_partial {
                return Err(StatlaError::NoConvergence {
                    requested: k,
                    converged: n_conv,
                    restarts: e.stats.restarts,
                });
            }
            // KS8.
            let take: Vec<(usize, bool)> = wanted
                .iter()
                .copied()
                .filter(|w| cands[w.0].converged)
                .collect();
            let values: Vec<Complex<T>> = take
                .iter()
                .map(|&(ci, conj)| Complex {
                    re: cands[ci].re,
                    im: if conj { -cands[ci].im } else { cands[ci].im },
                })
                .collect();
            let vectors = if opts.vectors {
                Some(e.vectors(&act, &cands, &take))
            } else {
                None
            };
            let mut stats = e.stats;
            stats.converged = values.len();
            #[cfg(test)]
            let k2 = e.k2.unwrap_or(0.0);
            #[cfg(not(test))]
            let k2 = 0.0;
            return Ok(Raw {
                values,
                vectors,
                stats,
                k2,
                locked: e.nl,
            });
        }
        // KS6.
        e.stats.restarts += 1;
        let n_conv_active = wanted
            .iter()
            .filter(|w| !cands[w.0].locked && cands[w.0].converged)
            .count();
        let kp0 = kept_dimension(k, m, n_conv, nl + n_conv_active);
        // Ranks of the active blocks: converged wanted, other wanted, rest.
        let mut group = vec![2usize; cands.len()];
        for &(ci, _) in &wanted {
            group[ci] = usize::from(!cands[ci].converged);
        }
        let mut pos_in_order = vec![0usize; cands.len()];
        for (r, &ci) in order.iter().enumerate() {
            pos_in_order[ci] = r;
        }
        let n_locked_blocks = cands.iter().filter(|c| c.locked).count();
        let rank: Vec<usize> = (n_locked_blocks..cands.len())
            .map(|ci| group[ci] * cands.len() + pos_in_order[ci])
            .collect();
        let (bl_r, _) = reorder(&mut act.t, &mut act.z, &rank, kp0.saturating_sub(nl), &so);
        let b_a = act.coupling();
        // Lock the leading converged wanted blocks whose couplings pass.
        let mut nl_new = nl;
        for &(q, s, r) in &bl_r {
            if r >= cands.len() {
                break; // group ≥ 1: not converged-wanted
            }
            let (re, im) = block_value(&act.t, (q, s));
            let bn = if s == 1 {
                b_a[q].abs()
            } else {
                hypot(b_a[q], b_a[q + 1])
            };
            if !passes(bn, hypot(re, im), &par) {
                break;
            }
            nl_new += s;
        }
        // KS7.
        let mut kp = kp0.max(nl_new).min(m - 1);
        if kp > nl {
            let last = kp - 1 - nl;
            if let Some(&(q, s, _)) = bl_r.iter().find(|b| b.0 <= last && last < b.0 + b.1)
                && s == 2
                && q == last
            {
                if kp + 1 < m {
                    kp += 1;
                } else {
                    kp -= 1;
                }
            }
        }
        let kp = kp.max(nl_new);
        let q = kp - nl;
        let mut vn = vec![T::ZERO; n * q];
        for c in 0..q {
            for t in 0..na {
                let f = act.z.data[t + c * na];
                if is_exact_zero(f) {
                    continue;
                }
                let src = &e.v[(nl + t) * n..(nl + t + 1) * n];
                for (o, &x) in vn[c * n..(c + 1) * n].iter_mut().zip(src) {
                    *o += x * f;
                }
            }
        }
        let last = e.v[m * n..(m + 1) * n].to_vec();
        e.v[nl * n..kp * n].copy_from_slice(&vn);
        e.v[kp * n..(kp + 1) * n].copy_from_slice(&last);
        // KS7′: one Gram–Schmidt sweep over the kept and the residual vectors.
        for c in nl..=kp {
            let (done, rest) = e.v.split_at_mut(c * n);
            let col = &mut rest[..n];
            let (nu, _) = orthogonalize_against(done, n, c, col, par.eta);
            if !is_exact_zero(nu) {
                for x in col.iter_mut() {
                    *x = *x / nu;
                }
            }
        }
        let hla = if sym { None } else { Some(act.hla()) };
        let mut hn = DenseBlock::zeros(m + 1, m);
        let ld = m + 1;
        for j in 0..nl {
            for i in 0..nl {
                hn.data[i + j * ld] = e.h.data[e.hi(i, j)];
            }
        }
        if let Some(hla) = &hla {
            for c in 0..q {
                for i in 0..nl {
                    hn.data[i + (nl + c) * ld] = hla.data[i + c * nl];
                }
            }
        }
        for c in 0..q {
            for i in 0..q {
                hn.data[nl + i + (nl + c) * ld] = act.t.data[i + c * na];
            }
            if nl + c >= nl_new {
                hn.data[kp + (nl + c) * ld] = b_a[c];
            }
        }
        e.h = hn;
        e.nl = nl_new;
        from = kp;
        debug_assert!(e.restart_shape(kp), "(K3) after a restart");
        #[allow(clippy::cast_precision_loss)]
        let k1 = C_K * (m as f64) * T::EPSILON.to_f64() / 2.0;
        debug_assert!(
            e.orthogonality(kp + 1) <= k1,
            "(K1) after a restart: {}",
            e.orthogonality(kp + 1)
        );
    }
}

impl<T: Real> Engine<T> {
    /// KS8: the eigenvectors of the values `take` (candidate, conjugate?).
    fn vectors(
        &self,
        act: &ActiveSchur<T>,
        cands: &[Blk<T>],
        take: &[(usize, bool)],
    ) -> (DenseBlock<T>, DenseBlock<T>) {
        let (n, m, nl) = (self.n, self.m, self.nl);
        let na = m - nl;
        let t = self.full_t(act);
        let bl = blocks(&t);
        let mut yr = DenseBlock::zeros(n, take.len());
        let mut yi = DenseBlock::zeros(n, take.len());
        for (col, &(ci, conj)) in take.iter().enumerate() {
            let bi = bl.iter().position(|b| b.0 == cands[ci].start).unwrap_or(0);
            let x = schur_eigvec(&t, &bl, bi);
            // z = blockdiag(I, Z_A) x.
            let mut z = vec![Cx::new(T::ZERO, T::ZERO); m];
            z[..nl].copy_from_slice(&x[..nl]);
            for r in 0..na {
                let mut acc = Cx::new(T::ZERO, T::ZERO);
                for c in 0..na {
                    let f = act.z.data[r + c * na];
                    acc = Cx::new(acc.re + f * x[nl + c].re, acc.im + f * x[nl + c].im);
                }
                z[nl + r] = acc;
            }
            let (re, im) = (
                &mut yr.data[col * n..(col + 1) * n],
                &mut yi.data[col * n..(col + 1) * n],
            );
            for (j, zj) in z.iter().enumerate() {
                let vj = &self.v[j * n..(j + 1) * n];
                if !is_exact_zero(zj.re) {
                    for (o, &x) in re.iter_mut().zip(vj) {
                        *o += x * zj.re;
                    }
                }
                if !is_exact_zero(zj.im) {
                    for (o, &x) in im.iter_mut().zip(vj) {
                        *o += x * zj.im;
                    }
                }
            }
            // Normalize in Cⁿ, largest entry real positive, conjugate if asked.
            let nr = hypot(norm2(re), norm2(im));
            let mut big = 0;
            let mut bm = T::ZERO;
            for i in 0..n {
                let a = hypot(re[i], im[i]);
                if lt(bm, a) {
                    bm = a;
                    big = i;
                }
            }
            let ph = if is_exact_zero(bm) {
                Cx::new(T::ONE, T::ZERO)
            } else {
                Cx::new(re[big] / bm, -im[big] / bm)
            };
            for i in 0..n {
                let w = Cx::new(re[i], im[i]).mul(ph);
                re[i] = w.re / nr;
                im[i] = if conj { -(w.im / nr) } else { w.im / nr };
            }
            // A real eigenvalue has a real eigenvector: drop the rounding.
            if is_exact_zero(cands[ci].im) {
                im.fill(T::ZERO);
            }
        }
        (yr, yi)
    }
}

/// `k` eigenpairs of a symmetric operator (the caller asserts the
/// symmetry; it is not checked).
///
/// Algorithm KS of Chapter 12.6 with the symmetric small solver — the
/// thick-restart Lanczos method (Algorithm L of the contract): full DGKS
/// reorthogonalization, Jacobi on the projected arrowhead (from the 100th
/// restart on polished in double-double and rounded once, Chapter
/// 12.6.1), Krylov–Schur
/// truncation with locking, ARPACK's test `ρ ≤ tol·max(ε^{2/3}, |θ|)`.
///
/// **Acceptance (L3.4):** on symmetric Matrix Market matrices the `k`
/// extreme eigenvalues agree with `scipy.sparse.linalg.eigsh` (ARPACK) to
/// `1e-10` relative where the eigenvalue gap is `≥ 1e-6 ‖A‖` (S9), and
/// every pair has `‖Ay − θy‖ ≤ (tol + c_K·ncv·u) ‖A‖` (`c_K = 8`), measured
/// on runs of few restarts. Long runs (hundreds of restarts, see
/// [`EigsStats::restarts`]) exceed it, and no form in the number of
/// restarts bounds every measured run: Chapter 12.6, "Long runs". A
/// residual recomputed by the caller (`‖A y − θ y‖`, one application per
/// pair) is bounded, whatever the number of restarts, by Lemma R there.
///
/// # Errors
/// [`StatlaError::InvalidArgument`] (not square; `which` not symmetric; `k
/// = 0`; `k ≥ n`; `ncv ∉ (k, n]`; `tol < 0` or not finite; `reorth_eta ∉
/// (0, 1)`; `v0 = 0`), [`StatlaError::Dimension`] (`v0`),
/// [`StatlaError::NonFinite`] (`v0`, or the operator returned NaN/∞),
/// [`StatlaError::Operator`], [`StatlaError::NoConvergence`].
pub fn eigsh<T, O>(
    op: &mut O,
    k: usize,
    which: Which,
    opts: &EigsOptions<T>,
) -> Result<SymEigen<T>, StatlaError>
where
    T: Real,
    O: LinearOperator<T> + ?Sized,
{
    let raw = run(op, k, which, opts, true, false)?;
    Ok(sym_result(raw))
}

fn sym_result<T: Real>(raw: Raw<T>) -> SymEigen<T> {
    SymEigen {
        values: raw.values.iter().map(|c| c.re).collect(),
        vectors: raw.vectors.map(|(r, _)| r),
        stats: raw.stats,
    }
}

/// `k` eigenpairs of a general real operator.
///
/// Algorithm KS of Chapter 12.6 (Krylov–Schur, Stewart 2001; S7): Arnoldi
/// with DGKS, real Schur form of the projected matrix by Francis
/// double-shift QR, reordering by Bai–Demmel swaps so that the wanted lead
/// (conjugate pairs never split), locking and purging by truncation,
/// eigenvectors by back substitution in the quasi-triangular form.
///
/// **Guarantee:** as [`eigsh()`]: every pair has `‖Ay − θy‖ ≤ (tol +
/// c_K·ncv·u)‖A‖` (`c_K = 8`), measured on runs of few restarts; long runs
/// (hundreds of restarts, see [`EigsStats::restarts`]) exceed it, and no
/// form in the number of restarts bounds every measured run (Chapter 12.6,
/// "Long runs"; Lemma R there bounds a residual the caller recomputes).
///
/// # Errors
/// As [`eigsh()`], with `k ≥ n − 1` invalid (ARPACK's `dnaupd` rule) and the
/// `which` set `LM SM LR SR LI SI`.
pub fn eigs<T, O>(
    op: &mut O,
    k: usize,
    which: Which,
    opts: &EigsOptions<T>,
) -> Result<Eigen<T>, StatlaError>
where
    T: Real,
    O: LinearOperator<T> + ?Sized,
{
    let raw = run(op, k, which, opts, false, false)?;
    Ok(gen_result(raw))
}

fn gen_result<T: Real>(raw: Raw<T>) -> Eigen<T> {
    let (vr, vi) = match raw.vectors {
        Some((r, i)) => (Some(r), Some(i)),
        None => (None, None),
    };
    Eigen {
        values: raw.values,
        vectors_re: vr,
        vectors_im: vi,
        stats: raw.stats,
    }
}

/// Lemma R of Chapter 12.6: `ρ̂_c = fl(‖A y_c − θ_c y_c‖₂)` for column `c`
/// of `(yr, yi)` (`yi` `None` or a zero column for a real value: one
/// application; two for a complex value).
fn recomputed<T, O>(
    op: &mut O,
    values: &[Complex<T>],
    yr: &DenseBlock<T>,
    yi: Option<&DenseBlock<T>>,
) -> Result<Vec<T>, StatlaError>
where
    T: Real,
    O: LinearOperator<T> + ?Sized,
{
    let n = yr.rows;
    if op.nrows() != n || op.ncols() != n {
        return Err(invalid(
            "residuals: the operator does not match the vectors",
        ));
    }
    let mut wr = vec![T::ZERO; n];
    let mut wi = vec![T::ZERO; n];
    let mut r = vec![T::ZERO; 2 * n];
    let mut out = Vec::with_capacity(values.len());
    for (c, th) in values.iter().enumerate() {
        let y = &yr.data[c * n..(c + 1) * n];
        let z = yi.map(|b| &b.data[c * n..(c + 1) * n]);
        op.apply(y, &mut wr).map_err(op_err)?;
        check_finite("operator output", &wr)?;
        let complex = z.is_some_and(|z| z.iter().any(|&v| !is_exact_zero(v)));
        if complex {
            let z = z.unwrap_or(&[]);
            op.apply(z, &mut wi).map_err(op_err)?;
            check_finite("operator output", &wi)?;
            for i in 0..n {
                r[i] = wr[i] - th.re * y[i] + th.im * z[i];
                r[n + i] = wi[i] - th.re * z[i] - th.im * y[i];
            }
            out.push(norm2(&r));
        } else {
            for i in 0..n {
                r[i] = wr[i] - th.re * y[i];
            }
            out.push(norm2(&r[..n]));
        }
    }
    Ok(out)
}

impl<T: Real> SymEigen<T> {
    /// The residuals `fl(‖A y − θ y‖₂)` of the returned pairs, recomputed
    /// with one application of `op` per pair (Lemma R of Chapter 12.6:
    /// a bound on the true residual that does not depend on the number of
    /// restarts). Not counted in [`SymEigen::stats`].
    ///
    /// # Errors
    /// [`StatlaError::InvalidArgument`] (no vectors, or `op` not of their
    /// order), [`StatlaError::NonFinite`], [`StatlaError::Operator`].
    pub fn residuals<O: LinearOperator<T> + ?Sized>(
        &self,
        op: &mut O,
    ) -> Result<Vec<T>, StatlaError> {
        let Some(v) = &self.vectors else {
            return Err(invalid("residuals: the eigenvectors were not computed"));
        };
        let values: Vec<Complex<T>> = self
            .values
            .iter()
            .map(|&re| Complex { re, im: T::ZERO })
            .collect();
        recomputed(op, &values, v, None)
    }
}

impl<T: Real> Eigen<T> {
    /// As [`SymEigen::residuals`], for complex pairs (`‖·‖₂` in `Cⁿ`; two
    /// applications per complex value).
    ///
    /// # Errors
    /// As [`SymEigen::residuals`].
    pub fn residuals<O: LinearOperator<T> + ?Sized>(
        &self,
        op: &mut O,
    ) -> Result<Vec<T>, StatlaError> {
        let Some(v) = &self.vectors_re else {
            return Err(invalid("residuals: the eigenvectors were not computed"));
        };
        recomputed(op, &self.values, v, self.vectors_im.as_ref())
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::operator::from_fn;
    use xingo_extprec::DoubleDouble as Dd;

    /// A symmetric `n × n` matrix `Q diag(d) Qᵀ` with three random reflectors.
    fn sym_matrix<T: Real>(d: &[f64], seed: u64) -> DenseBlock<T> {
        let n = d.len();
        let mut a = DenseBlock::<T>::zeros(n, n);
        for (i, &x) in d.iter().enumerate() {
            a.data[i + i * n] = T::from_f64(x);
        }
        let mut rng = Rng::new(seed);
        for _ in 0..3 {
            let w: Vec<T> = (0..n).map(|_| T::from_f64(rng.gaussian())).collect();
            let ww = dot(&w, &w);
            let two = T::from_f64(2.0);
            // A ← H A H.
            for j in 0..n {
                let col = &mut a.data[j * n..(j + 1) * n];
                let f = two * dot(col, &w) / ww;
                crate::small::axpy_sub(f, &w, col);
            }
            for i in 0..n {
                let s: T = (0..n).map(|j| a.data[i + j * n] * w[j]).sum();
                let f = two * s / ww;
                for j in 0..n {
                    a.data[i + j * n] -= f * w[j];
                }
            }
        }
        // Exact symmetry.
        for j in 0..n {
            for i in 0..j {
                let s = (a.data[i + j * n] + a.data[j + i * n]) * T::from_f64(0.5);
                a.data[i + j * n] = s;
                a.data[j + i * n] = s;
            }
        }
        a
    }

    fn residual<T: Real>(a: &DenseBlock<T>, y: &[T], yi: &[T], th: Complex<T>) -> f64 {
        let n = a.rows;
        let mut r = 0.0f64;
        for i in 0..n {
            let (mut sr, mut si) = (T::ZERO, T::ZERO);
            for j in 0..n {
                sr += a.data[i + j * n] * y[j];
                si += a.data[i + j * n] * yi[j];
            }
            let er = sr - (th.re * y[i] - th.im * yi[i]);
            let ei = si - (th.re * yi[i] + th.im * y[i]);
            r += (er * er + ei * ei).to_f64();
        }
        r.sqrt()
    }

    /// (K2) holds after every expansion; restarts and locking happen. The
    /// runs restart a few times only, which is where (K2) is stated (issue
    /// xingo-statla#9: long runs exceed it).
    #[test]
    fn krylov_relation_holds_after_every_expansion() {
        fn k2_of<T: Real>(sym: bool, u: f64) -> f64 {
            let d: Vec<f64> = (0..80).map(|j| 1.0 / (1.0 + 0.1 * f64::from(j))).collect();
            let mut a = sym_matrix::<T>(&d, 3);
            if !sym {
                // A nonsymmetric perturbation with complex eigenvalues.
                for j in 0..80 {
                    for i in 0..j {
                        a.data[i + j * 80] += T::from_f64(0.01 * ((i * 7 + j * 3) % 5) as f64);
                    }
                }
            }
            let raw = run(
                &mut a,
                6,
                Which::LargestMagnitude,
                &EigsOptions::default(),
                sym,
                true,
            )
            .unwrap();
            assert!(raw.stats.restarts > 0, "the test must restart");
            assert!(raw.locked > 0, "the test must lock");
            raw.k2 / (20.0 * u)
        }
        let u64_ = f64::EPSILON / 2.0;
        let udd = Dd::EPSILON.to_f64() / 2.0;
        let r = [
            k2_of::<f64>(true, u64_),
            k2_of::<f64>(false, u64_),
            k2_of::<Dd>(true, udd),
            k2_of::<Dd>(false, udd),
        ];
        eprintln!("(K2) / (ncv u): {r:.3?}");
        assert!(r.iter().all(|&x| x <= C_K), "{r:?}");
    }

    /// The five operators of task I09 (Chapter 12.6, "Long runs"), applied
    /// in `f64`: `(name, n, ncv, which, restart limit)` and the operator.
    #[allow(clippy::type_complexity)]
    fn i09_operators() -> Vec<(
        &'static str,
        usize,
        usize,
        Which,
        Option<usize>,
        Box<dyn FnMut(&[f64], &mut [f64])>,
    )> {
        fn tri(d: f64) -> Box<dyn FnMut(&[f64], &mut [f64])> {
            Box::new(move |x: &[f64], y: &mut [f64]| {
                let n = x.len();
                for i in 0..n {
                    let mut s = d * x[i];
                    if i > 0 {
                        s += -x[i - 1];
                    }
                    if i + 1 < n {
                        s += -x[i + 1];
                    }
                    y[i] = s;
                }
            })
        }
        fn reflected(d: Vec<f64>) -> Box<dyn FnMut(&[f64], &mut [f64])> {
            let n = d.len();
            let mut r = Rng::new(77);
            let ws: Vec<Vec<f64>> = (0..3)
                .map(|_| (0..n).map(|_| r.gaussian()).collect())
                .collect();
            let refl = |w: &[f64], x: &mut [f64]| {
                let f = 2.0 * dot(w, x) / dot(w, w);
                crate::small::axpy_sub(f, w, x);
            };
            Box::new(move |x: &[f64], y: &mut [f64]| {
                y.copy_from_slice(x);
                for w in &ws {
                    refl(w, y);
                }
                for (yi, di) in y.iter_mut().zip(&d) {
                    *yi *= di;
                }
                for w in ws.iter().rev() {
                    refl(w, y);
                }
            })
        }
        let g = 45;
        let lap2 = Box::new(move |x: &[f64], y: &mut [f64]| {
            for b in 0..g {
                for a in 0..g {
                    let i = a + b * g;
                    let mut s = 4.0 * x[i];
                    if a > 0 {
                        s -= x[i - 1];
                    }
                    if a + 1 < g {
                        s -= x[i + 1];
                    }
                    if b > 0 {
                        s -= x[i - g];
                    }
                    if b + 1 < g {
                        s -= x[i + g];
                    }
                    y[i] = s;
                }
            }
        });
        let clusters: Vec<f64> = (0..1000u32)
            .map(|i| 1.0 - 0.04 * f64::from(i / 50) + 1e-6 * f64::from(i % 50))
            .collect();
        let geo: Vec<f64> = (0..1000).map(|i| 0.995f64.powi(i)).collect();
        let lm = Which::LargestMagnitude;
        vec![
            ("1-D Laplacian", 2000, 24, lm, None, tri(2.0)),
            ("indefinite tridiagonal", 1000, 24, lm, None, tri(0.0)),
            ("2-D Laplacian", g * g, 14, lm, None, lap2),
            ("clusters", 1000, 24, lm, Some(3000), reflected(clusters)),
            (
                "geometric",
                1000,
                24,
                Which::SmallestAlgebraic,
                Some(3000),
                reflected(geo),
            ),
        ]
    }

    /// Issue xingo-statla#11: with the polish of KS3 (Algorithm SP) the
    /// relation (K2) of `eigsh` grows like `√R` on the five operators of task
    /// I09, where Algorithm SJ alone made it grow about linearly. The largest
    /// (K2) of the run (after every expansion, `paranoid`) is asserted
    /// against `c·m·u·√R`, `c = 0.5`; measured with the polish from restart
    /// 100 on, `0.152`, `0.188`, `0.141`, `0.314` and `0.041` (`0.164` at most
    /// polishing from the first restart); with SJ alone `c` was `2.09`,
    /// `1.43`, `0.60`, `4.32` and `0.19` (task I10), so the test fails on four
    /// of the five operators without the polish. Release builds only.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "thousands of restarts: release only")]
    fn long_runs_relation_grows_like_the_square_root_of_restarts() {
        let u = f64::EPSILON / 2.0;
        let mut worst = 0.0f64;
        for (name, n, ncv, which, limit, mut f) in i09_operators() {
            let mut op = from_fn(n, n, |x: &[f64], y: &mut [f64]| {
                f(x, y);
                Ok(())
            });
            let o = EigsOptions {
                ncv: Some(ncv),
                max_restarts: limit,
                allow_partial: true,
                ..EigsOptions::default()
            };
            let raw = run(&mut op, 10, which, &o, true, true).unwrap();
            let r = raw.stats.restarts as f64;
            assert!(r >= 300.0, "{name}: {r} restarts");
            let c = raw.k2 / (ncv as f64 * u * r.sqrt());
            eprintln!(
                "{name}: R = {r}, converged {}, max (K2) = {:.1} m u = {c:.3} m u √R",
                raw.stats.converged,
                raw.k2 / (ncv as f64 * u)
            );
            worst = worst.max(c);
        }
        assert!(worst <= LONG_RUN_K2, "{worst}");
    }

    /// The constant of the test above (`c` in `c·m·u·√R`).
    const LONG_RUN_K2: f64 = 0.5;

    /// The prescribed spectrum is found, in every `which`, to `ε‖A‖`.
    #[test]
    fn symmetric_which_set() {
        let d: Vec<f64> = (0..60).map(|j| -3.0 + 0.1 * f64::from(j)).collect();
        let mut a = sym_matrix::<f64>(&d, 7);
        let mut sorted = d.clone();
        sorted.sort_by(f64::total_cmp);
        let cases: [(Which, Vec<f64>); 5] = [
            (Which::LargestAlgebraic, vec![2.9, 2.8, 2.7, 2.6]),
            (Which::SmallestAlgebraic, vec![-3.0, -2.9, -2.8, -2.7]),
            (Which::LargestMagnitude, vec![-3.0, 2.9, -2.9, 2.8]),
            (Which::SmallestMagnitude, vec![0.0, -0.1, 0.1, -0.2]),
            (Which::BothEnds, vec![2.9, -3.0, 2.8, -2.9]),
        ];
        for (w, want) in cases {
            let f = eigsh(&mut a, 4, w, &EigsOptions::default()).unwrap();
            // Ties in |λ| (±0.1, ±2.9) may come in either order: compare sorted.
            let (mut got_s, mut want_s) = (f.values.clone(), want.clone());
            if matches!(w, Which::LargestMagnitude | Which::SmallestMagnitude) {
                got_s.sort_by(f64::total_cmp);
                want_s.sort_by(f64::total_cmp);
            }
            for (got, exp) in got_s.iter().zip(&want_s) {
                assert!((got - exp).abs() < 1e-13, "{w:?}: {:?}", f.values);
            }
            let y = f.vectors.as_ref().unwrap();
            for (c, &th) in f.values.iter().enumerate() {
                let col = &y.data[c * 60..(c + 1) * 60];
                let r = residual(&a, col, &[0.0; 60], Complex { re: th, im: 0.0 });
                assert!(r < 1e-13, "{w:?} {c}: {r}");
            }
        }
    }

    /// A block-diagonal matrix of rotations-and-scalings: conjugate pairs,
    /// `+i` first, adjacent; residuals small; every `which`.
    #[test]
    fn general_which_set_and_pairs() {
        let n = 40;
        let mut a = DenseBlock::<f64>::zeros(n, n);
        // Blocks [r cosφ, −r sinφ; r sinφ, r cosφ] and real diagonal entries.
        let mut exact: Vec<(f64, f64)> = Vec::new();
        let mut i = 0;
        let mut t = 0;
        while i < n {
            if t % 3 == 0 && i + 1 < n {
                let (re, im) = (0.5 + 0.07 * f64::from(t), 0.2 + 0.05 * f64::from(t));
                a.data[i + i * n] = re;
                a.data[(i + 1) + (i + 1) * n] = re;
                a.data[i + (i + 1) * n] = -im;
                a.data[(i + 1) + i * n] = im;
                exact.push((re, im));
                exact.push((re, -im));
                i += 2;
            } else {
                let re = -1.0 + 0.09 * f64::from(t);
                a.data[i + i * n] = re;
                exact.push((re, 0.0));
                i += 1;
            }
            t += 1;
        }
        // Similarity by a fixed orthogonal-ish mixing (a reflector).
        let w: Vec<f64> = (0..n).map(|j| 1.0 + (j % 7) as f64).collect();
        let ww = dot(&w, &w);
        for j in 0..n {
            let col = &mut a.data[j * n..(j + 1) * n];
            let f = 2.0 * dot(col, &w) / ww;
            crate::small::axpy_sub(f, &w, col);
        }
        for r in 0..n {
            let s: f64 = (0..n).map(|j| a.data[r + j * n] * w[j]).sum();
            let f = 2.0 * s / ww;
            for j in 0..n {
                a.data[r + j * n] -= f * w[j];
            }
        }
        for which in [
            Which::LargestMagnitude,
            Which::SmallestMagnitude,
            Which::LargestReal,
            Which::SmallestReal,
            Which::LargestImag,
            Which::SmallestImag,
        ] {
            let f = eigs(&mut a, 5, which, &EigsOptions::default()).unwrap();
            assert_eq!(f.values.len(), 5);
            let yr = f.vectors_re.as_ref().unwrap();
            let yi = f.vectors_im.as_ref().unwrap();
            for (c, th) in f.values.iter().enumerate() {
                let r = residual(
                    &a,
                    &yr.data[c * n..(c + 1) * n],
                    &yi.data[c * n..(c + 1) * n],
                    *th,
                );
                assert!(r < 1e-12, "{which:?} {c}: {r}");
                assert!(
                    exact
                        .iter()
                        .any(|e| (e.0 - th.re).abs() < 1e-12 && (e.1 - th.im).abs() < 1e-12),
                    "{which:?}: {th:?}"
                );
                if th.im > 0.0 && c + 1 < f.values.len() {
                    assert_eq!(f.values[c + 1].re, th.re);
                    assert_eq!(f.values[c + 1].im, -th.im);
                }
                assert!(th.im >= 0.0 || (c > 0 && f.values[c - 1].im == -th.im));
            }
            // The order of `which`.
            for w2 in f.values.windows(2) {
                let (a0, b0) = ((w2[0].re, w2[0].im.abs()), (w2[1].re, w2[1].im.abs()));
                assert!(!before(which, b0, a0), "{which:?}: {:?}", f.values);
            }
        }
    }

    #[test]
    fn arpack_codes_round_trip() {
        for c in ["LM", "SM", "LA", "SA", "BE", "LR", "SR", "LI", "SI"] {
            assert_eq!(Which::from_arpack(c).unwrap().arpack_code(), c);
        }
        assert_eq!(Which::from_arpack("lm"), None);
        assert_eq!(Which::from_arpack("XX"), None);
        let e = eps23::<f64>();
        let r = (e / f64::EPSILON.powf(2.0 / 3.0) - 1.0).abs();
        assert!(r < 1e-14, "{e} {r}");
    }

    #[test]
    #[allow(clippy::too_many_lines)] // one assertion per error path
    fn options_and_errors() {
        let d: Vec<f64> = (0..10).map(f64::from).collect();
        let mut a = sym_matrix::<f64>(&d, 1);
        let o = EigsOptions::<f64>::default();
        let inv = |r: Result<SymEigen<f64>, StatlaError>| {
            matches!(r, Err(StatlaError::InvalidArgument { .. }))
        };
        assert!(inv(eigsh(&mut a, 0, Which::LargestMagnitude, &o)));
        assert!(inv(eigsh(&mut a, 10, Which::LargestMagnitude, &o)));
        assert!(inv(eigsh(&mut a, 2, Which::LargestReal, &o)));
        assert!(matches!(
            eigs(&mut a, 9, Which::LargestMagnitude, &o),
            Err(StatlaError::InvalidArgument { .. })
        ));
        assert!(matches!(
            eigs(&mut a, 2, Which::BothEnds, &o),
            Err(StatlaError::InvalidArgument { .. })
        ));
        for bad in [
            EigsOptions {
                ncv: Some(2),
                ..o.clone()
            },
            EigsOptions {
                ncv: Some(11),
                ..o.clone()
            },
            EigsOptions {
                tol: -1.0,
                ..o.clone()
            },
            EigsOptions {
                tol: f64::INFINITY,
                ..o.clone()
            },
            EigsOptions {
                reorth_eta: 1.0,
                ..o.clone()
            },
            EigsOptions {
                v0: Some(vec![0.0; 10]),
                ..o.clone()
            },
        ] {
            assert!(
                inv(eigsh(&mut a, 2, Which::LargestMagnitude, &bad)),
                "{bad:?}"
            );
        }
        assert!(matches!(
            eigsh(
                &mut a,
                2,
                Which::LargestMagnitude,
                &EigsOptions {
                    v0: Some(vec![1.0; 3]),
                    ..o.clone()
                }
            ),
            Err(StatlaError::Dimension { what: "v0", .. })
        ));
        assert!(matches!(
            eigsh(
                &mut a,
                2,
                Which::LargestMagnitude,
                &EigsOptions {
                    v0: Some(vec![f64::NAN; 10]),
                    ..o.clone()
                }
            ),
            Err(StatlaError::NonFinite {
                what: "v0",
                index: 0
            })
        ));
        let mut rect = DenseBlock::<f64>::zeros(4, 3);
        assert!(matches!(
            eigsh(&mut rect, 1, Which::LargestMagnitude, &o),
            Err(StatlaError::InvalidArgument { .. })
        ));
        // Operator failure and NaN output.
        let mut failing = from_fn(10, 10, |_x: &[f64], _y: &mut [f64]| {
            Err(crate::OperatorError {
                message: "boom".into(),
            })
        });
        assert!(matches!(
            eigsh(&mut failing, 2, Which::LargestMagnitude, &o),
            Err(StatlaError::Operator(_))
        ));
        let mut nan = from_fn(10, 10, |_x: &[f64], y: &mut [f64]| {
            y.fill(f64::NAN);
            Ok(())
        });
        assert!(matches!(
            eigs(&mut nan, 2, Which::LargestMagnitude, &o),
            Err(StatlaError::NonFinite { .. })
        ));
        // Restart limit: NoConvergence, or partial.
        let dd: Vec<f64> = (0..200).map(|j| 1.0 + 1e-3 * f64::from(j)).collect();
        let mut slow = sym_matrix::<f64>(&dd, 5);
        let lim = EigsOptions {
            max_restarts: Some(1),
            ..o.clone()
        };
        assert!(matches!(
            eigsh(&mut slow, 6, Which::SmallestMagnitude, &lim),
            Err(StatlaError::NoConvergence { requested: 6, .. })
        ));
        let part = eigsh(
            &mut slow,
            6,
            Which::SmallestMagnitude,
            &EigsOptions {
                allow_partial: true,
                ..lim
            },
        )
        .unwrap();
        assert!(part.values.len() < 6);
        assert_eq!(part.stats.converged, part.values.len());
        // ncv = n: one pass, exact; vectors off.
        let f = eigsh(
            &mut a,
            3,
            Which::LargestAlgebraic,
            &EigsOptions {
                ncv: Some(10),
                vectors: false,
                ..o
            },
        )
        .unwrap();
        assert_eq!(f.stats.restarts, 0);
        assert!(f.vectors.is_none());
        for (g, e) in f.values.iter().zip([9.0, 8.0, 7.0]) {
            assert!((g - e).abs() < 1e-13);
        }
    }

    /// Determinism: the same bits twice; an invariant start vector
    /// (an eigenvector) breaks down and recovers from the seed's stream.
    #[test]
    fn reproducible_and_breakdown_safe() {
        let d: Vec<f64> = (0..30).map(|j| f64::from(j % 10)).collect();
        let mut a = DenseBlock::<f64>::zeros(30, 30);
        for (i, &x) in d.iter().enumerate() {
            a.data[i + i * 30] = x;
        }
        let mut e0 = vec![0.0; 30];
        e0[3] = 1.0;
        let o = EigsOptions {
            v0: Some(e0),
            ..EigsOptions::default()
        };
        let f = eigsh(&mut a, 3, Which::LargestAlgebraic, &o).unwrap();
        for g in &f.values {
            assert!((g - 9.0).abs() < 1e-14, "{:?}", f.values);
        }
        let g = eigsh(&mut a, 3, Which::LargestAlgebraic, &o).unwrap();
        assert_eq!(
            f.values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            g.values.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
}
