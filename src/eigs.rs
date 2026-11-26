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
//!      block (Algorithm SJ of 12.6.1); in `eigs`, the standardized real
//!      Schur form by Hessenberg reduction and Francis double-shift QR
//!      (HR + FQ). Then `b_A ← b_Aᵀ Z_A`, `H_LA ← H_LA Z_A`.
//! KS4. \[Ritz.\] The eigenvalues `θ` of the locked blocks and of `T_A`;
//!      for an active block the eigenvector `x` of `T_A` (Algorithm EV) gives
//!      the estimate `ρ = |b_Aᵀ x|/‖x‖` (≥ the true one: the locked
//!      components of the eigenvector of `H` are left out). The candidates
//!      are ordered by `which` (a conjugate pair counts as two values,
//!      adjacent, `+i` first; `BE` alternates the high and the low end,
//!      high first; ties are broken by ARPACK's secondary key — `LM`↔`LR`,
//!      `SM`↔`SR`, `LI`→`LM`, `SI`→`SM` — then by position); the
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
//!      `33 648u` after 10 000 restarts at `j = 24`). No form in `R` bounds every measured
//!      run (issue xingo-statla#9, "Guarantee").
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
//! distance of `θ` to the spectrum (Parlett, Thm. 4.5.1); for a general
//! `A` the eigenvalue error is that times its condition number.
//!
//! **Long runs** (issue xingo-statla#9, task I02). The rounding term grows
//! with the number `R` of restarts ([`EigsStats::restarts`]), because (K2)
//! does, and **no closed form in `R` is guaranteed**: two candidates fitted
//! to the first measurements were each exceeded by a later one —
//! `c_K(1 + √R/8) = 8 + √R` by `eigs` on a clustered spectrum (`102.9`
//! against `62.8`), `c_K(1 + √R/4) = 8 + 2√R` by `eigsh` on an indefinite
//! tridiagonal (`125.5` against `117.2`). After hundreds of restarts a
//! caller that needs the residual should recompute it, and Lemma R below
//! bounds the recomputed value whatever `R`. Measured in `f64`,
//! ratio `(‖r‖/‖A‖ − τ)/(m u)` against `√(‖A‖₁‖A‖_∞)`, the worst returned
//! pair (`allow_partial` where fewer than `k` converged, marked `*`):
//!
//! | driver | matrix | which | `k`, `m` | `R` | ratio | `8 + √R` | `8 + 2√R` |
//! |---|---|---|---|---:|---:|---:|---:|
//! | `eigs` | Grcar(100, 3) | LM | 16, 40 | 108 | 3.78 | 18.4 | 28.8 |
//! | `eigs` | Grcar(200, 3) | LM | 16, 40 | 988 | 12.10 | 39.4 | 70.9 |
//! | `eigs` | Grcar(200, 3) | LI | 16, 40 | 142 | 3.81 | 19.9 | 31.8 |
//! | `eigs` | Grcar(400, 3) | LR | 10, 24 | 1490 | 18.86 | 46.6 | 85.2 |
//! | `eigs` | Grcar(100, 3) | LM | 10, 24 `*` | 2000 | 22.14 | 52.7 | 97.4 |
//! | `eigs` | 1-D Laplacian, 2000 | LM | 10, 24 | 1271 | 13.82 | 43.7 | 79.3 |
//! | `eigs` | clusters of 50, 1000 | LM | 10, 24 `*` | 3000 | **102.86** | 62.8 | 117.5 |
//! | `eigsh` | 1-D Laplacian, 1000 | BE | 10, 24 `*` | 10000 | 31.64 | 108.0 | 208.0 |
//! | `eigsh` | 1-D Laplacian, 2000 | LM | 10, 24 | 1221 | 48.59 | 42.9 | 77.9 |
//! | `eigsh` | 1-D Laplacian, 3000 | LA | 10, 24 | 2659 | 93.50 | 59.6 | 111.1 |
//! | `eigsh` | 2-D Laplacian, 45² | LM | 10, 14 | 1984 | 18.37 | 52.5 | 97.1 |
//! | `eigsh` | 2-D Laplacian, 60² | SA | 10, 16 `*` | 3000 | 1.42 | 62.8 | 117.5 |
//! | `eigsh` | clusters of 50, 1000 | LM | 10, 24 | 2485 | 21.56 | 57.8 | 107.7 |
//! | `eigsh` | indefinite tridiagonal, 1000 | LM | 10, 24 | 880 | 36.70 | 37.7 | 67.3 |
//! | `eigsh` | indefinite tridiagonal, 2000 | BE | 10, 24 | 2983 | **125.53** | 62.6 | 117.2 |
//! | `eigsh` | graded tridiagonal, 1000 | LM | 10, 14 | 128 | 1.39 | 19.3 | 30.6 |
//! | `eigsh_near` | 1-D Laplacian, 2000, `σ = 1.0001` | — | 10, 12 | 189 | 0.71 | 21.7 | 35.5 |
//! | `eigsh_near` | 2-D Laplacian, 45², `σ = 2.0001` | — | 10, 12 | 141 | 0.23 | 19.9 | 31.7 |
//! | `eigs_near` | 1-D Laplacian, 2000, `σ = 1.0001` | — | 10, 12 | 189 | 0.74 | 21.7 | 35.5 |
//!
//! (The 1-D Laplacian is `tridiag(−1, 2, −1)`; the indefinite tridiagonal
//! is that minus `2I`; the graded one has diagonal `(1 − i/n)⁴` and
//! off-diagonal `0.1√(d_i d_{i+1})`; the clusters are 20 groups of 50
//! eigenvalues `1 − 0.04c + 10⁻⁶j`, rotated by three reflectors.) The
//! shift-invert drivers converge fast on `(A − σI)⁻¹` and stay below `1`.
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
//! *Why `eigsh` grows faster than `eigs`* (issue xingo-statla#10, task I03;
//! measured along the runs in `f64`, `‖A V − V H̄‖_F/‖A‖_est` over the kept
//! columns after each restart, in units of `u`; 1-D Laplacian of order
//! 2000 and the indefinite tridiagonal of order 1000, `LM`, `k = 10`,
//! `m = 24`). In `eigs` the relation error grows like `√R`; in `eigsh` it
//! grows about **linearly**, and the final ratio of criterion (b) is three
//! times larger on the same matrix:
//!
//! | run | (K2) at `R` = 100 / 300 / 600 / 900 | `R` | ratio |
//! |---|---|---:|---:|
//! | `eigs`, Laplacian | 499 / 743 / 967 / 1168 | 1271 | 13.8 |
//! | `eigsh`, Laplacian | 268 / 778 / 1937 / 3240 | 1221 | 48.6 |
//! | `eigsh` with HR + FQ in KS3, Laplacian | 504 / 975 / 1111 / 1128 | 1275 | 17.6 |
//! | `eigs`, indefinite | 339 / 707 / 1020 / 1084 | 983 | 21.9 |
//! | `eigsh`, indefinite | 226 / 864 / 2058 / — | 880 | 36.7 |
//! | `eigsh` with HR + FQ in KS3, indefinite | 348 / 603 / 847 / 1086 | 926 | 15.0 |
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
//! by the Rayleigh quotients `z_iᵀ H_AA z_i` (`51.9`). Using HR + FQ in
//! `eigsh` changes its results and its number of restarts, so it is a
//! proposal (task I03), not a change of this edition.
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
//! *Proof.* Entrywise, `r̂ = (ŵ − θ y) + e` with `|e| ≤ γ_3(|ŵ| + |a||y|
//! + |b||y|)` (each entry is a sum of three terms, two of them products,
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
//! number of applications, [`EigsStats::applies`]); how to offer it is
//! issue xingo-statla#10's open decision.
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
//! ratio `2.8`; 3 s in release, ignored in debug builds).
//!
//! ## Divergences from the references
//! * Restart by Krylov–Schur, not implicit QR (S7). Results are compared
//!   with ARPACK, not traces.
//! * The symmetric small solver is cyclic Jacobi, not tridiagonal QL (the
//!   projected matrix after a restart is an arrowhead, not tridiagonal; and
//!   Jacobi gives orthogonal eigenvectors to `O(ε)` in every case).
//! * Exactly `k` values are returned, as scipy does: when the `k`-th value
//!   is the first member of a conjugate pair, its conjugate is omitted
//!   (question Q-L34-1).
//! * `BE` returns the values in its own wanted order (high, low, high, …).
//!
//! ## References
//! Stewart, "A Krylov–Schur algorithm for large eigenproblems", SIMAX 23
//! (2001); Wu & Simon, "Thick-restart Lanczos method for large symmetric
//! eigenvalue problems", SIMAX 22 (2000); Lehoucq, Sorensen & Yang, *ARPACK
//! Users' Guide*, SIAM (1998) (`dsaupd`/`dnaupd`: `which`, `ncv`, the test
//! `‖r‖ ≤ tol·max(ε^{2/3}, |θ|)`, `nev′`); Sorensen, SIMAX 13 (1992);
//! Parlett, *The Symmetric Eigenvalue Problem* (1998); Daniel, Gragg,
//! Kaufman & Stewart (1976). Cited from memory.

use xingo_sparse::{DenseBlock, Real};

use crate::StatlaError;
use crate::error::op_err;
use crate::krylov::{expand_arnoldi, kept_dimension};
use crate::operator::LinearOperator;
use crate::random::Rng;
use crate::real::{is_exact_zero, le_exact, lt};
use crate::schur::{
    Cx, SchurOptions, block_value, blocks, hypot, identity, real_schur, reorder, schur_eigvec,
    sym_eig,
};
use crate::small::{check_finite, check_len, dot, norm2, orthogonalize_against};

/// The constant `c_K` of invariants (K1)–(K2) and of the residual guarantee
/// (measured in task L3.4, with a margin).
pub(crate) const C_K: f64 = 8.0;

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
/// before the stable sort by `which` (`LM`↔`LR`, `SM`↔`SR`, `LI`→`LM`,
/// `SI`→`SM`), so that equal keys (a spectrum of real values under `LI`,
/// `±λ` under `LM`) are resolved the same way at every restart.
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
            let (d, z) = sym_eig(&ha, so)?;
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
/// reorthogonalization, Jacobi on the projected arrowhead, Krylov–Schur
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
