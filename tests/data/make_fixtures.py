"""Fixtures of the acceptance tests of Chapter 12.6 (task L3.4) and of
criterion (iii) of Chapter 12.5 (`svds`, moved from L3.3).

Generates a few small Matrix Market matrices (no download: every matrix is
built here from a formula or a fixed seed) and, for each, a reference file
`<name>.ref` with ARPACK's answers through `scipy.sparse.linalg.eigsh`,
`eigs` and `svds` and the dense spectrum from LAPACK through NumPy. The
references are computed from the matrix *as read back* from the `.mtx`
file (17 significant digits, so the Rust tests read the same bits).

Run from this directory (the outputs are committed; CI needs no Python):

    uv run -q --with numpy --with scipy python3 make_fixtures.py
    uv run -q --with numpy --with scipy python3 make_fixtures.py --large   # the large ones only

Reference file format, one record per line, values with `repr` (17
significant digits, exact round trip):

    norm2 <‖A‖₂>                          (largest singular value, dense)
    dense <λ…>                            (symmetric: all eigenvalues, ascending)
    dense_re <…> / dense_im <…>           (general: all eigenvalues)
    sv_dense <σ…>                         (all singular values, descending)
    eigsh <which> <k> <λ…>                (ARPACK, scipy's order)
    eigsh_near <σ> <k> <λ…>               (ARPACK mode 3)
    eigs <which> <k> <re im …>
    eigs_near <σ> <k> <re im …>
    svds <k> <σ…>                         (descending)
    noconv <what>                         (ARPACK did not converge: no record)

The large fixtures (`--large`, issue xingo-statla#2) have no `.mtx` and
replace `dense`/`sv_dense` by `checksum`, `dense_near`, `dense_near_re`,
`dense_near_im`, `sv_top` and `count_below <σ>:<count>…` (see
`write_large`).
"""

import math
import sys

import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import scipy.sparse.linalg as sla

K = 6


def lap2d(nx, ny):
    """5-point Laplacian on an nx × ny grid (Dirichlet), symmetric."""
    tx = sp.diags([-np.ones(nx - 1), 2 * np.ones(nx), -np.ones(nx - 1)], [-1, 0, 1])
    ty = sp.diags([-np.ones(ny - 1), 2 * np.ones(ny), -np.ones(ny - 1)], [-1, 0, 1])
    return (sp.kron(sp.identity(ny), tx) + sp.kron(ty, sp.identity(nx))).tocsc()


def symrand(n, density, seed):
    """Random sparse symmetric indefinite matrix (fixed seed)."""
    rng = np.random.default_rng(seed)
    s = sp.random(n, n, density=density, random_state=rng, data_rvs=rng.standard_normal)
    d = sp.diags(rng.standard_normal(n))
    return (s + s.T + d).tocsc()


def tridiag_graded(n):
    """tridiag(−1, 2 + i/50, −1): symmetric positive definite, graded."""
    d = 2.0 + np.arange(n) / 50.0
    return sp.diags([-np.ones(n - 1), d, -np.ones(n - 1)], [-1, 0, 1]).tocsc()


def convdiff(nx, ny, px, py):
    """Central-difference convection–diffusion −Δu + β·∇u on a grid:
    `4` on the diagonal, `−1 ∓ p` off it (cell Péclet numbers p < 1: real
    eigenvalues, a nonnormal matrix)."""
    tx = sp.diags([(-1 - px) * np.ones(nx - 1), 2 * np.ones(nx), (-1 + px) * np.ones(nx - 1)], [-1, 0, 1])
    ty = sp.diags([(-1 - py) * np.ones(ny - 1), 2 * np.ones(ny), (-1 + py) * np.ones(ny - 1)], [-1, 0, 1])
    return (sp.kron(sp.identity(ny), tx) + sp.kron(ty, sp.identity(nx))).tocsc()


def nonsymrand(n, density, seed):
    """Random sparse nonsymmetric matrix plus a diagonal shift: complex
    eigenvalues (circular law), fixed seed."""
    rng = np.random.default_rng(seed)
    s = sp.random(n, n, density=density, random_state=rng, data_rvs=rng.standard_normal)
    return (s + sp.diags(0.3 * rng.standard_normal(n))).tocsc()


def rectrand(m, n, density, seed):
    rng = np.random.default_rng(seed)
    return sp.random(m, n, density=density, random_state=rng, data_rvs=rng.standard_normal).tocsc()


def fmt(xs):
    return " ".join(repr(float(x)) for x in xs)


def write(name, a, symmetric, near_sym=(), near_gen=(), which_sym=("LM", "SM", "LA", "SA", "BE"),
          which_gen=("LM", "SM", "LR", "SR", "LI", "SI")):
    path = f"{name}.mtx"
    scipy.io.mmwrite(path, a, symmetry="symmetric" if symmetric else "general", precision=17,
                     comment="xingo-statla L3.4 fixture (make_fixtures.py)")
    a = sp.csc_matrix(scipy.io.mmread(path))
    dense = a.toarray()
    lines = [f"# {name}: {a.shape[0]} x {a.shape[1]}, nnz {a.nnz}; numpy {np.__version__}, scipy {scipy.__version__}"]
    sv = np.linalg.svd(dense, compute_uv=False)
    lines.append(f"norm2 {fmt([sv[0]])}")
    lines.append(f"sv_dense {fmt(sv)}")
    square = a.shape[0] == a.shape[1]
    if square and symmetric:
        lam = np.linalg.eigvalsh(dense)
        lines.append(f"dense {fmt(lam)}")
        for w in which_sym:
            try:
                v = sla.eigsh(a, k=K, which=w, tol=0, return_eigenvectors=False)
                lines.append(f"eigsh {w} {K} {fmt(v)}")
            except sla.ArpackNoConvergence:
                lines.append(f"noconv eigsh {w}")
        for s in near_sym:
            v = sla.eigsh(a, k=K, sigma=s, which="LM", tol=0, return_eigenvectors=False)
            lines.append(f"eigsh_near {s!r} {K} {fmt(v)}")
    elif square:
        lam = np.linalg.eigvals(dense)
        lines.append(f"dense_re {fmt(lam.real)}")
        lines.append(f"dense_im {fmt(lam.imag)}")
        for w in which_gen:
            try:
                v = sla.eigs(a, k=K, which=w, tol=0, return_eigenvectors=False)
                lines.append(f"eigs {w} {K} {fmt(np.column_stack([v.real, v.imag]).ravel())}")
            except sla.ArpackNoConvergence:
                lines.append(f"noconv eigs {w}")
        for s in near_gen:
            v = sla.eigs(a, k=K, sigma=s, which="LM", tol=0, return_eigenvectors=False)
            lines.append(f"eigs_near {s!r} {K} {fmt(np.column_stack([v.real, v.imag]).ravel())}")
    s = sla.svds(a, k=K, tol=0, return_singular_vectors=False, solver="arpack")
    lines.append(f"svds {K} {fmt(sorted(s, reverse=True))}")
    with open(f"{name}.ref", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(name, a.shape, a.nnz, file=sys.stderr)


def write_large(name, a, symmetric, near=(), which_sym=("LM", "LA", "SA", "BE"),
                which_gen=("LM", "LR", "SR")):
    """A fixture of a few thousand rows built from a formula that the Rust
    tests rebuild bit for bit (issue xingo-statla#2): no `.mtx` (it would
    weigh megabytes), only `<name>.ref`, whose `checksum` record — the
    exactly rounded `Σ a_ij (i + 1)(2j + 1)` (`math.fsum`) — lets the
    tests check that they rebuilt the same matrix. Instead of the whole
    dense spectrum the reference keeps what the tests use: `dense_near`
    (symmetric) or `dense_near_re`/`dense_near_im` (general), the union of
    the 8 dense eigenvalues nearest each ARPACK value (the gap of criterion
    (a)), and `sv_top`, the `2K` largest dense singular values (the gap of
    criterion (d))."""
    a = sp.csc_matrix(a)
    coo = a.tocoo()
    checksum = math.fsum(float(v) * (int(i) + 1) * (2 * int(j) + 1)
                         for i, j, v in zip(coo.row, coo.col, coo.data))
    dense = a.toarray()
    lines = [f"# {name}: {a.shape[0]} x {a.shape[1]}, nnz {a.nnz}; numpy {np.__version__}, "
             f"scipy {scipy.__version__} (formula fixture, no .mtx)"]
    lines.append(f"checksum {checksum!r}")
    sv = np.linalg.svd(dense, compute_uv=False)
    lines.append(f"norm2 {fmt([sv[0]])}")
    lines.append(f"sv_top {fmt(sv[:2 * K])}")
    found = []
    if symmetric:
        lam = np.linalg.eigvalsh(dense)
        for w in which_sym:
            v = sla.eigsh(a, k=K, which=w, tol=0, return_eigenvectors=False)
            lines.append(f"eigsh {w} {K} {fmt(v)}")
            found.extend(v)
        for s in near:
            v = sla.eigsh(a, k=K, sigma=s, which="LM", tol=0, return_eigenvectors=False)
            lines.append(f"eigsh_near {s!r} {K} {fmt(v)}")
            found.extend(v)
        keep = set()
        for x in found:
            keep.update(np.argsort(np.abs(lam - x))[:8].tolist())
        lines.append(f"dense_near {fmt(lam[sorted(keep)])}")
        lines.append(f"count_below {' '.join(f'{s!r}:{int(np.sum(lam < s))}' for s in near)}")
    else:
        lam = np.linalg.eigvals(dense)
        for w in which_gen:
            v = sla.eigs(a, k=K, which=w, tol=0, return_eigenvectors=False)
            lines.append(f"eigs {w} {K} {fmt(np.column_stack([v.real, v.imag]).ravel())}")
            found.extend(v)
        for s in near:
            v = sla.eigs(a, k=K, sigma=s, which="LM", tol=0, return_eigenvectors=False)
            lines.append(f"eigs_near {s!r} {K} {fmt(np.column_stack([v.real, v.imag]).ravel())}")
            found.extend(v)
        keep = set()
        for x in found:
            keep.update(np.argsort(np.abs(lam - x))[:8].tolist())
        sel = lam[sorted(keep)]
        lines.append(f"dense_near_re {fmt(sel.real)}")
        lines.append(f"dense_near_im {fmt(sel.imag)}")
    s = sla.svds(a, k=K, tol=0, return_singular_vectors=False, solver="arpack")
    lines.append(f"svds {K} {fmt(sorted(s, reverse=True))}")
    with open(f"{name}.ref", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(name, a.shape, a.nnz, file=sys.stderr)


def main():
    if "--large" in sys.argv:
        # Issue xingo-statla#2: a few thousand rows (minutes of dense
        # LAPACK); only these, so that the small fixtures keep their bits.
        write_large("lap2d_60x50", lap2d(60, 50), True, near=(1.0,))
        write_large("tridiag_graded_4000", tridiag_graded(4000), True, near=(3.0,))
        write_large("convdiff_60x50", convdiff(60, 50, 0.1, 0.05), False, near=(4.0,))
        return
    write("lap2d_20x17", lap2d(20, 17), True, near_sym=(1.05, 3.3))
    write("symrand_300", symrand(300, 0.01, 7), True, near_sym=(0.0, 1.5))
    write("tridiag_graded_200", tridiag_graded(200), True, near_sym=(3.0,), which_sym=("LM", "LA", "SA", "BE"))
    write("convdiff_20x20", convdiff(20, 20, 0.3, 0.15), False, near_gen=(4.0,))
    write("nonsymrand_200", nonsymrand(200, 0.03, 11), False, near_gen=(0.25, -0.5))
    write("rectrand_300x200", rectrand(300, 200, 0.02, 5), False)


if __name__ == "__main__":
    main()
