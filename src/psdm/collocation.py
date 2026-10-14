"""Orthogonal collocation operators for the particle (radial) and bed (axial) coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse, special


@dataclass(frozen=True)
class Collocation:
    """Discretisation operators.

    Attributes
    ----------
    nr
        Radial collocation points, including the particle surface.
    nz
        Axial points in the whole bed (``ne * nz_element - (ne - 1)``).
    wr
        Radial quadrature weights, shape ``(nr,)``.
    br
        Radial Laplacian operator for a sphere, shape ``(nr, nr)``.
    az
        Axial first-derivative operator, shape ``(nz, nz)``.
    """

    nr: int
    nz: int
    wr: np.ndarray
    br: np.ndarray
    az: np.ndarray


def _radial(nr: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric (sphere) polynomials in ``r**2`` at Jacobi roots plus the surface."""
    a = 3  # spherical geometry
    rr = 0.5 * (special.roots_jacobi(nr - 1, 1.0, 0.5)[0] + 1.0)
    r = np.sqrt(np.append(rr, 1.0))

    powers = np.arange(nr)
    q = r[:, None] ** (2 * powers)
    d = np.zeros((nr, nr))
    for i in range(1, nr):
        d[:, i] = (2 * i) * (2 * i - 2 + a) * r ** (2 * i - 2)
    f = 1.0 / (2.0 * powers + a)
    qinv = np.linalg.inv(q)
    return f @ qinv, d @ qinv


def _lagrange_derivative(x: np.ndarray) -> np.ndarray:
    """First-derivative collocation matrix on points ``x`` via the monomial basis."""
    n = len(x)
    powers = np.arange(n)
    q = x[:, None] ** powers
    c = np.zeros((n, n))
    c[:, 1:] = powers[1:] * x[:, None] ** (powers[1:] - 1)
    return c @ np.linalg.inv(q)


def _axial_single(nz: int) -> np.ndarray:
    z = np.concatenate([[0.0], special.roots_sh_legendre(nz - 2)[0], [1.0]])
    # monomials in (2z - 1) are better conditioned than in z; the operator is identical
    s = 2.0 * z - 1.0
    return 2.0 * _lagrange_derivative(s)


def _axial_elements(ne: int, nz: int) -> np.ndarray:
    """Orthogonal collocation on ``ne`` finite elements, continuity at shared nodes."""
    a = _lagrange_derivative(np.concatenate([[0.0], special.roots_sh_legendre(nz - 2)[0], [1.0]]))
    z = ne * a
    n_total = ne * nz - (ne - 1)
    op = np.zeros((n_total, n_total))
    for k in range(ne):
        lo = k * (nz - 1)
        op[lo : lo + nz, lo : lo + nz] = z
    # rows where elements meet: enforce continuity of the first derivative
    q = a[-1, -1] - a[0, 0]
    for k in range(ne - 1):
        start = k * (nz - 1)
        shared = (k + 1) * (nz - 1)
        op[shared, :] = 0.0
        op[shared, start:shared] = z[-1, :-1] - z[-1, -1] / q * a[-1, :-1]
        op[shared, shared + 1 : shared + nz] = z[-1, -1] / q * a[0, 1:]
    op[0, :] = 0.0  # constant inlet boundary
    return op


def build(nr: int, nz: int, ne: int = 1) -> Collocation:
    """Build collocation operators.

    Parameters
    ----------
    nr
        Number of radial collocation points (including the surface).
    nz
        Number of axial collocation points per element (including both ends).
    ne
        Number of axial finite elements.
    """
    if nr < 2 or nz < 3 or ne < 1:
        raise ValueError("need nr >= 2, nz >= 3 and ne >= 1")
    wr, br = _radial(nr)
    az = _axial_single(nz) if ne == 1 else _axial_elements(ne, nz)
    return Collocation(nr=nr, nz=az.shape[0], wr=wr, br=br, az=az)


def jacobian_sparsity(n_comp: int, nr: int, nz: int) -> sparse.csr_matrix:
    """Sparsity pattern of the PSDM right-hand side.

    The state is ordered ``(compound, radial node + bulk, axial node)``; ``nz`` is the total
    number of axial nodes. Particle nodes couple to every particle node of every compound at
    the same axial position (competitive equilibrium) and to their own bulk concentration;
    bulk nodes couple to the bulk along the axis and to the particle surface.
    """
    shape = (n_comp, nr + 1, nz)
    index = np.arange(np.prod(shape)).reshape(shape)
    rows, cols = [], []

    def connect(r, c):
        r, c = np.broadcast_arrays(r, c)
        rows.append(r.ravel())
        cols.append(c.ravel())

    particle = index[:, :nr, :]  # (n_comp, nr, nz)
    bulk = index[:, nr, :]  # (n_comp, nz)
    # particle <-> all particle nodes (all compounds) at the same z
    for z in range(nz):
        p = particle[:, :, z].ravel()
        connect(p[:, None], p[None, :])
    # surface <-> own bulk
    connect(particle[:, nr - 1, 1:], bulk[:, 1:])
    # bulk <-> bulk along the axis, own compound
    for c in range(n_comp):
        connect(bulk[c, 1:, None], bulk[c, None, 1:])
    # bulk <-> surface of every compound at the same z
    for z in range(1, nz):
        connect(bulk[:, z, None], particle[:, nr - 1, z][None, :])

    n = index.size
    data = np.ones(sum(len(r) for r in rows), dtype=bool)
    mat = sparse.coo_matrix((data, (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return mat.tocsr()
