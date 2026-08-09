"""
Compositional data log-ratio transforms - ALR, CLR, ILR.

Soil texture (sand + silt + clay) is *compositional*: the parts carry only
relative information and are constrained to a constant sum (100 %).  Modelling
each part with an independent regression ignores that constraint, so the
predictions need not sum to 100 and the (singular) covariance structure is
mis-specified.  The log-ratio transforms of Aitchison map a D-part composition
from the simplex into ordinary real space, where standard regression is valid;
predictions are mapped back with the inverse, which is closed to the total by
construction - so sand + silt + clay always sums to 100 again.

Transforms provided (each with an exact inverse):

* **CLR** - centred log-ratio: ``clr(x)_i = ln(x_i / g(x))`` with ``g`` the
  geometric mean.  Yields ``D`` coordinates that sum to zero (a singular basis,
  fine as regression targets but not for methods needing full-rank covariance).
* **ALR** - additive log-ratio: ``alr(x)_i = ln(x_i / x_ref)`` for the ``D-1``
  non-reference parts.  Simple and interpretable; the geometry is oblique.
* **ILR** - isometric log-ratio: an orthonormal rotation of the CLR onto ``D-1``
  coordinates (Helmert contrast basis).  Isometric (preserves distances/angles),
  the statistically preferred default.

Zeros (a part measured as exactly 0) are handled by simple multiplicative
replacement before taking logs.

@author: Sharad Kumar Gupta
"""
import numpy as np

TRANSFORMS = ["CLR", "ALR", "ILR"]


# ── Basic simplex operations ────────────────────────────────────────────────
def close(x, total=1.0):
    """Close each row of ``x`` so it sums to ``total`` (the constant-sum rule)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    s = x.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return total * x / s


def replace_zeros(x, delta=None):
    """Multiplicative zero replacement (Martín-Fernández et al., 2003).

    Works on proportions: each zero part is set to a small ``delta`` and the
    non-zero parts are scaled down so the row still sums to 1 - preserving the
    ratios between the observed parts.  ``delta`` defaults to 65 % of the
    smallest positive proportion in the data (a common rule of thumb)."""
    P = close(np.asarray(x, dtype=float))
    P = np.where(P < 0, 0.0, P)
    if delta is None:
        pos = P[P > 0]
        delta = float(pos.min() * 0.65) if pos.size else 1e-6
    zeros = P <= 0
    if not zeros.any():
        return P
    n_zero = zeros.sum(axis=1, keepdims=True)
    sum_nz = np.where(zeros, 0.0, P).sum(axis=1, keepdims=True)
    scale = np.clip(1.0 - n_zero * delta, 1e-12, None)   # mass left for non-zeros
    factor = np.where(sum_nz > 0, scale / sum_nz, 0.0)
    return np.where(zeros, delta, P * factor)


# ── CLR ─────────────────────────────────────────────────────────────────────
def clr(x):
    """Centred log-ratio.  Returns ``(n, D)``; each row sums to 0."""
    P = replace_zeros(x)
    L = np.log(P)
    return L - L.mean(axis=1, keepdims=True)


def clr_inv(Y, total=1.0):
    """Inverse CLR → composition closed to ``total``."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[None, :]
    E = np.exp(Y - Y.max(axis=1, keepdims=True))   # shift for numerical stability
    return close(E, total)


# ── ALR ─────────────────────────────────────────────────────────────────────
def alr(x, ref=-1):
    """Additive log-ratio wrt part index ``ref`` (default: last).  ``(n, D-1)``."""
    P = replace_zeros(x)
    D = P.shape[1]
    ref = ref % D
    num = np.delete(P, ref, axis=1)
    den = P[:, ref:ref + 1]
    return np.log(num / den)


def alr_inv(Y, ref=-1, total=1.0):
    """Inverse ALR → composition closed to ``total`` (D = ``Y`` cols + 1)."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[None, :]
    D = Y.shape[1] + 1
    ref = ref % D
    E = np.exp(Y - np.maximum(Y.max(axis=1, keepdims=True), 0.0))
    ref_val = np.exp(-np.maximum(Y.max(axis=1, keepdims=True), 0.0))  # exp(0) shifted
    # Reinsert the reference column, then close.
    full = np.insert(E, ref, ref_val.ravel(), axis=1)
    return close(full, total)


# ── ILR (Helmert orthonormal contrast basis) ────────────────────────────────
def _helmert_basis(D):
    """Return the ``(D, D-1)`` orthonormal contrast basis ``V`` (V.T rows are the
    Helmert contrasts): columns are orthonormal and sum to zero, so ``clr @ V``
    is an isometry onto ``R^{D-1}`` and ``ilr @ V.T`` recovers the CLR."""
    H = np.zeros((D - 1, D))
    for i in range(D - 1):
        norm = 1.0 / np.sqrt((i + 1) * (i + 2))
        H[i, :i + 1] = -norm
        H[i, i + 1] = (i + 1) * norm
    return H.T   # (D, D-1)


def ilr(x):
    """Isometric log-ratio.  Returns ``(n, D-1)``."""
    C = clr(x)
    V = _helmert_basis(C.shape[1])
    return C @ V


def ilr_inv(Z, total=1.0):
    """Inverse ILR → composition closed to ``total`` (D = ``Z`` cols + 1)."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[None, :]
    D = Z.shape[1] + 1
    V = _helmert_basis(D)
    C = Z @ V.T
    return clr_inv(C, total)


# ── Unified dispatch ────────────────────────────────────────────────────────
def forward(x, transform, ref=-1):
    """Forward log-ratio transform.  ``transform`` in :data:`TRANSFORMS`.

    Returns ``(coords, n_coords)`` - ``D`` for CLR, ``D-1`` for ALR/ILR."""
    t = str(transform).upper()
    if t == "CLR":
        return clr(x)
    if t == "ALR":
        return alr(x, ref=ref)
    if t == "ILR":
        return ilr(x)
    raise ValueError(f"Unknown compositional transform '{transform}'. "
                     f"Choose one of {TRANSFORMS}.")


def inverse(coords, transform, total=100.0, ref=-1):
    """Inverse log-ratio transform → composition closed to ``total`` (default
    100 %), so the recovered parts always sum to ``total``."""
    t = str(transform).upper()
    if t == "CLR":
        return clr_inv(coords, total)
    if t == "ALR":
        return alr_inv(coords, ref=ref, total=total)
    if t == "ILR":
        return ilr_inv(coords, total)
    raise ValueError(f"Unknown compositional transform '{transform}'. "
                     f"Choose one of {TRANSFORMS}.")


def n_coords(transform, n_parts):
    """Number of log-ratio coordinates a ``n_parts``-composition maps to."""
    return n_parts if str(transform).upper() == "CLR" else n_parts - 1
