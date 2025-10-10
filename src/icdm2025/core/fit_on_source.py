# src/icdm2025/core/fit_on_source.py
from __future__ import annotations
import numpy as np

__all__ = ["lmfit", "fitdag"]

def lmfit(s: np.ndarray, y: int, x: list[int]) -> np.ndarray:
    """
    Fit regression of node y on parents x using covariance matrix s.
    Returns vector m of length p: m[parent] = beta, m[y] = residual variance.
    """
    p = s.shape[0]
    m = np.zeros(p, dtype=float)
    if not x:
        m[y] = float(s[y, y])
        return m
    S_xx = s[np.ix_(x, x)]
    S_xy = s[np.ix_(x, [y])].reshape(-1,)
    beta = np.linalg.solve(S_xx, S_xy)
    for idx, parent in enumerate(x):
        m[parent] = beta[idx]
    m[y] = float(s[y, y] - S_xy @ beta)
    return m

def fitdag(amat: np.ndarray, s: np.ndarray, parent_list: list[list[int]]) -> dict:
    """
    Fit a recursive Gaussian DAG via GLS on covariance matrix s.

    Convention: amat[j, i] == 1 means edge j → i (j is a parent of i).
    Returns dict with keys 'A', 'Delta', 'Shat'.
    """
    p = s.shape[0]
    A = np.zeros((p, p), dtype=float)
    Delta = np.zeros(p, dtype=float)

    for i in range(p):
        parents = parent_list[i]
        if not parents:
            Delta[i] = float(s[i, i])
            A[i, i]  = 1.0
        else:
            m = lmfit(s, y=i, x=parents)
            for parent in parents:
                A[i, parent] = -m[parent]
            A[i, i] = 1.0
            Delta[i] = m[i]

    Khat = A.T @ np.diag(1.0 / Delta) @ A
    Shat = np.linalg.inv(Khat)
    return {"A": A, "Delta": Delta, "Shat": Shat}
