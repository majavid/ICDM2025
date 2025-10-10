# Copyright (c) 2025 Mohammad Ali Javidian
# SPDX-License-Identifier: MIT
#
# This file is part of the ICDM2025 project.
# Licensed under the MIT License – see LICENSE in the repo root.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===========================
# (1) Helper functions (unchanged): lmfit, fitdag
# ===========================
def lmfit(s: np.ndarray, y: int, x: list, z=None) -> np.ndarray:
    p = s.shape[0]
    m = np.zeros(p)
    if not x:
        m[y] = s[y, y]
        return m
    S_xx = s[np.ix_(x, x)]
    S_xy = s[np.ix_(x, [y])].reshape(-1,)
    β = np.linalg.solve(S_xx, S_xy)
    for idx, parent in enumerate(x):
        m[parent] = β[idx]
    m[y] = s[y, y] - S_xy @ β
    return m

def fitdag(amat: np.ndarray, s: np.ndarray, n: int, constr: np.ndarray | None = None) -> dict:
    """
    Given adjacency (amat) and a covariance s for all nodes, refit linear-Gaussian DAG by OLS.
    Returns dict with A (I - B), Delta (noise variances), Shat, Khat.
    """
    p = s.shape[0]
    emat = (amat != 0).astype(int)
    A = emat.astype(float)
    Delta = np.zeros(p)

    for i in range(p):
        parents = [j for j in range(p) if emat[i, j] and j != i]
        if not parents:
            Delta[i] = s[i, i]
            A[i, :] = 0.0
            A[i, i] = 1.0
            continue
        m = lmfit(s, y=i, x=parents, z=None)
        A[i, :] = 0.0
        for parent in parents:
            A[i, parent] = -m[parent]
        A[i, i] = 1.0
        Delta[i] = m[i]

    # Model-implied covariance: Sigma = A^{-1} diag(Delta) A^{-T}
    Khat = A.T @ np.diag(1.0 / Delta) @ A
    Shat = np.linalg.inv(Khat)
    return {'A': A, 'Delta': Delta, 'Shat': Shat, 'Khat': Khat}

# ===========================
# (2) E-step: expected complete-data covariance with one fully-missing node T
# ===========================
def E_step(X_obs: np.ndarray, idx_t: int, Sigma_hat: np.ndarray) -> np.ndarray:
    """
    X_obs: n x (p-1) observed data (all variables except T)
    idx_t: index of the missing node T in 0..p-1
    Sigma_hat: current full p x p covariance
    Returns Q_hat = E[XX^T | X_O] averaged over samples (p x p)
    """
    n, p_minus_1 = X_obs.shape
    p = p_minus_1 + 1
    O = [i for i in range(p) if i != idx_t]
    t = idx_t

    Sigma_oo = Sigma_hat[np.ix_(O, O)]
    Sigma_ot = Sigma_hat[np.ix_(O, [t])]
    Sigma_to = Sigma_hat[np.ix_([t], O)]
    Sigma_tt = Sigma_hat[t, t]

    Sigma_oo_inv = np.linalg.inv(Sigma_oo)
    V_t = float(Sigma_tt - (Sigma_to @ Sigma_oo_inv @ Sigma_ot)[0, 0])

    Q_accum = np.zeros((p, p))
    for i in range(n):
        x_o = X_obs[i, :].reshape(-1, 1)
        mu_t = (Sigma_to @ Sigma_oo_inv @ x_o).item()

        M_i = np.zeros((p, p))
        M_i[np.ix_(O, O)] = x_o @ x_o.T
        M_i[np.ix_(O, [t])] = x_o * mu_t
        M_i[np.ix_([t], O)] = (x_o * mu_t).T
        M_i[t, t] = V_t + mu_t**2
        Q_accum += M_i

    return Q_accum / n

# ===========================
# (3) Utilities for ECME
# ===========================
def sigma_from_params(A: np.ndarray, Delta: np.ndarray) -> np.ndarray:
    """Sigma = A^{-1} diag(Delta) A^{-T}"""
    invA = np.linalg.inv(A)
    return invA @ np.diag(Delta) @ invA.T

def ecme_update_delta_t_observed_ll(A: np.ndarray,
                                    Delta: np.ndarray,
                                    idx_t: int,
                                    S_oo: np.ndarray,
                                    O: list[int],
                                    jitter: float = 1e-9) -> float:
    """
    ECME 'Either' step: with A, Delta_{-t} fixed, update Delta_t by maximizing
    the observed-data Gaussian log-likelihood for the observed block O.

    Closed-form solution using Sherman–Morrison:
      Let b = (A^{-1})_{O, t}, current Sigma_oo = S0 + Delta_t * b b^T,
      S0 := Sigma_oo - Delta_t * b b^T.
      Define u = S0^{-1} b, c = b^T u, d = u^T S_oo u.
      Then Delta_t_new = max(eps, (d - c) / c^2).

    Returns the new (positive) Delta_t.
    """
    p = A.shape[0]
    t = idx_t
    invA = np.linalg.inv(A)
    b_full = invA[:, t]
    b = b_full[O].reshape(-1, 1)

    # Current model-implied Sigma and its observed block
    Sigma = sigma_from_params(A, Delta)
    Sigma_oo = Sigma[np.ix_(O, O)]

    # Build S0 = Sigma_oo - Delta_t * b b^T
    Delta_t_old = float(Delta[t])
    S0 = Sigma_oo - Delta_t_old * (b @ b.T)
    # Symmetrize + jitter for numerical stability
    S0 = 0.5 * (S0 + S0.T)
    # Ensure positive definiteness
    try:
        S0_inv = np.linalg.inv(S0)
    except np.linalg.LinAlgError:
        S0 = S0 + jitter * np.eye(S0.shape[0])
        S0_inv = np.linalg.inv(S0)

    u = S0_inv @ b                       # (|O| x 1)
    c = float(b.T @ u)                   # scalar > 0
    d = float(u.T @ S_oo @ u)            # scalar >= 0

    eps = 1e-12
    if c <= eps:
        # Degenerate direction; keep old variance (or small floor)
        return max(eps, Delta_t_old)

    Delta_t_new = (d - c) / (c**2)
    if not np.isfinite(Delta_t_new) or (Delta_t_new <= eps):
        Delta_t_new = max(eps, Delta_t_old)

    return float(Delta_t_new)

# ===========================
# (4) ECME routine
# ===========================
def ECME_algorithm(X_obs: np.ndarray,
                   idx_t: int,
                   amat: np.ndarray,
                   Sigma_init: np.ndarray,
                   n: int,
                   tol: float = 1e-6,
                   max_iter: int = 9000,
                   verbose_every: int = 50):
    """
    ECME for single fully-missing node T.
    - E-step: compute Q_hat = E[XX^T | X_O, Sigma_hat]
    - CM-step (complete-data): (A, Delta) <- fitdag(amat, s=Q_hat)
    - CE-step (observed-data): optimize Delta_t to maximize observed-data loglik over O

    Returns: dag_fit (last CM-step), Sigma_hat, iters, converged
    """
    # Observed index set and S_oo (MLE scaling 1/n to align with likelihood)
    p = Sigma_init.shape[0]
    O = [i for i in range(p) if i != idx_t]
    Xc = X_obs - X_obs.mean(axis=0, keepdims=True)
    S_oo = (Xc.T @ Xc) / Xc.shape[0]   # (|O| x |O|)

    Sigma_hat = Sigma_init.copy()
    converged = False

    for it in range(1, max_iter + 1):
        Sigma_prev = Sigma_hat.copy()

        # --- E-step
        Q_hat = E_step(X_obs, idx_t, Sigma_hat)

        # --- CM-step (complete-data maximization via DAG OLS)
        dag_fit = fitdag(amat=amat, s=Q_hat, n=n, constr=None)
        A = dag_fit['A'].copy()
        Delta = dag_fit['Delta'].copy()

        # --- CE-step (observed-data maximization for Delta_t)
        Delta_t_new = ecme_update_delta_t_observed_ll(A, Delta, idx_t, S_oo, O)
        Delta[idx_t] = Delta_t_new

        # Update Sigma from (A, Delta)
        Sigma_hat = sigma_from_params(A, Delta)

        diff_norm = np.linalg.norm(Sigma_hat - Sigma_prev, ord='fro')
        if it == 1 or (verbose_every and it % verbose_every == 0):
            print(f"[ECME] Iter {it}: ||Σ_new − Σ_old||_F = {diff_norm:.6e} | Δ_t={Delta_t_new:.6e}")

        if diff_norm < tol:
            converged = True
            print(f"[ECME] Converged at iter {it} (||ΔΣ||_F = {diff_norm:.6e} < tol).")
            # Update dag_fit to reflect final Delta tweak
            dag_fit['Delta'] = Delta
            dag_fit['Shat']  = Sigma_hat
            dag_fit['Khat']  = np.linalg.inv(Sigma_hat)
            dag_fit['A']     = A
            break

    if not converged:
        print(f"[ECME] Did not converge within {max_iter} iters. Final ||ΔΣ||_F = {diff_norm:.6e}")

    return dag_fit, Sigma_hat, it, converged
