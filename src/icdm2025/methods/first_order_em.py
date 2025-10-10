# Copyright (c) 2025 Mohammad Ali Javidian
# SPDX-License-Identifier: MIT
#
# This file is part of the ICDM2025 project.
# Licensed under the MIT License – see LICENSE in the repo root.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# (1) Local OLS from covariance (node i on parents)
# =========================

def lmfit(s: np.ndarray, y: int, x: list[int]) -> np.ndarray:
    """
    Fit regression of node y on parents x using covariance matrix s.
    Returns vector m of length p: m[parent] = beta, m[y] = residual variance.
    """
    p = s.shape[0]
    m = np.zeros(p)
    if not x:
        m[y] = s[y, y]
        return m
    S_xx = s[np.ix_(x, x)]
    S_xy = s[np.ix_(x, [y])].reshape(-1,)
    beta = np.linalg.solve(S_xx, S_xy)
    for idx, parent in enumerate(x):
        m[parent] = beta[idx]
    m[y] = s[y, y] - S_xy @ beta
    return m

# =========================
# (2) Fit DAG-constrained Gaussian (A, Delta) from covariance
# =========================

def fitdag(amat: np.ndarray, s: np.ndarray, parent_list: list[list[int]]) -> dict:
    """
    Fit a Gaussian DAG (recursive) via GLS on covariance matrix s.
    parent_list[i] gives list of parent indices for node i.
    Returns dict with keys 'A', 'Delta', 'Shat'.
    """
    p = s.shape[0]
    A = np.zeros((p, p))
    Delta = np.zeros(p)
    for i in range(p):
        parents = parent_list[i]
        if not parents:
            Delta[i] = s[i, i]
            A[i, i] = 1.0
        else:
            m = lmfit(s, y=i, x=parents)
            for parent in parents:
                A[i, parent] = -m[parent]
            A[i, i] = 1.0
            Delta[i] = m[i]
    Khat = A.T @ np.diag(1.0 / Delta) @ A
    Shat = np.linalg.inv(Khat)
    return {'A': A, 'Delta': Delta, 'Shat': Shat}

# =========================
# (3) Helpers for conditional of T | O under (mu, Sigma)
# =========================

def conditional_params(Sigma: np.ndarray, idx_t: int, ridge: float = 0.0):
    """
    Returns observed index list O, W = Sigma_to Sigma_oo^{-1}, V_t, and Sigma_oo_inv.
    """
    p = Sigma.shape[0]
    O = [i for i in range(p) if i != idx_t]
    Sigma_oo = Sigma[np.ix_(O, O)].copy()
    if ridge > 0:
        Sigma_oo += ridge * np.eye(Sigma_oo.shape[0])
    Sigma_oo_inv = np.linalg.inv(Sigma_oo)
    Sigma_to = Sigma[idx_t, O].reshape(1, -1)
    W = (Sigma_to @ Sigma_oo_inv).ravel()  # (p-1,)
    V_t = Sigma[idx_t, idx_t] - float(Sigma_to @ Sigma_oo_inv @ Sigma_to.T)
    return O, W, V_t, Sigma_oo_inv

# =========================
# (4) Mean-aware E-step: expected complete-data covariance + updated mean
# =========================

def E_step_vec_mean_aware(
    X_obs: np.ndarray,
    idx_t: int,
    Sigma_hat: np.ndarray,
    mu: np.ndarray | None = None,
    ridge: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean-aware E-step that returns:
      - Q_hat: expected complete-data COVARIANCE (n-1 denominator)
      - mu_new: updated mean vector [mu_O = sample mean; mu_t = mean E[T|X_O]]

    Both are consistent with pandas.DataFrame.cov() (ddof=1).
    """
    p = Sigma_hat.shape[0]
    n = X_obs.shape[0]
    denom = max(n - 1, 1)

    # Conditional params of T|O
    O, W, V_t, _ = conditional_params(Sigma_hat, idx_t, ridge=ridge)

    # Means
    if mu is None:
        mu_O = X_obs.mean(axis=0)  # (p-1,)
        mu_t = 0.0
    else:
        mu_O = mu[O]
        mu_t = float(mu[idx_t])

    # Center observed and compute conditional means for T
    Xc = X_obs - mu_O                   # (n x (p-1))
    MU = (Xc @ W.reshape(-1, 1)) + mu_t # (n x 1), rowwise E[T|x_O]

    # Updated means
    mu_O_new = mu_O
    mu_t_new = float(MU.mean())
    mu_new = np.zeros(p)
    mu_new[O] = mu_O_new
    mu_new[idx_t] = mu_t_new

    # Expected complete-data covariance
    Q = np.zeros((p, p))
    # Q_oo: sample covariance of observed block (ddof=1)
    Q_oo = (Xc.T @ Xc) / denom
    # Q_ot: Cov(X_O, T) = Cov(X_O, E[T|X_O]) (since Var(T|X_O) independent of X_O)
    tc = MU - mu_t_new                  # center by updated mu_t
    Q_ot = (Xc.T @ tc) / denom          # ((p-1) x 1)
    Q_to = Q_ot.T
    # Q_tt: Var(T) = E[Var(T|X_O)] + Var(E[T|X_O])
    var_mu = float(((tc) ** 2).sum() / denom)
    Q_tt = V_t + var_mu

    # Pack and symmetrize
    Q[np.ix_(O, O)] = Q_oo
    Q[np.ix_(O, [idx_t])] = Q_ot
    Q[np.ix_([idx_t], O)] = Q_to
    Q[idx_t, idx_t] = Q_tt
    Q = 0.5 * (Q + Q.T)
    return Q, mu_new

# =========================
# (5) EM algorithm (mean-aware throughout; returns mu_hat and Sigma_hat)
# =========================

def EM_algorithm(
    X_obs: np.ndarray,
    idx_t: int,
    amat: np.ndarray,
    Sigma_init: np.ndarray,
    parent_list: list[list[int]],
    mu_init: np.ndarray | None = None,
    tol: float = 1e-6,
    max_iter: int = 50000,
    ridge: float = 0.0,
    print_every: int = 1000
) -> tuple[dict, np.ndarray, np.ndarray, int, bool]:
    """
    EM with mean-aware E-step (updates mu) and DAG-constrained M-step (updates Sigma).
    Returns (dag_fit, Sigma_hat, mu_hat, iters, converged).
    """
    Sigma_hat = Sigma_init.copy()
    mu_hat = mu_init.copy() if mu_init is not None else None
    converged = False

    for it in range(1, max_iter + 1):
        Sigma_prev = Sigma_hat.copy()

        # E-step: get expected covariance and updated mean
        Q_hat, mu_hat = E_step_vec_mean_aware(
            X_obs=X_obs,
            idx_t=idx_t,
            Sigma_hat=Sigma_hat,
            mu=mu_hat,
            ridge=ridge
        )

        # M-step: fit DAG from expected covariance
        dag_fit = fitdag(amat, Q_hat, parent_list)
        Sigma_hat = dag_fit['Shat']

        # Convergence on Sigma (you may also track ||mu_new - mu_old|| if desired)
        diff_norm = np.linalg.norm(Sigma_hat - Sigma_prev, ord='fro')
        if it == 1 or (print_every and it % print_every == 0):
            print(f"Iteration {it}: ||Σ_new − Σ_old||_F = {diff_norm:.2e}")
        if diff_norm < tol:
            converged = True
            print(f"Converged at iteration {it} (||ΔΣ||_F = {diff_norm:.2e}).")
            break

    if not converged:
        print(f"Did not converge within {max_iter} iterations. Final Δ = {diff_norm:.2e}")
    return dag_fit, Sigma_hat, mu_hat, it, converged
