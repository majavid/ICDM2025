# fit_dag_latent.py
"""
Python port of the R function `fitDagLatent` (Kiiveri-type EM for a recursive
SEM with one latent variable). Only the single-latent case is implemented.

Dependencies
------------
numpy >= 1.22
pandas >= 1.4
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.linalg import inv, solve, det


# --------------------------------------------------------------------------- #
#  Numeric hygiene & robust solvers                                           #
# --------------------------------------------------------------------------- #
def symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)

def nearest_spd(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Project a symmetric matrix to the nearest SPD by eigenvalue clipping."""
    A = symmetrize(A)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, eps)
    return (V * w) @ V.T

def safe_solve_psd(Sxx: np.ndarray, sxy: np.ndarray,
                   ridge: float = 1e-10, max_tries: int = 6) -> np.ndarray:
    """
    Solve Sxx * beta = sxy robustly. Try direct solve; if singular/ill-conditioned,
    add increasing ridge to the diagonal; fall back to pinv at the end.
    """
    Sxx = symmetrize(Sxx)
    # try direct
    try:
        return solve(Sxx, sxy)
    except np.linalg.LinAlgError:
        pass
    lam = ridge
    I = np.eye(Sxx.shape[0])
    for _ in range(max_tries):
        try:
            return solve(Sxx + lam * I, sxy)
        except np.linalg.LinAlgError:
            lam *= 10.0
    # final fallback
    return np.linalg.pinv(Sxx) @ sxy


# --------------------------------------------------------------------------- #
#  Small helpers                                                              #
# --------------------------------------------------------------------------- #
def random_corr(p: int) -> np.ndarray:
    """Generate a random positive-definite correlation matrix."""
    a = np.random.randn(p, p)
    s = a @ a.T
    d = np.sqrt(np.diag(s))
    return s / d[:, None] / d[None, :]

def setvar1(v: np.ndarray, z: int, paz_mask: np.ndarray, norm: int = 1) -> np.ndarray:
    """Normalise the latent so that either V[z,z] == 1 (norm=1)
    or the *residual* variance of z equals 1 (norm=2)."""
    v = v.copy()
    if norm == 1:
        scale = 1.0 / np.sqrt(v[z, z])
    else:  # norm == 2
        if paz_mask.any():
            sig = v[z, z] - v[z, paz_mask] @ solve(v[np.ix_(paz_mask, paz_mask)], v[paz_mask, z])
        else:
            sig = v[z, z]
        scale = 1.0 / np.sqrt(sig)
    v[z, :] *= scale
    v[:, z] *= scale
    return v

def cmqi(syy: np.ndarray, sigma: np.ndarray, z: int) -> np.ndarray:
    """
    Conditional covariance matrix C(M|Q) (Kiiveri, 1987).
    `syy`  : observed covariance (p × p)
    `sigma`: current full covariance ((p+1) × (p+1))
    `z`    : index of the (single) latent variable
    """
    p1 = sigma.shape[0]
    y_idx = [i for i in range(p1) if i != z]

    q = inv(sigma)
    qzz = q[z, z]
    qzy = q[z, y_idx]

    b = -qzy / qzz            # shape (p,)
    bs = b @ syy              # shape (p,)

    e = np.zeros_like(sigma)
    e[np.ix_(y_idx, y_idx)] = syy          # S_y|y
    e[y_idx, z] = bs                       # S_y|z
    e[z, y_idx] = bs                       # S_z|y (sym.)
    e[z, z] = bs @ b + 1.0 / qzz           # S_z|z
    return e


# --------------------------------------------------------------------------- #
#  Parent printing helper (amat[j, i] == 1 means j → i)                       #
# --------------------------------------------------------------------------- #
def print_parent_lists(nodes: list[str], amat_np: np.ndarray) -> None:
    """
    Print parents for each node using convention: amat[j, i] == 1 means j → i.
    """
    p = len(nodes)
    parent_list = [[j for j in range(p) if amat_np[j, i] == 1 and j != i] for i in range(p)]
    for i, parents in enumerate(parent_list):
        parent_names = [nodes[j] for j in parents]
        print(f"Node {i} ({nodes[i]}): parents idx={parents}  names={parent_names}")


# --------------------------------------------------------------------------- #
#  Regression helper with optional fixed coefficients                         #
# --------------------------------------------------------------------------- #
def lmfit(s: np.ndarray, y: int, x: list[int], z: list[int] | None = None) -> np.ndarray:
    """
    Regression coefficients of Y on X with optional coefficients on Z fixed to 1.
    Returns a length-p vector:
        out[x] = β̂,   out[z] = 1,   out[y] = residual variance σ̂²_y.
    """
    z = z or []
    p = s.shape[0]

    # β̂ = (Sxx)^{-1} (Sxy − Σ_{j∈z} Sxj)
    s = symmetrize(s)
    sxy = s[np.ix_(x, [y])].flatten()
    if z:
        sxy -= s[np.ix_(x, z)].sum(axis=1)

    Sxx = s[np.ix_(x, x)]
    bxy = safe_solve_psd(Sxx, sxy)  # robust solve

    out = np.zeros(p)
    out[x] = bxy
    out[z] = 1.0  # fixed

    # residual variance
    xz = x + z
    if xz:
        b = out[xz][:, None]
        sxz = symmetrize(s[np.ix_(xz, xz)])
        inner = sxz @ b - 2 * s[np.ix_(xz, [y])]
        res = s[y, y] + (b.T @ inner).item()
    else:
        res = s[y, y]
    out[y] = float(res)
    return out


# --------------------------------------------------------------------------- #
#  DAG fitting (GLS) with orientation amat[j,i]==1 (j is parent of i)        #
# --------------------------------------------------------------------------- #
def fitdag(amat: np.ndarray,
           s: np.ndarray,
           n: int,
           constr: np.ndarray | None = None,
           node_names: list[str] | None = None,
           verbose_fitdag: bool = False) -> dict:
    """
    Fast GLS fitting of a recursive DAG with independent errors.
    Returns A, Delta, Shat and Khat = Shat⁻¹.
    Convention: amat[j, i] = 1 means edge j → i (j is a parent of i).
    """
    p = s.shape[0]
    emat = (amat != 0).astype(int)
    a = np.zeros((p, p))
    delta = np.zeros(p)

    s = symmetrize(s)  # hygiene

    for i in range(p):
        # PARENTS OF i ARE COLUMNS j WHERE emat[j, i] == 1
        parents = [j for j in range(p) if emat[j, i] == 1 and j != i]

        if verbose_fitdag:
            if node_names is None:
                print(f"[fitdag] node {i} parents idx={parents}")
            else:
                print(f"[fitdag] node {i} ({node_names[i]}) parents idx={parents} "
                      f"names={[node_names[j] for j in parents]}")

        if not parents:
            delta[i] = s[i, i]
            a[i, :] = 0.0
            a[i, i] = 1.0
            continue

        # Only allow constraints on existing parents (prevents singular systems)
        if constr is not None:
            z_fix = [j for j in parents if constr[i, j] == 1]
        else:
            z_fix = []

        m = lmfit(s, y=i, x=parents, z=z_fix)

        a[i, :] = 0.0
        for parent in parents:
            a[i, parent] = -m[parent]  # minus sign
        a[i, i] = 1.0
        delta[i] = m[i]

    Khat = a.T @ np.diag(1.0 / delta) @ a
    Khat = symmetrize(Khat)
    Shat = nearest_spd(inv(Khat))  # guard against tiny asymmetries
    return dict(A=a, Delta=delta, Shat=Shat, Khat=Khat)


def deviance(k: np.ndarray, s: np.ndarray, n: int) -> float:
    """-2 × log-likelihood (up to a constant)."""
    sk = s @ k
    return (np.trace(sk) - np.log(det(sk)) - sk.shape[0]) * n


# --------------------------------------------------------------------------- #
#  EM with one latent                                                         #
# --------------------------------------------------------------------------- #
def fitDagLatent(amat: pd.DataFrame,
                 syy: pd.DataFrame,
                 sigma_old: np.ndarray,
                 n: int,
                 latent: str,
                 norm: int = 1,
                 seed: int | None = None,
                 maxit: int = 9000,
                 tol: float = 1e-6,
                 verbose: bool = False,
                 verbose_parents: bool = True,
                 verbose_fitdag: bool = False) -> dict:
    """
    EM algorithm for a recursive DAG with one latent variable.

    Parameters
    ----------
    amat   : adjacency matrix (DataFrame) – amat[j,i]==1 means j ⟶ i (parent → child)
    syy    : observed covariance matrix (DataFrame) (latent not included)
    sigma_old : initial full covariance (numpy array, size (p+1)×(p+1))
    n      : sample size
    latent : name of the latent variable (must appear in amat)
    norm   : 1  ⇒ set Var(z)=1;   2 ⇒ set residual Var(z)=1
    """
    if seed is not None:
        np.random.seed(seed)

    # sanitize labels
    amat.index = amat.index.astype(str).str.strip()
    amat.columns = amat.columns.astype(str).str.strip()
    syy.index = syy.index.astype(str).str.strip()
    syy.columns = syy.columns.astype(str).str.strip()

    nam_obs = syy.index.tolist()
    nod_all = amat.index.tolist()

    if not set(nod_all) - {latent} <= set(nam_obs):
        raise ValueError("Observed nodes in 'amat' do not all appear in 'syy'.")

    # reorder so that observed come first, latent last
    nod = [v for v in nod_all if v != latent] + [latent]
    amat = amat.loc[nod, nod]
    z_idx = nod.index(latent)

    # print parent lists once, in EM node order
    if verbose_parents:
        print("\nParent lists (amat[j,i]=1 means j→i), in EM node order:")
        print_parent_lists(nod, amat.values)

    # mask of parents of z
    paz_mask = (amat.iloc[z_idx] == 1).to_numpy()

    # observed covariance matrix (in the same order as nod, excluding latent)
    syy_obs = syy.loc[[v for v in nod if v != latent], [v for v in nod if v != latent]].to_numpy()

    # initial Σ: SPD + scaling
    sigma_old = nearest_spd(sigma_old)
    sigma_old = setvar1(sigma_old, z_idx, paz_mask, norm)

    dev = None
    # --- EM ----------------------------------------------------------------
    for it in range(1, maxit + 1):
        # E-step
        q_tilde = cmqi(syy_obs, sigma_old, z_idx)
        q_tilde = nearest_spd(q_tilde, eps=1e-10)

        # M-step
        fit = fitdag(amat.values, q_tilde, n,
                     node_names=nod,
                     verbose_fitdag=verbose_fitdag)
        sigma_new = fit["Shat"]
        sigma_new = setvar1(sigma_new, z_idx, paz_mask, norm)

        # deviance
        dev = deviance(fit["Khat"], q_tilde, n)

        if verbose and (it == 1 or it % 50 == 0):
            delta_norm = np.abs(sigma_new - sigma_old).sum()
            print(f"[EM] it={it:4d}  dev={dev:.6f}  ΔΣ₁={delta_norm:.3e}")

        if np.abs(sigma_new - sigma_old).sum() < tol:
            if verbose:
                print(f"[EM] converged at it={it}")
            break

        sigma_old = sigma_new

    print("\nDone.")

    # final normalisation and refit
    shat = setvar1(sigma_new, z_idx, paz_mask, norm)
    final = fitdag(amat.values, shat, n, node_names=nod, verbose_fitdag=False)

    # Optional sign normalization for latent (if your convention prefers it)
    shat[z_idx, :]       *= -1
    shat[:, z_idx]       *= -1
    final["A"][z_idx, :] *= -1
    final["A"][:, z_idx] *= -1

    # degrees of freedom (use observed p only)
    p_obs = len(nod) - 1              # exclude the latent
    edges = (amat.values == 1).sum()
    df = p_obs * (p_obs + 1) // 2 - edges - p_obs

    return dict(
        Shat=pd.DataFrame(shat, index=nod, columns=nod),
        Ahat=pd.DataFrame(final["A"], index=nod, columns=nod),
        Dhat=pd.Series(final["Delta"], index=nod, name="Delta"),
        dev=dev,
        df=df,
        iterations=it,
    )


# --------------------------------------------------------------------------- #
#  Example                                                                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # 1) Load data
    source_df      = pd.read_csv("source_1.csv")          # complete data (for init Sigma)
    target_df      = pd.read_csv("tgt_target_missing_1.csv")   # T missing or dropped
    target_true_df = pd.read_csv("tgt_target_true_1.csv")          # T complete (for eval)

    # 2) Load adjacency matrix (amat[j,i] == 1 means j → i)
    amat_df = pd.read_csv("adjacency_matrix.csv", index_col=0)
    amat_df.index  = amat_df.index.astype(str).str.strip()
    amat_df.columns = amat_df.columns.astype(str).str.strip()
    nodes   = list(amat_df.index)
    amat_df = amat_df.loc[nodes, nodes]  # ensure consistent ordering

    # (Optional) print parents once here too:
    print("\nParents from CSV order (sanity check):")
    print_parent_lists(nodes, amat_df.values)

    # 3) Observed covariance matrix (exclude latent 'T')
    obs_vars = [v for v in nodes if v != 'T']
    if 'T' in target_df.columns:
        syy = target_df[obs_vars].cov()
    else:
        syy = target_df.cov()

    # 4) Initial Sigma: from source fit (could alternatively use random_corr)
    syy_source = source_df.cov()
    fit_source = fitdag(amat_df.values, syy_source.values, n=source_df.shape[0])
    sigma0 = fit_source["Shat"]
    # Alternative: sigma0 = random_corr(len(nodes))

    # 5) EM fit with 'T' as latent; also print parent lists once (EM order)
    fit = fitDagLatent(
        amat=amat_df,
        syy=syy,
        sigma_old=sigma0,
        n=target_df.shape[0],
        latent='T',
        norm=2,
        seed=145,
        verbose=True,            # EM progress every 50 its
        verbose_parents=True,    # <-- PRINT PARENT LISTS (once, in EM order)
        verbose_fitdag=False     # set True to print parents during each M-step fit
    )

    # 6) Impute T via mean-aware conditional expectation (posterior mean)
    Shat_df  = fit['Shat']
    all_vars = list(Shat_df.columns)
    idx_T    = all_vars.index('T')
    idx_obs  = [i for i, v in enumerate(all_vars) if v != 'T']
    obs_vars = [v for v in all_vars if v != 'T']

    X_obs    = target_df[obs_vars].values
    mu_O     = X_obs.mean(axis=0)  # target-domain observed means

    Sigma    = Shat_df.values
    Sigma_oo = Sigma[np.ix_(idx_obs, idx_obs)]
    Sigma_to = Sigma[idx_T, idx_obs].reshape(1, -1)

    # ===== Choose how to set mu_t (latent mean) =====
    # Option A (default, modeling choice): assume zero mean for latent
    #mu_t = 0.0

    # Option B: borrow the source-domain mean (if you want a prior from source)
    mu_t = float(source_df['T'].mean())

    # Option C: set a custom value from domain knowledge
    # mu_t = <your_value_here>
    # ================================================

    # Mean-aware conditional: E[T | x_O] = mu_t + Sigma_to Sigma_oo^{-1} (x_O - mu_O)
    Sigma_oo_inv = np.linalg.inv(Sigma_oo)
    W = (Sigma_to @ Sigma_oo_inv).ravel()        # shape (p-1,)
    Xc = X_obs - mu_O                             # center observed sample-by-sample
    imputed_T = mu_t + (Xc @ W)                   # shape (n,)

    # 7) Evaluate
    true_T = target_true_df['T'].values
    mask   = (~np.isnan(true_T)) & (~np.isnan(imputed_T))
    true_T = true_T[mask]
    imputed_T = imputed_T[mask]

    mae = mean_absolute_error(true_T, imputed_T)
    mse = mean_squared_error(true_T, imputed_T)
    rmse = np.sqrt(mse)
    r2  = r2_score(true_T, imputed_T)

    print(f"\nFinal EM results (mean-aware):")
    print(f"MAE  = {mae:.4f}")
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R^2  = {r2:.4f}")


    # 8) Plot True vs Predicted T
    plt.figure(figsize=(5, 5))
    plt.scatter(true_T, imputed_T, alpha=0.6, label="Baseline")
    mn, mx = true_T.min(), true_T.max()
    plt.plot([mn, mx], [mn, mx], "k--", lw=2)
    plt.xlabel("True T", fontsize=24)
    plt.ylabel("Predicted T", fontsize=24)
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.tight_layout()
    plt.savefig("KiiveriCov.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # Optional sign flip (if you want to check identifiability flips)
    imputed_T_flipped = -imputed_T
    mae = mean_absolute_error(true_T, imputed_T_flipped)
    mse = mean_squared_error(true_T, imputed_T_flipped)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_T, imputed_T_flipped)

    print(f"\nFinal EM results (after sign flip):")
    print(f"MAE  = {mae:.4f}")
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R^2  = {r2:.4f}")

    plt.figure(figsize=(5, 5))
    plt.scatter(true_T, imputed_T_flipped, alpha=0.6, label="Baseline (flipped)")
    mn, mx = true_T.min(), true_T.max()
    plt.plot([mn, mx], [mn, mx], "k--", lw=2)
    plt.xlabel("True T", fontsize=24)
    plt.ylabel("Predicted T", fontsize=24)
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.tight_layout()
    plt.savefig("KiiveriCov2.pdf", format="pdf", bbox_inches="tight")
    plt.show()
