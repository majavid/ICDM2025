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

# =========================
# (6) Main: covariate-shift example with non-zero-mean handled everywhere
# =========================

if __name__ == "__main__":
    # ---- Load data ----
    source_df = pd.read_csv("source_1.csv")
    target_obs_df = pd.read_csv("cov_target_missing_1.csv")  # T missing
    target_true_df = pd.read_csv("cov_target_true_1.csv")        # T true

    # ---- Adjacency & nodes ----
    amat_df = pd.read_csv("adjacency_matrix.csv", index_col=0)
    nodes = list(amat_df.index)
    p = len(nodes)
    amat = amat_df.values.astype(int)

    # ---- Parent lists (sanity print) ----
    parent_list = [[j for j in range(p) if amat[j, i]] for i in range(p)]
    for i, parents in enumerate(parent_list):
        parent_names = [nodes[j] for j in parents]
        print(f"Node {i} ({nodes[i]}): parents idx={parents}  names={parent_names}")

    # ---- Source-domain fit (covariance under DAG) ----
    # Use pandas.cov() (ddof=1) for consistency with our E-step covariance
    S_source = source_df[nodes].cov().values
    dag_src = fitdag(amat, S_source, parent_list)
    Sigma_src = dag_src['Shat']

    # ---- Initial means ----
    # Source sample mean (length p)
    mu_source = source_df[nodes].mean().values

    # Target observed matrix and observed/latent indices
    idx_t = nodes.index("T")
    obs_vars = [n for n in nodes if n != "T"]
    X_obs = target_obs_df[obs_vars].values

    # Target observed means (for O block); initialize mu_t using conditional on source Σ and means
    mu_O_target = X_obs.mean(axis=0)  # (p-1,)
    O, W_src, _, _ = conditional_params(Sigma_src, idx_t, ridge=0.0)
    mu_t_init = float(mu_source[idx_t] + W_src @ (mu_O_target - mu_source[O]))

    mu_init = np.zeros(p)
    mu_init[O] = mu_O_target
    mu_init[idx_t] = mu_t_init
    print(f"Initialized means: mu_t={mu_init[idx_t]:.6f}")

    # ---- Run EM (mean-aware everywhere) ----
    print("=== Running mean-aware EM ===")
    dag_fit, Sigma_hat, mu_hat, iters, conv = EM_algorithm(
        X_obs=X_obs,
        idx_t=idx_t,
        amat=amat,
        Sigma_init=Sigma_src,
        parent_list=parent_list,
        mu_init=mu_init,     # start with target-observed mean and conditional for T
        tol=1e-6,
        max_iter=50000,
        ridge=0.0,
        print_every=1000
    )

    # ---- Mean-aware imputation and evaluation ----
    # E[T | x_O] = mu_t + Sigma_to Sigma_oo^{-1} (x_O - mu_O)
    O, W_hat, _, Sigma_oo_inv = conditional_params(Sigma_hat, idx_t, ridge=0.0)
    mu_O_hat = mu_hat[O]
    mu_t_hat = float(mu_hat[idx_t])

    Xc = X_obs - mu_O_hat
    imputed = mu_t_hat + (Xc @ W_hat)  # (n,)

    true_vals = target_true_df["T"].values
    mask = (~np.isnan(true_vals)) & (~np.isnan(imputed))
    true_vals = true_vals[mask]
    imputed = imputed[mask]

    mae = mean_absolute_error(true_vals, imputed)
    mse = mean_squared_error(true_vals, imputed)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_vals, imputed)

    print(f"\nEM final results: converged={conv} in {iters} iter")
    print(f"MAE={mae:.4f}, RMSE={rmse:.4f}, R^2={r2:.4f}")
    print(f"Final mean of T (mu_t_hat) = {mu_t_hat:.6f}")

    # ---- Plot: mean-aware prediction vs truth ----
    plt.figure(figsize=(6, 6))
    plt.scatter(true_vals, imputed, alpha=0.6)
    mn, mx = np.nanmin(true_vals), np.nanmax(true_vals)
    plt.plot([mn, mx], [mn, mx], 'k--', lw=2)
    plt.xlabel("True T", fontsize=24)
    plt.ylabel("Predicted T (mean-aware)", fontsize=24)
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig("EM_T_mean_aware.pdf", format="pdf", bbox_inches="tight")
    plt.show()
