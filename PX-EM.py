import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- (1) Helper functions: lmfit and fitdag (exactly as before) -------------
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

    Khat = A.T @ np.diag(1.0 / Delta) @ A
    Shat = np.linalg.inv(Khat)
    return {'A': A, 'Delta': Delta, 'Shat': Shat, 'Khat': Khat}

# --- (2) E-step (unchanged) -------------------------------------------------
def E_step(X_obs: np.ndarray, idx_t: int, Sigma_hat: np.ndarray) -> np.ndarray:
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

# --- (2b) Utilities ---------------------------------------------------------
def sigma_from_params(A: np.ndarray, Delta: np.ndarray) -> np.ndarray:
    invA = np.linalg.inv(A)
    return invA @ np.diag(Delta) @ invA.T

def observed_cov(X_obs: np.ndarray) -> np.ndarray:
    Xc = X_obs - X_obs.mean(axis=0, keepdims=True)
    n = max(1, Xc.shape[0])
    return (Xc.T @ Xc) / n

def px_expand_Q(Q: np.ndarray, idx_t: int, alpha: float) -> np.ndarray:
    """Apply expansion X' = G X with G=diag(..., α at t, ...) to complete-data moments."""
    p = Q.shape[0]
    G = np.ones(p); G[idx_t] = alpha
    Q_exp = Q.copy()
    # scale row/col t
    Q_exp[:, idx_t] *= alpha
    Q_exp[idx_t, :] *= alpha
    return Q_exp

def px_reduce_Sigma(Shat_expanded: np.ndarray, idx_t: int, alpha: float) -> np.ndarray:
    """Reduce back to original coordinates: Σ = G^{-1} Σ' G^{-1}."""
    p = Shat_expanded.shape[0]
    Ginv = np.ones(p); Ginv[idx_t] = 1.0 / max(alpha, 1e-12)
    # Σ = D Σ' D where D=diag(Ginv) because G is diagonal
    D = np.diag(Ginv)
    return D @ Shat_expanded @ D

# --- (2c) Choose α to accelerate (PX choice guided by observed likelihood) ---
def choose_alpha_px(A: np.ndarray, Delta: np.ndarray, idx_t: int, S_oo: np.ndarray, O: list[int]) -> float:
    """
    Use the same rank-one geometry as ECME to pick a good α.
    Let Σ_oo = S0 + Δ_t b b^T where b = (A^{-1})_{O,t}, S0 = Σ_oo - Δ_t b b^T.
    Define u = S0^{-1} b, c = b^T u, d = u^T S_oo u.
    The observed-likelihood-optimal Δ_t is Δ_t* = (d - c) / c^2  (if positive).
    We pick α = sqrt( clamp(Δ_t* / Δ_t, α_min^2, α_max^2) ) so the expanded M-step
    tends to achieve the target variance along the b-direction.
    """
    t = idx_t
    invA = np.linalg.inv(A)
    b = invA[O, t].reshape(-1, 1)

    # Current Σ from (A,Δ)
    Sigma = sigma_from_params(A, Delta)
    Sigma_oo = Sigma[np.ix_(O, O)]
    Delta_t = max(Delta[t], 1e-12)

    # S0 = Σ_oo - Δ_t b b^T
    S0 = Sigma_oo - Delta_t * (b @ b.T)
    S0 = 0.5 * (S0 + S0.T)
    try:
        S0_inv = np.linalg.inv(S0)
    except np.linalg.LinAlgError:
        S0_inv = np.linalg.pinv(S0)

    u = S0_inv @ b
    c = float(b.T @ u)
    d = float(u.T @ S_oo @ u)

    # ECME target variance; floor to keep positive
    Delta_t_star = max((d - c) / max(c**2, 1e-24), 1e-12)

    # Map to α; clamp to avoid extremes
    alpha2 = np.clip(Delta_t_star / Delta_t, 1e-3, 1e3)
    alpha = float(np.sqrt(alpha2))
    return alpha

# --- (3) PX-EM routine ------------------------------------------------------
def PX_EM_algorithm(X_obs: np.ndarray,
                    idx_t: int,
                    amat: np.ndarray,
                    Sigma_init: np.ndarray,
                    n: int,
                    tol: float = 1e-6,
                    max_iter: int = 9000,
                    verbose_every: int = 50):
    """
    PX-EM for one fully-missing node T.

    Iteration:
      1) E-step: Q = E[XX^T | X_O, Σ]
      2) Provisional M-step on Q -> (A,Δ)
      3) Choose expansion α via observed-likelihood geometry (rank-one, closed form)
      4) Expand Q -> Q' with α on the T coordinate
      5) M-step on Q' -> Shat' (covariance in expanded coords)
      6) Reduce: Σ_new = G^{-1} Shat' G^{-1}
    """
    p = Sigma_init.shape[0]
    O = [i for i in range(p) if i != idx_t]
    S_oo = observed_cov(X_obs)

    Sigma_hat = Sigma_init.copy()
    converged = False

    for it in range(1, max_iter + 1):
        Sigma_prev = Sigma_hat.copy()

        # (1) E-step
        Q_hat = E_step(X_obs, idx_t, Sigma_hat)

        # (2) Provisional M-step (gives current A, Δ used to pick α)
        dag_tmp = fitdag(amat=amat, s=Q_hat, n=n, constr=None)
        A_tmp = dag_tmp['A'].copy()
        Delta_tmp = dag_tmp['Delta'].copy()

        # (3) Pick α
        alpha = choose_alpha_px(A_tmp, Delta_tmp, idx_t, S_oo, O)

        # (4) Expand sufficient stats for T
        Q_exp = px_expand_Q(Q_hat, idx_t, alpha)

        # (5) M-step on expanded stats
        dag_fit_exp = fitdag(amat=amat, s=Q_exp, n=n, constr=None)
        Shat_exp = dag_fit_exp['Shat']

        # (6) Reduce back to original coordinates
        Sigma_hat = px_reduce_Sigma(Shat_exp, idx_t, alpha)

        # Convergence check
        diff_norm = np.linalg.norm(Sigma_hat - Sigma_prev, ord='fro')
        if it == 1 or (verbose_every and it % verbose_every == 0):
            print(f"[PX-EM] Iter {it}: ||Σ_new − Σ_old||_F = {diff_norm:.6e} | α={alpha:.3e}")
        if diff_norm < tol:
            converged = True
            print(f"[PX-EM] Converged at iteration {it} (||ΔΣ||_F = {diff_norm:.6e} < tol).")
            break

    if not converged:
        print(f"[PX-EM] Did not converge within {max_iter} iterations. Final ||ΔΣ||_F = {diff_norm:.6e}")

    # Optionally refit DAG on the final Σ to return A,Δ consistent with Σ_hat
    dag_fit_final = fitdag(amat=amat, s=Sigma_hat, n=n, constr=None)
    return dag_fit_final, Sigma_hat, it, converged

# === (4) Main driver: covariate-shift example ================================
if __name__ == "__main__":
    # 1. Read source and target-observed CSVs (T is missing in target_df):
    source_df = pd.read_csv("source_1.csv")                 # ["C1","C2","Z","X","T","P","Y"]
    target_df = pd.read_csv("cov_target_missing_1.csv")     # T is NaN
    target_true_T = pd.read_csv("cov_target_true_1.csv")    # for evaluation only

    # 2. Define the DAG
    nodes = ["C1", "C2", "Z", "X", "T", "P", "Y"]
    p = len(nodes)
    DAG = {
        "C1": [],
        "C2": [],
        "Z":  ["C1", "C2"],
        "X":  ["C1"],
        "T":  ["C1", "X", "Z"],
        "P":  ["T"],
        "Y":  ["T"]
    }
    amat = np.zeros((p, p), dtype=int)  # amat[i,j] = 1 if j → i
    for child, parents in DAG.items():
        i = nodes.index(child)
        for parent in parents:
            j = nodes.index(parent)
            amat[i, j] = 1

    # 3. Source fit -> Σ_source
    n_source = source_df.shape[0]
    S_source = source_df[nodes].cov().values
    dag_fit_source = fitdag(amat=amat, s=S_source, n=n_source, constr=None)
    Sigma_source = dag_fit_source['Shat']

    # 4. Observed target data (drop T)
    X_obs = target_df[["C1", "C2", "Z", "X", "P", "Y"]].values
    n_target = X_obs.shape[0]
    idx_t = nodes.index("T")

    # 5. Run PX-EM
    print("=== Running PX-EM with Σ_init = Σ_source ===")
    dag_fit_px, Sigma_px, it_px, conv_px = PX_EM_algorithm(
        X_obs=X_obs,
        idx_t=idx_t,
        amat=amat,
        Sigma_init=Sigma_source,
        n=n_target,
        tol=1e-6,
        max_iter=9000,
        verbose_every=50
    )

    # 6. Impute T and evaluate
    O = [i for i in range(p) if i != idx_t]
    Sigma_oo = Sigma_px[np.ix_(O, O)]
    Sigma_to = Sigma_px[np.ix_([idx_t], O)]
    Sigma_oo_inv = np.linalg.inv(Sigma_oo)

    imputed_T = np.zeros(n_target)
    for i in range(n_target):
        x_o = X_obs[i].reshape(-1, 1)
        imputed_T[i] = (Sigma_to @ Sigma_oo_inv @ x_o).item()

    true_T = target_true_T["T"].values
    mae = mean_absolute_error(true_T, imputed_T)
    mse = mean_squared_error(true_T, imputed_T)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_T, imputed_T)

    print(f"\nFinal PX-EM results (covariate-shift init):")
    print(f"  Converged?   = {conv_px}")
    print(f"  Iterations   = {it_px}")
    print(f"  MAE          = {mae:.4f}")
    print(f"  MSE          = {mse:.4f}")
    print(f"  RMSE         = {rmse:.4f}")
    print(f"  R^2          = {r2:.4f}")

    # 7. Plot True vs Imputed T
    plt.figure(figsize=(5, 5))
    plt.scatter(true_T, imputed_T, alpha=0.6, label="PX-EM")
    mn, mx = true_T.min(), true_T.max()
    plt.plot([mn, mx], [mn, mx], "k--", lw=2)
    plt.xlabel("True T", fontsize=24)
    plt.ylabel("Predicted T", fontsize=24)
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.tight_layout()
    plt.savefig("PXEM_Cov.pdf", format="pdf", bbox_inches="tight")
    plt.show()
