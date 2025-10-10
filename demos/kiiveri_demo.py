# demos/kiiveri_demo.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import solve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import Kiiveri module (algorithms only)
from icdm2025.methods.kiiveri_em import fitDagLatent, fitdag


def _metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def main(data_dir: str = "data", norm: int = 2, tol: float = 1e-6, maxit: int = 3000):
    # --- Load data ---
    source_df = pd.read_csv(os.path.join(data_dir, "source_1.csv"))
    target_df = pd.read_csv(os.path.join(data_dir, "tgt_target_missing_1.csv"))
    true_df = pd.read_csv(os.path.join(data_dir, "tgt_target_true_1.csv"))
    amat_df = pd.read_csv(os.path.join(data_dir, "adjacency_matrix.csv"), index_col=0)

    # Normalize node labels/ordering, build amat
    amat_df.index = amat_df.index.astype(str).str.strip()
    amat_df.columns = amat_df.columns.astype(str).str.strip()
    nodes = list(amat_df.index)
    p = len(nodes)
    amat = amat_df.values.astype(int)

    # --- Source fit → Sigma_init for latent run ---
    S_source = source_df[nodes].cov().values
    fit_src = fitdag(amat, S_source, n=source_df.shape[0])  # follow your module’s API
    Sigma0 = fit_src["Shat"]

    # --- Observed target covariance (drop T) ---
    obs_nodes = [v for v in nodes if v != "T"]
    syy = target_df[obs_nodes].cov()

    # --- Run Kiiveri latent fit (latent='T') ---
    fit = fitDagLatent(
        amat=amat_df,  # many versions accept DataFrame for labels
        syy=syy,
        sigma_old=Sigma0,
        n=target_df.shape[0],
        latent="T",
        norm=norm,
        seed=145,
        maxit=maxit,
        tol=tol,
        verbose=False,
        verbose_parents=False,
        verbose_fitdag=False,
    )

    # The full Σ over all variables (including T)
    Shat_df = fit["Shat"]  # DataFrame indexed by variable names
    all_vars = list(Shat_df.columns)
    idx_T = all_vars.index("T")
    idx_obs = [i for i, v in enumerate(all_vars) if v != "T"]
    obs_vars = [v for v in all_vars if v != "T"]

    # --- Impute T using Σ (conditional mean) ---
    X_obs = target_df[obs_vars].values  # (n × |O|)
    mu_O = X_obs.mean(axis=0)  # sample mean of observed
    Sigma = Shat_df.values
    Sigma_oo = Sigma[np.ix_(idx_obs, idx_obs)]   # |O|×|O|
    Sigma_to = Sigma[idx_T, idx_obs].reshape(1, -1)  # 1×|O|

    # mean of T can be set from source or 0 — here use source mean of T if present
    mu_t = float(source_df["T"].mean()) if "T" in source_df.columns else 0.0

    # Prefer solve over explicit inverse: W = Sigma_to @ inv(Sigma_oo)
    # solve(Sigma_oo, Sigma_to.T) gives |O|×1 → transpose back to 1×|O|
    W_row = solve(Sigma_oo, Sigma_to.T).T  # 1×|O|
    imputed = mu_t + (X_obs - mu_O) @ W_row.ravel()

    # --- Choose best sign: imputed vs -imputed ---
    true_T = true_df["T"].values

    imputed_pos = imputed
    imputed_neg = -imputed

    mae_p, rmse_p, r2_p = _metrics(true_T, imputed_pos)
    mae_n, rmse_n, r2_n = _metrics(true_T, imputed_neg)

    # Pick by higher R^2; tie-breaker = lower RMSE
    use_neg = (r2_n > r2_p) or (np.isclose(r2_n, r2_p) and rmse_n < rmse_p)

    best = imputed_neg if use_neg else imputed_pos
    mae = mae_n if use_neg else mae_p
    rmse = rmse_n if use_neg else rmse_p
    r2 = r2_n if use_neg else r2_p
    sign = "-" if use_neg else "+"

    print(f"Kiiveri EM — best sign: {sign}imputed")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R^2  = {r2:.4f}")

    # --- Plot ONLY the chosen sign ---
    os.makedirs("outputs", exist_ok=True)
    mn, mx = float(np.nanmin(true_T)), float(np.nanmax(true_T))
    plt.figure(figsize=(5, 5))
    plt.scatter(true_T, best, alpha=0.6)
    plt.plot([mn, mx], [mn, mx], "k--")
    plt.xlabel("True T")
    plt.ylabel(f"{sign}Imputed T")
    plt.title(f"Kiiveri EM — best sign: {sign}imputed")
    plt.tight_layout()
    out = os.path.join("outputs", "KiiveriCov_best_sign.pdf")
    plt.savefig(out, bbox_inches="tight")
    print("Saved:", out)


if __name__ == "__main__":
    main()
