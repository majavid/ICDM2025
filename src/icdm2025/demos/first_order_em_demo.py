# Copyright (c) 2025 Mohammad Ali Javidian
# SPDX-License-Identifier: MIT
#
# This file is part of the ICDM2025 project.
# Licensed under the MIT License – see LICENSE in the repo root.

# demos/first_order_em_demo.py
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from icdm2025.methods.first_order_em import EM_algorithm, conditional_params, fitdag


def main(data_dir: str = "data", tol: float = 1e-6, max_iter: int = 50_000) -> None:
    # --- Load data ---
    src = pd.read_csv(os.path.join(data_dir, "source_1.csv"))
    tgt_obs = pd.read_csv(os.path.join(data_dir, "cov_target_missing_1.csv"))
    tgt_true = pd.read_csv(os.path.join(data_dir, "cov_target_true_1.csv"))
    amat_df = pd.read_csv(os.path.join(data_dir, "adjacency_matrix.csv"), index_col=0)

    nodes = list(amat_df.index)
    p = len(nodes)
    amat = amat_df.values.astype(int)

    parent_list = [[j for j in range(p) if amat[j, i] == 1 and j != i] for i in range(p)]

    # --- Source fit for Sigma_init ---
    S_source = src[nodes].cov().values
    Sigma_src = fitdag(amat, S_source, parent_list)["Shat"]

    # --- Observed/target layout ---
    idx_t = nodes.index("T")
    obs_vars = [v for v in nodes if v != "T"]
    X_obs = tgt_obs[obs_vars].values

    # --- Mean initialization (source-aware) ---
    obs_idx, W_src, _, _ = conditional_params(Sigma_src, idx_t, ridge=0.0)
    mu_O_target = X_obs.mean(axis=0)
    mu_source = src[nodes].mean().values
    mu_t_init = float(mu_source[idx_t] + W_src @ (mu_O_target - mu_source[obs_idx]))

    mu_init = np.zeros(p)
    mu_init[obs_idx] = mu_O_target
    mu_init[idx_t] = mu_t_init

    # --- Run EM ---
    dag_fit, Sigma_hat, mu_hat, iters, converged = EM_algorithm(
        X_obs=X_obs,
        idx_t=idx_t,
        amat=amat,
        Sigma_init=Sigma_src,
        parent_list=parent_list,
        mu_init=mu_init,
        tol=tol,
        max_iter=max_iter,
        ridge=0.0,
        print_every=0,  # quieter/faster demo output
    )

    # --- Impute T using final params ---
    obs_idx, W_hat, _, _ = conditional_params(Sigma_hat, idx_t, ridge=0.0)
    mu_O_hat = mu_hat[obs_idx]
    mu_t_hat = float(mu_hat[idx_t])
    imputed = mu_t_hat + ((X_obs - mu_O_hat) @ W_hat)

    # --- Metrics ---
    true_T = tgt_true["T"].values
    mae = mean_absolute_error(true_T, imputed)
    rmse = np.sqrt(mean_squared_error(true_T, imputed))
    r2 = r2_score(true_T, imputed)
    print(
        f"1st-order EM: conv={converged}, iters={iters}, "
        f"MAE={mae:.4f}, RMSE={rmse:.4f}, R^2={r2:.4f}"
    )

    # --- Plot ---
    os.makedirs("outputs", exist_ok=True)
    mn, mx = float(np.nanmin(true_T)), float(np.nanmax(true_T))
    plt.figure(figsize=(6, 6))
    plt.scatter(true_T, imputed, alpha=0.6)
    plt.plot([mn, mx], [mn, mx], "k--")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "EM_T_mean_aware.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
