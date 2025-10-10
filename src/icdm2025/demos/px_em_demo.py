# Copyright (c) 2025 Mohammad Ali Javidian
# SPDX-License-Identifier: MIT
#
# This file is part of the ICDM2025 project.
# Licensed under the MIT License – see LICENSE in the repo root.
# demos/px_em_demo.py
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from icdm2025.core.fit_on_source import fitdag  # reuse GLS DAG fit for Sigma_init
from icdm2025.methods.px_em import PX_EM_algorithm


def build_parent_list(amat: np.ndarray) -> list[list[int]]:
    p = amat.shape[0]
    return [[j for j in range(p) if amat[j, i] == 1 and j != i] for i in range(p)]


def main(data_dir: str = "data", tol: float = 1e-6, max_iter: int = 9000):
    # --- Load data ---
    source_df = pd.read_csv(os.path.join(data_dir, "source_1.csv"))
    target_df = pd.read_csv(os.path.join(data_dir, "cov_target_missing_1.csv"))
    true_df = pd.read_csv(os.path.join(data_dir, "cov_target_true_1.csv"))
    amat_df = pd.read_csv(os.path.join(data_dir, "adjacency_matrix.csv"), index_col=0)

    nodes = list(amat_df.index)
    p = len(nodes)
    amat = amat_df.values.astype(int)
    parent_list = build_parent_list(amat)
    idx_t = nodes.index("T")

    # --- Sigma_init from source via GLS under DAG ---
    S_source = source_df[nodes].cov().values
    Sigma_src = fitdag(amat, S_source, parent_list)["Shat"]

    # --- Observed target matrix (drop T) ---
    obs_cols = [v for v in nodes if v != "T"]
    X_obs = target_df[obs_cols].values
    n_tgt = X_obs.shape[0]

    # --- Run PX-EM ---
    dag_fit, Sigma_hat, iters, converged = PX_EM_algorithm(
        X_obs=X_obs,
        idx_t=idx_t,
        amat=amat,
        Sigma_init=Sigma_src,
        n=n_tgt,
        tol=tol,
        max_iter=max_iter,
        verbose_every=50,
    )

    # --- Impute T using final Sigma ---
    O = [i for i in range(p) if i != idx_t]
    Sigma_oo = Sigma_hat[np.ix_(O, O)]
    Sigma_to = Sigma_hat[np.ix_([idx_t], O)]
    imputed = (Sigma_to @ np.linalg.inv(Sigma_oo) @ X_obs.T).ravel()

    # --- Metrics ---
    true_T = true_df["T"].values
    mae = mean_absolute_error(true_T, imputed)
    rmse = np.sqrt(mean_squared_error(true_T, imputed))
    r2 = r2_score(true_T, imputed)
    print(f"PX-EM: conv={converged}, iters={iters}, MAE={mae:.4f}, RMSE={rmse:.4f}, R^2={r2:.4f}")

    # --- Plot ---
    os.makedirs("outputs", exist_ok=True)
    mn, mx = float(np.nanmin(true_T)), float(np.nanmax(true_T))
    plt.figure(figsize=(5, 5))
    plt.scatter(true_T, imputed, alpha=0.6)
    plt.plot([mn, mx], [mn, mx], "k--")
    plt.tight_layout()
    out = os.path.join("outputs", "PXEM_Cov.pdf")
    plt.savefig(out, bbox_inches="tight")
    print("Saved:", out)


if __name__ == "__main__":
    main()
