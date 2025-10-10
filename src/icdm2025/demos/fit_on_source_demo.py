# Copyright (c) 2025 Mohammad Ali Javidian
# SPDX-License-Identifier: MIT
#
# This file is part of the ICDM2025 project.
# Licensed under the MIT License – see LICENSE in the repo root.
# demos/fit_on_source_demo.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from icdm2025.core.fit_on_source import fitdag  # uses lmfit internally

def main(data_dir: str = "data"):
    # 1) Load source data
    source_df = pd.read_csv(os.path.join(data_dir, "source_1.csv"))
    n_source  = source_df.shape[0]

    # 2) Load adjacency matrix and prepare nodes / amat
    amat_df = pd.read_csv(os.path.join(data_dir, "adjacency_matrix.csv"), index_col=0)
    amat_df.index   = amat_df.index.astype(str).str.strip()
    amat_df.columns = amat_df.columns.astype(str).str.strip()

    nodes = list(amat_df.index)
    p     = len(nodes)
    amat  = amat_df.values.astype(int)

    # parent_list with convention amat[j, i] == 1 implies j → i
    parent_list = [[j for j in range(p) if amat[j, i] == 1 and j != i] for i in range(p)]

    # Print parents for each node
    for i, parents in enumerate(parent_list):
        parent_names = [nodes[j] for j in parents]
        print(f"Node {i} ({nodes[i]}): parents idx={parents}  names={parent_names}")

    # 3) Fit the DAG on the source domain
    S_source = source_df[nodes].cov().values
    dag_fit  = fitdag(amat=amat, s=S_source, parent_list=parent_list)
    A_src    = dag_fit["A"]

    # 4) Extract β for T in SAME order as X_obs columns (parents of T)
    idx_T          = nodes.index("T")
    parents_idx_T  = parent_list[idx_T]
    parents_T      = [nodes[j] for j in parents_idx_T]
    beta_vec_ht    = np.array([-A_src[idx_T, j] for j in parents_idx_T], dtype=float)

    print("\nRegression coefficients learned for T on the SOURCE domain:")
    for j, parent in zip(parents_idx_T, parents_T):
        coeff = -A_src[idx_T, j]
        print(f"   beta[{parent} -> T] = {coeff:.6f}")
    print()

    # 5) Load target (T missing) and impute using mean-aware form
    target_df = pd.read_csv(os.path.join(data_dir, "tgt_target_missing_1.csv"))

    missing_cols = [c for c in parents_T if c not in target_df.columns]
    if missing_cols:
        raise ValueError(f"'tgt_target_missing_1.csv' missing required parent columns: {missing_cols}")

    X_obs = target_df[parents_T].values  # (n × num_parents)

    # Mean-aware prediction
    mu_parents_tgt = target_df[parents_T].mean(axis=0).values
    mu_t_baseline  = float(mu_parents_tgt @ beta_vec_ht)              # = beta^T mu_parents
    imputed_T_ma   = mu_t_baseline + (X_obs - mu_parents_tgt) @ beta_vec_ht
    imputed_T      = X_obs @ beta_vec_ht

    max_abs_diff = float(np.max(np.abs(imputed_T_ma - imputed_T)))
    print(f"Sanity check (mean-aware vs plain X@beta): max |diff| = {max_abs_diff:.3e}\n")

    # 6) Evaluate against true T
    targetT_df = pd.read_csv(os.path.join(data_dir, "tgt_target_true_1.csv"))
    true_T    = targetT_df["T"].values

    mask        = (~np.isnan(true_T)) & (~np.isnan(imputed_T_ma))
    true_T     = true_T[mask]
    imputed_T  = imputed_T_ma[mask]

    mae  = mean_absolute_error(true_T, imputed_T)
    mse  = mean_squared_error(true_T, imputed_T)
    rmse = np.sqrt(mse)
    r2   = r2_score(true_T, imputed_T)

    print("“Fit-on-Source (Parents-only)” baseline for T (mean-aware):")
    print(f"  MAE  = {mae:.4f}")
    print(f"  MSE  = {mse:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}\n")

    # 7) Scatter plot
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(true_T, imputed_T, alpha=0.6, label="Baseline (parents-only)")
    mn, mx = np.nanmin(true_T), np.nanmax(true_T)
    plt.plot([mn, mx], [mn, mx], "k--", lw=2)
    plt.xlabel("True T", fontsize=24)
    plt.ylabel("Predicted T", fontsize=24)
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.tight_layout()
    out = os.path.join("outputs", "T_baseline_scatter.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight")
    print("Saved:", out)

if __name__ == "__main__":
    main()
