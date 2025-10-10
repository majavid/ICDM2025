# Copyright (c) 2025 Mohammad Ali Javidian
# SPDX-License-Identifier: MIT
#
# This file is part of the ICDM2025 project.
# Licensed under the MIT License – see LICENSE in the repo root.

# src/icdm2025/demos/fit_on_source_demo.py
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from icdm2025.core.fit_on_source import fitdag


def main(data_dir: str = "data") -> None:
    # 1) Load source data
    source_df = pd.read_csv(os.path.join(data_dir, "source_1.csv"))

    # 2) Load adjacency matrix and prepare nodes / amat
    amat_df = pd.read_csv(os.path.join(data_dir, "adjacency_matrix.csv"), index_col=0)
    amat_df.index = amat_df.index.astype(str).str.strip()
    amat_df.columns = amat_df.columns.astype(str).str.strip()
    nodes = list(amat_df.index)
    p = len(nodes)
    amat = amat_df.values.astype(int)

    # 2a) Build parent list
    parent_list = [[j for j in range(p) if amat[j, i] == 1 and j != i] for i in range(p)]

    # 3) Fit the DAG on the source domain (covariance with ddof=1)
    S_source = source_df[nodes].cov().values
    dag_fit = fitdag(amat=amat, s=S_source, parent_list=parent_list)
    A_src = dag_fit["A"]

    # 4) Extract β-coefficients for T
    idx_T = nodes.index("T")
    parents_idx_T = parent_list[idx_T]
    parents_T = [nodes[j] for j in parents_idx_T]
    beta_vec_ht = np.array([-A_src[idx_T, j] for j in parents_idx_T], dtype=float)

    # 5) Load target (T missing) and impute using parents-only
    target_df = pd.read_csv(os.path.join(data_dir, "tgt_target_missing_1.csv"))
    missing_cols = [c for c in parents_T if c not in target_df.columns]
    if missing_cols:
        raise ValueError(f"'tgt_target_missing_1.csv' is missing required columns: {missing_cols}")

    X_obs = target_df[parents_T].values
    imputed_T = X_obs @ beta_vec_ht

    # 6) Evaluate against true T
    targetT_df = pd.read_csv(os.path.join(data_dir, "tgt_target_true_1.csv"))
    true_T = targetT_df["T"].values

    mae = mean_absolute_error(true_T, imputed_T)
    rmse = np.sqrt(mean_squared_error(true_T, imputed_T))
    r2 = r2_score(true_T, imputed_T)

    print("Fit-on-Source (parents-only) baseline for T:")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}")

    # 7) Plot: True vs Predicted T
    os.makedirs("outputs", exist_ok=True)
    mn, mx = float(np.nanmin(true_T)), float(np.nanmax(true_T))
    plt.figure(figsize=(6, 6))
    plt.scatter(true_T, imputed_T, alpha=0.6, label="Parents-only")
    plt.plot([mn, mx], [mn, mx], "k--", lw=2)
    plt.xlabel("True T")
    plt.ylabel("Predicted T")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "T_baseline_scatter.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
