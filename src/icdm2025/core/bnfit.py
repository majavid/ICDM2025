# Copyright (c) 2025 Mohammad Ali Javidian
# SPDX-License-Identifier: MIT
#
# This file is part of the ICDM2025 project.
# Licensed under the MIT License – see LICENSE in the repo root.
# src/icdm2025/core/bnfit.py
from __future__ import annotations

# Re-export import-safe implementations (no I/O here)
from .fit_on_source import lmfit, fitdag

__all__ = ["lmfit", "fitdag"]

# Optional: dev-only smoke test (no filesystem access)
if __name__ == "__main__":
    import numpy as np
    # Tiny check: 2-node DAG with no parents
    s = np.array([[1.0, 0.0], [0.0, 1.0]])
    amat = np.zeros((2, 2), dtype=int)
    parent_list = [[], []]
    out = fitdag(amat, s, parent_list)
    print("bnfit dev smoke OK; Shat shape:", out["Shat"].shape)
