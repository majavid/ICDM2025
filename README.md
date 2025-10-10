# EM-Based Transfer Learning for Gaussian Causal Models Under Covariate and Target Shift

Code and experiments for our paper:

> **“EM-Based Transfer Learning for Gaussian Causal Models Under Covariate and Target Shift” (ICDM 2025, regular paper)**

---

## ✨ What’s in this repo
- Implementations:
  - Kiiveri EM (classical latent-variable EM)
  - **1st-order EM** (gradient/GEM variant)
  - ECME and PX-EM accelerations
- Reproducible experiments for:
  - Synthetic 7-node SEM
  - 64-node MAGIC-IRRI network
  - Sachs single-cell signaling dataset
- Utilities for DAG-constrained Gaussian SEM fitting

> If you’re here to **impute a fully missing target variable `T`** in a known DAG under domain shift, jump to [Quickstart](#quickstart).

---

## 📦 Environment

- Python >= 3.10
- See [`requirements.txt`](./requirements.txt) for runtime deps
- Docs deps in [`docs/requirements.txt`](./docs/requirements.txt)

Docker (optional):
- We provide a Docker build recipe in `DockerfileDocs` (rename to `Dockerfile` if you prefer).
- Add any system libs your extras need (BLAS, graphviz, etc.).

---

## 🚀 Quickstart

```bash
# 1) create env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) run a small synthetic demo
python src/demos/demo_synthetic_first_order_em.py \
  --seed 42 --iters 1_000 --tol 1e-6

# 3) reproduce a Sachs run (example)
python src/experiments/sachs/run_em_pipeline.py \
  --config configs/sachs/t_as_latent.yaml

Outputs (plots, CSV metrics) land in outputs/ by default.

📁 Repo layout
.
├─ src/
│  ├─ core/          # SEM/DAG primitives: lmfit, fitdag, E-step, etc.
│  ├─ methods/       # kiiveri_em.py, first_order_em.py, ecme.py, px_em.py
│  ├─ experiments/   # scripts to reproduce paper tables/figures
│  └─ demos/         # small runnable examples
├─ tests/            # unit tests (pytest)
├─ configs/          # YAML configs for experiments
├─ data/             # (empty) put datasets or symlinks here; see below
├─ docs/             # Sphinx (or mkdocs) documentation
├─ requirements.txt
├─ docs/requirements.txt
├─ DockerfileDocs
└─ README.md

📚 Datasets

Sachs: see instructions in data/README.md (we do not commit raw data).

MAGIC-IRRI: pointers and preprocessing scripts in data/magic_irri/.

Synthetic: generated on the fly by scripts in src/experiments/synthetic/.

🧠 Methods overview

Kiiveri EM: classic EM with closed-form E-step for one latent (the target T) and GLS M-step.

First-order EM: one gradient ascent step on the DAG-constrained covariance per iteration — O(p²) per step.

ECME: observed-likelihood maximization for the variance of T to accelerate convergence.

PX-EM: parameter expansion in the T direction to improve curvature.

Each method exposes a unified interface:
dag_fit, Sigma_hat, (mu_hat), iters, converged = method.fit(X_obs, idx_t, amat, Sigma_init, **kwargs)

🧪 Reproducing paper results

See src/experiments/* and the corresponding configs/*.yaml.
We export:

Per-experiment CSVs with MAE, RMSE, R²

Scatter plots (true vs imputed T)

Logs with likelihood/gap diagnostics

Example:
python src/experiments/magic_irri/run_magic.py --config configs/magic_irri/cov_shift.yaml

🔬 Citing

If you use this code, please cite:
@inproceedings{javidian2025emtransfer,
  title={EM-Based Transfer Learning for Gaussian Causal Models Under Covariate and Target Shift},
  author={Javidian, Mohammad Ali},
  booktitle={IEEE International Conference on Data Mining (ICDM)},
  year={2025}
}

📄 License

MIT — see the [LICENSE](./LICENSE) file.
SPDX-License-Identifier: MIT

🤝 Contributing

PRs and issues welcome. Run pytest before submitting.

📨 Contact

Mohammad Ali Javidian — javidianma@appstate.edu
