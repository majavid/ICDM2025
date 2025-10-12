# EM-Based Transfer Learning for Gaussian Causal Models Under Covariate & Target Shift

Code and experiments for the paper:

> **“EM-Based Transfer Learning for Gaussian Causal Models Under Covariate and Target Shift”** (ICDM 2025, regular paper)

**Docs:** https://majavid.github.io/ICDM2025/
![Run demos](https://github.com/majavid/ICDM2025/actions/workflows/run-demos.yml/badge.svg)
[![Docs](https://github.com/majavid/ICDM2025/actions/workflows/docs.yml/badge.svg)](https://majavid.github.io/ICDM2025/)

---

## ✨ What’s in this repo

- **Methods**
  - Kiiveri EM (classical latent-variable EM)
  - **First-order EM** (GEM/gradient variant)
  - ECME and PX-EM accelerations
- **Reproducible demos/configs** (CLI) for:
  - Synthetic 7-node SEM
  - 64-node MAGIC-IRRI network
  - Sachs single-cell signaling dataset
- **Utilities** for DAG-constrained Gaussian SEM fitting
- **Supplementary Materials (PDF):**
[In repo](SupplementaryMaterials.pdf) ·
[Hosted](https://majavid.github.io/ICDM2025/_static/SupplementaryMaterials.pdf)


If you’re here to **impute a fully missing target `T`** under domain shift with a known DAG, jump to **Quickstart**.

---

## 📦 Environment

- Python **≥ 3.10**
- Runtime deps: see [`requirements.txt`](./requirements.txt)
- Docs deps: see [`docs/requirements.txt`](./docs/requirements.txt)

> Windows note: activate your venv with `.\.venv\Scripts\activate`. On macOS/Linux use `source .venv/bin/activate`.

---

## 🚀 Quickstart

### 1) Create & activate a virtualenv, install in editable mode

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
pip install -e .[dev,docs]  # CLI, tests, docs
````

### 2) Run demos (tiny synthetic example)

```bash
# Baseline: fit on source (parents-only)
icdm2025-demo fit-on-source --data_dir data

# First-order EM (mean-aware)
icdm2025-demo first-order-em --data_dir data

# ECME and PX-EM
icdm2025-demo ecme --data_dir data
icdm2025-demo px-em --data_dir data

# Kiiveri latent EM (fitDagLatent)
icdm2025-demo kiiveri --data_dir data
```

Artifacts (plots) land in `outputs/`.

### 3) Reproduce via config (paper-style runs)

```bash
icdm2025-run --config configs/first_order_em.yaml
icdm2025-run --config configs/ecme.yaml
icdm2025-run --config configs/px_em.yaml
icdm2025-run --config configs/kiiveri.yaml
```

---

## 📁 Repo layout

```
.
├─ src/icdm2025/
│  ├─ core/            # DAG/SEM primitives (fitdag, etc.)
│  ├─ methods/         # first_order_em.py, ecme.py, px_em.py, kiiveri_em.py
│  ├─ demos/           # CLI-backed demo scripts
│  └─ experiments/     # run_from_config.py (used by icdm2025-run)
├─ configs/            # YAML configs for reproducible runs
├─ data/               # small CSVs for demos; larger datasets not committed
├─ docs/               # Sphinx sources (docs/source/) → GitHub Pages
├─ tests/              # pytest smoke tests
├─ outputs/            # generated figures/artifacts (gitignored)
├─ requirements.txt
└─ README.md
```

---

## 📚 Datasets

* **Synthetic:** CSVs included under `data/` for quick demos.
* **Sachs:** follow instructions in `docs/` (or add your own CSVs under `data/` with the same column names).
* **MAGIC-IRRI:** provide CSVs under `data/` (see configs for expected columns / adjacency).

You can replace the provided `data/*.csv` with your own, as long as the **column names** and **adjacency matrix** (`data/adjacency_matrix.csv`) match.

---

## 🧠 Methods (very brief)

* **Kiiveri EM:** classic EM with closed-form E-step for one latent (`T`) and GLS M-step.
* **First-order EM:** gradient/GEM update on the DAG-constrained covariance (cheap O(p²) step).
* **ECME:** maximizes observed likelihood wrt the variance of `T` for faster convergence.
* **PX-EM:** parameter expansion along `T` to improve curvature/conditioning.

All integrate with a common pipeline; see `icdm2025-run` configs for knobs like `tol`, `max_iter`, `norm`, etc.

---

## 🧪 Reproducing results

* Use the **configs** in `configs/*.yaml` with `icdm2025-run` (above).
* Outputs include **MAE / RMSE / R²** CSV lines and **PDF plots** (true vs imputed `T`).

---

## 🧾 Citation

If you use this software, please cite:

```bibtex
@inproceedings{javidian2025emtransfer,
  title={EM-Based Transfer Learning for Gaussian Causal Models Under Covariate and Target Shift},
  author={Javidian, Mohammad Ali},
  booktitle={IEEE International Conference on Data Mining (ICDM)},
  year={2025}
}
```

Also see the repository’s [`CITATION.cff`](./CITATION.cff).

---

## 📄 License

**MIT** — see the [LICENSE](./LICENSE).
SPDX-License-Identifier: MIT

---

## 🤝 Contributing

PRs and issues welcome. Before submitting:

```bash
pytest -q
pre-commit run --all-files
```

---

## 📨 Contact

**Mohammad Ali Javidian** — [javidianma@appstate.edu](mailto:javidianma@appstate.edu)
