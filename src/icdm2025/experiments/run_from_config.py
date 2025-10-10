# src/icdm2025/experiments/run_from_config.py
from __future__ import annotations
import argparse, sys, os, yaml

def run(method: str, data_dir: str, outputs_dir: str, **kwargs):
    if method == "fit_on_source":
        from icdm2025.demos.fit_on_source_demo import main as fn
        return fn(data_dir=data_dir)
    if method == "first_order_em":
        from icdm2025.demos.first_order_em_demo import main as fn
        return fn(
            data_dir=data_dir,
            tol=kwargs.get("tol", 1e-6),
            max_iter=kwargs.get("max_iter", 50000),
        )
    if method == "ecme":
        from icdm2025.demos.ecme_demo import main as fn
        return fn(
            data_dir=data_dir,
            tol=kwargs.get("tol", 1e-6),
            max_iter=kwargs.get("max_iter", 3000),
        )
    if method == "px_em":
        from icdm2025.demos.px_em_demo import main as fn
        return fn(
            data_dir=data_dir,
            tol=kwargs.get("tol", 1e-6),
            max_iter=kwargs.get("max_iter", 9000),
        )
    if method == "kiiveri":
        from icdm2025.demos.kiiveri_demo import main as fn
        return fn(
            data_dir=data_dir,
            norm=kwargs.get("norm", 2),
            tol=kwargs.get("tol", 1e-6),
            maxit=kwargs.get("maxit", 3000),
        )
    raise SystemExit(f"Unknown method: {method}")

def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="Run ICDM2025 experiment from YAML config")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args(argv or sys.argv[1:])

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    data_dir = cfg.get("data_dir", "data")
    outputs_dir = cfg.get("outputs_dir", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    method = cfg["method"]
    extras = {k: v for k, v in cfg.items() if k not in {"data_dir", "outputs_dir", "method"}}
    return run(method=method, data_dir=data_dir, outputs_dir=outputs_dir, **extras)

if __name__ == "__main__":
    raise SystemExit(main())
