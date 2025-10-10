# Copyright (c) 2025 Mohammad Ali Javidian
# SPDX-License-Identifier: MIT
#
# This file is part of the ICDM2025 project.
# Licensed under the MIT License – see LICENSE in the repo root.
# src/icdm2025/cli.py
from __future__ import annotations
import argparse
import sys

def _add_common(parser: argparse.ArgumentParser):
    parser.add_argument("--data_dir", default="data", help="Path to data folder")

def cmd_fit_on_source(args):
    from icdm2025.demos.fit_on_source_demo import main
    main(data_dir=args.data_dir)

def cmd_first_order_em(args):
    from icdm2025.demos.first_order_em_demo import main
    main(data_dir=args.data_dir)

def cmd_ecme(args):
    from icdm2025.demos.ecme_demo import main
    main(data_dir=args.data_dir)

def cmd_px_em(args):
    from icdm2025.demos.px_em_demo import main
    main(data_dir=args.data_dir)

def cmd_kiiveri(args):
    from icdm2025.demos.kiiveri_demo import main
    main(data_dir=args.data_dir)

def main(argv: list[str] | None = None):
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(prog="icdm2025-demo", description="Run ICDM 2025 demos")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("fit-on-source", help="Baseline: fit on source (parents-only)")
    _add_common(sp); sp.set_defaults(func=cmd_fit_on_source)

    sp = sub.add_parser("first-order-em", help="First-order EM (mean-aware)")
    _add_common(sp); sp.set_defaults(func=cmd_first_order_em)

    sp = sub.add_parser("ecme", help="ECME")
    _add_common(sp); sp.set_defaults(func=cmd_ecme)

    sp = sub.add_parser("px-em", help="PX-EM")
    _add_common(sp); sp.set_defaults(func=cmd_px_em)

    sp = sub.add_parser("kiiveri", help="Kiiveri latent EM (fitDagLatent)")
    _add_common(sp); sp.set_defaults(func=cmd_kiiveri)

    args = p.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
