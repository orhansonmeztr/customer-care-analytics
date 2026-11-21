"""Command line entry point: ``care <step>`` or ``care run-all``."""

from __future__ import annotations

import argparse
import json
import time

from .paths import get_paths

STEPS = ["generate", "quality", "warehouse", "forecast", "segment", "churn", "anomaly"]


def _run(step: str, args) -> dict:
    paths = get_paths(args.data_dir)
    if step == "generate":
        from .generate import GenConfig, generate

        return generate(paths, GenConfig(scale=args.scale, seed=args.seed))
    if step == "quality":
        from .quality import run_quality

        return run_quality(paths)
    if step == "warehouse":
        from .warehouse import build_warehouse

        return build_warehouse(paths)
    if step == "forecast":
        from .forecast import run_forecast

        return run_forecast(paths, n_folds=args.folds)
    if step == "segment":
        from .segmentation import run_segmentation

        return run_segmentation(paths)
    if step == "churn":
        from .churn import run_churn

        return run_churn(paths)
    if step == "anomaly":
        from .anomaly import run_anomaly

        return run_anomaly(paths)
    raise ValueError(step)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="care", description=__doc__)
    ap.add_argument("step", choices=STEPS + ["run-all"])
    ap.add_argument("--data-dir", default=None, help="default: $CARE_DATA_DIR or ./data")
    ap.add_argument("--scale", type=float, default=1.0, help="data size (generate only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=12, help="forecast backtest folds")
    args = ap.parse_args(argv)
    steps = STEPS if args.step == "run-all" else [args.step]
    for step in steps:
        t0 = time.time()
        out = _run(step, args)
        print(f"[{step}] {time.time() - t0:.1f}s")
        print(json.dumps(out, indent=2, default=str)[:1500])


if __name__ == "__main__":
    main()
