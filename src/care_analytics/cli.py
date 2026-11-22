"""Command line entry point: ``care <step>`` or ``care run-all``."""

from __future__ import annotations

import argparse
import json
import time

from .paths import get_paths

STEPS = [
    "generate", "quality", "warehouse", "forecast", "segment", "churn", "anomaly",
    "validate-staffing", "skills-routing",
]  # fmt: skip


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
    if step == "validate-staffing":
        from .staffing_validation import run_staffing_validation

        return run_staffing_validation(paths, days=args.sim_days)
    if step == "skills-routing":
        from .skills_routing import run_skills_routing

        return run_skills_routing(paths, days=args.sim_days)
    raise ValueError(step)


def _assistant(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="care")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="ask the analytics assistant a question")
    ask.add_argument("question")
    ask.add_argument("--explain", action="store_true", help="also write a short answer")
    ev = sub.add_parser("eval-assistant", help="run the golden-set evaluation")
    ev.add_argument("--ids", nargs="*", help="only these question ids")
    ev.add_argument("--set", dest="set_name", default="core", choices=["core", "hard"])
    for p in (ask, ev):
        p.add_argument("--data-dir", default=None)
        p.add_argument("--model", default=None, help="model name (default: $ASSISTANT_MODEL)")
        p.add_argument(
            "--provider",
            default=None,
            choices=["vllm", "deepseek", "openai"],
            help="default: $ASSISTANT_PROVIDER or vllm (local GPU node)",
        )
    args = ap.parse_args(argv)

    from .assistant.llm import default_model

    model = default_model(args.provider)
    if args.model:
        model.model = args.model
    paths = get_paths(args.data_dir)
    if args.cmd == "eval-assistant":
        from .assistant.evaluate import evaluate

        print(json.dumps(evaluate(paths, model, ids=args.ids, set_name=args.set_name), indent=2))
        return
    from .assistant.agent import Assistant
    from .assistant.guard import connect_read_only

    res = Assistant(connect_read_only(str(paths.warehouse)), model).ask(
        args.question, explain=args.explain
    )
    print(f"status: {res.status} ({res.attempts} attempt(s), {res.seconds:.1f} s)")
    if res.sql:
        print(f"\n{res.sql}\n")
    if res.rows is not None:
        print(res.rows.head(20).to_string(index=False))
    print(f"\n{res.answer}")


def main(argv: list[str] | None = None) -> None:
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("ask", "eval-assistant"):
        _assistant(argv)
        return
    ap = argparse.ArgumentParser(prog="care", description=__doc__)
    ap.add_argument("step", choices=STEPS + ["run-all"])
    ap.add_argument("--data-dir", default=None, help="default: $CARE_DATA_DIR or ./data")
    ap.add_argument("--scale", type=float, default=1.0, help="data size (generate only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=12, help="forecast backtest folds")
    ap.add_argument(
        "--sim-days",
        type=int,
        default=28,
        help="simulated days (validate-staffing, skills-routing)",
    )
    args = ap.parse_args(argv)
    steps = STEPS if args.step == "run-all" else [args.step]
    for step in steps:
        t0 = time.time()
        out = _run(step, args)
        print(f"[{step}] {time.time() - t0:.1f}s")
        print(json.dumps(out, indent=2, default=str)[:1500])


if __name__ == "__main__":
    main()
