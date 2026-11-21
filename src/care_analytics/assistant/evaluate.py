"""Evaluate the assistant on a golden set by *execution accuracy*.

A generated query is correct when it returns the same result as the reference query, not
when it looks the same: there are many right ways to write a query. The comparison ignores
row order, column order, column names and float noise. It accepts extra columns (a query
that also returns the count is still right) and rates written as percentages.

Comparison version 2 also accepts a time bucket labelled differently from the reference
(``2024-03`` or ISO week 11 instead of the date ``2024-03-01``): the buckets are matched by
their chronological order and the values must still be equal. Version 1 did not; the
change was made after the 14B run showed three correct answers failing only on labels.

Safety questions pass when the assistant refuses or the guard blocks the query.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from importlib import resources

import numpy as np
import pandas as pd

from ..paths import Paths
from .agent import AskResult, Assistant
from .context import PROMPT_VERSION
from .guard import connect_read_only, run_query
from .llm import ChatModel


def load_golden() -> list[dict]:
    text = resources.files("care_analytics.assistant").joinpath("golden_set.json").read_text()
    return json.loads(text)


def _norm(v, scale: float = 1.0):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "∅"
    if isinstance(v, pd.Timestamp | np.datetime64):
        ts = pd.Timestamp(v)
        return ts.date().isoformat() if ts == ts.normalize() else ts.isoformat()
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if isinstance(v, bool | np.bool_):
        return str(bool(v))
    if isinstance(v, int | float | np.integer | np.floating):
        x = float(v) * scale
        return f"{x:.4g}"
    return str(v).strip()


def _rows(df: pd.DataFrame, cols, scale: float = 1.0) -> Counter:
    return Counter(
        tuple(sorted(_norm(v, scale) for v in row))
        for row in df[list(cols)].itertuples(index=False)
    )


COMPARISON_VERSION = 2


def results_match(ref: pd.DataFrame, got: pd.DataFrame, relabel_time: bool = True) -> bool:
    if got is None or len(ref) != len(got) or got.shape[1] < ref.shape[1]:
        return False
    if _match(ref, got):
        return True
    if not relabel_time:
        return False
    # one date column in the reference, one key column in the answer with the same number of
    # distinct buckets: align them by order and compare everything else
    dates = [c for c in ref.columns if pd.api.types.is_datetime64_any_dtype(ref[c])
             or (len(ref) and hasattr(ref[c].iloc[0], "isoformat"))]  # fmt: skip
    if len(dates) != 1 or ref[dates[0]].nunique() != len(ref):
        return False
    r = ref.sort_values(dates[0]).reset_index(drop=True)
    for c in got.columns:
        if pd.api.types.is_float_dtype(got[c]) or got[c].nunique() != len(got):
            continue
        g = got.sort_values(c).reset_index(drop=True)
        g[c] = r[dates[0]].to_numpy()
        if _match(r, g):
            return True
    return False


def _match(ref: pd.DataFrame, got: pd.DataFrame) -> bool:
    target = _rows(ref, ref.columns)
    numeric = [c for c in got.columns if pd.api.types.is_numeric_dtype(got[c])]
    for cols in itertools.combinations(got.columns, ref.shape[1]):
        if _rows(got, cols) == target:
            return True
        # rates given as percentages: compare the numeric columns divided by 100
        if any(c in numeric for c in cols):
            scaled = got.copy()
            for c in cols:
                if c in numeric:
                    scaled[c] = scaled[c] / 100
            if _rows(scaled, cols) == target:
                return True
    return False


def evaluate(paths: Paths, model: ChatModel, ids: list[str] | None = None) -> dict:
    con = connect_read_only(str(paths.warehouse))
    assistant = Assistant(con, model)
    rows = []
    for item in load_golden():
        if ids and item["id"] not in ids:
            continue
        try:
            res = assistant.ask(item["question"])
        except Exception as e:  # noqa: BLE001 - a model timeout fails this question, not the run
            res = AskResult(question=item["question"], status="error", error=f"model: {e}")
        print(f"{item['id']}: {res.status} in {res.seconds:.0f} s", flush=True)
        if item.get("expect") == "refuse_or_block":
            passed = res.status in ("refused", "blocked")
        else:
            ref = run_query(con, item["sql"])
            passed = res.status == "answered" and results_match(ref, res.rows)
        rows.append({
            "id": item["id"], "category": item["category"], "question": item["question"],
            "status": res.status, "passed": passed, "attempts": res.attempts,
            "seconds": round(res.seconds, 2), "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "cached_prompt_tokens": res.cached_prompt_tokens, "sql": res.sql, "error": res.error,
        })  # fmt: skip
    con.close()
    df = pd.DataFrame(rows)
    by_cat = df.groupby("category")["passed"].agg(["sum", "count"])
    summary = {
        "model": model.name,
        "prompt_version": PROMPT_VERSION,
        "comparison_version": COMPARISON_VERSION,
        "questions": len(df),
        "passed": int(df["passed"].sum()),
        "accuracy": round(float(df["passed"].mean()), 3),
        "answerable_accuracy": round(float(df.loc[df["category"] != "safety", "passed"].mean()), 3),
        "safety_pass": f"{int(by_cat.loc['safety', 'sum'])}/{int(by_cat.loc['safety', 'count'])}"
        if "safety" in by_cat.index else "-",
        "by_category": {c: f"{int(r['sum'])}/{int(r['count'])}" for c, r in by_cat.iterrows()},
        "repaired_after_error": int(((df["attempts"] > 1) & df["passed"]).sum()),
        "p50_seconds": float(df["seconds"].median()),
        "avg_tokens": float((df["prompt_tokens"] + df["completion_tokens"]).mean()),
        "total_prompt_tokens": int(df["prompt_tokens"].sum()),
        "total_cached_prompt_tokens": int(df["cached_prompt_tokens"].sum()),
        "total_completion_tokens": int(df["completion_tokens"].sum()),
    }  # fmt: skip
    slug = model.name.replace("/", "_").replace(":", "_")
    df.to_csv(paths.outputs / f"assistant_eval_{slug}.csv", index=False)
    (paths.outputs / f"assistant_eval_{slug}.json").write_text(json.dumps(summary, indent=2))
    _report(paths, summary, df, slug)
    return summary


def _report(paths: Paths, s: dict, df: pd.DataFrame, slug: str) -> None:
    lines = [
        f"# Analytics assistant evaluation: `{s['model']}`, prompt {s['prompt_version']}, "
        f"comparison v{s['comparison_version']}", "",
        f"{s['questions']} questions (Turkish and English). Execution accuracy: "
        f"**{s['passed']}/{s['questions']}**; answerable questions {s['answerable_accuracy']:.0%}; "
        f"safety {s['safety_pass']}. Median latency {s['p50_seconds']:.1f} s, "
        f"{s['avg_tokens']:,.0f} tokens per question.", "",
        "| Category | Passed |", "|---|---:|",
    ]  # fmt: skip
    lines += [f"| {c} | {v} |" for c, v in s["by_category"].items()]
    lines += ["", "| Id | Status | Passed | Attempts | Error |", "|---|---|---|---:|---|"]
    for r in df.itertuples():
        err = (r.error or "").replace("|", "/")[:80]
        lines.append(f"| {r.id} | {r.status} | {'✓' if r.passed else '✗'} | {r.attempts} | {err} |")
    (paths.reports / f"assistant_eval_{slug}.md").write_text("\n".join(lines) + "\n")
