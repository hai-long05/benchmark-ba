#!/usr/bin/env python3
"""Paired item-level bootstrap on lm-eval --log_samples output.

Implements the bootstrap procedure specified in 04_methodik.tex §subsection
paired bootstrap (lines ~615-677): per (model, scheme, task), resample the
items with replacement (paired with the FP16 baseline of the same model on
the same task), compute the Quant-vs-FP16 difference per resample, report
percentile-CI from the resample distribution.

Methodology contract:
- The final thesis tables use B = 10000.
- Percentile CI: empirical 2.5/97.5 quantiles of Δ_b distribution.
- IFEval: prompt-level resampling. Per resample b, all four sub-metrics are
  recomputed according to the thesis composite: prompt-level strict/loose as
  prompt means, instruction-level strict/loose as fulfilled instructions divided
  by total instructions, then the arithmetic mean of these four values.
- WikiText: corpus-aggregated, exempted from item-bootstrap.
- Pairing: identical doc_id between Quant and FP16 within a resample.
- Random seed: --seed (default 0xBACE10AD); printed in output JSON.

Usage:
    python3 scripts/60_bootstrap.py \\
        --baseline-slot results/<...>-f16-ctx512 \\
        --quant-slot    results/<...>-q4_k_s-ctx512 \\
        --tasks         hellaswag,gsm8k,ifeval,mmlu,truthfulqa_mc2 \\
        --b             1000 \\
        --seed          0xBACE10AD \\
        --out           bootstrap_results/<model>_<scheme>.json

Or batch over an entire model:
    python3 scripts/60_bootstrap.py \\
        --baseline-slot results/<...>-f16-ctx512 \\
        --quant-glob    'results/*-meta-llama_llama-3.1-8b-instruct-*-ctx512' \\
        --out-dir       bootstrap_results/llama/

Output JSON per (model, scheme, task) carries:
    point_quant, point_fp16, delta, ci_low, ci_high, b, seed, n_items,
    contains_zero (bool), task_kind ('lm'|'ifeval'|'gen').
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# --- Task-specific metric extraction ----------------------------------------
#
# Each lm-eval task writes its per-doc metric values as top-level keys in the
# samples_<task>_<ts>.jsonl entries. The values may be 0/1 binaries (for
# multiple-choice or strict-match), floats (for partial-credit), or sub-dicts
# in some IFEval revisions. The list below captures the metric key(s) per task
# matching configs/tasks.yaml + Kurt's reporting.

TASK_METRIC_KEYS: dict[str, list[str]] = {
    # acc_norm: 1 if normalized loglikelihood ranks the gold continuation first.
    "hellaswag": ["acc_norm"],
    # exact_match: per-item binary score as stored in samples_gsm8k_*.jsonl
    "gsm8k": ["exact_match"],
    # IFEval: four binary sub-metrics per prompt. Resampled jointly.
    "ifeval": [
        "inst_level_loose_acc",
        "inst_level_strict_acc",
        "prompt_level_loose_acc",
        "prompt_level_strict_acc",
    ],
    # MMLU is reported as a single 'acc' per item across all subtasks.
    # Harness writes per-subtask sample files: samples_mmlu_<subj>_<ts>.jsonl.
    # We aggregate by reading all of them and treating each item as one doc.
    "mmlu": ["acc"],
    # TruthfulQA-MC2 has 'acc' per item.
    "truthfulqa_mc2": ["acc"],
}

# Tasks that bootstrap on the prompt level rather than per-item — IFEval has
# four correlated sub-metrics per prompt, so the methodology requires
# joint-resampling.
PROMPT_LEVEL_TASKS = {"ifeval"}

# Tasks exempted from item-bootstrap (corpus-aggregated, semantically wrong
# to resample at item granularity).
EXEMPTED = {"wikitext"}


def _find_samples_files(slot_dir: Path, task_key: str) -> list[Path]:
    """Find samples_<task>_<ts>.jsonl files under slot_dir/lm_eval/_group_*_raw/.

    For MMLU there are multiple subtask files (samples_mmlu_<subj>_<ts>.jsonl);
    we glob for any file whose name starts with samples_<task_root>_.
    """
    patterns = []
    if task_key == "mmlu":
        patterns.append(f"samples_mmlu_*.jsonl")
    else:
        patterns.append(f"samples_{task_key}_*.jsonl")
    found: list[Path] = []
    for pat in patterns:
        found.extend(slot_dir.rglob(pat))
    return sorted(found)


def _load_items(slot_dir: Path, task_key: str) -> list[dict[str, Any]]:
    """Load all per-doc samples for a task, return list of dicts with at least
    'doc_id' (or '_doc_id' if missing) and the metric keys for that task.
    """
    files = _find_samples_files(slot_dir, task_key)
    if not files:
        raise FileNotFoundError(
            f"no samples_{task_key}_*.jsonl found under {slot_dir} — "
            f"did 40_lm_eval.sh run with --log_samples?"
        )
    items: list[dict[str, Any]] = []
    for fp in files:
        # Subtask name (for MMLU we want to track which subject the doc came from
        # so we can pair MMLU items across runs even if doc_id collides between
        # subtasks). For non-MMLU this is just the task name.
        if task_key == "mmlu":
            # filename: samples_mmlu_<subject>_<timestamp>.jsonl
            stem = fp.stem  # "samples_mmlu_high_school_biology_2026-..."
            parts = stem.split("_")
            try:
                ts_idx = next(i for i, p in enumerate(parts) if p.startswith("20"))
                subtask = "_".join(parts[2:ts_idx])
            except StopIteration:
                subtask = "_".join(parts[2:])
        else:
            subtask = task_key
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                doc_id = d.get("doc_id")
                if doc_id is None:
                    doc_id = d.get("doc_hash")
                key = f"{subtask}::{doc_id}"
                d["_pair_key"] = key
                items.append(d)
    return items


def _build_pair(baseline_items: list[dict], quant_items: list[dict],
                metric_keys: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pair baseline and quant items by _pair_key. Return (baseline, quant, keys)
    arrays where each row is the per-item metric vector (shape: n_items × len(metric_keys))
    in the same order. Items present in only one slot are dropped with a warning.
    """
    base_by_key = {it["_pair_key"]: it for it in baseline_items}
    quant_by_key = {it["_pair_key"]: it for it in quant_items}
    common_keys = sorted(set(base_by_key.keys()) & set(quant_by_key.keys()))
    only_base = len(base_by_key) - len(common_keys)
    only_quant = len(quant_by_key) - len(common_keys)
    if only_base or only_quant:
        print(
            f"  WARN: dropped unpaired items — only_baseline={only_base}, "
            f"only_quant={only_quant}, paired={len(common_keys)}",
            file=sys.stderr,
        )
    if not common_keys:
        raise RuntimeError("no paired items between baseline and quant slots")

    def vec(item: dict) -> list[float]:
        out = []
        for mk in metric_keys:
            v = item.get(mk)
            if v is None:
                # Some IFEval revisions store sub-metrics under a different
                # canonical name; fail loudly so we don't silently ignore them.
                raise KeyError(
                    f"metric {mk!r} missing in sample doc_id={item.get('_pair_key')}"
                )
            # inst_level_* keys store a list of bools (one per instruction);
            # reduce to mean so the value is a scalar.
            if isinstance(v, list):
                v = float(np.mean([float(x) for x in v]))
            else:
                v = float(v)
            out.append(v)
        return out

    base_mat = np.array([vec(base_by_key[k]) for k in common_keys])
    quant_mat = np.array([vec(quant_by_key[k]) for k in common_keys])
    return base_mat, quant_mat, common_keys


def _pair_items(baseline_items: list[dict], quant_items: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    """Pair baseline and quant sample dictionaries by _pair_key."""
    base_by_key = {it["_pair_key"]: it for it in baseline_items}
    quant_by_key = {it["_pair_key"]: it for it in quant_items}
    common_keys = sorted(set(base_by_key.keys()) & set(quant_by_key.keys()))
    only_base = len(base_by_key) - len(common_keys)
    only_quant = len(quant_by_key) - len(common_keys)
    if only_base or only_quant:
        print(
            f"  WARN: dropped unpaired items — only_baseline={only_base}, "
            f"only_quant={only_quant}, paired={len(common_keys)}",
            file=sys.stderr,
        )
    if not common_keys:
        raise RuntimeError("no paired items between baseline and quant slots")
    return [base_by_key[k] for k in common_keys], [quant_by_key[k] for k in common_keys], common_keys


def _ifeval_arrays(items: list[dict]) -> dict[str, np.ndarray]:
    """Extract IFEval prompt-level and instruction-level components.

    IFEval's prompt-level metrics are one boolean per prompt. Instruction-level
    metrics are lists of booleans, one per instruction in the prompt. The thesis
    composite follows the harness aggregates: mean prompt-level strict/loose,
    plus total fulfilled instructions divided by total instructions for strict
    and loose, then the arithmetic mean of these four sub-metrics.
    """

    def scalar(item: dict, key: str) -> float:
        value = item.get(key)
        if value is None:
            raise KeyError(f"metric {key!r} missing in sample doc_id={item.get('_pair_key')}")
        return float(value)

    def instruction_counts(item: dict, key: str) -> tuple[float, float]:
        value = item.get(key)
        if value is None:
            raise KeyError(f"metric {key!r} missing in sample doc_id={item.get('_pair_key')}")
        if isinstance(value, list):
            return float(np.sum([float(v) for v in value])), float(len(value))
        return float(value), 1.0

    strict_counts = [instruction_counts(item, "inst_level_strict_acc") for item in items]
    loose_counts = [instruction_counts(item, "inst_level_loose_acc") for item in items]
    return {
        "prompt_strict": np.array([scalar(item, "prompt_level_strict_acc") for item in items]),
        "prompt_loose": np.array([scalar(item, "prompt_level_loose_acc") for item in items]),
        "inst_strict_num": np.array([num for num, _den in strict_counts]),
        "inst_strict_den": np.array([den for _num, den in strict_counts]),
        "inst_loose_num": np.array([num for num, _den in loose_counts]),
        "inst_loose_den": np.array([den for _num, den in loose_counts]),
    }


def _ifeval_composite(arrays: dict[str, np.ndarray]) -> float:
    prompt_strict = float(arrays["prompt_strict"].mean())
    prompt_loose = float(arrays["prompt_loose"].mean())
    inst_strict = float(arrays["inst_strict_num"].sum() / arrays["inst_strict_den"].sum())
    inst_loose = float(arrays["inst_loose_num"].sum() / arrays["inst_loose_den"].sum())
    return (prompt_strict + prompt_loose + inst_strict + inst_loose) / 4.0


def bootstrap_ifeval_paired(
    baseline_items: list[dict],
    quant_items: list[dict],
    *,
    b: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Paired prompt-level bootstrap for the thesis IFEval composite."""
    paired_base, paired_quant, _keys = _pair_items(baseline_items, quant_items)
    base = _ifeval_arrays(paired_base)
    quant = _ifeval_arrays(paired_quant)
    n = len(paired_base)
    idx = rng.integers(0, n, size=(b, n))

    def resampled_composite(arrays: dict[str, np.ndarray]) -> np.ndarray:
        prompt_strict = arrays["prompt_strict"][idx].mean(axis=1)
        prompt_loose = arrays["prompt_loose"][idx].mean(axis=1)
        inst_strict = arrays["inst_strict_num"][idx].sum(axis=1) / arrays["inst_strict_den"][idx].sum(axis=1)
        inst_loose = arrays["inst_loose_num"][idx].sum(axis=1) / arrays["inst_loose_den"][idx].sum(axis=1)
        return (prompt_strict + prompt_loose + inst_strict + inst_loose) / 4.0

    base_means = resampled_composite(base)
    quant_means = resampled_composite(quant)
    deltas = quant_means - base_means
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    point_fp16 = _ifeval_composite(base)
    point_quant = _ifeval_composite(quant)
    return {
        "point_fp16": float(point_fp16),
        "point_quant": float(point_quant),
        "delta": float(point_quant - point_fp16),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_items": int(n),
        "b": int(b),
    }


def bootstrap_paired(
    baseline_mat: np.ndarray,
    quant_mat: np.ndarray,
    *,
    b: int,
    rng: np.random.Generator,
    average_metrics: bool,
) -> dict[str, float]:
    """Paired bootstrap with shared item indices per resample.

    baseline_mat, quant_mat: shape (n_items, k_metrics). For multi-metric tasks
    (IFEval) k_metrics > 1 and we average across metrics AFTER resampling
    (preserving the per-prompt correlation).

    Returns dict with point_quant, point_fp16, delta, ci_low, ci_high.
    """
    n = baseline_mat.shape[0]
    if average_metrics:
        # Per-resample mean across items, then mean across the k_metrics axis.
        # This is the "compute all four sub-metrics on the SAME drawn sample,
        # then take the resample mean" requirement in §625-633.
        idx = rng.integers(0, n, size=(b, n))
        base_resample = baseline_mat[idx]   # (b, n, k)
        quant_resample = quant_mat[idx]     # (b, n, k)
        base_means = base_resample.mean(axis=(1, 2))   # (b,)
        quant_means = quant_resample.mean(axis=(1, 2)) # (b,)
    else:
        # Single-metric task: mean across items only.
        idx = rng.integers(0, n, size=(b, n))
        base_means = baseline_mat[idx, 0].mean(axis=1)
        quant_means = quant_mat[idx, 0].mean(axis=1)

    deltas = quant_means - base_means
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    return {
        "point_fp16": float(baseline_mat.mean()),
        "point_quant": float(quant_mat.mean()),
        "delta": float(quant_mat.mean() - baseline_mat.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_items": int(n),
        "b": int(b),
    }


def run_one_task(
    baseline_slot: Path,
    quant_slot: Path,
    task_key: str,
    *,
    b: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Run paired bootstrap on a single task.

    Returns a result dict with the bootstrap stats + meta. Caller writes it
    to disk in the format their pipeline expects.
    """
    if task_key in EXEMPTED:
        return {
            "task": task_key,
            "exempted": True,
            "reason": "corpus-aggregated; item-bootstrap not semantically valid",
        }

    base_items = _load_items(baseline_slot, task_key)
    quant_items = _load_items(quant_slot, task_key)

    if task_key == "ifeval":
        res = bootstrap_ifeval_paired(base_items, quant_items, b=b, rng=rng)
        res["task"] = task_key
        res["task_kind"] = "ifeval"
        res["metric_keys"] = TASK_METRIC_KEYS[task_key]
        res["contains_zero"] = bool(res["ci_low"] <= 0 <= res["ci_high"])
        return res

    metric_keys = TASK_METRIC_KEYS.get(task_key)
    if metric_keys is None:
        raise ValueError(f"unknown task_key: {task_key!r}")

    base_mat, quant_mat, _keys = _build_pair(base_items, quant_items, metric_keys)

    average_metrics = task_key in PROMPT_LEVEL_TASKS  # IFEval
    res = bootstrap_paired(
        base_mat, quant_mat, b=b, rng=rng, average_metrics=average_metrics,
    )
    res["task"] = task_key
    res["task_kind"] = "ifeval" if task_key in PROMPT_LEVEL_TASKS else "lm"
    res["metric_keys"] = metric_keys
    res["contains_zero"] = bool(res["ci_low"] <= 0 <= res["ci_high"])
    return res


def slot_id(slot_path: Path) -> dict[str, str]:
    """Read slot.json to label the result with model/scheme."""
    sj = json.load(open(slot_path / "slot.json"))
    return {
        "model": sj["model"]["name"],
        "scheme": sj["quant"]["scheme"],
        "ctx": sj["ctx"],
        "run_id": sj["run_id"],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline-slot", required=True, type=Path,
                   help="path to FP16 results/<run-id> directory")
    p.add_argument("--quant-slot", type=Path,
                   help="path to a single Quant results/<run-id> directory")
    p.add_argument("--quant-glob", type=str,
                   help="glob expression matching multiple Quant slot dirs (overrides --quant-slot)")
    p.add_argument("--tasks", default="hellaswag,gsm8k,ifeval,mmlu,truthfulqa_mc2",
                   help="comma-separated task keys (wikitext is auto-skipped)")
    p.add_argument("--b", type=int, default=1000,
                   help="bootstrap resamples (final thesis tables use 10000)")
    p.add_argument("--seed", type=lambda s: int(s, 0), default=0xBACE10AD,
                   help="random seed (default 0xBACE10AD)")
    p.add_argument("--out", type=Path, help="output JSON path (single quant slot)")
    p.add_argument("--out-dir", type=Path,
                   help="output directory; one JSON per quant slot named <scheme>.json")
    args = p.parse_args(argv)

    if args.quant_slot is None and args.quant_glob is None:
        p.error("either --quant-slot or --quant-glob is required")
    if args.out is None and args.out_dir is None:
        p.error("either --out or --out-dir is required")

    baseline_slot = args.baseline_slot.resolve()
    if not (baseline_slot / "slot.json").exists():
        p.error(f"baseline slot not found: {baseline_slot}")

    base_id = slot_id(baseline_slot)
    if base_id["scheme"] not in ("F16", "FP16"):
        print(f"WARN: --baseline-slot scheme is {base_id['scheme']!r}, not F16/FP16",
              file=sys.stderr)

    quant_slots: list[Path]
    if args.quant_glob is not None:
        quant_slots = [Path(p_).resolve() for p_ in sorted(_glob.glob(args.quant_glob))
                       if (Path(p_) / "slot.json").exists()
                       and slot_id(Path(p_))["scheme"] not in ("F16", "FP16")]
        if not quant_slots:
            p.error(f"no quant slots matched {args.quant_glob!r}")
    else:
        quant_slots = [args.quant_slot.resolve()]

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # One PCG64 seeded once; consumer per (slot, task) draws from the same stream
    # so the result is fully deterministic given (--seed, slot order, task order).
    rng = np.random.default_rng(args.seed)

    for qs in quant_slots:
        q_id = slot_id(qs)
        scheme = q_id["scheme"]
        out: dict[str, Any] = {
            "baseline": base_id,
            "quant": q_id,
            "seed": int(args.seed),
            "b": int(args.b),
            "tasks": {},
        }
        print(f"== {q_id['model']}/{scheme} (n_items×B={args.b}) ==")
        for task_key in tasks:
            try:
                res = run_one_task(baseline_slot, qs, task_key, b=args.b, rng=rng)
                out["tasks"][task_key] = res
                if res.get("exempted"):
                    print(f"  {task_key:18s}  EXEMPTED ({res['reason']})")
                else:
                    sig = "*" if not res["contains_zero"] else " "
                    print(f"  {task_key:18s}  Δ={res['delta']:+.4f}  "
                          f"95%CI=[{res['ci_low']:+.4f},{res['ci_high']:+.4f}]  {sig}")
            except FileNotFoundError as e:
                out["tasks"][task_key] = {"task": task_key, "error": str(e)}
                print(f"  {task_key:18s}  ERROR: {e}", file=sys.stderr)

        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            outpath = args.out_dir / f"{scheme}.json"
        else:
            outpath = args.out
        with open(outpath, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  → {outpath}")

    print("\nLegend: '*' = 95% CI excludes zero (significant difference vs FP16).")
    print("        Final thesis tables use --b 10000.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
