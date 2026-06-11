"""Aggregate llama-bench --output json across repeats into median + IQR for pp and tg.

llama-bench's JSON is an array of result rows; each row has n_prompt, n_gen, and avg_ts (tokens/s).
Rows with n_gen == 0 are prompt-processing (pp) measurements; rows with n_prompt == 0 are
token-generation (tg) measurements. We aggregate per phase across repeats.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _samples(rows: list[dict[str, Any]], phase: str) -> list[float]:
    if phase == "pp":
        return [float(r["avg_ts"]) for r in rows if r.get("n_gen", 0) == 0 and r.get("n_prompt", 0) > 0]
    if phase == "tg":
        return [float(r["avg_ts"]) for r in rows if r.get("n_prompt", 0) == 0 and r.get("n_gen", 0) > 0]
    raise ValueError(f"unknown phase: {phase}")


def _aggregate_phase(samples: list[float]) -> tuple[float | None, float | None, list[float]]:
    if not samples:
        return None, None, samples
    median = float(np.median(samples))
    if len(samples) < 2:
        return median, None, samples
    q75, q25 = np.percentile(samples, [75, 25])
    iqr = float(q75 - q25)
    return median, iqr, samples


def aggregate(rows: list[dict[str, Any]], ctx: int) -> dict[str, Any]:
    pp_samples = _samples(rows, "pp")
    tg_samples = _samples(rows, "tg")
    pp_median, pp_iqr, _ = _aggregate_phase(pp_samples)
    tg_median, tg_iqr, _ = _aggregate_phase(tg_samples)

    warnings: list[str] = []
    if len(pp_samples) < 2 or len(tg_samples) < 2:
        warnings.append(f"repeats=1 — IQR undefined (pp={len(pp_samples)}, tg={len(tg_samples)})")
    if pp_median and pp_iqr is not None and pp_iqr > 0.05 * pp_median:
        warnings.append(f"pp IQR {pp_iqr:.3f} exceeds 5% of median {pp_median:.3f}")
    if tg_median and tg_iqr is not None and tg_iqr > 0.05 * tg_median:
        warnings.append(f"tg IQR {tg_iqr:.3f} exceeds 5% of median {tg_median:.3f}")

    return {
        "pp": {"ctx": ctx, "median": pp_median, "iqr": pp_iqr, "samples": pp_samples},
        "tg": {"median": tg_median, "iqr": tg_iqr, "samples": tg_samples},
        "warnings": warnings,
    }
