#!/usr/bin/env python3
"""Aggregate one slot directory's stage outputs into a single results.json row.

Reads:
  <slot>/slot.json
  <slot>/bench.json
  <slot>/lm_eval/<task>.json   (zero or more)

Writes:
  <slot>/results.json

Refuses to merge when the slot's recorded env hash does not match the current env hash.
The expected env hash is read from $EXPECTED_ENV_HASH if set (used by tests); otherwise it
is computed from the env/ directory adjacent to the current working directory.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


_BITS_RE = re.compile(r"Q(\d)")


def nominal_bits(scheme: str) -> int:
    m = _BITS_RE.match(scheme.upper())
    if not m:
        raise ValueError(f"cannot parse bits from scheme: {scheme}")
    return int(m.group(1))


def _current_env_hash() -> str:
    if "EXPECTED_ENV_HASH" in os.environ:
        return os.environ["EXPECTED_ENV_HASH"]
    # Call the shared helper.
    helper = Path(__file__).parent / "lib" / "env_hash.sh"
    out = subprocess.run(
        ["bash", "-c", f"source {helper} && env_hash env"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def _extract_task_metrics(lm_eval_path: Path) -> dict[str, Any]:
    """Pull metrics out of an lm-eval results.json into a flat dict.

    lm-eval keys are of the form "<metric>,<filter>". The trivial filter ",none"
    carries no information and is stripped. Non-trivial filters (e.g. ",strict-match"
    and ",flexible-extract" on GSM8K) are preserved with an underscore separator
    so the two scores do not collide on the same metric name.
    """
    data = json.loads(lm_eval_path.read_text())
    results = data.get("results", {})
    flat: dict[str, dict[str, float]] = {}
    for task_name, metrics in results.items():
        entry: dict[str, float] = {}
        for key, value in metrics.items():
            if key == "alias" or not isinstance(value, (int, float)):
                continue
            if "," in key:
                metric, _, filter_name = key.partition(",")
                base = metric if filter_name == "none" else f"{metric}_{filter_name.replace('-', '_')}"
            else:
                base = key
            entry[base] = float(value)
        flat[task_name] = entry
    return flat


def aggregate(slot_dir: Path) -> dict[str, Any]:
    slot = json.loads((slot_dir / "slot.json").read_text())
    bench = json.loads((slot_dir / "bench.json").read_text())

    expected = _current_env_hash()
    if slot["host_id"] != expected:
        raise RuntimeError(
            f"env drift: slot.host_id={slot['host_id']!r} but current env_hash={expected!r}"
        )

    tasks: dict[str, dict[str, float]] = {}
    lm_eval_dir = slot_dir / "lm_eval"
    if lm_eval_dir.is_dir():
        for f in sorted(lm_eval_dir.glob("*.json")):
            tasks.update(_extract_task_metrics(f))

    # WikiText-2 PPL (when present) lives in tasks under the wikitext task.
    ppl = None
    if "wikitext" in tasks:
        ppl = tasks["wikitext"].get("word_perplexity") or tasks["wikitext"].get("perplexity")

    return {
        "run_id": slot["run_id"],
        "model": slot["model"]["name"],
        "quant": slot["quant"]["scheme"],
        "bits_nominal": nominal_bits(slot["quant"]["scheme"]),
        "ctx": slot["ctx"],
        "size_mib": slot["quant"]["size_mib"],
        "size_reduction_pct": slot["quant"]["size_reduction_pct"],
        "quant_time_s": slot["quant"]["quant_time_s"],
        "pp": bench["pp"],
        "tg": bench["tg"],
        "tasks": tasks,
        "ppl_wikitext2": ppl,
        "env": {
            "host_id": slot["host_id"],
            "llama_cpp_commit": slot["llama_cpp_commit"],
            "lm_eval_version": slot["lm_eval_version"],
        },
        "warnings": list(bench.get("warnings", [])),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: {argv[0]} <slot-dir>", file=sys.stderr)
        return 2
    slot_dir = Path(argv[1]).resolve()
    if not slot_dir.is_dir():
        print(f"slot dir not found: {slot_dir}", file=sys.stderr)
        return 2
    try:
        result = aggregate(slot_dir)
    except RuntimeError as e:
        print(f"aggregate refused: {e}", file=sys.stderr)
        return 3
    out = slot_dir / "results.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
