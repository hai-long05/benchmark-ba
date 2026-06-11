"""Tests for scripts/lib/bench_parse.py — aggregating llama-bench --output json across repeats."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "lib"))
import bench_parse  # noqa: E402


def test_aggregate_pp_and_tg_median_iqr():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "llama_bench_repeats.json").read_text())
    out = bench_parse.aggregate(fixture, ctx=512)

    # pp samples: 92, 93, 94 -> median 93, IQR = 1.0
    assert out["pp"]["ctx"] == 512
    assert sorted(out["pp"]["samples"]) == [92.0, 93.0, 94.0]
    assert out["pp"]["median"] == 93.0
    assert out["pp"]["iqr"] == 1.0

    # tg samples: 4.5, 4.6, 4.7 -> median 4.6, IQR = 0.1
    assert sorted(out["tg"]["samples"]) == [4.5, 4.6, 4.7]
    assert abs(out["tg"]["median"] - 4.6) < 1e-9
    assert abs(out["tg"]["iqr"] - 0.1) < 1e-9

    # Bench noise within 5% of median -> no warnings
    assert out["warnings"] == []


def test_iqr_undefined_with_one_repeat():
    one = json.loads((Path(__file__).parent / "fixtures" / "llama_bench_one_shot.json").read_text())
    out = bench_parse.aggregate(one, ctx=512)
    assert out["pp"]["iqr"] is None
    assert out["tg"]["iqr"] is None
    assert any("repeats=1" in w for w in out["warnings"])


def test_high_iqr_emits_warning():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "llama_bench_repeats.json").read_text())
    # Make pp samples wildly noisy by mutating avg_ts entries.
    pp_rows = [r for r in fixture if r.get("n_gen", 0) == 0]
    pp_rows[0]["avg_ts"] = 50.0
    pp_rows[1]["avg_ts"] = 100.0
    pp_rows[2]["avg_ts"] = 150.0
    out = bench_parse.aggregate(fixture, ctx=512)
    assert any("pp" in w and "IQR" in w for w in out["warnings"])
