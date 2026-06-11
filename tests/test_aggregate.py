"""Tests for scripts/50_aggregate.py — merge slot/bench/lm_eval into results.json."""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
AGG = REPO / "scripts" / "50_aggregate.py"


def _copy_fixture(tmp_path: Path) -> Path:
    src = REPO / "tests" / "fixtures" / "slot_demo"
    dst = tmp_path / "slot"
    shutil.copytree(src, dst)
    return dst


def test_aggregate_writes_results_json(tmp_path, monkeypatch):
    slot = _copy_fixture(tmp_path)
    # Pretend the env hash matches what slot.json expects.
    monkeypatch.setenv("EXPECTED_ENV_HASH", "EXPECTED_HOST_ID")
    res = subprocess.run(
        [sys.executable, str(AGG), str(slot)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    out = json.loads((slot / "results.json").read_text())

    assert out["model"] == "Qwen3-0.6B"
    assert out["quant"] == "Q4_K_S"
    assert out["bits_nominal"] == 4
    assert out["ctx"] == 512
    assert out["size_mib"] == 412.3
    assert out["pp"]["median"] == 93.0
    assert out["tg"]["median"] == 4.6
    assert out["tasks"]["hellaswag"]["acc_norm"] == 0.7187
    assert out["tasks"]["hellaswag"]["acc_norm_stderr"] == 0.0045
    assert out["env"]["lm_eval_version"] == "0.4.9.2"
    assert out["warnings"] == []


def test_aggregate_refuses_env_drift(tmp_path, monkeypatch):
    slot = _copy_fixture(tmp_path)
    monkeypatch.setenv("EXPECTED_ENV_HASH", "DIFFERENT_HASH")
    res = subprocess.run(
        [sys.executable, str(AGG), str(slot)],
        capture_output=True, text=True,
    )
    assert res.returncode != 0
    assert "env" in res.stderr.lower() or "drift" in res.stderr.lower()


def test_bits_nominal_mapping():
    sys.path.insert(0, str(REPO / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("aggregate", AGG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.nominal_bits("Q3_K_S") == 3
    assert mod.nominal_bits("Q4_0") == 4
    assert mod.nominal_bits("Q4_K_M") == 4
    assert mod.nominal_bits("Q5_1") == 5
    assert mod.nominal_bits("Q6_K") == 6
    assert mod.nominal_bits("Q8_0") == 8


def _load_agg_module():
    spec = importlib.util.spec_from_file_location("aggregate", AGG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_extract_metrics_gsm8k_filter_collision(tmp_path):
    """GSM8K emits both strict-match and flexible-extract; both must survive."""
    sys.path.insert(0, str(REPO / "scripts"))
    mod = _load_agg_module()

    fake = tmp_path / "gsm8k.json"
    fake.write_text(json.dumps({
        "results": {
            "gsm8k": {
                "alias": "gsm8k",
                "exact_match,strict-match": 0.42,
                "exact_match_stderr,strict-match": 0.013,
                "exact_match,flexible-extract": 0.51,
                "exact_match_stderr,flexible-extract": 0.014,
            }
        }
    }))
    out = mod._extract_task_metrics(fake)
    assert out["gsm8k"]["exact_match_strict_match"] == 0.42
    assert out["gsm8k"]["exact_match_flexible_extract"] == 0.51
    assert out["gsm8k"]["exact_match_stderr_strict_match"] == 0.013
    assert out["gsm8k"]["exact_match_stderr_flexible_extract"] == 0.014


def test_extract_metrics_none_filter_stripped(tmp_path):
    """The trivial ',none' filter is stripped (preserves existing test compatibility)."""
    sys.path.insert(0, str(REPO / "scripts"))
    mod = _load_agg_module()

    fake = tmp_path / "hellaswag.json"
    fake.write_text(json.dumps({
        "results": {
            "hellaswag": {
                "alias": "hellaswag",
                "acc_norm,none": 0.7187,
                "acc_norm_stderr,none": 0.0045,
            }
        }
    }))
    out = mod._extract_task_metrics(fake)
    assert out["hellaswag"]["acc_norm"] == 0.7187
    assert out["hellaswag"]["acc_norm_stderr"] == 0.0045
