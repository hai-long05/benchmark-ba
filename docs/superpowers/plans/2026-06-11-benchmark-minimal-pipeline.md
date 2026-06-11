# Benchmark Setup — Minimal Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the minimal end-to-end benchmark pipeline (env-capture → model-fetch → quantize → bench → lm-eval → aggregate) that proves one (model, quant, ctx, task) slot runs to completion on Mac, ready to scale to the 84-slot sweep on GCP.

**Architecture:** Filesystem-as-database. Six small single-purpose scripts (00–50) plus one orchestrator (`run_slot.sh`). Each stage reads files, writes files, exits. State on disk; idempotent re-runs skip valid outputs. The result tree directly serves as Kurt §4.6's reproducibility package.

**Tech Stack:** bash (orchestration + glue), Python 3.11+ (aggregation, JSON shaping), `llama.cpp` (build from source — quantize, bench), `lm-evaluation-harness==0.4.9.2` (accuracy + PPL), `huggingface-hub` CLI (model fetch), `numpy` (median/IQR), `pyyaml` (task config), `pytest` (script-level tests).

**Spec:** `docs/superpowers/specs/2026-06-11-benchmark-design.md`.

**Smoke target (acceptance test):** `Qwen/Qwen3-0.6B` × `Q4_K_S` × `ctx=512` × `hellaswag`, < 30 min on Mac.

---

## File Structure

Created (committed to git):
- `README.md` — project overview, smoke-test in three commands.
- `.gitignore` — excludes `models/`, `quantized/`, `results/`, `env/`, virtualenv, `__pycache__`.
- `requirements.txt` — Python deps with version pins.
- `configs/quants.txt` — 13 GGUF schemes, one per line (for the future sweep).
- `configs/tasks.yaml` — lm-eval task IDs + version + shot count, pinned per exposé §4.4.
- `scripts/00_capture_env.sh` — write `env/host.json`, `env/llama_cpp.json`, `env/lm_eval.json`.
- `scripts/10_fetch_model.sh` — download HF model, convert to FP16 GGUF if needed.
- `scripts/20_quantize.sh` — wrap `llama-quantize`, write `.sha256` + `.quant.json` side-cars.
- `scripts/30_bench.sh` — wrap `llama-bench`, repeat N times, aggregate median + IQR.
- `scripts/40_lm_eval.sh` — wrap `lm_eval --model gguf --log_samples`.
- `scripts/50_aggregate.py` — merge slot.json + bench.json + lm_eval/*.json → results.json.
- `scripts/lib/env_hash.sh` — shared helper: SHA256 of the `env/` directory (sourced by 00 and run_slot).
- `scripts/run_slot.sh` — orchestrate 10 → 20 → 30 → 40 → 50 for one (model, quant, ctx) triple.
- `tests/test_aggregate.py` — pytest covering `50_aggregate.py` shape + drift refusal.
- `tests/test_bench_parse.py` — pytest covering bench JSON parsing/aggregation.
- `tests/fixtures/` — sample llama-bench JSON, sample lm-eval results.

Not committed (gitignored, generated at runtime):
- `models/`, `quantized/`, `results/<run-id>/`, `env/`, `.venv/`.

Not in scope this iteration (deferred): `scripts/sweep.sh`, statistical layer, plotting.

---

## Task 1: Repository skeleton + `.gitignore` + git init

**Files:**
- Create: `.gitignore`, `README.md` (placeholder), `requirements.txt`.

- [ ] **Step 1: Initialize git**

Run from `/Users/I589258/Documents/benchmark`:
```bash
git init
git config user.name "Hai Long Do Pham"   # if not already set globally
git branch -M main
```

- [ ] **Step 2: Write `.gitignore`**

Create `.gitignore`:
```
# Generated artefacts
models/
quantized/
results/
env/

# Python
.venv/
__pycache__/
*.pyc
.pytest_cache/

# Editor / OS
.DS_Store
.idea/
.vscode/
```

- [ ] **Step 3: Write minimal `README.md`**

Create `README.md`:
```markdown
# Benchmark — llama.cpp GGUF quantization sweep

Bachelor thesis benchmark, replicates Kurt (2026, arXiv:2601.14277) cross-family on Qwen3-4B, Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct.

Spec: `docs/superpowers/specs/2026-06-11-benchmark-design.md`.

## Smoke test (Mac)

```
./scripts/00_capture_env.sh
./scripts/run_slot.sh Qwen/Qwen3-0.6B Q4_K_S 512
cat results/*/results.json | jq .
```

Expected wall-clock: < 30 min.
```

- [ ] **Step 4: Write `requirements.txt`**

Create `requirements.txt`:
```
lm-eval==0.4.9.2
huggingface-hub>=0.24,<1.0
numpy>=1.26,<3.0
pyyaml>=6.0,<7.0
pytest>=8.0,<9.0
```

- [ ] **Step 5: Commit**

```bash
git add .gitignore README.md requirements.txt docs/
git commit -m "chore: repo skeleton, gitignore, requirements, spec"
```

Expected: commit succeeds; `git status` clean.

---

## Task 2: Build llama.cpp from source (CPU-only) + verify binaries

**Files:**
- Modify: `README.md` (add llama.cpp build section).

- [ ] **Step 1: Clone llama.cpp into a sibling directory**

Run:
```bash
cd /Users/I589258/Documents
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git rev-parse HEAD > /tmp/llama_cpp_commit.txt
cat /tmp/llama_cpp_commit.txt
```

Expected: commit SHA printed. Record this — `00_capture_env.sh` will pin it later.

- [ ] **Step 2: Build CPU-only**

Run from `llama.cpp` checkout (Mac uses CPU path deliberately, per spec §9):
```bash
cmake -B build -DGGML_METAL=OFF -DGGML_BLAS=OFF -DLLAMA_CURL=OFF
cmake --build build --config Release -j
```

Expected: `build/bin/llama-quantize`, `build/bin/llama-bench`, `build/bin/llama-cli` exist.

- [ ] **Step 3: Smoke-test the binaries exist and run `--help`**

Run:
```bash
./build/bin/llama-quantize --help 2>&1 | head -5
./build/bin/llama-bench --help 2>&1 | head -5
```

Expected: usage text from each. Non-zero exit is fine (`--help` sometimes returns 1); the goal is text output, not exit code.

- [ ] **Step 4: Add `LLAMA_CPP_DIR` convention to README**

Edit `README.md`, append:
```markdown

## Dependencies

- `llama.cpp` checkout: scripts read `$LLAMA_CPP_DIR` (default `../llama.cpp`).
  Build with `cmake -B build -DGGML_METAL=OFF -DGGML_BLAS=OFF && cmake --build build -j`.
- Python 3.11+: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
```

- [ ] **Step 5: Commit**

```bash
cd /Users/I589258/Documents/benchmark
git add README.md
git commit -m "docs: document llama.cpp build + LLAMA_CPP_DIR convention"
```

---

## Task 3: Python venv + install dependencies

**Files:** none changed in repo (the venv is gitignored).

- [ ] **Step 1: Create venv and install**

Run from `/Users/I589258/Documents/benchmark`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Expected: install succeeds.

- [ ] **Step 2: Verify `lm_eval` version is exactly 0.4.9.2**

Run:
```bash
lm_eval --help 2>&1 | head -3
python -c "import lm_eval; print(lm_eval.__version__)"
```

Expected: prints `0.4.9.2`. If not, `pip install --force-reinstall lm-eval==0.4.9.2`.

- [ ] **Step 3: Verify `huggingface-cli` is installed**

Run:
```bash
huggingface-cli --help 2>&1 | head -3
```

Expected: usage text.

(No commit — only installed deps, repo unchanged.)

---

## Task 4: `configs/quants.txt` and `configs/tasks.yaml`

**Files:**
- Create: `configs/quants.txt`, `configs/tasks.yaml`.

- [ ] **Step 1: Write `configs/quants.txt`**

Create `configs/quants.txt` (one scheme per line, in Kurt's order):
```
Q3_K_S
Q3_K_M
Q3_K_L
Q4_0
Q4_1
Q4_K_S
Q4_K_M
Q5_0
Q5_1
Q5_K_S
Q5_K_M
Q6_K
Q8_0
```

- [ ] **Step 2: Write `configs/tasks.yaml`**

Create `configs/tasks.yaml` (pinned per exposé §4.4 + Kurt Table B):
```yaml
# lm-eval-harness v0.4.9.2 task IDs, versions, shots — pinned per Kurt (2026) and exposé §4.4.
tasks:
  hellaswag:
    task_id: hellaswag
    num_fewshot: 0
    primary_metric: acc_norm
  gsm8k:
    task_id: gsm8k
    num_fewshot: 5
    primary_metric: exact_match,flexible-extract
  ifeval:
    task_id: ifeval
    num_fewshot: 0
    primary_metric: prompt_level_strict_acc   # one of four; aggregator records all four
  mmlu:
    task_id: mmlu
    num_fewshot: 0
    primary_metric: acc
  truthfulqa_mc2:
    task_id: truthfulqa_mc2
    num_fewshot: 0
    primary_metric: acc
  wikitext:
    task_id: wikitext
    num_fewshot: 0
    primary_metric: word_perplexity
```

- [ ] **Step 3: Commit**

```bash
git add configs/
git commit -m "feat(configs): pin quant schemes and lm-eval task IDs"
```

---

## Task 5: `scripts/lib/env_hash.sh` — shared SHA256 helper

**Files:**
- Create: `scripts/lib/env_hash.sh`.

- [ ] **Step 1: Write the helper**

Create `scripts/lib/env_hash.sh`:
```bash
#!/usr/bin/env bash
# Compute a stable SHA256 of the env/ directory contents.
# Sourced by 00_capture_env.sh and run_slot.sh — do not execute directly.

env_hash() {
  local env_dir="${1:-env}"
  if [ ! -d "$env_dir" ]; then
    echo "env_hash: $env_dir not found" >&2
    return 1
  fi
  # Sorted file list, then SHA256 of concatenated SHAs of each file.
  ( cd "$env_dir" && find . -type f -name '*.json' -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 shasum -a 256 \
    | shasum -a 256 \
    | awk '{print $1}' )
}
```

- [ ] **Step 2: Make executable + sanity-check**

Run:
```bash
chmod +x scripts/lib/env_hash.sh
mkdir -p /tmp/envtest && echo '{"a":1}' > /tmp/envtest/x.json
bash -c 'source scripts/lib/env_hash.sh && env_hash /tmp/envtest'
rm -rf /tmp/envtest
```

Expected: 64-character hex hash printed.

- [ ] **Step 3: Commit**

```bash
git add scripts/lib/env_hash.sh
git commit -m "feat(scripts): add env_hash helper"
```

---

## Task 6: `scripts/00_capture_env.sh` — capture host, llama.cpp, lm-eval

**Files:**
- Create: `scripts/00_capture_env.sh`.

- [ ] **Step 1: Write the script**

Create `scripts/00_capture_env.sh`:
```bash
#!/usr/bin/env bash
# Capture environment metadata into env/. Idempotent.
set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-../llama.cpp}"
LM_EVAL_REQUIRED="0.4.9.2"

mkdir -p env

# host.json
{
  echo '{'
  echo "  \"captured_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"uname\": \"$(uname -a | sed 's/"/\\"/g')\","
  if command -v lscpu >/dev/null 2>&1; then
    echo "  \"cpu_model\": \"$(lscpu | awk -F: '/Model name/ {gsub(/^ +/,"",$2); print $2; exit}')\","
    echo "  \"cpu_cores\": $(lscpu | awk -F: '/^CPU\(s\):/ {gsub(/ /,"",$2); print $2; exit}'),"
    flags=$(lscpu | awk -F: '/Flags/ {print $2}' | tr -s ' ' | sed 's/^ //;s/"/\\"/g')
    echo "  \"flags\": \"$flags\","
    gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
    echo "  \"governor\": \"$gov\","
  else
    echo "  \"cpu_model\": \"$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)\","
    echo "  \"cpu_cores\": $(sysctl -n hw.physicalcpu 2>/dev/null || echo 0),"
    echo "  \"flags\": \"$(sysctl -n machdep.cpu.features 2>/dev/null || echo unknown)\","
    echo "  \"governor\": \"n/a-darwin\","
  fi
  if command -v free >/dev/null 2>&1; then
    echo "  \"ram_total_mib\": $(free -m | awk '/Mem:/ {print $2}'),"
  else
    echo "  \"ram_total_mib\": $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 )),"
  fi
  echo "  \"platform\": \"$(uname -s)\""
  echo '}'
} > env/host.json

# llama_cpp.json
if [ -d "$LLAMA_CPP_DIR/.git" ]; then
  commit=$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)
else
  commit="UNKNOWN-no-git"
fi
{
  echo '{'
  echo "  \"path\": \"$LLAMA_CPP_DIR\","
  echo "  \"commit\": \"$commit\","
  echo "  \"build_dir\": \"$LLAMA_CPP_DIR/build\""
  echo '}'
} > env/llama_cpp.json

# lm_eval.json
ver=$(python -c "import lm_eval; print(lm_eval.__version__)" 2>/dev/null || echo "MISSING")
if [ "$ver" != "$LM_EVAL_REQUIRED" ]; then
  echo "ERROR: lm_eval version is '$ver', expected '$LM_EVAL_REQUIRED' (per exposé §4.4)" >&2
  exit 1
fi
{
  echo '{'
  echo "  \"version\": \"$ver\""
  echo '}'
} > env/lm_eval.json

# Compute and print env hash for caller convenience.
# shellcheck source=lib/env_hash.sh
source "$(dirname "$0")/lib/env_hash.sh"
echo "env_hash: $(env_hash env)"
echo "env captured: env/host.json env/llama_cpp.json env/lm_eval.json"
```

- [ ] **Step 2: Make executable**

Run:
```bash
chmod +x scripts/00_capture_env.sh
```

- [ ] **Step 3: Run it (must be in venv with lm_eval installed)**

Run:
```bash
source .venv/bin/activate
LLAMA_CPP_DIR=../llama.cpp ./scripts/00_capture_env.sh
```

Expected: prints `env_hash: <64 hex>` and `env captured: ...`. Three JSON files exist under `env/`.

- [ ] **Step 4: Validate JSON well-formed**

Run:
```bash
python -c "import json; [json.load(open(f)) for f in ['env/host.json','env/llama_cpp.json','env/lm_eval.json']]; print('OK')"
```

Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add scripts/00_capture_env.sh
git commit -m "feat(scripts): 00_capture_env.sh writes host/llama_cpp/lm_eval metadata"
```

---

## Task 7: `scripts/10_fetch_model.sh` — download HF model + convert to FP16 GGUF

**Files:**
- Create: `scripts/10_fetch_model.sh`.

- [ ] **Step 1: Write the script**

Create `scripts/10_fetch_model.sh`:
```bash
#!/usr/bin/env bash
# Download a Hugging Face model and ensure an FP16 GGUF exists under models/.
# Usage: 10_fetch_model.sh <hf-id>
#   e.g. 10_fetch_model.sh Qwen/Qwen3-0.6B
# Output: models/<safe-name>-fp16.gguf + .sha256 + .meta.json
set -euo pipefail

HF_ID="${1:?Usage: $0 <hf-id>}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-../llama.cpp}"
SAFE_NAME=$(echo "$HF_ID" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
OUT_GGUF="models/${SAFE_NAME}-fp16.gguf"
OUT_SHA="${OUT_GGUF}.sha256"
OUT_META="${OUT_GGUF}.meta.json"
DL_DIR="models/_hf/${SAFE_NAME}"

mkdir -p models models/_hf

# Idempotency: if the GGUF + sha exist and match, exit 0.
if [ -f "$OUT_GGUF" ] && [ -f "$OUT_SHA" ]; then
  expected=$(awk '{print $1}' "$OUT_SHA")
  actual=$(shasum -a 256 "$OUT_GGUF" | awk '{print $1}')
  if [ "$expected" = "$actual" ]; then
    echo "model already present and verified: $OUT_GGUF"
    exit 0
  fi
  echo "WARN: SHA mismatch on $OUT_GGUF, re-downloading" >&2
fi

# Download (snapshot pins to a specific revision via --revision once we know it; default = main, recorded below).
echo "downloading $HF_ID -> $DL_DIR"
huggingface-cli download "$HF_ID" --local-dir "$DL_DIR" --local-dir-use-symlinks False
revision=$(cat "$DL_DIR/.cache/huggingface/download/.last_commit" 2>/dev/null \
  || git -C "$DL_DIR" rev-parse HEAD 2>/dev/null \
  || echo "UNKNOWN")

# Convert to FP16 GGUF if the repo doesn't already ship one.
existing_gguf=$(find "$DL_DIR" -maxdepth 2 -type f -name '*fp16*.gguf' -o -name '*f16*.gguf' 2>/dev/null | head -1)
if [ -n "${existing_gguf:-}" ]; then
  echo "found pre-built GGUF in repo: $existing_gguf"
  cp "$existing_gguf" "$OUT_GGUF"
else
  CONVERT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
  if [ ! -f "$CONVERT" ]; then
    echo "ERROR: cannot find $CONVERT — set LLAMA_CPP_DIR" >&2
    exit 1
  fi
  echo "converting safetensors -> FP16 GGUF"
  python "$CONVERT" "$DL_DIR" --outfile "$OUT_GGUF" --outtype f16
fi

# Hash + metadata
shasum -a 256 "$OUT_GGUF" > "$OUT_SHA"
size_mib=$(python -c "import os; print(round(os.path.getsize('$OUT_GGUF') / (1024*1024), 2))")

# License: try to find LICENSE/LICENSE.* in download.
license=$(find "$DL_DIR" -maxdepth 2 -iname 'LICENSE*' -type f | head -1 | xargs -I{} basename {} 2>/dev/null || echo "unknown")

cat > "$OUT_META" <<EOF
{
  "hf_id": "$HF_ID",
  "revision": "$revision",
  "license_file": "$license",
  "size_mib": $size_mib,
  "safe_name": "$SAFE_NAME"
}
EOF

echo "fetched: $OUT_GGUF ($size_mib MiB)"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/10_fetch_model.sh
```

- [ ] **Step 3: Run with the smoke target**

Run:
```bash
source .venv/bin/activate
LLAMA_CPP_DIR=../llama.cpp ./scripts/10_fetch_model.sh Qwen/Qwen3-0.6B
```

Expected: `models/qwen_qwen3-0.6b-fp16.gguf` exists, ~1.2 GiB, plus `.sha256` and `.meta.json`. Re-running prints `model already present and verified` and exits.

- [ ] **Step 4: Validate metadata + idempotency**

Run:
```bash
cat models/qwen_qwen3-0.6b-fp16.gguf.meta.json | python -m json.tool
./scripts/10_fetch_model.sh Qwen/Qwen3-0.6B   # second run, must be a no-op
```

Expected: meta.json has hf_id, revision, size_mib. Second run prints "already present".

- [ ] **Step 5: Commit**

```bash
git add scripts/10_fetch_model.sh
git commit -m "feat(scripts): 10_fetch_model.sh downloads HF model and emits FP16 GGUF + sha + meta"
```

---

## Task 8: `scripts/20_quantize.sh` — wrap llama-quantize

**Files:**
- Create: `scripts/20_quantize.sh`.

- [ ] **Step 1: Write the script**

Create `scripts/20_quantize.sh`:
```bash
#!/usr/bin/env bash
# Quantize an FP16 GGUF using llama-quantize. Idempotent.
# Usage: 20_quantize.sh <fp16.gguf> <SCHEME>
#   e.g. 20_quantize.sh models/qwen_qwen3-0.6b-fp16.gguf Q4_K_S
# Output: quantized/<base>-<SCHEME>.gguf + .sha256 + .quant.json
set -euo pipefail

IN_GGUF="${1:?Usage: $0 <fp16.gguf> <SCHEME>}"
SCHEME="${2:?Usage: $0 <fp16.gguf> <SCHEME>}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-../llama.cpp}"
QUANT_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"

[ -f "$IN_GGUF" ] || { echo "ERROR: $IN_GGUF not found" >&2; exit 1; }
[ -x "$QUANT_BIN" ] || { echo "ERROR: $QUANT_BIN not found — build llama.cpp" >&2; exit 1; }

mkdir -p quantized
base=$(basename "$IN_GGUF" .gguf | sed 's/-fp16$//')
OUT_GGUF="quantized/${base}-${SCHEME,,}.gguf"     # lowercase scheme for path
OUT_SHA="${OUT_GGUF}.sha256"
OUT_META="${OUT_GGUF}.quant.json"

# Idempotency
if [ -f "$OUT_GGUF" ] && [ -f "$OUT_SHA" ]; then
  expected=$(awk '{print $1}' "$OUT_SHA")
  actual=$(shasum -a 256 "$OUT_GGUF" | awk '{print $1}')
  if [ "$expected" = "$actual" ]; then
    echo "quantized already present and verified: $OUT_GGUF"
    exit 0
  fi
  rm -f "$OUT_GGUF" "$OUT_SHA" "$OUT_META"
fi

# Run + time it
start=$(date +%s.%N)
"$QUANT_BIN" "$IN_GGUF" "$OUT_GGUF" "$SCHEME"
end=$(date +%s.%N)
quant_time_s=$(python -c "print(round($end - $start, 3))")

# Hash + metadata
shasum -a 256 "$OUT_GGUF" > "$OUT_SHA"
size_mib=$(python -c "import os; print(round(os.path.getsize('$OUT_GGUF') / (1024*1024), 2))")
in_size_mib=$(python -c "import os; print(round(os.path.getsize('$IN_GGUF') / (1024*1024), 2))")
size_reduction_pct=$(python -c "print(round((1 - $size_mib / $in_size_mib) * 100, 2))")

cat > "$OUT_META" <<EOF
{
  "scheme": "$SCHEME",
  "input_gguf": "$IN_GGUF",
  "input_size_mib": $in_size_mib,
  "output_size_mib": $size_mib,
  "size_reduction_pct": $size_reduction_pct,
  "quant_time_s": $quant_time_s
}
EOF

echo "quantized: $OUT_GGUF ($size_mib MiB, ${size_reduction_pct}% reduction, ${quant_time_s}s)"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/20_quantize.sh
```

- [ ] **Step 3: Run with smoke target**

```bash
LLAMA_CPP_DIR=../llama.cpp ./scripts/20_quantize.sh models/qwen_qwen3-0.6b-fp16.gguf Q4_K_S
```

Expected: `quantized/qwen_qwen3-0.6b-q4_k_s.gguf` exists + sha + quant.json. Reduction approximately 65–70 %.

- [ ] **Step 4: Verify idempotency**

```bash
./scripts/20_quantize.sh models/qwen_qwen3-0.6b-fp16.gguf Q4_K_S
```

Expected: prints "quantized already present and verified".

- [ ] **Step 5: Commit**

```bash
git add scripts/20_quantize.sh
git commit -m "feat(scripts): 20_quantize.sh wraps llama-quantize with sha + metadata"
```

---

## Task 9: `tests/test_bench_parse.py` — failing test for bench JSON aggregation

**Files:**
- Create: `tests/__init__.py` (empty), `tests/fixtures/llama_bench_repeats.json`, `tests/test_bench_parse.py`.
- Create: `scripts/lib/bench_parse.py` (will be implemented to make this pass in Task 10).

- [ ] **Step 1: Capture a real `llama-bench --output json` sample**

Run (uses the quantized model from Task 8 — quick, ~30 s):
```bash
mkdir -p tests/fixtures
../llama.cpp/build/bin/llama-bench \
  -m quantized/qwen_qwen3-0.6b-q4_k_s.gguf \
  -p 512 -n 128 -r 1 -o json \
  > tests/fixtures/llama_bench_one_shot.json
cat tests/fixtures/llama_bench_one_shot.json | python -m json.tool | head -30
```

Expected: a JSON array of result objects with `n_prompt`, `n_gen`, `avg_ts`, etc. If schema is unfamiliar, inspect the file before writing the test.

- [ ] **Step 2: Build a 3-repeat fixture by hand from the sample**

Read `tests/fixtures/llama_bench_one_shot.json`, then construct `tests/fixtures/llama_bench_repeats.json` containing six entries: three pp-rows (n_prompt=512, n_gen=0) with `avg_ts` values of 92.0, 93.0, 94.0; and three tg-rows (n_prompt=0, n_gen=128) with `avg_ts` values of 4.5, 4.7, 4.6. Use the real schema from Step 1 — only the values change. Write it as proper JSON (an array).

- [ ] **Step 3: Write `tests/__init__.py` (empty file)**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 4: Write the failing test**

Create `tests/test_bench_parse.py`:
```python
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
```

- [ ] **Step 5: Run the test, expect failure**

Run:
```bash
source .venv/bin/activate
pytest tests/test_bench_parse.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'bench_parse'`.

(No commit yet — TDD red.)

---

## Task 10: Implement `scripts/lib/bench_parse.py` to make Task 9 tests pass

**Files:**
- Create: `scripts/lib/bench_parse.py`.

- [ ] **Step 1: Write the implementation**

Create `scripts/lib/bench_parse.py`:
```python
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
```

- [ ] **Step 2: Run the tests**

```bash
source .venv/bin/activate
pytest tests/test_bench_parse.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add scripts/lib/bench_parse.py tests/test_bench_parse.py tests/fixtures/llama_bench_one_shot.json tests/fixtures/llama_bench_repeats.json tests/__init__.py
git commit -m "feat(bench): aggregate llama-bench JSON across repeats with median/IQR + warnings"
```

---

## Task 11: `scripts/30_bench.sh` — run llama-bench N times, call aggregator

**Files:**
- Create: `scripts/30_bench.sh`.

- [ ] **Step 1: Write the script**

Create `scripts/30_bench.sh`:
```bash
#!/usr/bin/env bash
# Run llama-bench <repeats> times, aggregate to bench.json.
# Usage: 30_bench.sh <gguf> <ctx> <out_dir> [<repeats>]
#   e.g. 30_bench.sh quantized/qwen_qwen3-0.6b-q4_k_s.gguf 512 results/run-id 3
set -euo pipefail

GGUF="${1:?Usage: $0 <gguf> <ctx> <out_dir> [<repeats>]}"
CTX="${2:?ctx required}"
OUT_DIR="${3:?out_dir required}"
REPEATS="${4:-3}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-../llama.cpp}"
BENCH_BIN="$LLAMA_CPP_DIR/build/bin/llama-bench"

[ -f "$GGUF" ] || { echo "ERROR: $GGUF not found" >&2; exit 1; }
[ -x "$BENCH_BIN" ] || { echo "ERROR: $BENCH_BIN not found" >&2; exit 1; }

mkdir -p "$OUT_DIR/bench_raw"

# Run llama-bench REPEATS times, each invocation does one pp + one tg sample.
combined="$OUT_DIR/bench_raw/all.json"
echo "[" > "$combined"
for i in $(seq 1 "$REPEATS"); do
  raw="$OUT_DIR/bench_raw/run-$i.json"
  "$BENCH_BIN" -m "$GGUF" -p "$CTX" -n 128 -r 1 -o json > "$raw"
  # Append rows from this run (strip the outer brackets), comma between runs.
  python -c "
import json, sys
rows = json.load(open('$raw'))
sys.stdout.write(','.join(json.dumps(r) for r in rows))
" >> "$combined"
  if [ "$i" -lt "$REPEATS" ]; then echo "," >> "$combined"; fi
done
echo "]" >> "$combined"

# Aggregate via the python helper.
python -c "
import json, sys
sys.path.insert(0, 'scripts/lib')
import bench_parse
rows = json.load(open('$combined'))
agg = bench_parse.aggregate(rows, ctx=$CTX)
json.dump(agg, open('$OUT_DIR/bench.json', 'w'), indent=2)
print('bench:', agg['pp']['median'], 'pp tok/s,', agg['tg']['median'], 'tg tok/s,', len(agg['warnings']), 'warnings')
"
```

- [ ] **Step 2: Make executable + run**

```bash
chmod +x scripts/30_bench.sh
LLAMA_CPP_DIR=../llama.cpp ./scripts/30_bench.sh \
  quantized/qwen_qwen3-0.6b-q4_k_s.gguf 512 /tmp/bench_smoke 3
cat /tmp/bench_smoke/bench.json | python -m json.tool
```

Expected: prints summary line; `bench.json` has pp/tg with 3 samples each, median, iqr.

- [ ] **Step 3: Commit**

```bash
rm -rf /tmp/bench_smoke
git add scripts/30_bench.sh
git commit -m "feat(scripts): 30_bench.sh runs llama-bench N times and aggregates"
```

---

## Task 12: `scripts/40_lm_eval.sh` — wrap lm_eval with task config

**Files:**
- Create: `scripts/40_lm_eval.sh`.

- [ ] **Step 1: Write the script**

Create `scripts/40_lm_eval.sh`:
```bash
#!/usr/bin/env bash
# Run lm-evaluation-harness on one task against a quantized GGUF.
# Usage: 40_lm_eval.sh <gguf> <task_key> <out_dir>
#   <task_key> matches a key under `tasks:` in configs/tasks.yaml (e.g. hellaswag).
set -euo pipefail

GGUF="${1:?Usage: $0 <gguf> <task_key> <out_dir>}"
TASK_KEY="${2:?task_key required}"
OUT_DIR="${3:?out_dir required}"
TASKS_YAML="${TASKS_YAML:-configs/tasks.yaml}"

[ -f "$GGUF" ] || { echo "ERROR: $GGUF not found" >&2; exit 1; }
[ -f "$TASKS_YAML" ] || { echo "ERROR: $TASKS_YAML not found" >&2; exit 1; }

# Read task config via Python (yaml).
read -r task_id num_fewshot < <(python -c "
import yaml, sys
cfg = yaml.safe_load(open('$TASKS_YAML'))['tasks']
if '$TASK_KEY' not in cfg:
    sys.exit(f\"unknown task_key: $TASK_KEY\")
t = cfg['$TASK_KEY']
print(t['task_id'], t['num_fewshot'])
")

mkdir -p "$OUT_DIR/lm_eval"

# Idempotency: skip if results.json already exists.
if [ -f "$OUT_DIR/lm_eval/${TASK_KEY}.json" ]; then
  echo "lm_eval already done: $OUT_DIR/lm_eval/${TASK_KEY}.json"
  exit 0
fi

# Greedy decoding (temperature=0). Item-level logs go alongside.
lm_eval \
  --model gguf \
  --model_args "model=$GGUF" \
  --tasks "$task_id" \
  --num_fewshot "$num_fewshot" \
  --batch_size 1 \
  --log_samples \
  --output_path "$OUT_DIR/lm_eval/${TASK_KEY}_raw" \
  --gen_kwargs "temperature=0,do_sample=False" \
  2>&1 | tee "$OUT_DIR/lm_eval/${TASK_KEY}.log"

# The harness writes a results JSON deep inside output_path; locate and copy it.
results_json=$(find "$OUT_DIR/lm_eval/${TASK_KEY}_raw" -name 'results*.json' -type f | head -1)
if [ -z "$results_json" ]; then
  echo "ERROR: lm-eval produced no results JSON" >&2
  exit 1
fi
cp "$results_json" "$OUT_DIR/lm_eval/${TASK_KEY}.json"
echo "lm_eval done: $OUT_DIR/lm_eval/${TASK_KEY}.json"
```

- [ ] **Step 2: Make executable + smoke run**

```bash
chmod +x scripts/40_lm_eval.sh
source .venv/bin/activate
./scripts/40_lm_eval.sh quantized/qwen_qwen3-0.6b-q4_k_s.gguf hellaswag /tmp/eval_smoke
cat /tmp/eval_smoke/lm_eval/hellaswag.json | python -m json.tool | head -40
```

Expected: lm_eval runs (this is the longest single step — minutes on Mac CPU). Final JSON has `results.hellaswag.acc_norm` and an stderr value.

- [ ] **Step 3: Commit**

```bash
rm -rf /tmp/eval_smoke
git add scripts/40_lm_eval.sh
git commit -m "feat(scripts): 40_lm_eval.sh wraps lm-evaluation-harness, greedy, log_samples"
```

---

## Task 13: `tests/test_aggregate.py` — failing tests for the aggregator

**Files:**
- Create: `tests/fixtures/slot_demo/` (slot.json + bench.json + lm_eval/hellaswag.json).
- Create: `tests/test_aggregate.py`.

- [ ] **Step 1: Build the fixture slot directory**

Run:
```bash
mkdir -p tests/fixtures/slot_demo/lm_eval
```

Create `tests/fixtures/slot_demo/slot.json`:
```json
{
  "run_id": "fixture-2026-06-11T00-00-00Z-qwen3-0.6b-q4_k_s-ctx512",
  "model": {
    "name": "Qwen3-0.6B",
    "hf_id": "Qwen/Qwen3-0.6B",
    "fp16_sha256": "0000000000000000000000000000000000000000000000000000000000000000"
  },
  "quant": {
    "scheme": "Q4_K_S",
    "gguf_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
    "size_mib": 412.3,
    "size_reduction_pct": 65.4,
    "quant_time_s": 4.1
  },
  "ctx": 512,
  "repeats": 3,
  "host_id": "EXPECTED_HOST_ID",
  "llama_cpp_commit": "abc1234",
  "lm_eval_version": "0.4.9.2"
}
```

Create `tests/fixtures/slot_demo/bench.json`:
```json
{
  "pp": {"ctx": 512, "median": 93.0, "iqr": 1.0, "samples": [92.0, 93.0, 94.0]},
  "tg": {"median": 4.6, "iqr": 0.1, "samples": [4.5, 4.6, 4.7]},
  "warnings": []
}
```

Create `tests/fixtures/slot_demo/lm_eval/hellaswag.json` (real lm-eval shape, trimmed):
```json
{
  "results": {
    "hellaswag": {
      "acc,none": 0.5754,
      "acc_stderr,none": 0.0049,
      "acc_norm,none": 0.7187,
      "acc_norm_stderr,none": 0.0045,
      "alias": "hellaswag"
    }
  },
  "versions": {"hellaswag": 1.0},
  "n-shot": {"hellaswag": 0},
  "config": {"model": "gguf"}
}
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_aggregate.py`:
```python
"""Tests for scripts/50_aggregate.py — merge slot/bench/lm_eval into results.json."""
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
```

- [ ] **Step 3: Run, expect failure**

```bash
pytest tests/test_aggregate.py -v
```

Expected: FAIL — `50_aggregate.py` does not exist.

(No commit — TDD red.)

---

## Task 14: Implement `scripts/50_aggregate.py` to make Task 13 pass

**Files:**
- Create: `scripts/50_aggregate.py`.

- [ ] **Step 1: Write the implementation**

Create `scripts/50_aggregate.py`:
```python
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
    """Pull primary metric + stderr out of an lm-eval results.json into a flat dict."""
    data = json.loads(lm_eval_path.read_text())
    results = data.get("results", {})
    flat: dict[str, dict[str, float]] = {}
    for task_name, metrics in results.items():
        entry: dict[str, float] = {}
        for key, value in metrics.items():
            if key == "alias" or not isinstance(value, (int, float)):
                continue
            # Normalize "metric,filter" -> "metric"
            base = key.split(",")[0]
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

    # `_extract_task_metrics` uses "acc_norm"-style names; the test expects
    # those exact keys, which matches lm-eval's own stripped key.
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

- [ ] **Step 2: Run the tests**

```bash
pytest tests/test_aggregate.py -v
```

Expected: all 3 tests PASS. If `test_aggregate_writes_results_json` fails on a missing `acc_norm`/`acc_norm_stderr`, inspect: the harness writes `acc_norm,none`; `_extract_task_metrics` strips the `,none` suffix, so the test's expected keys (`acc_norm`, `acc_norm_stderr`) line up. If they don't, fix the test or the extraction — keep them consistent.

- [ ] **Step 3: Commit**

```bash
git add scripts/50_aggregate.py tests/test_aggregate.py tests/fixtures/slot_demo/
git commit -m "feat(aggregate): merge slot+bench+lm_eval -> results.json with env drift refusal"
```

---

## Task 15: `scripts/run_slot.sh` — orchestrate the full slot end-to-end

**Files:**
- Create: `scripts/run_slot.sh`.

- [ ] **Step 1: Write the script**

Create `scripts/run_slot.sh`:
```bash
#!/usr/bin/env bash
# Orchestrate one (model, quant, ctx) slot end-to-end.
# Usage: run_slot.sh <hf-id> <SCHEME> <ctx> [<repeats>] [<task1,task2,...>]
#   e.g. run_slot.sh Qwen/Qwen3-0.6B Q4_K_S 512 3 hellaswag
set -euo pipefail

HF_ID="${1:?Usage: $0 <hf-id> <SCHEME> <ctx> [<repeats>] [<task1,task2,...>]}"
SCHEME="${2:?SCHEME required}"
CTX="${3:?ctx required}"
REPEATS="${4:-3}"
TASKS_CSV="${5:-hellaswag}"

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-../llama.cpp}"
export LLAMA_CPP_DIR

# 1. Ensure env captured.
if [ ! -f env/host.json ] || [ ! -f env/llama_cpp.json ] || [ ! -f env/lm_eval.json ]; then
  echo "==> capturing env"
  ./scripts/00_capture_env.sh
fi

# shellcheck source=lib/env_hash.sh
source scripts/lib/env_hash.sh
HOST_ID=$(env_hash env)
LLAMA_COMMIT=$(python -c "import json; print(json.load(open('env/llama_cpp.json'))['commit'])")
LM_EVAL_VER=$(python -c "import json; print(json.load(open('env/lm_eval.json'))['version'])")

# 2. Fetch model.
echo "==> fetch model"
./scripts/10_fetch_model.sh "$HF_ID"
SAFE_NAME=$(echo "$HF_ID" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
FP16_GGUF="models/${SAFE_NAME}-fp16.gguf"
FP16_SHA=$(awk '{print $1}' "${FP16_GGUF}.sha256")

# 3. Quantize.
echo "==> quantize"
./scripts/20_quantize.sh "$FP16_GGUF" "$SCHEME"
QUANT_GGUF="quantized/${SAFE_NAME}-${SCHEME,,}.gguf"
QUANT_SHA=$(awk '{print $1}' "${QUANT_GGUF}.sha256")
QUANT_META="${QUANT_GGUF}.quant.json"

# 4. Build run-id and slot directory.
TS=$(date -u +%Y-%m-%dT%H-%M-%SZ)
RUN_ID="${TS}-${SAFE_NAME}-${SCHEME,,}-ctx${CTX}"
SLOT_DIR="results/${RUN_ID}"
mkdir -p "$SLOT_DIR"

# 5. Write slot.json.
python <<PY
import json
quant_meta = json.load(open("$QUANT_META"))
slot = {
  "run_id": "$RUN_ID",
  "model": {"name": "$(basename "$HF_ID")", "hf_id": "$HF_ID", "fp16_sha256": "$FP16_SHA"},
  "quant": {
    "scheme": "$SCHEME",
    "gguf_sha256": "$QUANT_SHA",
    "size_mib": quant_meta["output_size_mib"],
    "size_reduction_pct": quant_meta["size_reduction_pct"],
    "quant_time_s": quant_meta["quant_time_s"],
  },
  "ctx": $CTX,
  "repeats": $REPEATS,
  "host_id": "$HOST_ID",
  "llama_cpp_commit": "$LLAMA_COMMIT",
  "lm_eval_version": "$LM_EVAL_VER",
}
json.dump(slot, open("$SLOT_DIR/slot.json", "w"), indent=2)
print("wrote", "$SLOT_DIR/slot.json")
PY

# 6. Bench.
echo "==> bench"
./scripts/30_bench.sh "$QUANT_GGUF" "$CTX" "$SLOT_DIR" "$REPEATS"

# 7. lm-eval each task.
IFS=',' read -ra TASKS <<< "$TASKS_CSV"
for t in "${TASKS[@]}"; do
  echo "==> lm_eval $t"
  ./scripts/40_lm_eval.sh "$QUANT_GGUF" "$t" "$SLOT_DIR"
done

# 8. Aggregate.
echo "==> aggregate"
EXPECTED_ENV_HASH="$HOST_ID" python scripts/50_aggregate.py "$SLOT_DIR"

echo "DONE: $SLOT_DIR/results.json"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/run_slot.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_slot.sh
git commit -m "feat(scripts): run_slot.sh orchestrates one (model, quant, ctx, tasks) slot"
```

---

## Task 16: End-to-end smoke test (the actual acceptance gate from spec §8)

**Files:** none changed; this is a verification.

- [ ] **Step 1: Clean any prior smoke artefacts**

```bash
rm -rf env results
```

- [ ] **Step 2: Run the full smoke**

```bash
source .venv/bin/activate
LLAMA_CPP_DIR=../llama.cpp ./scripts/00_capture_env.sh
LLAMA_CPP_DIR=../llama.cpp ./scripts/run_slot.sh Qwen/Qwen3-0.6B Q4_K_S 512 3 hellaswag
```

Expected wall-clock: < 30 min on Apple Silicon. Exits 0.

- [ ] **Step 3: Verify pass criteria from spec §8**

```bash
RESULT=$(ls results/*/results.json | head -1)
python <<PY
import json, sys
r = json.load(open("$RESULT"))
checks = [
  ("quant.gguf_sha256", "quant" in r and len(r.get("env",{}).get("host_id",""))==64 or r.get("env",{}).get("host_id")),
  ("pp.median present", isinstance(r["pp"]["median"], (int,float))),
  ("pp.iqr present",   r["pp"]["iqr"] is not None),
  ("pp.samples len 3", len(r["pp"]["samples"]) == 3),
  ("tg.median present", isinstance(r["tg"]["median"], (int,float))),
  ("tg.iqr present",   r["tg"]["iqr"] is not None),
  ("tg.samples len 3", len(r["tg"]["samples"]) == 3),
  ("hellaswag.acc_norm present", "acc_norm" in r["tasks"].get("hellaswag", {})),
  ("hellaswag stderr present", "acc_norm_stderr" in r["tasks"].get("hellaswag", {})),
  ("lm_eval_version 0.4.9.2", r["env"]["lm_eval_version"] == "0.4.9.2"),
  ("llama_cpp_commit set",     bool(r["env"]["llama_cpp_commit"])),
]
ok = all(v for _,v in checks)
for n,v in checks:
    print(("OK  " if v else "FAIL"), n)
sys.exit(0 if ok else 1)
PY
```

Expected: all checks `OK`, script exits 0.

- [ ] **Step 4: Verify reproducibility-package contents on disk**

```bash
ls -la env/ results/*/ results/*/lm_eval/
shasum -a 256 quantized/qwen_qwen3-0.6b-q4_k_s.gguf
```

Expected: env/ has 3 JSON files; results/<run-id>/ has slot.json, bench.json, lm_eval/, results.json; quantized GGUF has a SHA matching the side-car file.

- [ ] **Step 5: Tag the milestone**

```bash
git tag -a v0.1-smoke -m "minimal pipeline smoke passes on Mac with Qwen3-0.6B/Q4_K_S/ctx512/hellaswag"
```

(No new commits — verifying existing artefacts.)

---

## Task 17: Update `README.md` with the verified smoke-test recipe

**Files:**
- Modify: `README.md`.

- [ ] **Step 1: Replace README with the verified recipe**

Overwrite `README.md`:
```markdown
# Benchmark — llama.cpp GGUF quantization sweep

Bachelor thesis benchmark, replicates Kurt (2026, [arXiv:2601.14277](https://arxiv.org/abs/2601.14277)) cross-family on Qwen3-4B, Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct on a server-class x86 CPU.

**Spec:** `docs/superpowers/specs/2026-06-11-benchmark-design.md`
**Plan:** `docs/superpowers/plans/2026-06-11-benchmark-minimal-pipeline.md`

## Setup (once)

```bash
# 1. Build llama.cpp (CPU-only) into a sibling directory
git clone https://github.com/ggml-org/llama.cpp.git ../llama.cpp
cmake -S ../llama.cpp -B ../llama.cpp/build -DGGML_METAL=OFF -DGGML_BLAS=OFF -DLLAMA_CURL=OFF
cmake --build ../llama.cpp/build --config Release -j

# 2. Python venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke test (Mac, < 30 min)

```bash
source .venv/bin/activate
export LLAMA_CPP_DIR=../llama.cpp
./scripts/00_capture_env.sh
./scripts/run_slot.sh Qwen/Qwen3-0.6B Q4_K_S 512 3 hellaswag
ls results/*/results.json
```

The smoke test proves the pipeline shape; it does *not* replicate Kurt's numbers — that happens later on the GCP server.

## Layout

| Path | Purpose |
|---|---|
| `scripts/` | One stage per script: 00 env, 10 fetch, 20 quantize, 30 bench, 40 lm_eval, 50 aggregate, plus `run_slot.sh` orchestrator |
| `configs/` | `quants.txt` (13 schemes for the future sweep), `tasks.yaml` (lm-eval task IDs pinned per exposé §4.4) |
| `tests/` | pytest covering `bench_parse.py` aggregation and `50_aggregate.py` merge + env-drift refusal |
| `models/`, `quantized/`, `results/`, `env/` | gitignored runtime artefacts |
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with verified smoke-test recipe"
```

---

## Self-Review

**1. Spec coverage**

| Spec §  | Topic                              | Plan task |
|---------|------------------------------------|-----------|
| §1      | Purpose: minimal pipeline          | All 17 tasks |
| §2      | Final targets (models, quants, tasks) | Task 4 (configs) |
| §3      | Architecture: 5 stages + orchestrator | Tasks 6–15 |
| §4      | Directory layout                    | Task 1 (.gitignore), 5–15 (everything else) |
| §5.1    | `00_capture_env.sh`                 | Task 6 |
| §5.2    | `10_fetch_model.sh`                 | Task 7 |
| §5.3    | `20_quantize.sh`                    | Task 8 |
| §5.4    | `30_bench.sh` + median/IQR + warnings | Tasks 9–11 |
| §5.5    | `40_lm_eval.sh`                     | Task 12 |
| §5.6    | `50_aggregate.py` + env drift       | Tasks 13–14 |
| §5.7    | `run_slot.sh`                       | Task 15 |
| §6      | Idempotency                         | Skip-if-SHA-matches in Tasks 7, 8, 12 |
| §7      | Error handling (3 modes)            | Tasks 10 (warnings), 14 (env drift), bash `set -euo pipefail` everywhere |
| §8      | Smoke-test pass criteria            | Task 16 |
| §9      | Mac vs GCP portability              | Task 2 build flags + Task 6 governor handling |
| §11     | Acceptance criteria                 | Tasks 16 + 17 |

All spec requirements have a task. ✓

**2. Placeholder scan**

No "TBD"/"TODO"/"implement later" — every step has actual content. The two "deferred to next iteration" notes (sweep, statistical layer) are explicitly out of scope per spec §1, not placeholders.

**3. Type / name consistency**

- `bench_parse.aggregate(rows, ctx)` is defined in Task 10 and called in Tasks 11 + 14 with the same signature. ✓
- `slot.json` schema in Task 15 (`run_slot.sh`) matches the schema asserted in Task 13's fixture and Task 14's `aggregate()`. ✓
- `nominal_bits()` used in Task 13's test is defined in Task 14. ✓
- `env_hash` function defined in Task 5, sourced in Tasks 6 and 15, called from Python in Task 14 via the same shell helper. ✓
- File path conventions: `quantized/<safe_name>-<lowercase_scheme>.gguf` consistent across Tasks 8, 15. ✓

Plan is internally consistent.
