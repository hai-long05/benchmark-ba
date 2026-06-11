#!/usr/bin/env bash
# Orchestrate one (model, quant, ctx) slot end-to-end.
# Usage: run_slot.sh <hf-id> <SCHEME> <ctx> [<repeats>] [<task1,task2,...>]
#   e.g. run_slot.sh Qwen/Qwen3-0.6B Q4_K_S 512 3 hellaswag
#
# Special schemes: F16 / FP16 — uses the unquantized FP16 GGUF as-is, no quantize step.
#
# Env vars:
#   PERF_ONLY=1      — run only env-capture + bench, skip lm_eval. Used for the
#                      ctx=2048 perf sweep where accuracy is identical to ctx=512.
#   LM_EVAL_LIMIT=N  — pass through to 40_lm_eval.sh.
#   LLAMA_CPP_DIR    — path to llama.cpp checkout (default ../llama.cpp).
set -euo pipefail

HF_ID="${1:?Usage: $0 <hf-id> <SCHEME> <ctx> [<repeats>] [<task1,task2,...>]}"
SCHEME="${2:?SCHEME required}"
CTX="${3:?ctx required}"
REPEATS="${4:-3}"
TASKS_CSV="${5:-hellaswag}"

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-../llama.cpp}"
export LLAMA_CPP_DIR
PERF_ONLY="${PERF_ONLY:-0}"

# 1. Ensure env captured.
if [ ! -f env/host.json ] || [ ! -f env/llama_cpp.json ] || [ ! -f env/lm_eval.json ]; then
  echo "==> capturing env"
  ./scripts/00_capture_env.sh
fi

# shellcheck source=lib/env_hash.sh
source scripts/lib/env_hash.sh
HOST_ID=$(env_hash env)
LLAMA_COMMIT=$(python3 -c "import json; print(json.load(open('env/llama_cpp.json'))['commit'])")
LM_EVAL_VER=$(python3 -c "import json; print(json.load(open('env/lm_eval.json'))['version'])")

# 2. Fetch model.
echo "==> fetch model"
./scripts/10_fetch_model.sh "$HF_ID"
SAFE_NAME=$(echo "$HF_ID" | tr '/' '_' | tr -d "'\"" | tr '[:upper:]' '[:lower:]')
FP16_GGUF="models/${SAFE_NAME}-fp16.gguf"
FP16_SHA=$(awk '{print $1}' "${FP16_GGUF}.sha256")

# 3. Quantize (or pass-through for FP16).
scheme_upper=$(echo "$SCHEME" | tr '[:lower:]' '[:upper:]')
scheme_lower=$(echo "$SCHEME" | tr '[:upper:]' '[:lower:]')

if [ "$scheme_upper" = "F16" ] || [ "$scheme_upper" = "FP16" ]; then
  echo "==> FP16 baseline (no quantization)"
  QUANT_GGUF="$FP16_GGUF"
  QUANT_SHA="$FP16_SHA"
  # Synthesise a quant.json on the fly with zero quant time / no reduction.
  size_mib=$(python3 -c "import os, sys; print(round(os.path.getsize(sys.argv[1]) / (1024*1024), 2))" "$FP16_GGUF")
  QUANT_META=$(mktemp /tmp/quant_meta.fp16.XXXXXX.json)
  python3 - "$FP16_GGUF" "$size_mib" <<'PY' > "$QUANT_META"
import json, sys
gguf, size_mib = sys.argv[1], float(sys.argv[2])
print(json.dumps({
  "scheme": "F16",
  "input_gguf": gguf,
  "input_size_mib": size_mib,
  "output_size_mib": size_mib,
  "size_reduction_pct": 0.0,
  "quant_time_s": 0,
}, indent=2))
PY
  trap 'rm -f "$QUANT_META"' EXIT
else
  echo "==> quantize"
  ./scripts/20_quantize.sh "$FP16_GGUF" "$SCHEME"
  QUANT_GGUF="quantized/${SAFE_NAME}-${scheme_lower}.gguf"
  QUANT_SHA=$(awk '{print $1}' "${QUANT_GGUF}.sha256")
  QUANT_META="${QUANT_GGUF}.quant.json"
fi

# 4. Build run-id and slot directory.
TS=$(date -u +%Y-%m-%dT%H-%M-%SZ)
RUN_ID="${TS}-${SAFE_NAME}-${scheme_lower}-ctx${CTX}"
SLOT_DIR="results/${RUN_ID}"
mkdir -p "$SLOT_DIR"

# 5. Write slot.json via Python (safe escaping).
MODEL_NAME=$(basename "$HF_ID")
python3 - "$RUN_ID" "$MODEL_NAME" "$HF_ID" "$FP16_SHA" "$SCHEME" "$QUANT_SHA" \
  "$QUANT_META" "$CTX" "$REPEATS" "$HOST_ID" "$LLAMA_COMMIT" "$LM_EVAL_VER" "$SLOT_DIR" <<'PY'
import json, sys
(run_id, model_name, hf_id, fp16_sha, scheme, quant_sha,
 quant_meta_path, ctx, repeats, host_id, llama_commit, lm_eval_ver, slot_dir) = sys.argv[1:14]
qm = json.load(open(quant_meta_path))
slot = {
  "run_id": run_id,
  "model": {"name": model_name, "hf_id": hf_id, "fp16_sha256": fp16_sha},
  "quant": {
    "scheme": scheme,
    "gguf_sha256": quant_sha,
    "size_mib": qm["output_size_mib"],
    "size_reduction_pct": qm["size_reduction_pct"],
    "quant_time_s": qm["quant_time_s"],
  },
  "ctx": int(ctx),
  "repeats": int(repeats),
  "host_id": host_id,
  "llama_cpp_commit": llama_commit,
  "lm_eval_version": lm_eval_ver,
}
with open(f"{slot_dir}/slot.json", "w") as f:
    json.dump(slot, f, indent=2)
print(f"wrote {slot_dir}/slot.json")
PY

# 6. Bench.
echo "==> bench"
./scripts/30_bench.sh "$QUANT_GGUF" "$CTX" "$SLOT_DIR" "$REPEATS"

# 7. lm-eval — single process, all tasks at once. Skip entirely if PERF_ONLY.
if [ "$PERF_ONLY" = "1" ]; then
  echo "==> PERF_ONLY=1 — skipping lm_eval"
else
  echo "==> lm_eval (tasks: $TASKS_CSV)"
  ./scripts/40_lm_eval.sh "$QUANT_GGUF" "$TASKS_CSV" "$SLOT_DIR"
fi

# 8. Aggregate.
echo "==> aggregate"
EXPECTED_ENV_HASH="$HOST_ID" python3 scripts/50_aggregate.py "$SLOT_DIR"

echo "DONE: $SLOT_DIR/results.json"
