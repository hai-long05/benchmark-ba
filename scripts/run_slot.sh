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
LLAMA_COMMIT=$(python3 -c "import json; print(json.load(open('env/llama_cpp.json'))['commit'])")
LM_EVAL_VER=$(python3 -c "import json; print(json.load(open('env/lm_eval.json'))['version'])")

# 2. Fetch model.
echo "==> fetch model"
./scripts/10_fetch_model.sh "$HF_ID"
SAFE_NAME=$(echo "$HF_ID" | tr '/' '_' | tr -d "'\"" | tr '[:upper:]' '[:lower:]')
FP16_GGUF="models/${SAFE_NAME}-fp16.gguf"
FP16_SHA=$(awk '{print $1}' "${FP16_GGUF}.sha256")

# 3. Quantize.
echo "==> quantize"
./scripts/20_quantize.sh "$FP16_GGUF" "$SCHEME"
scheme_lower=$(echo "$SCHEME" | tr '[:upper:]' '[:lower:]')
QUANT_GGUF="quantized/${SAFE_NAME}-${scheme_lower}.gguf"
QUANT_SHA=$(awk '{print $1}' "${QUANT_GGUF}.sha256")
QUANT_META="${QUANT_GGUF}.quant.json"

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

# 7. lm-eval each task.
IFS=',' read -ra TASKS <<< "$TASKS_CSV"
for t in "${TASKS[@]}"; do
  echo "==> lm_eval $t"
  ./scripts/40_lm_eval.sh "$QUANT_GGUF" "$t" "$SLOT_DIR"
done

# 8. Aggregate.
echo "==> aggregate"
EXPECTED_ENV_HASH="$HOST_ID" python3 scripts/50_aggregate.py "$SLOT_DIR"

echo "DONE: $SLOT_DIR/results.json"
