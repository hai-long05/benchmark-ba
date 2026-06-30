#!/usr/bin/env bash
# Run the full thesis sweep for one model. Idempotent — completed slots are skipped.
#
# Usage: sweep.sh <hf-id> [<tasks-csv>] [<jobs>]
#   hf-id     - Hugging Face model id (e.g. meta-llama/Llama-3.1-8B-Instruct)
#   tasks-csv - comma-separated task keys (default: final thesis task set)
#   jobs      - parallel slot count (default: 1; on c3-standard-176 use 8)
#
# Behaviour:
#   1. Fetches model + quantizes all configured schemes once (sequential, deps-aware).
#   2. Runs accuracy + ctx=512 performance slots (parallel via GNU parallel if jobs>1).
#   3. Optionally runs perf-only slots at ctx=2048 (parallel).
#
# Env vars (forwarded to run_slot.sh):
#   LM_EVAL_LIMIT, LM_EVAL_N_CTX, LM_EVAL_BATCH, LM_EVAL_N_THREADS, LLAMA_CPP_DIR,
#   TOKENIZER_REPO, CHAT_TEMPLATE_KWARGS
set -euo pipefail

HF_ID="${1:?Usage: $0 <hf-id> [<tasks-csv>] [<jobs>]}"
TASKS_CSV="${2:-hellaswag,gsm8k,ifeval,truthfulqa_mc2}"
JOBS="${3:-1}"
QUANTS_FILE="${QUANTS_FILE:-configs/quants.txt}"

[ -f "$QUANTS_FILE" ] || { echo "ERROR: $QUANTS_FILE not found" >&2; exit 1; }

mapfile -t SCHEMES < <(grep -v '^$' "$QUANTS_FILE")
echo "==> sweep: model=$HF_ID, ${#SCHEMES[@]} configs, tasks=$TASKS_CSV, jobs=$JOBS"

# 1. Sequential warm-up: fetch model + quantize each configured scheme once.
#    Doing this in one process avoids parallel HF downloads (they would all
#    clobber the same models/_hf/<...>/ dir) and parallel llama-quantize runs
#    (which would peg disk I/O without speedup since llama-quantize is
#    single-threaded).
echo "==> warm-up: fetch + quantize"
./scripts/10_fetch_model.sh "$HF_ID"
SAFE=$(echo "$HF_ID" | tr '/' '_' | tr -d "'\"" | tr '[:upper:]' '[:lower:]')
FP16_GGUF="models/${SAFE}-fp16.gguf"
for SCHEME in "${SCHEMES[@]}"; do
  scheme_upper=$(echo "$SCHEME" | tr '[:lower:]' '[:upper:]')
  if [ "$scheme_upper" = "F16" ] || [ "$scheme_upper" = "FP16" ]; then
    continue   # FP16 baseline uses the FP16 GGUF as-is; no quantize step.
  fi
  ./scripts/20_quantize.sh "$FP16_GGUF" "$SCHEME"
done

# 2. Accuracy + ctx=512 perf, in parallel.
echo "==> accuracy slots (ctx=512), parallel=$JOBS"
export HF_ID TASKS_CSV LLAMA_CPP_DIR LM_EVAL_LIMIT LM_EVAL_N_CTX LM_EVAL_BATCH LM_EVAL_N_THREADS \
       TOKENIZER_REPO CHAT_TEMPLATE_KWARGS BENCH_THREADS BENCH_NUMA KURT_GGUFS_DIR

if command -v parallel >/dev/null 2>&1 && [ "$JOBS" -gt 1 ]; then
  printf '%s\n' "${SCHEMES[@]}" | parallel -j "$JOBS" --halt soon,fail=1 \
    "LLAMA_CPP_DIR='${LLAMA_CPP_DIR:-../llama.cpp}' \
     LM_EVAL_LIMIT='${LM_EVAL_LIMIT:-}' \
     LM_EVAL_N_CTX='${LM_EVAL_N_CTX:-2048}' \
     LM_EVAL_BATCH='${LM_EVAL_BATCH:-1}' \
     LM_EVAL_N_THREADS='${LM_EVAL_N_THREADS:-}' \
     TOKENIZER_REPO='${TOKENIZER_REPO:-}' \
     CHAT_TEMPLATE_KWARGS='${CHAT_TEMPLATE_KWARGS:-}' \
     BENCH_THREADS='${BENCH_THREADS:-}' \
     KURT_GGUFS_DIR='${KURT_GGUFS_DIR:-}' \
     ./scripts/run_slot.sh '$HF_ID' {} 512 3 '$TASKS_CSV'"
else
  for SCHEME in "${SCHEMES[@]}"; do
    ./scripts/run_slot.sh "$HF_ID" "$SCHEME" 512 3 "$TASKS_CSV"
  done
fi

# 3. Performance-only at ctx=2048 (skip lm_eval — accuracy is ctx-independent).
# Set SKIP_CTX2048_PERF=1 to skip this phase entirely; the final thesis reports
# ctx=512 performance for the main sweep.
if [ "${SKIP_CTX2048_PERF:-0}" = "1" ]; then
  echo "==> SKIP_CTX2048_PERF=1 — skipping ctx=2048 perf-only phase"
elif command -v parallel >/dev/null 2>&1 && [ "$JOBS" -gt 1 ]; then
  echo "==> perf-only slots (ctx=2048), parallel=$JOBS"
  printf '%s\n' "${SCHEMES[@]}" | parallel -j "$JOBS" --halt soon,fail=1 \
    "PERF_ONLY=1 LLAMA_CPP_DIR='${LLAMA_CPP_DIR:-../llama.cpp}' \
     ./scripts/run_slot.sh '$HF_ID' {} 2048 3 '$TASKS_CSV'"
else
  echo "==> perf-only slots (ctx=2048), sequential"
  for SCHEME in "${SCHEMES[@]}"; do
    PERF_ONLY=1 ./scripts/run_slot.sh "$HF_ID" "$SCHEME" 2048 3 "$TASKS_CSV"
  done
fi

echo "==> sweep complete: $HF_ID"
ls -d results/*${SAFE}* 2>/dev/null | wc -l | xargs printf "results dirs for this model: %s\n"
