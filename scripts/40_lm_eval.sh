#!/usr/bin/env bash
# Run lm-evaluation-harness on one task against a quantized GGUF.
# Usage: 40_lm_eval.sh <gguf> <task_key> <out_dir>
#   <task_key> matches a key under `tasks:` in configs/tasks.yaml (e.g. hellaswag).
#
# Implementation:
#   Loads the GGUF directly via llama-cpp-python in-process and computes exact
#   per-token logprobs from the full softmax (the same scoring path lm_eval's
#   `hf` backend uses for non-GGUF models). The custom model class is at
#   scripts/lib/lm_eval_gguf_runner.py and is registered as `--model gguf_local`
#   by the wrapper at scripts/lm_eval_wrapper.py.
#
# Why not lm_eval's stock `--model gguf`?
#   That backend talks to a llama-server via HTTP and was designed for older
#   server builds that echoed prompt-token logprobs in /v1/completions. Modern
#   llama-server doesn't, so an earlier version of this script wrapped the
#   server in a Python proxy that emulated the legacy behaviour token-by-token.
#   The proxy's fallback (when the actual continuation token wasn't in the
#   server's top_logprobs) recorded the *generated* token's logprob instead —
#   producing systematically wrong scores. The HellaSwag gap was 32 percentage
#   points on Llama-3.1-8B-Q4_K_S vs Kurt's published number. Fixed by skipping
#   the HTTP layer entirely.
#
#   Tradeoff: this requires `pip install llama-cpp-python`. That dependency is
#   already in requirements.txt.
#
# Wall-clock on M1 Pro:
#   ~5 min for 200 hellaswag items on Qwen3-0.6B-Q4_K_S
#   ~30 min for 200 hellaswag items on Llama-3.1-8B-Q4_K_S
set -euo pipefail

GGUF="${1:?Usage: $0 <gguf> <task_key> <out_dir>}"
TASK_KEY="${2:?task_key required}"
OUT_DIR="${3:?out_dir required}"
TASKS_YAML="${TASKS_YAML:-configs/tasks.yaml}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-}"   # optional: set to N to limit examples
LM_EVAL_N_CTX="${LM_EVAL_N_CTX:-2048}"
LM_EVAL_BATCH="${LM_EVAL_BATCH:-1}"

[ -f "$GGUF" ] || { echo "ERROR: $GGUF not found" >&2; exit 1; }
[ -f "$TASKS_YAML" ] || { echo "ERROR: $TASKS_YAML not found" >&2; exit 1; }

# Resolve the wrapper script alongside this one (so it works from any cwd).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WRAPPER="$SCRIPT_DIR/lm_eval_wrapper.py"
[ -f "$WRAPPER" ] || { echo "ERROR: $WRAPPER not found" >&2; exit 1; }

# Read task config (task_id + num_fewshot) from configs/tasks.yaml.
read -r task_id num_fewshot < <(python3 - "$TASKS_YAML" "$TASK_KEY" <<'PY'
import yaml, sys
yaml_path, key = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(yaml_path))['tasks']
if key not in cfg:
    sys.exit(f"unknown task_key: {key}")
t = cfg[key]
print(t['task_id'], t['num_fewshot'])
PY
)

mkdir -p "$OUT_DIR/lm_eval"

# Idempotency: skip if results JSON already exists.
if [ -f "$OUT_DIR/lm_eval/${TASK_KEY}.json" ]; then
  echo "lm_eval already done: $OUT_DIR/lm_eval/${TASK_KEY}.json"
  exit 0
fi

# Resolve GGUF to an absolute path so it survives lm_eval's tempdir games.
GGUF_ABS="$(python3 -c "import os, sys; print(os.path.abspath(sys.argv[1]))" "$GGUF")"

# Build optional --limit clause.
limit_clause=()
if [ -n "$LM_EVAL_LIMIT" ]; then
  limit_clause=(--limit "$LM_EVAL_LIMIT")
fi

echo "==> lm_eval $TASK_KEY (model=gguf_local, gguf=$GGUF_ABS, n_ctx=$LM_EVAL_N_CTX${LM_EVAL_LIMIT:+, limit=$LM_EVAL_LIMIT})"

python3 "$WRAPPER" \
  --model gguf_local \
  --model_args "pretrained=${GGUF_ABS},n_ctx=${LM_EVAL_N_CTX},verbose=False" \
  --tasks "$task_id" \
  --num_fewshot "$num_fewshot" \
  --batch_size "$LM_EVAL_BATCH" \
  --log_samples \
  --output_path "$OUT_DIR/lm_eval/${TASK_KEY}_raw" \
  "${limit_clause[@]}" \
  2>&1 | tee "$OUT_DIR/lm_eval/${TASK_KEY}.log"

# The harness writes a results JSON deep inside output_path; locate and copy it.
results_json=$(find "$OUT_DIR/lm_eval/${TASK_KEY}_raw" -name 'results*.json' -type f | head -1)
if [ -z "$results_json" ]; then
  echo "ERROR: lm-eval produced no results JSON" >&2
  exit 1
fi
cp "$results_json" "$OUT_DIR/lm_eval/${TASK_KEY}.json"
echo "lm_eval done: $OUT_DIR/lm_eval/${TASK_KEY}.json"
