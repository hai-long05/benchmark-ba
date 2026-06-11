#!/usr/bin/env bash
# Run lm-evaluation-harness on one or more tasks against a quantized GGUF in a
# single Python process. Loading the model once and running all tasks in one
# invocation is ~1.4x faster than re-launching per task on big models, plus it
# avoids re-parsing the GGUF header each time.
#
# Usage: 40_lm_eval.sh <gguf> <task_keys_csv> <out_dir>
#   <task_keys_csv> matches keys under `tasks:` in configs/tasks.yaml,
#   comma-separated (e.g. "hellaswag,gsm8k,ifeval").
#
# Implementation:
#   Loads the GGUF directly via llama-cpp-python in-process and computes exact
#   per-token logprobs from the full softmax. Custom model class:
#   scripts/lib/lm_eval_gguf_runner.py, registered as `--model gguf_local` by
#   scripts/lm_eval_wrapper.py.
#
# Env vars:
#   LM_EVAL_LIMIT      — cap items per task (smoke runs only).
#   LM_EVAL_N_CTX      — llama context length (default 2048; set 4096 for wikitext).
#   LM_EVAL_BATCH      — lm_eval --batch_size (default 1; tune up on big servers).
#   LM_EVAL_N_THREADS  — llama-cpp-python n_threads (default: omit = auto).
set -euo pipefail

GGUF="${1:?Usage: $0 <gguf> <task_keys_csv> <out_dir>}"
TASKS_CSV="${2:?task_keys_csv required (comma-separated)}"
OUT_DIR="${3:?out_dir required}"
TASKS_YAML="${TASKS_YAML:-configs/tasks.yaml}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-}"
LM_EVAL_N_CTX="${LM_EVAL_N_CTX:-2048}"
LM_EVAL_BATCH="${LM_EVAL_BATCH:-1}"
LM_EVAL_N_THREADS="${LM_EVAL_N_THREADS:-}"

[ -f "$GGUF" ] || { echo "ERROR: $GGUF not found" >&2; exit 1; }
[ -f "$TASKS_YAML" ] || { echo "ERROR: $TASKS_YAML not found" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WRAPPER="$SCRIPT_DIR/lm_eval_wrapper.py"
[ -f "$WRAPPER" ] || { echo "ERROR: $WRAPPER not found" >&2; exit 1; }

mkdir -p "$OUT_DIR/lm_eval"

# Resolve task keys to (task_id, num_fewshot) pairs.
# Group keys by num_fewshot so we can submit them in one or two lm_eval invocations
# (lm_eval accepts a single --num_fewshot flag per call).
python3 - "$TASKS_YAML" "$TASKS_CSV" <<'PY' > /tmp/_lm_eval_groups.json.$$
import json, sys, yaml
yaml_path, csv = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(yaml_path))['tasks']
keys = [k.strip() for k in csv.split(',') if k.strip()]
unknown = [k for k in keys if k not in cfg]
if unknown:
    sys.exit(f"unknown task_keys: {unknown}")
groups = {}
for k in keys:
    t = cfg[k]
    fs = int(t['num_fewshot'])
    groups.setdefault(fs, []).append((k, t['task_id']))
print(json.dumps(groups))
PY
GROUPS_FILE="/tmp/_lm_eval_groups.json.$$"
trap 'rm -f "$GROUPS_FILE"' EXIT

# Filter out tasks whose final results.json already exists (idempotency).
TODO_GROUPS=$(python3 - "$GROUPS_FILE" "$OUT_DIR/lm_eval" <<'PY'
import json, os, sys
groups_path, lm_eval_dir = sys.argv[1], sys.argv[2]
groups = json.load(open(groups_path))
remaining = {}
for fs, pairs in groups.items():
    keep = [(k, tid) for (k, tid) in pairs if not os.path.exists(os.path.join(lm_eval_dir, f"{k}.json"))]
    if keep:
        remaining[fs] = keep
print(json.dumps(remaining))
PY
)

if [ "$TODO_GROUPS" = "{}" ]; then
  echo "lm_eval: all requested tasks already have results in $OUT_DIR/lm_eval/"
  exit 0
fi

GGUF_ABS="$(python3 -c "import os, sys; print(os.path.abspath(sys.argv[1]))" "$GGUF")"

# Build model_args. Include n_threads only if explicitly set.
MODEL_ARGS="pretrained=${GGUF_ABS},n_ctx=${LM_EVAL_N_CTX},verbose=False"
if [ -n "$LM_EVAL_N_THREADS" ]; then
  MODEL_ARGS="${MODEL_ARGS},n_threads=${LM_EVAL_N_THREADS}"
fi

limit_clause=()
if [ -n "$LM_EVAL_LIMIT" ]; then
  limit_clause=(--limit "$LM_EVAL_LIMIT")
fi

# For each fewshot group, run all tasks in that group in one lm_eval call.
# This is what makes the script faster: model loads once per group, not per task.
echo "$TODO_GROUPS" | python3 -c "
import json, sys
groups = json.load(sys.stdin)
for fs, pairs in groups.items():
    keys = ','.join(k for k, _ in pairs)
    task_ids = ','.join(tid for _, tid in pairs)
    print(f'{fs}\t{keys}\t{task_ids}')
" | while IFS=$'\t' read -r FS KEYS TASK_IDS; do
  GROUP_RAW="$OUT_DIR/lm_eval/_group_fs${FS}_raw"
  GROUP_LOG="$OUT_DIR/lm_eval/_group_fs${FS}.log"
  echo "==> lm_eval (n_fewshot=$FS, tasks: $KEYS)"

  python3 "$WRAPPER" \
    --model gguf_local \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASK_IDS" \
    --num_fewshot "$FS" \
    --batch_size "$LM_EVAL_BATCH" \
    --log_samples \
    --output_path "$GROUP_RAW" \
    "${limit_clause[@]}" \
    2>&1 | tee "$GROUP_LOG"

  # Find the harness's results JSON and split it back into per-task JSONs.
  results_json=$(find "$GROUP_RAW" -name 'results*.json' -type f | head -1)
  if [ -z "$results_json" ]; then
    echo "ERROR: lm-eval produced no results JSON for group fs=$FS" >&2
    exit 1
  fi

  python3 - "$results_json" "$KEYS" "$TASK_IDS" "$OUT_DIR/lm_eval" <<'PY'
import json, os, sys
src, keys_csv, ids_csv, out_dir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
keys = keys_csv.split(',')
ids = ids_csv.split(',')
data = json.load(open(src))
# `results` is keyed by task_id (the lm_eval id, not our task_key). Build a per-task
# slice that matches the original schema, then write one file per task_key.
all_results = data.get('results', {})
all_versions = data.get('versions', {})
all_nshot = data.get('n-shot', {})
configs = data.get('configs', {})
for key, tid in zip(keys, ids):
    if tid not in all_results:
        # Some tasks like 'mmlu' are groups that expand to subtasks; collect them.
        sub = {k: v for k, v in all_results.items() if k.startswith(tid)}
        if not sub:
            print(f'WARN: no results for task_id={tid} (key={key})', file=sys.stderr)
            continue
        slice_data = {
            'results': sub,
            'versions': {k: v for k, v in all_versions.items() if k in sub},
            'n-shot': {k: v for k, v in all_nshot.items() if k in sub},
            'config': data.get('config', {}),
        }
    else:
        slice_data = {
            'results': {tid: all_results[tid]},
            'versions': {tid: all_versions.get(tid)},
            'n-shot': {tid: all_nshot.get(tid)},
            'config': data.get('config', {}),
        }
    with open(os.path.join(out_dir, f'{key}.json'), 'w') as f:
        json.dump(slice_data, f, indent=2)
    print(f'lm_eval done: {os.path.join(out_dir, key + ".json")}')
PY
done
