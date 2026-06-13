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
# Chat-template handling (Kurt 2026 §3.1 replication):
#   When TOKENIZER_REPO is set, prompted tasks (apply_chat_template:true in
#   tasks.yaml) are dispatched with --apply_chat_template --fewshot_as_multiturn
#   and tokenizer_repo wired into model_args. wikitext is dispatched in a
#   separate subprocess without these flags. Tasks group by (num_fewshot,
#   apply_chat_template) so each lm_eval invocation has a single fewshot value
#   and a single template policy.
#
# Env vars:
#   LM_EVAL_LIMIT          — cap items per task (smoke runs only).
#   LM_EVAL_N_CTX          — llama context length (default 2048; set 4096 for wikitext).
#   LM_EVAL_BATCH          — lm_eval --batch_size (default 1; tune up on big servers).
#   LM_EVAL_N_THREADS      — llama-cpp-python n_threads (default: omit = auto).
#   TOKENIZER_REPO         — HF id whose tokenizer holds the chat template (e.g.
#                            meta-llama/Llama-3.1-8B-Instruct). Required when any
#                            requested task has apply_chat_template:true.
#   CHAT_TEMPLATE_KWARGS   — JSON dict forwarded into apply_chat_template (e.g.
#                            '{"enable_thinking": false}' for Qwen3 to suppress
#                            reasoning blocks during loglikelihood scoring).
set -euo pipefail

GGUF="${1:?Usage: $0 <gguf> <task_keys_csv> <out_dir>}"
TASKS_CSV="${2:?task_keys_csv required (comma-separated)}"
OUT_DIR="${3:?out_dir required}"
TASKS_YAML="${TASKS_YAML:-configs/tasks.yaml}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-}"
LM_EVAL_N_CTX="${LM_EVAL_N_CTX:-2048}"
LM_EVAL_BATCH="${LM_EVAL_BATCH:-1}"
LM_EVAL_N_THREADS="${LM_EVAL_N_THREADS:-}"
TOKENIZER_REPO="${TOKENIZER_REPO:-}"
CHAT_TEMPLATE_KWARGS="${CHAT_TEMPLATE_KWARGS:-}"

[ -f "$GGUF" ] || { echo "ERROR: $GGUF not found" >&2; exit 1; }
[ -f "$TASKS_YAML" ] || { echo "ERROR: $TASKS_YAML not found" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WRAPPER="$SCRIPT_DIR/lm_eval_wrapper.py"
[ -f "$WRAPPER" ] || { echo "ERROR: $WRAPPER not found" >&2; exit 1; }

mkdir -p "$OUT_DIR/lm_eval"

# Resolve task keys to (task_id, num_fewshot, apply_chat_template) triples.
# Group keys by (num_fewshot, apply_chat_template) so each lm_eval invocation
# has a single fewshot value and a single template policy.
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
    tmpl = bool(t.get('apply_chat_template', False))
    group_key = f"{fs}|{int(tmpl)}"
    groups.setdefault(group_key, []).append((k, t['task_id'], fs, tmpl))
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
for gkey, entries in groups.items():
    keep = [e for e in entries if not os.path.exists(os.path.join(lm_eval_dir, f"{e[0]}.json"))]
    if keep:
        remaining[gkey] = keep
print(json.dumps(remaining))
PY
)

if [ "$TODO_GROUPS" = "{}" ]; then
  echo "lm_eval: all requested tasks already have results in $OUT_DIR/lm_eval/"
  exit 0
fi

# Sanity-check: any group with apply_chat_template=1 needs TOKENIZER_REPO.
NEEDS_TMPL=$(echo "$TODO_GROUPS" | python3 -c "
import json, sys
g = json.load(sys.stdin)
print(1 if any(k.endswith('|1') for k in g) else 0)
")
if [ "$NEEDS_TMPL" = "1" ] && [ -z "$TOKENIZER_REPO" ]; then
  echo "ERROR: tasks request apply_chat_template:true but TOKENIZER_REPO is not set" >&2
  echo "  Set TOKENIZER_REPO=<hf-id> (e.g. meta-llama/Llama-3.1-8B-Instruct)" >&2
  exit 1
fi

GGUF_ABS="$(python3 -c "import os, sys; print(os.path.abspath(sys.argv[1]))" "$GGUF")"

# Build base model_args. Include n_threads only if explicitly set. Templated
# groups append tokenizer_repo (and chat_template_kwargs when set).
BASE_MODEL_ARGS="pretrained=${GGUF_ABS},n_ctx=${LM_EVAL_N_CTX},verbose=False"
if [ -n "$LM_EVAL_N_THREADS" ]; then
  BASE_MODEL_ARGS="${BASE_MODEL_ARGS},n_threads=${LM_EVAL_N_THREADS}"
fi

limit_clause=()
if [ -n "$LM_EVAL_LIMIT" ]; then
  limit_clause=(--limit "$LM_EVAL_LIMIT")
fi

# For each group, run all tasks in it in one lm_eval call.
echo "$TODO_GROUPS" | python3 -c "
import json, sys
groups = json.load(sys.stdin)
for gkey, entries in groups.items():
    fs, tmpl = gkey.split('|')
    keys = ','.join(e[0] for e in entries)
    task_ids = ','.join(e[1] for e in entries)
    print(f'{fs}\t{tmpl}\t{keys}\t{task_ids}')
" | while IFS=$'\t' read -r FS TMPL KEYS TASK_IDS; do
  TMPL_TAG="notmpl"
  [ "$TMPL" = "1" ] && TMPL_TAG="tmpl"
  GROUP_RAW="$OUT_DIR/lm_eval/_group_fs${FS}_${TMPL_TAG}_raw"
  GROUP_LOG="$OUT_DIR/lm_eval/_group_fs${FS}_${TMPL_TAG}.log"
  echo "==> lm_eval (n_fewshot=$FS, chat_template=$TMPL, tasks: $KEYS)"

  # Build the per-group model_args + CLI flags.
  MODEL_ARGS="$BASE_MODEL_ARGS"
  template_clause=()
  if [ "$TMPL" = "1" ]; then
    MODEL_ARGS="${MODEL_ARGS},tokenizer_repo=${TOKENIZER_REPO}"
    if [ -n "$CHAT_TEMPLATE_KWARGS" ]; then
      # JSON dict; embed as a model_args value. Comma-free JSON is required
      # because lm_eval splits model_args on `,`. We pass the JSON via a
      # placeholder (single value) — no commas in our actual use case
      # ('{"enable_thinking": false}' has no commas).
      case "$CHAT_TEMPLATE_KWARGS" in
        *,*) echo "ERROR: CHAT_TEMPLATE_KWARGS must not contain commas (lm_eval splits model_args on ,)" >&2; exit 1;;
      esac
      MODEL_ARGS="${MODEL_ARGS},chat_template_kwargs=${CHAT_TEMPLATE_KWARGS}"
    fi
    template_clause=(--apply_chat_template --fewshot_as_multiturn)
  fi

  python3 "$WRAPPER" \
    --model gguf_local \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASK_IDS" \
    --num_fewshot "$FS" \
    --batch_size "$LM_EVAL_BATCH" \
    --log_samples \
    --output_path "$GROUP_RAW" \
    "${template_clause[@]}" \
    "${limit_clause[@]}" \
    2>&1 | tee "$GROUP_LOG"

  # Find the harness's results JSON and split it back into per-task JSONs.
  results_json=$(find "$GROUP_RAW" -name 'results*.json' -type f | head -1)
  if [ -z "$results_json" ]; then
    echo "ERROR: lm-eval produced no results JSON for group fs=$FS tmpl=$TMPL" >&2
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
