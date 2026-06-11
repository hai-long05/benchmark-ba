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

# Run llama-bench REPEATS times. Each invocation does one pp + one tg sample (with -r 1).
for i in $(seq 1 "$REPEATS"); do
  raw="$OUT_DIR/bench_raw/run-$i.json"
  "$BENCH_BIN" -m "$GGUF" -p "$CTX" -n 128 -r 1 -o json > "$raw"
done

# Concatenate all rows from each run into a single array, then aggregate.
combined="$OUT_DIR/bench_raw/all.json"
python3 - "$OUT_DIR/bench_raw" "$REPEATS" "$combined" <<'PY'
import json, os, sys
raw_dir, repeats, combined_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
all_rows = []
for i in range(1, repeats + 1):
    p = os.path.join(raw_dir, f"run-{i}.json")
    with open(p) as f:
        all_rows.extend(json.load(f))
with open(combined_path, "w") as f:
    json.dump(all_rows, f, indent=2)
PY

# Aggregate via the python helper.
python3 - "$combined" "$CTX" "$OUT_DIR/bench.json" <<'PY'
import json, sys
sys.path.insert(0, "scripts/lib")
import bench_parse
combined_path, ctx, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(combined_path) as f:
    rows = json.load(f)
agg = bench_parse.aggregate(rows, ctx=ctx)
with open(out_path, "w") as f:
    json.dump(agg, f, indent=2)
print(f"bench: pp={agg['pp']['median']} tok/s, tg={agg['tg']['median']} tok/s, {len(agg['warnings'])} warnings")
PY
