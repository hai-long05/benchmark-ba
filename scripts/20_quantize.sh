#!/usr/bin/env bash
# Quantize an FP16 GGUF using llama-quantize. Idempotent.
# Usage: 20_quantize.sh <fp16.gguf> <SCHEME>
#   e.g. 20_quantize.sh models/qwen_qwen3-0.6b-fp16.gguf Q4_K_S
# Output: quantized/<base>-<scheme>.gguf + .sha256 + .quant.json
#
# Override: when KURT_GGUFS_DIR is set, the quantized GGUF is sourced from
# <KURT_GGUFS_DIR>/Llama-3.1-8B-Instruct-<SCHEME>.gguf (case-insensitive match)
# instead of being produced locally with llama-quantize. Bit-exact-replication
# path matching Kurt's published GGUF release.
set -euo pipefail

IN_GGUF="${1:?Usage: $0 <fp16.gguf> <SCHEME>}"
SCHEME="${2:?Usage: $0 <fp16.gguf> <SCHEME>}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-../llama.cpp}"
KURT_GGUFS_DIR="${KURT_GGUFS_DIR:-}"
QUANT_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"

[ -f "$IN_GGUF" ] || { echo "ERROR: $IN_GGUF not found" >&2; exit 1; }
# llama-quantize is only needed when we're actually quantizing locally; with
# KURT_GGUFS_DIR set we copy a pre-built GGUF and skip the quantizer entirely.
if [ -z "$KURT_GGUFS_DIR" ] && [ ! -x "$QUANT_BIN" ]; then
  echo "ERROR: $QUANT_BIN not found — build llama.cpp" >&2
  exit 1
fi

# F16 / FP16 is the unquantized baseline — caller (run_slot.sh) handles it
# directly without invoking this script. Fail loudly if it's passed here.
scheme_upper=$(echo "$SCHEME" | tr '[:lower:]' '[:upper:]')
if [ "$scheme_upper" = "F16" ] || [ "$scheme_upper" = "FP16" ]; then
  echo "ERROR: 20_quantize.sh does not handle F16/FP16; run_slot.sh routes that scheme directly" >&2
  exit 1
fi

# Pick a sha256 tool that exists.
if command -v sha256sum >/dev/null 2>&1; then
  SHA_CMD="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
  SHA_CMD="shasum -a 256"
else
  echo "ERROR: no sha256sum or shasum found" >&2
  exit 1
fi

mkdir -p quantized
base=$(basename "$IN_GGUF" .gguf | sed 's/-fp16$//')
scheme_lower=$(echo "$SCHEME" | tr '[:upper:]' '[:lower:]')
OUT_GGUF="quantized/${base}-${scheme_lower}.gguf"
OUT_SHA="${OUT_GGUF}.sha256"
OUT_META="${OUT_GGUF}.quant.json"

# Idempotency: require all three artefacts (gguf + sha + meta) to be present and matching.
if [ -f "$OUT_GGUF" ] && [ -f "$OUT_SHA" ] && [ -f "$OUT_META" ]; then
  expected=$(awk '{print $1}' "$OUT_SHA")
  actual=$($SHA_CMD "$OUT_GGUF" | awk '{print $1}')
  if [ "$expected" = "$actual" ]; then
    echo "quantized already present and verified: $OUT_GGUF"
    exit 0
  fi
fi
# Stale or partial — clear and re-run.
rm -f "$OUT_GGUF" "$OUT_SHA" "$OUT_META"

# KURT_GGUFS_DIR override: copy Kurt's published quantized GGUF instead of
# producing one locally. Replicates Kurt 2026 bit-exactly.
if [ -n "$KURT_GGUFS_DIR" ]; then
  KURT_QUANT=$(find "$KURT_GGUFS_DIR" -maxdepth 2 -type f -iname "*${SCHEME}*.gguf" 2>/dev/null \
    | grep -v -i 'fp16\|f16' | head -1)
  if [ -z "$KURT_QUANT" ]; then
    echo "ERROR: KURT_GGUFS_DIR=$KURT_GGUFS_DIR set but no GGUF matching scheme '$SCHEME' found" >&2
    exit 1
  fi
  echo "using Kurt's quantized GGUF: $KURT_QUANT"
  cp "$KURT_QUANT" "$OUT_GGUF"
  $SHA_CMD "$OUT_GGUF" > "$OUT_SHA"
  size_mib=$(python3 -c "import os, sys; print(round(os.path.getsize(sys.argv[1]) / (1024*1024), 2))" "$OUT_GGUF")
  in_size_mib=$(python3 -c "import os, sys; print(round(os.path.getsize(sys.argv[1]) / (1024*1024), 2))" "$IN_GGUF")
  python3 - "$SCHEME" "$IN_GGUF" "$in_size_mib" "$size_mib" "0" "$KURT_QUANT" <<'PY' > "$OUT_META"
import json, sys
scheme, input_gguf, in_mib, out_mib, qt, source = sys.argv[1:7]
in_mib_f, out_mib_f = float(in_mib), float(out_mib)
reduction_pct = round((1.0 - out_mib_f / in_mib_f) * 100.0, 2) if in_mib_f else 0.0
print(json.dumps({
  "scheme": scheme,
  "input_gguf": input_gguf,
  "input_size_mib": in_mib_f,
  "output_size_mib": out_mib_f,
  "size_reduction_pct": reduction_pct,
  "quant_time_s": int(qt),
  "source": "kurt_ggufs",
  "source_path": source,
}, indent=2))
PY
  reduction_pct=$(python3 -c "import sys; out, inp = float(sys.argv[1]), float(sys.argv[2]); print(round((1.0 - out/inp) * 100, 2) if inp else 0.0)" "$size_mib" "$in_size_mib")
  echo "quantized: $OUT_GGUF (${size_mib} MiB, ${reduction_pct}% reduction, [from KURT_GGUFS_DIR])"
  exit 0
fi

# Run + time it
start=$(date +%s)
"$QUANT_BIN" "$IN_GGUF" "$OUT_GGUF" "$SCHEME"
end=$(date +%s)
quant_time_s=$((end - start))

# Hash + sizes
$SHA_CMD "$OUT_GGUF" > "$OUT_SHA"
size_mib=$(python3 -c "import os, sys; print(round(os.path.getsize(sys.argv[1]) / (1024*1024), 2))" "$OUT_GGUF")
in_size_mib=$(python3 -c "import os, sys; print(round(os.path.getsize(sys.argv[1]) / (1024*1024), 2))" "$IN_GGUF")

# Write meta via Python so escaping is safe.
python3 - "$SCHEME" "$IN_GGUF" "$in_size_mib" "$size_mib" "$quant_time_s" <<'PY' > "$OUT_META"
import json, sys
scheme, input_gguf, in_mib, out_mib, qt = sys.argv[1:6]
in_mib_f, out_mib_f = float(in_mib), float(out_mib)
reduction_pct = round((1.0 - out_mib_f / in_mib_f) * 100.0, 2) if in_mib_f else 0.0
print(json.dumps({
  "scheme": scheme,
  "input_gguf": input_gguf,
  "input_size_mib": in_mib_f,
  "output_size_mib": out_mib_f,
  "size_reduction_pct": reduction_pct,
  "quant_time_s": int(qt),
}, indent=2))
PY

reduction_pct=$(python3 -c "import sys; out, inp = float(sys.argv[1]), float(sys.argv[2]); print(round((1.0 - out/inp) * 100, 2) if inp else 0.0)" "$size_mib" "$in_size_mib")
echo "quantized: $OUT_GGUF (${size_mib} MiB, ${reduction_pct}% reduction, ${quant_time_s}s)"
