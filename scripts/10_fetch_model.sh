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
