#!/usr/bin/env bash
# Download a Hugging Face model and ensure an FP16 GGUF exists under models/.
# Usage: 10_fetch_model.sh <hf-id>
#   e.g. 10_fetch_model.sh Qwen/Qwen3-0.6B
# Output: models/<safe-name>-fp16.gguf + .sha256 + .meta.json
#
# Override: when KURT_GGUFS_DIR is set, the FP16 GGUF is sourced from
# <KURT_GGUFS_DIR>/Llama-3.1-8B-Instruct-FP16.gguf (or whatever the directory
# contains as the FP16 baseline) instead of being fetched + converted from HF.
# This is the bit-exact-replication path: use the GGUFs Kurt published at
# huggingface.co/uygarkurt/Llama-3.1-8B-Instruct-GGUF rather than rebuilding
# them locally.
set -euo pipefail

HF_ID="${1:?Usage: $0 <hf-id>}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-../llama.cpp}"
KURT_GGUFS_DIR="${KURT_GGUFS_DIR:-}"
SAFE_NAME=$(echo "$HF_ID" | tr '/' '_' | tr -d "'\"" | tr '[:upper:]' '[:lower:]')
OUT_GGUF="models/${SAFE_NAME}-fp16.gguf"
OUT_SHA="${OUT_GGUF}.sha256"
OUT_META="${OUT_GGUF}.meta.json"
DL_DIR="models/_hf/${SAFE_NAME}"

# Pick a sha256 tool that exists.
if command -v sha256sum >/dev/null 2>&1; then
  SHA_CMD="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
  SHA_CMD="shasum -a 256"
else
  echo "ERROR: no sha256sum or shasum found" >&2
  exit 1
fi

mkdir -p models models/_hf

# Idempotency: if the GGUF + sha exist and match, exit 0.
if [ -f "$OUT_GGUF" ] && [ -f "$OUT_SHA" ]; then
  expected=$(awk '{print $1}' "$OUT_SHA")
  actual=$($SHA_CMD "$OUT_GGUF" | awk '{print $1}')
  if [ "$expected" = "$actual" ]; then
    echo "model already present and verified: $OUT_GGUF"
    exit 0
  fi
  echo "WARN: SHA mismatch on $OUT_GGUF, re-downloading" >&2
fi

# KURT_GGUFS_DIR override: copy Kurt's published FP16 GGUF in lieu of fetching
# from HF + converting locally. Replicates Kurt 2026 §3.1 with bit-identical
# weights from his open-source release at huggingface.co/uygarkurt/...-GGUF.
if [ -n "$KURT_GGUFS_DIR" ]; then
  # Find the FP16 file in Kurt's directory. Common names: *fp16*.gguf, *FP16*.gguf, *f16*.gguf.
  KURT_FP16=$(find "$KURT_GGUFS_DIR" -maxdepth 2 -type f \
    \( -iname '*fp16*.gguf' -o -iname '*f16*.gguf' \) 2>/dev/null | head -1)
  if [ -z "$KURT_FP16" ]; then
    echo "ERROR: KURT_GGUFS_DIR=$KURT_GGUFS_DIR set but no fp16/f16 GGUF found there" >&2
    exit 1
  fi
  echo "using Kurt's FP16 GGUF: $KURT_FP16"
  cp "$KURT_FP16" "$OUT_GGUF"
  $SHA_CMD "$OUT_GGUF" > "$OUT_SHA"
  size_mib=$(python3 -c "import os, sys; print(round(os.path.getsize(sys.argv[1]) / (1024*1024), 2))" "$OUT_GGUF")
  python3 - "$HF_ID" "kurt-released" "Llama-3.1-Community" "$size_mib" "$SAFE_NAME" "$KURT_FP16" <<'PY' > "$OUT_META"
import json, sys
hf_id, revision, license_file, size_mib, safe_name, source = sys.argv[1:7]
print(json.dumps({
  "hf_id": hf_id,
  "revision": revision,
  "license_file": license_file,
  "size_mib": float(size_mib),
  "safe_name": safe_name,
  "source": "kurt_ggufs",
  "source_path": source,
}, indent=2))
PY
  echo "fetched: $OUT_GGUF ($size_mib MiB) [from KURT_GGUFS_DIR]"
  exit 0
fi

# Download. The deprecated --local-dir-use-symlinks flag is dropped (default behavior is correct in current huggingface_hub).
echo "downloading $HF_ID -> $DL_DIR"
huggingface-cli download "$HF_ID" --local-dir "$DL_DIR"

# Capture the revision. huggingface_hub writes per-file .metadata files whose first line is the commit hash.
revision=$(awk 'FNR==1 && /^[0-9a-f]{40}$/ {print; exit}' \
  "$DL_DIR/.cache/huggingface/download/"*.metadata 2>/dev/null || echo "")
[ -n "$revision" ] || revision="UNKNOWN"

# Convert to FP16 GGUF if the repo doesn't already ship one.
existing_gguf=$(find "$DL_DIR" -maxdepth 2 -type f \( -name '*fp16*.gguf' -o -name '*f16*.gguf' \) 2>/dev/null | head -1)
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
  python3 "$CONVERT" "$DL_DIR" --outfile "$OUT_GGUF" --outtype f16
fi

# Hash + size
$SHA_CMD "$OUT_GGUF" > "$OUT_SHA"
size_mib=$(python3 -c "import os, sys; print(round(os.path.getsize(sys.argv[1]) / (1024*1024), 2))" "$OUT_GGUF")

# License: try to find LICENSE/LICENSE.* in download.
license=$(find "$DL_DIR" -maxdepth 2 -iname 'LICENSE*' -type f 2>/dev/null | head -1)
if [ -n "$license" ]; then
  license=$(basename "$license")
else
  license="unknown"
fi

# Write meta.json via Python so escaping is safe.
python3 - "$HF_ID" "$revision" "$license" "$size_mib" "$SAFE_NAME" <<'PY' > "$OUT_META"
import json, sys
hf_id, revision, license_file, size_mib, safe_name = sys.argv[1:6]
print(json.dumps({
  "hf_id": hf_id,
  "revision": revision,
  "license_file": license_file,
  "size_mib": float(size_mib),
  "safe_name": safe_name,
}, indent=2))
PY

echo "fetched: $OUT_GGUF ($size_mib MiB)"
