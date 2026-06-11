#!/usr/bin/env bash
# Compute a stable SHA256 of the env/ directory contents.
# Sourced by 00_capture_env.sh and run_slot.sh — do not execute directly.
#
# Usage: env_hash [<dir>]   (default: env)
# Hashes each *.json file's (sha256 + path), concatenated in C-locale-sorted order, and re-hashes the result.
# A rename or content change both affect the output.

env_hash() {
  local env_dir="${1:-env}"
  if [ ! -d "$env_dir" ]; then
    echo "env_hash: $env_dir not found" >&2
    return 1
  fi

  # Pick a sha256 tool that exists on the platform.
  local sha_cmd
  if command -v sha256sum >/dev/null 2>&1; then
    sha_cmd="sha256sum"
  elif command -v shasum >/dev/null 2>&1; then
    sha_cmd="shasum -a 256"
  else
    echo "env_hash: no sha256sum or shasum found" >&2
    return 1
  fi

  # Refuse to hash an empty directory — caller expects real env files.
  local count
  count=$(find "$env_dir" -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
  if [ "$count" -eq 0 ]; then
    echo "env_hash: no *.json files in $env_dir" >&2
    return 1
  fi

  ( set -o pipefail
    cd "$env_dir" && find . -type f -name '*.json' -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 $sha_cmd \
      | $sha_cmd \
      | awk '{print $1}'
  )
}
