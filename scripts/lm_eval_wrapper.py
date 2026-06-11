#!/usr/bin/env python3
"""Wrapper that registers the local GGUF adapter, then defers to lm_eval CLI.

`lm_eval --include_path` only picks up task YAML, not Python @register_model
modules. So we import the adapter here (which triggers the registration via
its decorator) and then forward all argv to lm_eval's cli_evaluate.

Usage matches `lm_eval` exactly — same flags, same args.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make scripts/lib importable.
sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

# Importing this module triggers @register_model("gguf_local").
import lm_eval_gguf_runner  # noqa: F401

from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    sys.exit(cli_evaluate())
