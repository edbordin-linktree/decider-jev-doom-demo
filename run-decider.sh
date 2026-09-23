#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'This demo requires an Apple Silicon Mac.' >&2
  exit 1
fi
if ! command -v uv >/dev/null; then
  echo 'Install uv first: brew install uv' >&2
  exit 1
fi
exec uv run --project demo/doom --locked --no-dev python demo/doom/launch_decider.py "$@"
