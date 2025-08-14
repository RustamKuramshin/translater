#!/usr/bin/env bash
set -euo pipefail

# Simple runner for the PDF translator
# Usage examples:
#   ./run.sh --input doc.pdf --output doc.ru.md
#   ./run.sh --config config.yaml --input doc.pdf

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set. Export it before running." >&2
  echo "       Example: export OPENAI_API_KEY=sk-..." >&2
  exit 1
fi

# Prefer installed console script if available
if command -v pdf-translater >/dev/null 2>&1; then
  exec pdf-translater "$@"
fi

# Fallback to running the local script directly
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/translater.py" --input hands-high-performance-spring-5.pdf --output out.md --config config.yaml --page-end 10
