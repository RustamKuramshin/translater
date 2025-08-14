#!/usr/bin/env bash
set -euo pipefail

# Installer for the PDF translator dependencies using requirements.txt
# Usage:
#   ./install.sh        # installs into current environment
#   OPENAI_API_KEY=... ./install.sh  # optional, only needed at run time

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQS_FILE="${SCRIPT_DIR}/requirements.txt"

if [[ ! -f "${REQS_FILE}" ]]; then
  echo "ERROR: requirements.txt not found next to install.sh" >&2
  exit 1
fi

# Choose Python
PYTHON_BIN=${PYTHON_BIN:-python3}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Could not find Python (python3). Set PYTHON_BIN to your Python executable and retry." >&2
  exit 1
fi

# Ensure pip available
if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  echo "ERROR: pip is not available for $PYTHON_BIN" >&2
  exit 1
fi

echo "Installing dependencies from requirements.txt..."
"$PYTHON_BIN" -m pip install -r "$REQS_FILE"

echo "Done. You can now run: ./run.sh --input your.pdf --output out.md --config config.yaml"
