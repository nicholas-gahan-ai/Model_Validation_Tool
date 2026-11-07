#!/usr/bin/env bash
# Start the Gahan AI Validation Viewer (desktop app). Pass --cli ... for a headless run.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Run ./setup_linux.sh first."; exit 1
fi
exec .venv/bin/python app.py "$@"
