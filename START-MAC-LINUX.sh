#!/usr/bin/env bash
# ============================================================
#  Agentic Video Tutor - launcher for macOS / Linux
#  Run with:  bash START-MAC-LINUX.sh
#  (or make it double-clickable: chmod +x START-MAC-LINUX.sh)
# ============================================================
cd "$(dirname "$0")"

PY=python3; command -v python3 >/dev/null 2>&1 || PY=python
if ! command -v $PY >/dev/null 2>&1; then
  echo "Python not found. Install it: https://www.python.org/downloads/ (or 'brew install python')"
  read -r -p "Press Enter to close..."; exit 1
fi

echo "Installing dependencies (first run only, ~30s)..."
$PY -m pip install -q -r requirements.txt || {
  echo "pip install failed - trying with --user..."
  $PY -m pip install -q --user -r requirements.txt || { echo "Install failed. Check internet and Python version (need 3.10+)."; read -r -p "Press Enter to close..."; exit 1; }
}

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo
  read -r -p "Paste your Anthropic API key (starts with sk-ant-) and press Enter: " ANTHROPIC_API_KEY
  export ANTHROPIC_API_KEY
fi

echo
echo "Starting server... open http://localhost:8000 in your browser."
echo "Keep this window open while you use the app. Ctrl+C to stop."
( sleep 2; command -v open >/dev/null && open http://localhost:8000 || command -v xdg-open >/dev/null && xdg-open http://localhost:8000 ) &
exec $PY server.py
