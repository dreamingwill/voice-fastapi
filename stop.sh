#!/bin/bash
set -euo pipefail

PROJECT_DIR="/home/orangepi/projects/voice/voice-fastapi"
PROCESS_PATTERN="python main.py"
cd "$PROJECT_DIR"

echo "Looking for voice-fastapi service processes..."
pids=$(pgrep -f "$PROCESS_PATTERN" || true)

if [ -z "$pids" ]; then
  echo "No running voice-fastapi service was found."
  exit 0
fi

echo "Stopping voice-fastapi service (PID(s): $pids)..."
kill $pids

for _ in $(seq 1 10); do
  sleep 1
  if ! pgrep -f "$PROCESS_PATTERN" >/dev/null; then
    echo "voice-fastapi service stopped."
    exit 0
  fi
done

echo "Service did not stop gracefully, forcing termination..."
remaining_pids=$(pgrep -f "$PROCESS_PATTERN" || true)
if [ -n "$remaining_pids" ]; then
  kill -9 $remaining_pids
  echo "Voice-fastapi service forcefully stopped (PID(s): $remaining_pids)."
else
  echo "Service already stopped."
fi
