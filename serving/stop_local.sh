#!/usr/bin/env bash
# Stop the local model server and its tunnel.
for p in /tmp/localspace.pid /tmp/ngrok.pid; do
  [ -f "$p" ] && kill "$(cat "$p")" 2>/dev/null && rm -f "$p"
done
pkill -f "python app.py" 2>/dev/null
pkill -f "ngrok http" 2>/dev/null
echo "stopped"
