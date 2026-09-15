#!/usr/bin/env bash
# Serve the fine-tuned model from this machine behind a stable public URL.
#
# ngrok's free tier includes one reserved domain, so unlike a Cloudflare quick
# tunnel the URL does not change between restarts and Vercel can point at it
# permanently.
#
#   set -a && . ../.env && set +a
#   bash start_local.sh
#
# Needs roughly 5GB free. Do not run it while a training job is going: both
# compete for unified memory and each ends up swapping.

set -euo pipefail
cd "$(dirname "$0")"

: "${NGROK_AUTHTOKEN:?run: set -a && . ../.env && set +a}"
NGROK_DOMAIN="${NGROK_DOMAIN:-bustled-hertha-unprojective.ngrok-free.dev}"
GGUF="${GGUF_PATH:-$HOME/.cache/text2sql/qwen3-4b-text2sql-q8.gguf}"
PORT="${PORT:-7860}"
PY=../.venv/bin/python

if [ ! -f "$GGUF" ]; then
  echo "GGUF not found at $GGUF"
  echo "Fetch it once with:"
  echo "  mkdir -p \"$(dirname "$GGUF")\""
  echo "  $PY -c \"from huggingface_hub import hf_hub_download; import shutil; \\"
  echo "    shutil.copy(hf_hub_download('bharatverse11/qwen3-4b-text2sql-gguf', \\"
  echo "    'qwen3-4b-text2sql-q8_0.gguf'), '$GGUF')\""
  exit 1
fi

echo "==> starting the model server on :$PORT"
USE_GGUF=1 GGUF_PATH="$GGUF" N_THREADS="${N_THREADS:-8}" \
  MODEL_ID=bharatverse11/qwen3-4b-text2sql MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-320}" \
  nohup $PY app.py > /tmp/localspace.log 2>&1 &
echo $! > /tmp/localspace.pid

until curl -s -o /dev/null "http://localhost:$PORT/" 2>/dev/null; do sleep 3; done
echo "    model server up"

echo "==> opening the tunnel on $NGROK_DOMAIN"
ngrok config add-authtoken "$NGROK_AUTHTOKEN" >/dev/null
nohup ngrok http "$PORT" --url "$NGROK_DOMAIN" --log stdout > /tmp/ngrok.log 2>&1 &
echo $! > /tmp/ngrok.pid
sleep 6

if curl -s -o /dev/null -w '%{http_code}' -m 20 "https://$NGROK_DOMAIN/" \
     -H "ngrok-skip-browser-warning: 1" | grep -q 200; then
  echo "    live at https://$NGROK_DOMAIN"
  echo
  echo "Vercel is already pointed here, so https://text2sql-qwen3.vercel.app works."
  echo "Stop with: bash stop_local.sh"
else
  echo "    tunnel did not answer; check /tmp/ngrok.log"
fi
