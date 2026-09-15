"""Stream a running Kaggle notebook's logs to a local file.

`kaggle kernels output` only works once a version has finished, so a long run is
invisible while it matters most. The SDK exposes GetKernelSessionLogsStream,
which the CLI does not surface: it returns Server-Sent Events while the session
is live. This follows that stream and appends to a file you can grep.

    python kaggle_tail.py --user gb1105 --slug qwen3-4b-text-to-sql-fine-tune \
        --out /tmp/kaggle_live.log
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from kagglesdk import KaggleClient
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest


def follow(user: str, slug: str, out: Path, wait: int) -> None:
    req = ApiGetKernelSessionLogsStreamRequest()
    req.user_name = user
    req.kernel_slug = slug
    req.wait_for_logs_url_seconds = wait

    with KaggleClient() as client:
        resp = client.kernels.kernels_api_client.get_kernel_session_logs_stream(req)
        with out.open("a", buffering=1) as fh:
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw[5:].strip() if raw.startswith("data:") else raw.strip()
                if line == "END_OF_LOG":
                    fh.write("[stream] END_OF_LOG\n")
                    return
                # payloads arrive as JSON objects with the text under a few keys
                try:
                    obj = json.loads(line)
                    line = obj.get("data") or obj.get("message") or obj.get("text") or line
                except Exception:
                    pass
                fh.write(line.rstrip() + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--out", type=Path, default=Path("/tmp/kaggle_live.log"))
    ap.add_argument("--wait", type=int, default=60)
    ap.add_argument("--retries", type=int, default=100,
                    help="reconnect this many times; the stream drops periodically")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(args.retries):
        try:
            follow(args.user, args.slug, args.out, args.wait)
            return  # END_OF_LOG
        except Exception as e:
            with args.out.open("a") as fh:
                fh.write(f"[stream] reconnect {attempt + 1}: {type(e).__name__}\n")
            time.sleep(10)
    print("gave up reconnecting", file=sys.stderr)


if __name__ == "__main__":
    main()
