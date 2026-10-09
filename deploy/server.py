# Copyright (c) 2026 Awomo-WAM Team. Licensed under the Apache License, Version 2.0.
"""Policy server, shared by all benchmarks.

    python deploy/server.py --policy eval_robocasa365.awomo_05i.policy --checkpoint <dir> --port 18000

`--policy` names a benchmark adapter module; it must provide `load_policy(checkpoint, **base_model_paths)`
returning an object with `act(request) -> (actions, info)`. Only the base-model paths given on the command line are
passed on. One client is served at a time; a client that disconnects (even mid-request) does not stop the server.
The metadata sent on connect identifies the model and the checkpoint (sha256 of its SHA256SUMS).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import logging
import socket
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "deploy")]

from client import recv, send


def serve(policy, host: str, port: int) -> None:
    log = logging.getLogger("server")
    with socket.create_server((host, port), backlog=8) as server:
        log.info("listening on %s:%d", host, port)
        while True:
            connection, _ = server.accept()
            with connection:
                try:
                    send(connection, {"type": "metadata", **policy.metadata})
                    while True:
                        try:
                            request = recv(connection)
                        except ConnectionError:
                            break
                        try:
                            actions, info = policy.act(request)
                            reply = {"type": "action", "actions": actions, "info": info}
                        except Exception as error:  # noqa: BLE001 - report to the client and keep serving
                            log.error("act failed: %s", traceback.format_exc())
                            reply = {"type": "error", "message": f"{type(error).__name__}: {error}"}
                        send(connection, reply)
                except OSError as error:  # the client went away; wait for the next one
                    log.warning("client disconnected: %s", error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="e.g. eval_robocasa365.awomo_05i.policy")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qwen-path", help="Awomo-0.5I: local Qwen3-VL-4B-Instruct (default: download)")
    parser.add_argument("--vae-path", help="Awomo-0.5I: local FLUX.2 VAE (default: download)")
    parser.add_argument("--wan-path", help="Awomo-0.5V: local Wan2.2-TI2V-5B (default: download)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = {name: getattr(args, name) for name in ("qwen_path", "vae_path", "wan_path") if getattr(args, name)}
    policy = importlib.import_module(args.policy).load_policy(args.checkpoint, **paths)
    sums = Path(args.checkpoint) / "SHA256SUMS"
    policy.metadata = {**policy.metadata, "checkpoint": hashlib.sha256(sums.read_bytes()).hexdigest()}
    serve(policy, args.host, args.port)


if __name__ == "__main__":
    main()
