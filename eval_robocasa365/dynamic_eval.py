# Copyright (c) 2026 Awomo-WAM Team. Licensed under the Apache License, Version 2.0.
"""File-queue evaluation: `init` writes the episode list, each `worker` claims episodes, `merge` writes summary.json.

An episode that raises is retried up to MAX_ATTEMPTS times, then recorded in errors/ (re-running `init` retries it).
An output directory belongs to one model, checkpoint and evaluation config: `init` records the config in config.json
and refuses a different one; the first worker records the server's identity in identity.json, and workers (also after
a reconnect) refuse a different server. summary.json carries the identity and the config. Queue and result files are
written atomically.

    python eval_robocasa365/dynamic_eval.py init   --out OUT [--task-set target50] [--trials 30] [--tasks ...]
    python eval_robocasa365/dynamic_eval.py worker --out OUT --port 18000
    python eval_robocasa365/dynamic_eval.py merge  --out OUT
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import defaultdict
from pathlib import Path

import entry

MAX_ATTEMPTS = 3
IDENTITY_KEYS = ("model", "benchmark", "checkpoint")
SEED_RULE = f"env seed {entry.SEED}; episode seed = {entry.SEED} + 50 * task index within its group + trial"


def write_json(path: Path, value) -> None:
    """Write via a temporary file and rename, so readers never see a partial file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=1))
    os.replace(tmp, path)


def check_identity(out: Path, metadata: dict) -> None:
    """Record the server's model/checkpoint for `out`, or exit if `out` holds results of another one."""
    identity = {key: metadata.get(key) for key in IDENTITY_KEYS}
    path = out / "identity.json"
    if not path.exists():
        if any((out / "results").glob("*.json")):
            raise SystemExit(f"{out} has results but no identity.json; use a new output directory")
        tmp = out / f".identity.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(identity, indent=1))
        try:
            os.link(tmp, path)  # atomic: only the first worker creates identity.json
        except FileExistsError:
            pass
        finally:
            tmp.unlink()
    recorded = json.loads(path.read_text())
    if recorded != identity:
        raise SystemExit(f"{out} holds results of {recorded}, but the policy server is {identity}; "
                         "use a separate output directory for each model and checkpoint")


def connect(out: Path, port: int):
    """Connect to the policy server and check it is the one `out` belongs to (also after every reconnect)."""
    client = entry.Client(port=port)
    try:
        check_identity(out, client.metadata)
    except SystemExit:
        client.close()
        raise
    return client


def init(out: Path, task_set: str, trials: int, tasks: list[str] | None) -> None:
    config = {"task_set": task_set, "trials": trials, "tasks": sorted(tasks) if tasks else None, "split": "pretrain",
              "seeds": SEED_RULE}
    path = out / "config.json"
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise SystemExit(f"{out} was started with {json.loads(path.read_text())}, not {config}; "
                             "use a separate output directory for a different task set, trial count or task list")
    elif (out / "results").is_dir() and any((out / "results").glob("*.json")):
        raise SystemExit(f"{out} has results but no config.json; use a new output directory")
    for name in ("pending", "running", "results", "errors"):
        (out / name).mkdir(parents=True, exist_ok=True)
    write_json(path, config)
    for failed in (out / "errors").glob("*.json"):  # retry episodes that failed in an earlier run
        failed.unlink()
    for stale in (out / "running").glob("*.json*"):  # requeue episodes of an interrupted run
        stale.rename(out / "pending" / (stale.name.split(".json")[0] + ".json"))
    queued = 0
    for job in entry.list_jobs(task_set, trials, tasks):
        name = f"{job['group']}__{job['task']}__{job['trial']:03d}.json"
        if not (out / "results" / name).exists():
            write_json(out / "pending" / name, job)
            queued += 1
    print(f"{queued} episodes queued in {out}")


def worker(out: Path, port: int) -> None:
    client = connect(out, port)
    tag = f"{os.getpid()}"
    while True:
        pending = sorted((out / "pending").glob("*.json"))
        if not pending:
            break
        running = out / "running" / f"{pending[0].name}.{tag}"
        try:
            pending[0].rename(running)  # atomic claim
        except FileNotFoundError:
            continue
        job = json.loads(running.read_text())
        started = time.time()
        try:
            result = entry.run_episode(client, job)
        except Exception:  # noqa: BLE001 - requeue the episode (or give up after MAX_ATTEMPTS) and reconnect
            error = traceback.format_exc()
            print(f"ERROR {job['task']} trial {job['trial']}\n{error}", flush=True)
            job["attempts"] = int(job.get("attempts", 0)) + 1
            if job["attempts"] >= MAX_ATTEMPTS:
                write_json(out / "errors" / pending[0].name, {**job, "error": error})
            else:
                write_json(out / "pending" / pending[0].name, job)
            running.unlink()
            client.close()
            client = connect(out, port)  # the server may have been restarted with another checkpoint
            continue
        write_json(out / "results" / pending[0].name, result)
        running.unlink()
        print(f"{job['task']} trial {job['trial']}: {'success' if result['success'] else 'fail'} "
              f"({result['steps']} steps, {time.time() - started:.0f}s)", flush=True)
    client.close()


def merge(out: Path) -> int:
    """Write summary.json; return 1 while episodes are unfinished or failed, else 0."""
    groups, tasks = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for path in (out / "results").glob("*.json"):
        r = json.loads(path.read_text())
        for table, key in ((groups, r["group"]), (tasks, r["task"])):
            table[key][0] += int(r["success"])
            table[key][1] += 1

    def rate(s, n):
        return round(100.0 * s / max(n, 1), 2)

    def recorded(name):
        path = out / name
        return json.loads(path.read_text()) if path.exists() else None

    total = [sum(v[0] for v in groups.values()), sum(v[1] for v in groups.values())]
    summary = {"identity": recorded("identity.json"), "config": recorded("config.json"),
               "groups": {k: {"success": s, "episodes": n, "success_rate": rate(s, n)} for k, (s, n) in sorted(groups.items())},
               "overall": {"success": total[0], "episodes": total[1], "success_rate": rate(*total)},
               "tasks": {k: {"success": s, "episodes": n} for k, (s, n) in sorted(tasks.items())},
               "unfinished": len(list((out / "pending").glob("*.json"))) + len(list((out / "running").glob("*"))),
               "errors": sorted(path.stem for path in (out / "errors").glob("*.json"))}
    write_json(out / "summary.json", summary)
    for name, value in [*summary["groups"].items(), ("overall", summary["overall"])]:
        print(f"{name:17s} {value['success']:4d}/{value['episodes']:4d} = {value['success_rate']:.2f}%")
    if summary["unfinished"]:
        print(f"{summary['unfinished']} episodes unfinished: rerun the launch script to resume")
    if summary["errors"]:
        print(f"{len(summary['errors'])} episodes failed {MAX_ATTEMPTS} times (see errors/): rerun the launch script to retry")
    return 1 if summary["unfinished"] or summary["errors"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("init", "worker", "merge"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--task-set", default="target50", choices=("target50", *entry.GROUPS))
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--port", type=int, default=18000)
    args = parser.parse_args()
    if args.command == "init":
        init(args.out, args.task_set, args.trials, args.tasks)
    elif args.command == "worker":
        worker(args.out, args.port)
    else:
        raise SystemExit(merge(args.out))


if __name__ == "__main__":
    main()
