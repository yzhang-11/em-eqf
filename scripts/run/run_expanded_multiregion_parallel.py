"""Run independent target cohorts concurrently, then score the complete study."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
import yaml

from run_expanded_multiregion import ROOT, Study, file_sha256, read_json, save_json


def init_plan(study):
    plan = dict(study.plan, plan_id=study.plan_id)
    path = study.runs / "plan.json"
    if path.exists() and read_json(path) != plan:
        raise ValueError("Parallel execution requires the original run plan")
    save_json(path, plan)
    save_json(study.reports / "plan.json", plan)
    for name, digest in study.plan["source_files"].items():
        destination = study.runs / "frozen_project" / name
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        if file_sha256(destination) != digest:
            raise ValueError(f"Frozen source hash mismatch: {name}")


def target_directory(runs, target):
    return runs / "parallel" / "targets" / target


def link_existing(runs, directory, target):
    directory.mkdir(parents=True, exist_ok=True)
    for existing in runs.glob(f"{target}__*"):
        if not existing.is_dir():
            continue
        link = directory / existing.name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(existing.resolve(), target_is_directory=True)
        elif not link.is_symlink() or link.resolve() != existing.resolve():
            raise ValueError(f"Two copies of the same fit: {link}, {existing}")


def fit_target(study, target, lock_fd):
    # Children retain the coordinator lock if the coordinator exits unexpectedly.
    if os.fstat(lock_fd).st_ino != (study.runs / ".lock").stat().st_ino:
        raise ValueError("Worker needs the coordinator lock")
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if target not in study.targets or read_json(study.runs / "plan.json")["plan_id"] != study.plan_id:
        raise ValueError("Worker target or plan mismatch")
    directory = target_directory(study.runs, target)
    link_existing(study.runs, directory, target)
    study.runs = directory
    study.targets = [target]
    study.fit(study.load("development.pt"))


def publish_target(study, target):
    directory = target_directory(study.runs, target)
    state = read_json(directory / "training.json")
    completed = state["completed"]
    expected = (study.plan["neural_fits"] + study.plan["etas_fits"]) // len(study.targets)
    if (state["plan_id"] != study.plan_id or len(set(completed)) != expected
            or len(completed) != expected or any(not label.startswith(f"{target}__") for label in completed)):
        raise ValueError(f"Incomplete target manifest: {target}")
    for label in completed:
        source = directory / label
        destination = study.runs / label
        result = read_json(source / "result.json")
        if (result["target"] != target or result["label"] != label or result["plan_id"] != study.plan_id
                or file_sha256(source / "best_model.pth") != result["checkpoint_sha256"]):
            raise ValueError(f"Invalid completed fit: {source}")
        if source.is_symlink():
            if source.resolve() != destination.resolve():
                raise ValueError(f"Unexpected run link: {source}")
        else:
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination)
            source.rename(destination)
            source.symlink_to(destination.resolve(), target_is_directory=True)
    print(f"TARGET_COMPLETE {target} {len(completed)}", flush=True)
    return state


def run_parallel(study, config, workers, lock_fd):
    init_plan(study)
    directory = study.runs / "parallel"
    logs = directory / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    save_json(directory / "execution.json", dict(plan_id=study.plan_id, workers=workers,
              launcher_sha256=file_sha256(__file__), targets=study.targets))
    queue = iter(study.targets)
    active, results = {}, {}
    try:
        while len(results) < len(study.targets):
            while len(active) < workers:
                target = next(queue, None)
                if target is None:
                    break
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--config", str(config),
                           "--worker", target, "--lock-fd", str(lock_fd)]
                with (logs / f"{target}.log").open("ab") as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                             pass_fds=(lock_fd,))
                active[target] = child
                print(f"TARGET_START {target} {child.pid}", flush=True)
            for target, child in list(active.items()):
                status = child.poll()
                if status is None:
                    continue
                if status:
                    raise RuntimeError(f"{target} failed ({status}). See {logs / (target + '.log')}")
                results[target] = publish_target(study, target)
                del active[target]
            if active:
                time.sleep(2)
    finally:
        for child in active.values():
            if child.poll() is None:
                child.terminate()
        for child in active.values():
            child.wait()
    save_json(study.runs / "training.json", dict(
        plan_id=study.plan_id,
        completed=[label for target in study.targets for label in results[target]["completed"]],
        selected=[row for target in study.targets for row in results[target]["selected"]],
    ))
    study.score()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers < 1 or (args.worker is None) != (args.lock_fd is None):
        parser.error("Expected a positive worker count and complete internal worker arguments")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    config = args.config.resolve()
    study = Study(yaml.safe_load(config.read_text(encoding="utf-8")))
    if study.neural and not torch.cuda.is_available():
        parser.error("This Mamba3 study requires CUDA")
    if args.worker is not None:
        fit_target(study, args.worker, args.lock_fd)
        return
    study.runs.mkdir(parents=True, exist_ok=True)
    study.reports.mkdir(parents=True, exist_ok=True)
    with (study.runs / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_parallel(study, config, min(args.workers, len(study.targets)), lock.fileno())


if __name__ == "__main__":
    main()
