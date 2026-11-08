"""Validation-only sensitivity checks against the sealed expanded-region baselines."""
import argparse
import copy
import fcntl
import math
from pathlib import Path
import subprocess
import sys
import time

import torch

BASE_RUNS = Path("/root/autodl-fs/em_eqf_training/expanded_multiregion_20261106/runs")
FROZEN = (BASE_RUNS / "frozen_project").resolve()
sys.path.insert(0, str(FROZEN / "scripts/run"))
from run_expanded_multiregion import Study, digest, file_sha256, read_json, save_json

VALIDATION_EPS = .002
NS_PER_DAY = 86_400_000_000_000


def setup():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    sealed = read_json(BASE_RUNS / "plan.json")
    study = Study(sealed["config"])
    # The frozen checkout moves the path placeholder used by model_settings.
    study.backbone["catalog_cfg"]["root_dir"] = sealed["backbone"]["catalog_cfg"]["root_dir"]
    study.plan_id = digest(study.plan)
    if sealed["plan_id"] != study.plan_id:
        raise ValueError("Original run plan no longer matches")
    selected = read_json(study.runs / "training.json")["selected"]
    baseline = {(r["target"], r["mode"], r["seed"]): study.result(r["label"]) for r in selected}
    return study, baseline


def design(study, baseline):
    tasks = []
    for target in study.targets:
        first = baseline[target, "local", 0]
        for edge, rate in ((max(study.rates), .001), (min(study.rates), .00001)):
            if first["lr"] == edge:
                tasks.append(dict(target=target, variant="local_lr_edge", seed=0, lr=rate, updates=800))
        terminal = [seed for seed in study.seeds if baseline[target, "local", seed]["best_step"] == 800]
        if terminal:
            seed = terminal[0]
            tasks.append(dict(target=target, variant="local_budget", seed=seed,
                              lr=baseline[target, "local", seed]["lr"], updates=1600))
        etas = baseline[target, "etas", None]
        curve = {r["step"]: r["validation"]["macro"]["total"] for r in etas["history"]}
        if etas["best_step"] == 500 and curve[400] - curve[500] > VALIDATION_EPS:
            tasks.append(dict(target=target, variant="etas_budget", seed=None, updates=1500))
        if etas["best_step"] <= 100 or target == "JMA_WestHonshu":
            tasks.append(dict(target=target, variant="etas_init", seed=None, updates=500))
    for task in tasks:
        task["key"] = f"{task['target']}__{task['variant']}"
        mode = "local" if task["variant"].startswith("local") else "etas"
        original = baseline[task["target"], mode, task["seed"]]
        task["baseline"] = {k: original[k] for k in ("label", "best_step", "val_total", "checkpoint_sha256")}
    return tasks


def fit_task(study, data, path, task):
    target = task["target"]
    study = copy.copy(study)
    study.options = dict(study.options)
    study.etas_config = dict(study.etas_config)
    study.runs = path / "runs" / task["key"]
    study.runs.mkdir(parents=True, exist_ok=True)
    if task["variant"].startswith("local"):
        study.options["target_steps"] = task["updates"]
    else:
        study.etas_epochs = task["updates"]
        if task["variant"] == "etas_init":
            region = data["regions"][target]
            physical_ns = region["physical_ns"]
            # Initialize the background at 20% of the observed scored training rate.
            cal, end = data["calibration_events"], region["train_end"]
            days = (physical_ns[end - 1] - physical_ns[cal - 1]) / NS_PER_DAY
            study.etas_config["base_rate_init"] = float(.2 * (end - cal) / days)
    study.plan_id = digest(dict(parent=study.plan_id, task=task, actual_etas_config=study.etas_config,
                                runner_sha256=file_sha256(__file__)))
    print("AUDIT_START", task, flush=True)
    if task["variant"].startswith("local"):
        result = study.neural_fit(data, target, "local", task["seed"], task["lr"])
    else:
        result = study.etas_fit(data, target)
    row = dict(task, diagnostic_validation=result["val_total"], diagnostic_best_step=result["best_step"],
               gain=task["baseline"]["val_total"] - result["val_total"], test_read=False)
    if task["variant"] == "etas_init":
        row["base_rate_init"] = study.etas_config["base_rate_init"]
    save_json(path / "completed" / f"{task['key']}.json", row)
    print("AUDIT_FINISH", row, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--lane", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this baseline check")
    path = args.output.resolve()
    study, baseline = setup()
    tasks = design(study, baseline)
    if args.lane is not None:
        plan = read_json(path / "plan.json")
        assert plan["tasks"] == tasks
        data = study.load("development.pt")
        remaining = [task for task in tasks if task["key"] not in plan["reused"]]
        for task in remaining[args.lane::4]:
            fit_task(study, data, path, task)
        return
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reused = {}
        if args.reuse:
            previous = read_json(args.reuse / "plan.json")
            assert previous["parent_plan"] == study.plan_id and previous["tasks"] == tasks
            for task in tasks:
                key = task["key"]
                saved = args.reuse / "completed" / f"{key}.json"
                if saved.exists():
                    row = read_json(saved)
                    assert all(row[k] == v for k, v in task.items()) and not row["test_read"]
                    result_path, = (args.reuse / "runs" / key).glob("*/result.json")
                    result = read_json(result_path)
                    assert row["diagnostic_validation"] == result["val_total"]
                    assert file_sha256(result_path.parent / "best_model.pth") == result["checkpoint_sha256"]
                    reused[key] = dict(row, reused_from=str(args.reuse), script_sha256=previous["script_sha256"])
        plan = dict(parent_plan=study.plan_id, script_sha256=file_sha256(__file__),
                    test_read=False, tasks=tasks, reused=reused)
        if (path / "plan.json").exists():
            assert read_json(path / "plan.json") == plan
        save_json(path / "plan.json", plan)
        children = []
        try:
            for lane in range(4):
                with (path / f"lane_{lane}.log").open("ab") as log:
                    child = subprocess.Popen([sys.executable, "-u", __file__, "--output", str(path),
                                              "--lane", str(lane)], stdout=log, stderr=subprocess.STDOUT)
                    children.append(child)
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError("Baseline audit worker failed")
                time.sleep(2)
            assert all(child.returncode == 0 for child in children)
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                child.wait()
        rows = [reused[task["key"]] if task["key"] in reused
                else read_json(path / "completed" / f"{task['key']}.json") for task in tasks]
        assert all(math.isfinite(r["gain"]) for r in rows)
        save_json(path / "summary.json", dict(plan, passed=True, rows=rows))
        print("BASELINE_VALIDATION_AUDIT_COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    main()
