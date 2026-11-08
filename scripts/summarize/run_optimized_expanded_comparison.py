"""Expand validation budgets for local, transfer and ETAS, then score the locked selections."""
import argparse
import copy
import fcntl
from pathlib import Path
import subprocess
import sys
import time

import torch

from audit_expanded_baseline_optimization import BASE_RUNS, NS_PER_DAY, setup
from run_expanded_multiregion import ScoredETAS, build_model, digest, file_sha256, read_json, save_json

STORE = BASE_RUNS.parent
REPORTS = Path("/root/autodl-tmp/em_eqf/experiments/revision/multiregion_pretraining/reports/expanded_20261106")
KEYS = ("time", "mag", "total")
RULES = dict(lr_extensions=[1e-5, 1e-3], target_budgets=[800, 1600, 3200, 4800],
             etas_budgets=[500, 1500, 3000, 6000], late_gain=.002,
             neural_selection="Seed 0. Extend the LR grid at a winning edge; test 1600 at the best 800-step LR. "
             "If a winning longer budget is still improving near its end, try the next budget. "
             "Use the selected LR and budget for seeds 1 and 2.",
             etas_selection="Compare original and 20% of training rate initialization at 500 epochs. "
             "Include completed validation audits. Extend the best eligible candidate if its best point "
             "is near the end; after 1500 also require late improvement.",
             test_selection="All targets and methods locked by validation before opening the formal test file.")


def candidate_key(record):
    if record["mode"] == "etas":
        return record["target"], "etas", None, record["config"]["base_rate_init"], record["updates"]
    return record["target"], record["mode"], record["seed"], record["lr"], record["updates"]


class Comparison:
    def __init__(self, output):
        self.study, self.original = setup()
        self.output = output.resolve()
        self.design = dict(parent_plan=self.study.plan_id, rules=RULES, script_sha256=file_sha256(__file__))
        self.study_id = digest(self.design)
        self.candidates = {}
        manifest = read_json(BASE_RUNS / "training.json")
        for label in manifest["completed"]:
            self.add_candidate(BASE_RUNS / label / "result.json")
        audit = read_json(STORE / "optimization_audit_frozen/summary.json")
        assert audit["passed"] and audit["parent_plan"] == self.study.plan_id and not audit["test_read"]
        for row in audit["rows"]:
            location = Path(row.get("reused_from", STORE / "optimization_audit_frozen"))
            path, = (location / "runs" / row["key"]).glob("*/result.json")
            record = read_json(path)
            assert record["val_total"] == row["diagnostic_validation"] and record["best_step"] == row["diagnostic_best_step"]
            self.add_candidate(path)

    def add_candidate(self, path):
        record = read_json(path)
        key = candidate_key(record)
        current = self.candidates.get(key)
        if current is None or record["val_total"] < current[0]["val_total"]:
            self.candidates[key] = record, path.parent

    def get(self, data, target, mode, seed, rate, updates):
        key = target, mode, seed, rate, updates
        if key not in self.candidates:
            study = copy.copy(self.study)
            study.options = dict(study.options, target_steps=updates)
            study.etas_epochs = updates if mode == "etas" else study.etas_epochs
            study.etas_config = dict(study.etas_config)
            tag = f"{target}__{mode}__s{seed}__r{rate:g}__steps{updates}"
            study.runs = self.output / "runs" / tag
            study.runs.mkdir(parents=True, exist_ok=True)
            study.plan_id = digest(dict(study_id=self.study_id, task=key))
            if mode == "etas":
                study.etas_config["base_rate_init"] = rate
                record = study.etas_fit(data, target)
            else:
                parent = None
                if mode == "fine":
                    parent = self.original[target, "zero", seed]
                    origin = BASE_RUNS / parent["label"]
                    link = study.runs / parent["label"]
                    if not link.exists():
                        link.symlink_to(origin.resolve(), target_is_directory=True)
                    assert link.resolve() == origin.resolve()
                    assert file_sha256(origin / "best_model.pth") == parent["checkpoint_sha256"]
                record = study.neural_fit(data, target, mode, seed, rate, parent)
            self.add_candidate(study.runs / record["label"] / "result.json")
        record, folder = self.candidates[key]
        checkpoint = folder / "best_model.pth"
        assert file_sha256(checkpoint) == record["checkpoint_sha256"]
        return dict(record, checkpoint=str(checkpoint))

    def fit_target(self, target):
        data = self.study.load("development.pt")
        all_trials = []
        selected = []
        for mode in ("local", "fine"):
            rates = sorted(self.study.rates)
            trials = [self.get(data, target, mode, 0, rate, 800) for rate in rates]
            winner = min(trials, key=lambda r: r["val_total"])
            if winner["lr"] in (rates[0], rates[-1]):
                rate = RULES["lr_extensions"][int(winner["lr"] == rates[-1])]
                trials.append(self.get(data, target, mode, 0, rate, 800))
                winner = min(trials, key=lambda r: r["val_total"])
            for budget in RULES["target_budgets"][1:]:
                trial = self.get(data, target, mode, 0, winner["lr"], budget)
                trials.append(trial)
                winner = min(trials, key=lambda r: r["val_total"])
                if winner["updates"] != budget or not late_improvement(winner, 100):
                    break
            all_trials.extend(trials)
            for seed in self.study.seeds:
                selected.append(winner if seed == 0 else self.get(
                    data, target, mode, seed, winner["lr"], winner["updates"]))
        region = data["regions"][target]
        cal, end = data["calibration_events"], region["train_end"]
        days = (region["physical_ns"][end - 1] - region["physical_ns"][cal - 1]) / NS_PER_DAY
        rate = float(.2 * (end - cal) / days)
        initial = self.study.etas_config["base_rate_init"]
        trials = [self.get(data, target, "etas", None, value, 500) for value in (initial, rate)]
        for key in sorted(self.candidates, key=str):
            if key[0] == target and key[1] == "etas" and key[-1] > 500:
                trials.append(self.get(data, *key))
        winner = min(trials, key=lambda r: r["val_total"])
        for budget in RULES["etas_budgets"][1:]:
            if winner["updates"] >= budget:
                continue
            if winner["updates"] == 500:
                if winner["best_step"] < 490:
                    break
            elif not late_improvement(winner, 10):
                break
            trials.append(self.get(data, target, "etas", None, winner["config"]["base_rate_init"], budget))
            previous = winner
            winner = min(trials, key=lambda r: r["val_total"])
            if winner is previous:
                break
        all_trials.extend(trials)
        selected.append(winner)
        for seed in self.study.seeds:
            local = next(r for r in selected if r["mode"] == "local" and r["seed"] == seed)
            fine = next(r for r in selected if r["mode"] == "fine" and r["seed"] == seed)
            source = self.original[target, "zero", seed]
            assert local["initial_sha256"] == fine["initial_sha256"] == source["initial_sha256"]
            assert fine["parent_sha256"] == source["checkpoint_sha256"]
            if local["updates"] == fine["updates"]:
                assert local["draw_sha256"] == fine["draw_sha256"]
        save_json(self.output / "selections" / f"{target}.json", dict(
            study_id=self.study_id, target=target, selected=selected, validation_trials=all_trials, test_read=False))
        print("TARGET_SELECTION_COMPLETE", target, [(r["mode"],r["seed"],r["updates"],r["val_total"]) for r in selected], flush=True)

    def score(self):
        selections = [read_json(self.output / "selections" / f"{target}.json") for target in self.study.targets]
        assert all(x["study_id"] == self.study_id and not x["test_read"] and len(x["selected"]) == 7 for x in selections)
        data = self.study.load("test.pt")
        original = read_json(REPORTS / "scores.json")
        assert original["plan_id"] == self.study.plan_id
        original_scores = {(r["target"],r["mode"],r["seed"]):r for r in original["checkpoints"]}
        rows = []
        for x in selections:
            target = x["target"]
            for record in x["selected"]:
                seed, mode = record["seed"], record["mode"]
                ref = original_scores[target, mode, seed]
                checkpoint = Path(record["checkpoint"])
                assert file_sha256(checkpoint) == record["checkpoint_sha256"]
                if record["checkpoint_sha256"] == ref["checkpoint_sha256"]:
                    value = {key: ref[key] for key in KEYS}
                    windows = ref["windows"]
                else:
                    model = (ScoredETAS(record["config"], self.study.device) if mode == "etas" else
                             build_model("stats_film", seed, data["vocabulary"], self.study.backbone, self.study.device))
                    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
                    region = data["regions"][target]
                    metrics, per_window = self.study.evaluate(model, data, {target: (region["val_end"], region["test_end"])})
                    value = metrics["macro"]
                    windows = per_window[target]
                assert [(w["first"], w["events"]) for w in windows] == [(w["first"], w["events"]) for w in ref["windows"]]
                rows.append(dict(target=target, mode=mode, seed=seed, updates=record["updates"],
                                 best_step=record["best_step"], lr=record["lr"], events=ref["events"],
                                 val_total=record["val_total"], checkpoint=str(checkpoint),
                                 checkpoint_sha256=record["checkpoint_sha256"], **value, windows=windows))
        assert len(rows) == len(self.study.targets) * 7
        save_json(self.output / "scores.json", dict(study_id=self.study_id, parent_plan=self.study.plan_id, rows=rows))
        print("OPTIMIZED_SCORE_COMPLETE", len(rows), flush=True)


def late_improvement(record, frequency):
    budget = record["updates"]
    if record["best_step"] < budget * .9:
        return False
    previous = budget * 3 // 4 // frequency * frequency
    point = next(x["validation"]["macro"]["total"] for x in record["history"] if x["step"] == previous)
    return point - record["val_total"] > RULES["late_gain"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    comparison = Comparison(args.output)
    path = comparison.output
    if args.target:
        assert read_json(path / "plan.json")["study_id"] == comparison.study_id
        comparison.fit_target(args.target)
        return
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = dict(comparison.design, study_id=comparison.study_id)
        if (path / "plan.json").exists():
            assert read_json(path / "plan.json") == plan
        save_json(path / "plan.json", plan)
        regions = comparison.study.manifest["regions"]
        pending = iter(sorted(comparison.study.targets, key=lambda t: regions[t]["counts"]["train"], reverse=True))
        active = {}
        try:
            while True:
                while len(active) < 4:
                    target = next(pending, None)
                    if target is None:
                        break
                    if (path / "selections" / f"{target}.json").exists():
                        continue
                    with (path / f"{target}.log").open("ab") as log:
                        child = subprocess.Popen([sys.executable, "-u", __file__, "--output", str(path),
                                                  "--target", target], stdout=log, stderr=subprocess.STDOUT)
                    active[target] = child
                    print("OPTIMIZED_TARGET_START", target, child.pid, flush=True)
                if not active:
                    break
                for target, child in list(active.items()):
                    status = child.poll()
                    if status not in (None, 0):
                        raise RuntimeError(f"{target} failed ({status}); see {target}.log")
                    if status == 0:
                        del active[target]
                time.sleep(2)
        finally:
            for child in active.values():
                if child.poll() is None:
                    child.terminate()
            for child in active.values():
                child.wait()
        comparison.score()


if __name__ == "__main__":
    main()
