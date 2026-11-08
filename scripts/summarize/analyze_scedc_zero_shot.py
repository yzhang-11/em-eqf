"""Replay saved SCEDC checkpoints and diagnose time and magnitude calibration."""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
import torch

STORE = Path("/root/autodl-fs/em_eqf_training/mamba3_decay_transfer_20261105")
NS_PER_DAY = 86_400_000_000_000
SECONDS_PER_DAY = 86_400


def load_study():
    spec = importlib.util.spec_from_file_location("default_transfer", STORE / "frozen_runner.py")
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    study.configure()
    study.setup()
    return study


@torch.no_grad()
def extract(study, model, data, split):
    from run_multidomain_pretraining import make_batch

    rows = []
    model.eval()
    windows = study.c.evaluation_pools(data, split, 1024)["scedc"]
    for left in range(0, len(windows), 8):
        batch = make_batch(windows[left:left + 8])
        mask = batch.nll_event_mask.bool()
        context = model.get_context(batch)
        time = model.get_inter_time_dist(context)
        magnitude = model.get_magnitude_dist(context)
        rows.append({name: value[mask].cpu().numpy() for name, value in dict(
            idx=batch.event_index,
            gap=batch.inter_times,
            mag=batch.mag,
            log_rate=time.component_distribution.scale.log(),
            shape=time.component_distribution.shape,
            log_weight=time.mixture_distribution.logits,
            b=magnitude.b,
        ).items()})
    data = {name: np.concatenate([row[name] for row in rows], axis=0) for name in rows[0]}
    assert np.all(np.diff(data["idx"]) == 1)
    return data


def time_nll(data, log_scale=0.):
    x = np.log(np.maximum(data["gap"] * math.exp(-log_scale), 1e-10))[:, None]
    shape = data["shape"]
    power = data["log_rate"] + shape * x
    log_pdf = np.log(shape) + power - x - np.exp(np.minimum(power, 700))
    return -logsumexp(data["log_weight"] + log_pdf, axis=-1) + log_scale


def pit(data):
    x = np.log(np.maximum(data["gap"], 1e-10))[:, None]
    power = data["log_rate"] + data["shape"] * x
    log_survival = logsumexp(data["log_weight"] - np.exp(np.minimum(power, 700)), axis=-1)
    return np.clip(-np.expm1(log_survival), 0, 1)


def mark_nll(data, mc, b):
    beta = b * math.log(10)
    return (beta * (data["mag"] - mc) - np.log(-np.expm1(-.1 * beta))
            + np.log(-np.expm1(-(10 - mc) * beta)))


def describe(spec, first, stop):
    gaps = np.diff(spec["ns"][first - 1:stop]) / NS_PER_DAY
    mag = spec["magnitude"][first:stop]
    mc = spec["metadata"][0]
    return dict(events=len(mag), mc=mc, calib_b=spec["metadata"][1],
                calib_tau_days=spec["metadata"][2], median_gap_seconds=float(np.median(gaps) * SECONDS_PER_DAY),
                under_minute=float(np.mean(gaps * SECONDS_PER_DAY < 60)),
                under_hour=float(np.mean(gaps * SECONDS_PER_DAY < 3600)),
                empirical_discrete_b=float(np.log1p(1 / np.mean((mag - mc) / .1)) / (.1 * math.log(10))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study = load_study()
    data = torch.load(study.DATA, map_location="cpu", weights_only=False)
    target = data["regions"]["scedc"]
    train = describe(target, 256, target["train_end"])
    validation = describe(target, target["train_end"], target["val_end"])
    test = describe(target, target["val_end"], target["test_end"])
    source_data = torch.load(study.control.source_data("scedc"), map_location="cpu", weights_only=False)
    sources = {name: describe(spec, 256, spec["train_end"]) for name, spec in source_data["regions"].items()}
    mc = target["metadata"][0]
    models, loss = [], {}
    for mode, budget in [("local", 800), ("mixed_zero", 6400), ("fine", 800)]:
        for seed in study.SEEDS:
            saved = study.read(study.REPORTS / f"neural/scedc_{mode}_steps_{budget}_seed_{seed}.json")
            checkpoint = Path(saved["checkpoint"])
            assert study.c.sha(checkpoint) == saved["checkpoint_sha256"]
            model, _ = study.build(study.FAMILY, seed, data["vocabulary"])
            model.load_state_dict(torch.load(checkpoint, map_location="cuda:0", weights_only=False)["model_state_dict"], strict=True)
            val = extract(study, model, data, "val")
            scored = extract(study, model, data, "test")
            assert len(scored["idx"]) == test["events"] and scored["idx"][0] == target["val_end"]
            original_time = time_nll(scored)
            original_mag = mark_nll(scored, mc, scored["b"])
            time_error = abs(original_time.mean() - saved["test"]["time"])
            mag_error = abs(original_mag.mean() - saved["test"]["mag"])
            assert max(time_error, mag_error) < 2e-4, (mode, seed, time_error, mag_error)
            rank = pit(scored)
            row = dict(mode=mode, seed=seed, time=float(original_time.mean()), mag=float(original_mag.mean()),
                       pit_mean=float(rank.mean()), pit_below_05=float(np.mean(rank < .05)),
                       pit_above_95=float(np.mean(rank > .95)), median_b=float(np.median(scored["b"])),
                       avg_b=float(scored["b"].mean()),
                       fixed_train_b_test_mag=float(mark_nll(scored, mc, train["empirical_discrete_b"]).mean()),
                       per_event_replay_error=dict(time=float(time_error), mag=float(mag_error)))
            if mode == "mixed_zero":
                # Scale only the predicted interval distribution and GR b. Fit both
                # scalars on target validation, and evaluate each once on test.
                t_fit = minimize_scalar(lambda x: time_nll(val, x).mean(), bounds=(-math.log(4), math.log(4)), method="bounded")
                b_fit = minimize_scalar(lambda x: mark_nll(val, mc, val["b"] * math.exp(x)).mean(),
                                        bounds=(-math.log(4), math.log(4)), method="bounded")
                assert t_fit.success and b_fit.success
                row.update(validation_time_multiplier=math.exp(t_fit.x), validation_b_multiplier=math.exp(b_fit.x),
                           calibrated_test_time=float(time_nll(scored, t_fit.x).mean()),
                           calibrated_test_mag=float(mark_nll(scored, mc, scored["b"] * math.exp(b_fit.x)).mean()))
            loss[mode, seed] = np.stack([original_time, original_mag], axis=1)
            models.append(row)
            print("SCEDC_REPLAY", mode, seed, row, flush=True)
            del model
            torch.cuda.empty_cache()
    delta = np.mean([loss["mixed_zero", seed] - loss["local", seed] for seed in study.SEEDS], axis=0)

    def group(label, mask):
        return dict(group=label, events=int(mask.sum()), proportion=float(mask.mean()),
                    mean_zero_minus_local=delta[mask].mean(axis=0).tolist(),
                    contribution_per_all_events=(delta[mask].sum(axis=0) / len(delta)).tolist())

    seconds = scored["gap"] * SECONDS_PER_DAY
    gap_groups = [group(label, (seconds >= low) & (seconds < high)) for label, low, high in [
        ("<1 min", 0, 60), ("1 min to <1 h", 60, 3600), ("1 h to <24 h", 3600, 86400),
        (">=24 h", 86400, np.inf),
    ]]
    months = pd.to_datetime(target["physical_ns"][scored["idx"]]).strftime("%Y-%m")
    month_groups = [group(month, months == month) for month in sorted(set(months))]
    result = dict(train=train, validation=validation, test=test, sources=sources, models=models,
                  gap_groups=gap_groups, month_groups=month_groups, time_and_mag_order=["time", "mag"],
                  calibration_protocol="Two scalars fit on validation only, model weights fixed. Historical test is used for diagnosis.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("SCEDC_DIAGNOSIS_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
