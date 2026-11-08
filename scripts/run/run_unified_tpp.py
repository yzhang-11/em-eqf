"""Fit one single-region TPP model, or score its frozen fit on the test period."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
from omegaconf import OmegaConf
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/run")]
from scripts.run.run_expanded_multiregion import ScoredETAS, build_model, prepare_model_config
from scripts.data.prepare_expanded_multiregion import discrete_b, file_sha256
from src.data.unified_tpp import NS_PER_DAY, calendar_batch, load_catalogs
from src.models.tpp.nmrp import NMRP
from src.models.tpp.recurrent.model_v1 import RecurrentTPP
from src.train.scheduler import CosineWithWarmupFloor

MODELS = ("mamba3", "etas", "recast", "i-nmrp", "j-nmrp")
PROFILE = ROOT / "config/mixer_tpp_mamba3_event_time.yaml"
ETAS_CONFIG = ROOT / "experiments/revision/generation_baselines/configs/etas_scsn_main_seed_0.yaml"
RECAST_CONFIG = (ROOT / "experiments/revision/multiregion_pretraining/configs"
                 / "recast_single_region_tpp_20261105/original_recast_template.yaml")


class SharedClockRECAST(RecurrentTPP):
    def _run_rnn(self, features, hidden_state=None):
        return super()._run_rnn(features.to(self.rnn.weight_ih_l0), hidden_state)


def make_model(spec, data, region, device):
    torch.manual_seed(spec["seed"])
    name = spec["model"]
    lo, hi = region["train_start"], region["train_end"]
    mc, _, tau = region["metadata"][:3]
    gaps = np.diff(region["ns"][lo - 1:hi]) / NS_PER_DAY
    marks = region["magnitude"][lo:hi]
    if name == "mamba3":
        backbone = prepare_model_config(PROFILE, 64, 2)
        return build_model("stats_film", spec["seed"], data["vocabulary"], backbone, device)
    if name == "etas":
        cfg = yaml.safe_load(ETAS_CONFIG.read_text())
        cfg.update(mag_completeness=mc, richter_b_mle=discrete_b(marks, mc),
                   etas_grad_checkpoint=True, etas_query_chunk_size=128, etas_history_chunk_size=2048)
        return ScoredETAS(cfg, device).native
    if name == "recast":
        cfg = OmegaConf.load(RECAST_CONFIG)
        cfg.update(d_model=32, num_rnn_layers=1, rnn_type="GRU", rnn_dropout=.3, num_components=32,
                   Mc=mc, mag_completeness=mc, richter_b_mle=discrete_b(marks, mc), bg_model=None,
                   rnn_chunk_len=4096, scale_range="positive", loss_reduction="per_event",
                   tau_mean=tau, mag_mean=float(region["magnitude"][:region["calibration_end"]].mean()),
                   time_max=float(gaps.max()), use_double_precision=False)
        return SharedClockRECAST(cfg, device=device).to(device)
    return NMRP(gaps, marks, joint=name == "j-nmrp",
                history=spec.get("nmrp_history", spec["history"])).to(device)


def intervals(region, start, stop, block_size):
    ns = region["ns"]
    lo, hi = map(int, np.searchsorted(ns, [start, stop]))
    # Cut immediately after the previous target, before the next is observed.
    cuts = [start, *(int(ns[index - 1]) + 1 for index in range(lo + block_size, hi, block_size)), stop]
    return [(left, right, min(block_size, hi - lo - i * block_size))
            for i, (left, right) in enumerate(zip(cuts, cuts[1:]))]


def losses(model, name, batch, reduction):
    if name == "mamba3":
        return model.nll_loss(batch, reduction=reduction)
    else:
        return model.nll_loss(batch, reduction=reduction, return_dict=True)


@torch.no_grad()
def evaluate(model, spec, region, region_id, start, end, device):
    model.eval()
    batch = calendar_batch(region, start, end, region_id=region_id).to(device)
    lo, hi = map(int, np.searchsorted(region["ns"], [start, end]))
    count = int(batch.nll_event_mask.sum())
    if count == 0 or count != hi - lo or int(batch.end_idx[0]) != hi:
        raise ValueError("Full-sequence score mask disagrees with calendar event count")
    value = losses(model, spec["model"], batch, "sum")["time"]
    if not torch.isfinite(value):
        raise ValueError("Nonfinite time NLL")
    return {"time_nll_per_event": float(value) / count, "events": count, "history_events": lo,
            "start_ns": start, "end_ns": end, "protocol": "full_sequence_calendar"}


def fit(model, spec, region, region_id, folder, device, *, draw_function=None, evaluation_function=None):
    dates = region["time_bounds_ns"]
    if draw_function is None:
        train = intervals(region, dates["train_start"], dates["train_end"], spec["train_events"])
        rng = random.Random(spec["seed"])
        draws = rng.choices(train, weights=[item[2] for item in train], k=spec["steps"])
    else:
        train, draws = draw_function(region, spec)
    history = spec["history"] if len(train) > 1 else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=.01)
    scheduler = CosineWithWarmupFloor(optimizer, spec["steps"] // 10, spec["steps"], .05 * spec["lr"])
    best = float("inf")
    for step, (left, right, _) in enumerate(draws, 1):
        batch = calendar_batch(region, left, right, history=history, region_id=region_id).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        objective = losses(model, spec["model"], batch, "per_event")["time"].mean()
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(objective) or not torch.isfinite(norm):
            raise ValueError(f"Nonfinite fit at step {step}")
        optimizer.step()
        if isinstance(model, NMRP):
            model.project_weights()
        scheduler.step()
        if step % spec["validate_every"] == 0 or step == spec["steps"]:
            val = (evaluation_function or evaluate)(model, spec, region, region_id, dates["train_end"], dates["val_end"], device)
            print(json.dumps(dict(step=step, **val)), flush=True)
            if val["time_nll_per_event"] < best:
                best = val["time_nll_per_event"]
                pending = folder / "best.tmp"
                torch.save(dict(spec=spec, state=model.state_dict(), validation=val, step=step), pending)
                pending.replace(folder / "best.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "validate", "test"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--train-events", type=int, default=2048, help="Maximum target count per training block")
    parser.add_argument("--history", type=int, help="Number of preceding input events per training block")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    manifest = args.data / "dataset.json"
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    audited = json.loads(manifest.read_text())["artifacts"]
    if file_sha256(args.data / "development.pt") != audited["development.pt"]:
        raise ValueError("Development artifact differs from the audited catalog")
    development = load_catalogs(args.data)
    if args.phase == "fit":
        if not args.model or args.target not in development["regions"]:
            parser.error("fit requires a model and an available target")
        if min(args.steps, args.validate_every, args.train_events) < 1 or args.lr <= 0:
            parser.error("fit budgets and learning rate must be positive")
        history = args.history if args.history is not None or args.model == "etas" else 1024
        if history is not None and history < 1:
            parser.error("history must be positive")
        if args.model in ("i-nmrp", "j-nmrp") and history < 50:
            parser.error("I/J-NMRP requires at least 50 history events")
        spec = dict(model=args.model, target=args.target, seed=args.seed, lr=args.lr,
                    steps=args.steps, validate_every=args.validate_every, history=history,
                    train_events=args.train_events, manifest=manifest_hash, objective="time_nll")
        if args.model in ("i-nmrp", "j-nmrp"):
            spec["nmrp_history"] = 50
        args.output.mkdir(parents=True, exist_ok=False)
    else:
        if not (args.output / "complete.json").is_file():
            raise ValueError("Fit must be complete before scoring")
        best_path = args.output / "best.pt"
        complete = json.loads((args.output / "complete.json").read_text())
        if complete["checkpoint_sha256"] != file_sha256(best_path):
            raise ValueError("Checkpoint differs from completed fit")
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
        spec = checkpoint["spec"]
        if spec["manifest"] != manifest_hash:
            raise ValueError("Catalog differs from training")
    target = spec["target"]
    region = development["regions"][target]
    region_id = development["vocabulary"][target]
    model = make_model(spec, development, region, device)
    if args.phase == "fit":
        fit(model, spec, region, region_id, args.output, device)
        complete = {"spec": spec, "checkpoint_sha256": file_sha256(args.output / "best.pt")}
        (args.output / "complete.json").write_text(json.dumps(complete, indent=2) + "\n")
    elif args.phase == "validate":
        model.load_state_dict(checkpoint["state"])
        dates = region["time_bounds_ns"]
        value = evaluate(model, spec, region, region_id, dates["train_end"], dates["val_end"], device)
        (args.output / "validation_full_sequence.json").write_text(json.dumps(value, indent=2) + "\n")
        print(json.dumps(value))
    else:
        model.load_state_dict(checkpoint["state"])
        if file_sha256(args.data / "test.pt") != audited["test.pt"]:
            raise ValueError("Test artifact differs from the audited catalog")
        test = load_catalogs(args.data, test=True)["regions"][target]
        dates = test["time_bounds_ns"]
        value = evaluate(model, spec, test, region_id, dates["val_end"], dates["test_end"], device)
        (args.output / "test.json").write_text(json.dumps(value, indent=2) + "\n")
        print(json.dumps(value))


if __name__ == "__main__":
    main()
