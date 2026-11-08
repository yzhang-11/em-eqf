"""Paired source pretraining with fixed or mixed history, evaluated on development only."""
import argparse
import math
from pathlib import Path

import torch
import yaml

from run_expanded_multiregion import Study, build_model, calendar_bound, file_sha256, make_batch, read_json, save_json


@torch.no_grad()
def history_curve(study, model, data, target):
    region = data["regions"][target]
    lo, hi = region["train_end"], region["val_end"]
    curve = {}
    for context in (1024, 4096, 8192, None):
        batch = make_batch(data, target, [(lo, hi - lo)], context, study.device)
        time = float(model.nll_loss(batch, reduction="per_event")["time"].item())
        if not math.isfinite(time):
            raise ValueError(f"Nonfinite target validation likelihood: {target}")
        curve[str(context) if context is not None else "full"] = time
    return dict(events=hi-lo, prehistory=lo, start_ns=calendar_bound(region, lo),
                end_ns=calendar_bound(region, hi), time_nll=curve)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--arm", choices=("fixed", "mixed"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["ablation_runner_sha256"] = file_sha256(__file__)
    output = args.output.resolve() / args.target / args.arm
    config["paths"].update(runs=str(output), reports=str(output))
    config["training"].update(
        targets=[args.target], seeds=[0], source_steps_per_region=300,
        source_validate_every=550, etas_epochs=0,
        history_choices=[1024, 4096, 8192] if args.arm == "mixed" else [1024],
    )
    study = Study(config)
    output.mkdir(parents=True, exist_ok=True)
    plan = output / "plan.json"
    if plan.exists():
        previous = read_json(plan)
        if previous["plan_id"] != study.plan_id:
            raise ValueError("Warmup ablation configuration changed; use a new output path")
    save_json(plan, dict(study.plan, plan_id=study.plan_id))
    data = study.load("development.pt")
    fitted = study.neural_fit(data, args.target, "source", 0, study.options["source_lr"])
    model = build_model("stats_film", 0, data["vocabulary"], study.backbone, study.device).eval()
    study.load_weights(model, fitted)
    report = dict(target=args.target, arm=args.arm, fit=fitted,
                  heldout_validation=history_curve(study, model, data, args.target))
    save_json(output / "comparison.json", report)
    print(args.target, args.arm, report["heldout_validation"], flush=True)


if __name__ == "__main__":
    main()
