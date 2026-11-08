"""Run paired regional TPP transfer and native ETAS on a prepared catalog cohort."""
import argparse
from collections import Counter
import csv
import fcntl
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import random
import shutil
import sys

import numpy as np
from omegaconf import OmegaConf
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "scripts/data", ROOT / "scripts/tuning"):
    sys.path.insert(0, str(directory))
from prepare_expanded_multiregion import discrete_b, file_sha256, read_json
from mamba3_transfer_common import sequence, state_hash
from run_multidomain_pretraining import build_model, model_settings
from run_multiregion_pretraining import save_json
from src.data import Batch
from src.data.unified_tpp import calendar_batch
from src.train.scheduler import CosineWithWarmupFloor

METRICS = ("time", "mag", "total")
BASELINES = ("mamba3", "recast", "rtpp-v2", "transformer-t-xpos", "mamba2-t-xpos")
MAMBA2_PROFILE = ROOT / "config/mixer_tpp_mamba2_rotary.yaml"
RECAST_PROFILE = (ROOT / "experiments/revision/multiregion_pretraining/configs"
                  / "recast_single_region_tpp_20261105/original_recast_template.yaml")
RTPPV2_PROFILE = ROOT / "config/rtpp_v2.yaml"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def spans(low, high, size):
    return [(first, min(size, high - first)) for first in range(low, high, size)]


def calendar_bound(region, index):
    for key in ("train_start", "source_train_end", "train_end", "val_end", "test_end"):
        if region.get(key) == index:
            return region["time_bounds_ns"][key]
    return int(region["ns"][index - 1]) + 1


def make_batch(data, region, windows, context, device):
    spec = data["regions"][region]
    if "time_bounds_ns" in spec:
        (first, count), = windows
        batch = calendar_batch(spec, calendar_bound(spec, first), calendar_bound(spec, first + count),
                               history=context, region_id=data["vocabulary"][region])
        indices = batch.event_index[batch.nll_event_mask.bool()]
        if not torch.equal(indices, torch.arange(first, first + count)):
            raise ValueError(f"Incorrect calendar target mask for {region} at event {first}")
        return batch.to(device)
    items = [sequence(spec, data["vocabulary"][region], first, count, context)
             for first, count in windows]
    batch = Batch.from_list(items)
    for row, (first, count) in enumerate(windows):
        indices = batch.event_index[row][batch.nll_event_mask[row].bool()]
        if not torch.equal(indices, torch.arange(first, first + count)):
            raise ValueError(f"Incorrect target mask for {region} at event {first}")
    if batch.inter_times.dtype != torch.float64:
        raise ValueError("Expected FP64 event clock")
    return batch.to(device)


def prepare_model_config(profile, width, layers):
    cfg = model_settings("stats_film")
    selected = OmegaConf.load(profile)
    for key in ("mixer_model_config", "provide_physical_dt", "mask_input_features", "mixer_time_parametrization"):
        cfg[key] = selected[key]
    mixer = cfg.mixer_model_config
    ssm = mixer.ssm_cfg
    if (ssm.layer != "Mamba3EventTime" or ssm.time_update_mode != "decay_only"
            or ssm.use_event_time_phase or not ssm.is_mimo or ssm.mimo_rank != 2
            or mixer.use_residual or cfg.mixer_time_parametrization != "legacy"
            or not cfg.provide_physical_dt or list(cfg.mask_input_features) != ["log_inter_times"]):
        raise ValueError("The model profile must match the project Mamba3 default")
    cfg.d_model = mixer.d_model = width
    mixer.n_layer = layers
    mixer.d_intermediate = mixer.input_proj_hidden = 2 * width
    cfg.resume_path = None
    cfg.load_specific_parts = []
    return OmegaConf.to_container(cfg, resolve=True)


def prepare_baseline_config(profile, width, layers, baseline, recurrent_width=32):
    backbone = prepare_model_config(profile, width, layers)
    if baseline == "mamba3":
        return backbone, []
    backbone.update(mask_input_features=[], provide_physical_dt=False)
    mixer = backbone["mixer_model_config"]
    if baseline == "mamba2-t-xpos":
        selected = OmegaConf.load(MAMBA2_PROFILE)
        mixer["ssm_cfg"] = OmegaConf.to_container(selected.mixer_model_config.ssm_cfg, resolve=True)
        return backbone, [MAMBA2_PROFILE]
    if baseline == "transformer-t-xpos":
        mixer["attn_layer_idx"] = list(range(layers))
        mixer["use_residual"] = True
        return backbone, []
    if type(recurrent_width) is not int or recurrent_width < 1:
        raise ValueError("Expected a positive integer recurrent_width")
    recurrent_profile = RTPPV2_PROFILE if baseline == "rtpp-v2" else RECAST_PROFILE
    recast = OmegaConf.load(recurrent_profile)
    # Region-specific input and output statistics come from the batch.
    recast.update(d_model=recurrent_width, num_rnn_layers=1, rnn_type="GRU", rnn_dropout=.3,
                  num_components=32, tau_mean=1., mag_mean=0., time_max=1.,
                  tau_min=0., tau_max=1., tau_q05=1., tau_q025=1.,
                  richter_b_mle=1., mag_completeness=0., rnn_chunk_len=4096,
                  bg_model=None, scale_range="positive", loss_reduction="per_event")
    for key in ("features_input_keys", "mask_input_features", "provide_physical_dt"):
        recast[key] = backbone[key]
    recast.extra_input_keys = []
    return OmegaConf.to_container(recast, resolve=True), [recurrent_profile]


class ScoredETAS(torch.nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        from src.models.builders import ModelBuilder

        self.native = ModelBuilder.by_name("etas")()(OmegaConf.create(cfg), device).double()
        self.native.b.fill_(cfg["richter_b_mle"])

    def nll_loss(self, batch, *, reduction):
        from src.models.tpp.region_mixer_tpp import RegionalGridGR

        time = self.native.nll_loss(batch, reduction=reduction, return_dict=True)["time"]
        mark = RegionalGridGR(self.native.b, self.native.M_c, self.native.b.new_tensor(.1))
        mask = batch.nll_event_mask.bool()
        mag = -mark.log_prob(batch.mag, mask=mask).sum(-1) / mask.sum(-1)
        return {"time": time, "mag": mag, "total": time + mag}


class Study:
    def __init__(self, config):
        self.options = config["training"]
        self.data_dir = ROOT / config["paths"]["data"]
        self.runs = ROOT / config["paths"]["runs"]
        self.reports = ROOT / config["paths"]["reports"]
        self.manifest = read_json(self.data_dir / "dataset.json")
        self.full_sequence = config.get("protocol") == "full_sequence_calendar"
        self.baseline = self.options.get("baseline", "mamba3")
        if self.baseline not in BASELINES or (self.baseline != "mamba3" and not self.full_sequence):
            raise ValueError("Expected a supported baseline and the full-sequence protocol")
        if self.full_sequence:
            if self.manifest["schema"] != "unified-tpp-v1" or self.manifest["group"] != "multiregion":
                raise ValueError("Expected the audited unified multiregion catalog")
        elif self.manifest["schema"] != "expanded-regional-tpp-v1" or self.manifest["settings"] != config["data"]:
            raise ValueError("Data manifest and study configuration differ")
        self.metrics = ("time",) if self.full_sequence else METRICS
        self.selection_metric = "time" if self.full_sequence else "total"
        self.history_choices = self.options.get("history_choices", [self.options["history"]])
        if (not self.history_choices or any(type(h) is not int or h < 1 for h in self.history_choices)
                or len(set(self.history_choices)) != len(self.history_choices)):
            raise ValueError("Expected distinct positive warmup lengths")
        self.regions = sorted(self.manifest["regions"])
        self.targets = self.regions if self.options["targets"] == "all" else self.options["targets"]
        self.seeds = self.options["seeds"]
        self.rates = self.options["target_lrs"]
        self.neural = self.options["neural"]
        self.etas_epochs = self.options["etas_epochs"]
        if not self.targets or len(set(self.targets)) != len(self.targets) or set(self.targets) - set(self.regions):
            raise ValueError("Targets must be unique prepared regions")
        if not self.neural and not self.etas_epochs:
            raise ValueError("Enable neural fitting or ETAS")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or min(self.seeds) < 0:
            raise ValueError("Expected unique nonnegative seeds")
        if not self.rates or len(set(self.rates)) != len(self.rates) or any(
                not math.isfinite(rate) or rate <= 0 for rate in [*self.rates, self.options["source_lr"]]):
            raise ValueError("Expected distinct positive learning rates")
        positive = ("history", "train_targets", "eval_targets", "eval_batch_size", "width", "layers",
                    "source_steps_per_region", "target_steps", "source_validate_every",
                    "target_validate_every", "etas_validate_every")
        if (self.etas_epochs < 0 or any(type(self.options[key]) is not int or self.options[key] < 1 for key in positive)
                or self.options["width"] % 32):
            raise ValueError("Expected positive integer window sizes and budgets, and width divisible by 32")
        self.device = torch.device("cuda:0" if self.neural or self.full_sequence else "cpu")
        profile = ROOT / self.options["model_profile"]
        etas_profile = ROOT / self.options["etas_profile"]
        self.backbone, baseline_profiles = prepare_baseline_config(
            profile, self.options["width"], self.options["layers"], self.baseline,
            recurrent_width=self.options.get("recurrent_width", 32),
        )
        if self.baseline in ("recast", "rtpp-v2"):
            self.backbone["region_condition_stats"] = self.options.get("region_condition_stats", True)
            self.backbone["region_magnitude_preprocess"] = self.options.get("region_magnitude_preprocess", "gr")
        self.etas_config = OmegaConf.to_container(OmegaConf.load(etas_profile), resolve=True)
        files = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__).resolve(), profile, etas_profile] + baseline_profiles
        files += [ROOT / name for name in (
            "scripts/data/prepare_expanded_multiregion.py",
            "scripts/tuning/mamba3_transfer_common.py",
            "scripts/run/run_multidomain_pretraining.py",
            "scripts/run/run_multiregion_pretraining.py",
            "experiments/revision/main/configs/pretrain_azdx_main_seed_0.yaml",
        )]
        if self.full_sequence:
            files.append(ROOT / "scripts/run/run_unified_tpp.py")
        self.plan = dict(config=config, dataset_manifest_sha256=file_sha256(self.data_dir / "dataset.json"),
                         source_files={str(path.relative_to(ROOT)): file_sha256(path) for path in files},
                         backbone=self.backbone, etas_template=self.etas_config,
                         targets=self.targets, source_regions={r: self.sources(r) for r in self.targets},
                         device=str(self.device), torch_version=str(torch.__version__),
                         neural_fits=len(self.targets) * (len(self.seeds) + 2 * (
                             len(self.rates) + len(self.seeds) - 1)) if self.neural else 0,
                         etas_fits=len(self.targets) if self.etas_epochs else 0)
        self.plan_id = digest(self.plan)

    def sources(self, target):
        return [region for region in self.regions if region != target]

    def _build_model(self, seed, vocabulary):
        if self.baseline in ("recast", "rtpp-v2"):
            from src.models.tpp.region_recast import RegionalRECAST, RegionalRTPPV2

            torch.manual_seed(seed)
            model_class = RegionalRECAST if self.baseline == "recast" else RegionalRTPPV2
            return model_class(OmegaConf.create(self.backbone), self.device)
        return build_model("stats_film", seed, vocabulary, self.backbone, self.device)

    def load(self, filename):
        path = self.data_dir / filename
        if file_sha256(path) != self.manifest["artifacts"][filename]:
            raise ValueError(f"Data hash mismatch: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    def validation(self, data, regions, source=False):
        bounds = ("source_train_end", "train_end") if source else ("train_end", "val_end")
        return {name: tuple(data["regions"][name][key] for key in bounds) for name in regions}

    @torch.no_grad()
    def evaluate(self, model, data, bounds):
        model.eval()
        values, records = {}, {}
        for name, (low, high) in bounds.items():
            if self.full_sequence:
                batch = make_batch(data, name, [(low, high - low)], None, self.device)
                score = float(model.nll_loss(batch, reduction="per_event")["time"].item())
                if not math.isfinite(score):
                    raise ValueError(f"Nonfinite time likelihood: {name}")
                values[name] = {"time": score}
                records[name] = [dict(first=low, events=high-low, history_events=low, time=score)]
                continue
            windows = iter(spans(low, high, self.options["eval_targets"]))
            rows = []
            while group := list(islice(windows, self.options["eval_batch_size"])):
                batch = make_batch(data, name, group, self.options["history"], self.device)
                losses = model.nll_loss(batch, reduction="per_event")
                matrix = torch.stack([losses[k] for k in METRICS], dim=1).cpu().tolist()
                rows.extend(dict(first=first, events=count, **dict(zip(METRICS, row)))
                            for (first, count), row in zip(group, matrix))
            if sum(row["events"] for row in rows) != high - low or high <= low:
                raise ValueError(f"Incorrect scored count: {name}")
            values[name] = {key: math.fsum(row[key] * row["events"] for row in rows) / (high - low)
                            for key in METRICS}
            if not all(math.isfinite(x) for x in values[name].values()):
                raise ValueError(f"Nonfinite likelihood: {name}")
            records[name] = rows
        macro = {key: float(np.mean([row[key] for row in values.values()])) for key in self.metrics}
        return {"regions": values, "macro": macro}, records

    def result(self, label):
        folder = self.runs / label
        result = read_json(folder / "result.json")
        if (result["plan_id"] != self.plan_id or result["label"] != label
                or file_sha256(folder / "best_model.pth") != result["checkpoint_sha256"]):
            raise ValueError(f"Checkpoint provenance mismatch: {folder}")
        return result

    def load_weights(self, model, result):
        state = torch.load(self.runs / result["label"] / "best_model.pth",
                           map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)

    def finish(self, folder, history, best_step, **fields):
        result = dict(plan_id=self.plan_id, label=folder.name, history=history, best_step=best_step,
                      checkpoint_sha256=file_sha256(folder / "best_model.pth"), **fields)
        save_json(folder / "result.json", result)
        return result

    def neural_fit(self, data, target, mode, seed, lr, parent=None):
        source = mode == "source"
        regions = self.sources(target) if source else [target]
        steps_per_region = self.options["source_steps_per_region" if source else "target_steps"]
        total_steps = steps_per_region * len(regions)
        folder = self.runs / f"{target}__{mode}__s{seed}__lr{lr:g}"
        parent_hash = parent["checkpoint_sha256"] if parent else None
        if (folder / "result.json").exists():
            result = self.result(folder.name)
            if result["parent_sha256"] != parent_hash:
                raise ValueError(f"Pretrained parent changed for {folder.name}")
            return result
        folder.mkdir(exist_ok=True)
        model = self._build_model(seed, data["vocabulary"])
        initial = state_hash(model)
        if parent:
            self.load_weights(model, parent)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
        scheduler = (CosineWithWarmupFloor(optimizer, total_steps // 10, total_steps, .05 * lr)
                     if self.full_sequence else None)
        draws = {}
        for name in regions:
            spec = data["regions"][name]
            cutoff = spec["source_train_end" if source else "train_end"]
            pool = spans(spec.get("train_start", data["calibration_events"]), cutoff,
                         self.options["train_targets"])
            rng = random.Random(seed * 100003 + data["vocabulary"][name] * 1009)
            draws[name] = rng.choices(pool, weights=[count for _, count in pool], k=steps_per_region)
        order_rng = random.Random(seed + 31337)
        history_rng = random.Random(seed + 12493)
        requested_history, effective_history = Counter(), Counter()
        validation = self.validation(data, regions, source=source)
        validate_every = self.options["source_validate_every" if source else "target_validate_every"]
        history, best, best_step = [], math.inf, 0
        step = 0
        for cycle in range(steps_per_region):
            order = regions.copy()
            order_rng.shuffle(order)
            for name in order:
                step += 1
                window = draws[name][cycle]
                context = history_rng.choice(self.history_choices)
                batch = make_batch(data, name, [window], context, self.device)
                requested_history[context] += 1
                effective_history[min(context, window[0] - (0 if self.full_sequence else 1))] += 1
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = model.nll_loss(batch, reduction="per_event")[self.selection_metric].mean()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not (torch.isfinite(loss) & torch.isfinite(norm)):
                    raise ValueError(f"Nonfinite neural loss or gradient: {folder.name}, step {step}")
                if scheduler is None:
                    factor = min(1., step / max(1, total_steps // 10))
                    factor *= max(.05, .5 * (1 + math.cos(math.pi * step / total_steps)))
                    optimizer.param_groups[0]["lr"] = lr * factor
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if step % validate_every == 0 or step == total_steps:
                    value, _ = self.evaluate(model, data, validation)
                    history.append(dict(step=step, validation=value))
                    print(folder.name, step, value["macro"], flush=True)
                    if value["macro"][self.selection_metric] < best:
                        best, best_step = value["macro"][self.selection_metric], step
                        atomic_checkpoint(folder / "best_model.pth", model.state_dict())
        return self.finish(folder, history, best_step, target=target, mode=mode, seed=seed,
                           lr=lr, updates=total_steps, val_total=best, source_regions=regions if source else [],
                           initial_sha256=initial, parent_sha256=parent_hash, draw_sha256=digest(draws),
                           requested_history=dict(requested_history), effective_history=dict(effective_history),
                           events_exposed={r: sum(n for _, n in windows) for r, windows in draws.items()},
                           parameters=sum(p.numel() for p in model.parameters()))

    def etas_model(self, data, target):
        region = data["regions"][target]
        mc = region["metadata"][0]
        magnitude = region["magnitude"][data["calibration_events"]:region["train_end"]]
        cfg = dict(self.etas_config, epochs=self.etas_epochs, mag_completeness=mc,
                   richter_b_mle=discrete_b(magnitude, mc), etas_grad_checkpoint=False)
        if self.full_sequence:
            cfg.update(learning_rate=self.options.get("etas_lr", 1e-3), etas_grad_checkpoint=True,
                       etas_query_chunk_size=128, etas_history_chunk_size=2048)
        return ScoredETAS(cfg, self.device), cfg

    def etas_fit(self, data, target):
        from src.train.config_setup import get_scheduler

        folder = self.runs / f"{target}__etas"
        if (folder / "result.json").exists():
            return self.result(folder.name)
        folder.mkdir(exist_ok=True)
        model, cfg = self.etas_model(data, target)
        spec = data["regions"][target]
        if self.full_sequence:
            from run_unified_tpp import fit

            options = dict(model="etas", seed=self.seeds[0], history=None,
                           train_events=self.options["train_targets"], steps=self.etas_epochs,
                           validate_every=self.options["etas_validate_every"], lr=cfg["learning_rate"])
            fit(model.native, options, spec, data["vocabulary"][target], folder, self.device)
            best = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
            model.native.load_state_dict(best["state"])
            atomic_checkpoint(folder / "best_model.pth", model.state_dict())
            (folder / "best.pt").unlink()
            history = [dict(step=best["step"], validation=best["validation"])]
            return self.finish(folder, history, best["step"], target=target, mode="etas", seed=None,
                               lr=cfg["learning_rate"], updates=self.etas_epochs,
                               val_total=best["validation"]["time_nll_per_event"], config=cfg)
        windows = spans(data["calibration_events"], spec["train_end"], self.options["train_targets"])
        batches = [make_batch(data, target, [window], self.options["history"], "cpu") for window in windows]
        durations = [float((batch.t_end - batch.t_nll_start).item()) for batch in batches]
        total_duration = sum(durations)
        if min(durations) <= 0:
            raise ValueError(f"{target}: zero-duration ETAS block")
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                      weight_decay=cfg["weight_decay"], betas=(.9, .99))
        scheduler = get_scheduler(cfg["scheduler_type"], optimizer, OmegaConf.create(cfg), train_dataloader=[None])
        history, best, best_step = [], math.inf, 0
        for epoch in range(1, self.etas_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for cpu_batch, duration in zip(batches, durations):
                batch = cpu_batch.to(self.device)
                loss = model.native.nll_loss(batch, reduction="per_time", return_dict=True)["time"].mean()
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite ETAS loss: {target}, epoch {epoch}")
                (loss * (duration / total_duration)).backward()
                if self.device.type == "cuda":
                    batch.to("cpu")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
            if not torch.isfinite(norm):
                raise ValueError(f"Nonfinite ETAS gradient: {target}, epoch {epoch}")
            optimizer.step()
            scheduler.step()
            if epoch == 1 or epoch % self.options["etas_validate_every"] == 0 or epoch == self.etas_epochs:
                value, _ = self.evaluate(model, data, self.validation(data, [target]))
                history.append(dict(step=epoch, validation=value))
                print(folder.name, epoch, value["macro"], flush=True)
                if value["macro"]["total"] < best:
                    best, best_step = value["macro"]["total"], epoch
                    atomic_checkpoint(folder / "best_model.pth", model.state_dict())
        return self.finish(folder, history, best_step, target=target, mode="etas", seed=None,
                           lr=cfg["learning_rate"], updates=self.etas_epochs, val_total=best, config=cfg)

    def audit_windows(self, data):
        totals = {}
        for name in self.regions:
            spec = data["regions"][name]
            subsets = {
                "source_train": spans(data["calibration_events"], spec["source_train_end"], self.options["train_targets"]),
                "target_train": spans(data["calibration_events"], spec["train_end"], self.options["train_targets"]),
                "source_validation": spans(spec["source_train_end"], spec["train_end"], self.options["eval_targets"]),
                "target_validation": spans(spec["train_end"], spec["val_end"], self.options["eval_targets"]),
            }
            for windows in subsets.values():
                for window in windows:
                    make_batch(data, name, [window], self.options["history"], "cpu")
            totals[name] = {key: sum(count for _, count in windows) for key, windows in subsets.items()}
        save_json(self.reports / "window_audit.json", dict(passed=True, plan_id=self.plan_id, events=totals))
        return totals

    def fit(self, data):
        completed, selected = [], []
        for target in self.targets:
            if self.neural:
                parents = {}
                for seed in self.seeds:
                    parent = self.neural_fit(data, target, "source", seed, self.options["source_lr"])
                    parents[seed] = parent
                    completed.append(parent["label"])
                    selected.append(dict(target=target, mode="zero", seed=seed, label=parent["label"]))
                paired = {}
                for mode in ("local", "fine"):
                    first = self.seeds[0]
                    parent = parents[first] if mode == "fine" else None
                    trials = [self.neural_fit(data, target, mode, first, lr, parent) for lr in self.rates]
                    completed.extend(r["label"] for r in trials)
                    winner = min(trials, key=lambda r: r["val_total"])
                    for seed in self.seeds:
                        parent = parents[seed] if mode == "fine" else None
                        fit = winner if seed == first else self.neural_fit(data, target, mode, seed, winner["lr"], parent)
                        if seed != first:
                            completed.append(fit["label"])
                        selected.append(dict(target=target, mode=mode, seed=seed, label=fit["label"]))
                        paired[mode, seed] = fit
                for seed in self.seeds:
                    local, fine = paired["local", seed], paired["fine", seed]
                    if not (local["initial_sha256"] == fine["initial_sha256"] == parents[seed]["initial_sha256"]
                            and local["draw_sha256"] == fine["draw_sha256"]
                            and fine["parent_sha256"] == parents[seed]["checkpoint_sha256"]):
                        raise ValueError(f"Broken local/fine pairing: {target}, seed {seed}")
            if self.etas_epochs:
                fit = self.etas_fit(data, target)
                completed.append(fit["label"])
                selected.append(dict(target=target, mode="etas", seed=None, label=fit["label"]))
        save_json(self.runs / "training.json", dict(plan_id=self.plan_id, completed=completed, selected=selected))

    def score(self):
        training = read_json(self.runs / "training.json")
        completed = training["completed"]
        expected = self.plan["neural_fits"] + self.plan["etas_fits"]
        if training["plan_id"] != self.plan_id or len(completed) != expected or len(set(completed)) != expected:
            raise ValueError("All planned fits must finish before scoring")
        results = {label: self.result(label) for label in completed}
        expected_modes = [(mode, seed) for mode in ("local", "zero", "fine") for seed in self.seeds] if self.neural else []
        if self.etas_epochs:
            expected_modes.append(("etas", None))
        expected_scores = {(target, mode, seed) for target in self.targets for mode, seed in expected_modes}
        selected = training["selected"]
        if len(selected) != len(expected_scores) or {
                (row["target"], row["mode"], row["seed"]) for row in selected} != expected_scores:
            raise ValueError("Selection manifest is incomplete")
        for row in selected:
            fit = results[row["label"]]
            expected_mode = "source" if row["mode"] == "zero" else row["mode"]
            if (fit["target"], fit["mode"], fit["seed"]) != (row["target"], expected_mode, row["seed"]):
                raise ValueError("Selected checkpoint does not match the requested target or mode")
        data = self.load("test.pt")
        rows, details = [], []
        for row in selected:
            target = row["target"]
            fit = results[row["label"]]
            if row["mode"] == "etas":
                model = ScoredETAS(fit["config"], self.device)
            else:
                model = self._build_model(row["seed"], data["vocabulary"])
            self.load_weights(model, fit)
            region = data["regions"][target]
            first, end = region["val_end"], region["test_end"]
            values, windows = self.evaluate(model, data, {target: (first, end)})
            summary = dict(row, lr=fit["lr"], best_step=fit["best_step"], events=end-first,
                           checkpoint_sha256=fit["checkpoint_sha256"], **values["macro"])
            rows.append(summary)
            details.append(dict(summary, windows=windows[target]))
        temporary = self.reports / "metrics.tmp"
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(self.reports / "metrics.csv")
        summary = []
        for target in self.targets:
            for mode in dict.fromkeys(mode for mode, _ in expected_modes):
                group = [row for row in rows if row["target"] == target and row["mode"] == mode]
                mean = {key: float(np.mean([row[key] for row in group])) for key in self.metrics}
                sd = {key: float(np.std([row[key] for row in group], ddof=1)) for key in self.metrics} if len(group) > 1 else None
                summary.append(dict(target=target, mode=mode, mean=mean, seed_sd=sd))
        save_json(self.reports / "scores.json", dict(plan_id=self.plan_id, summary=summary, checkpoints=details))
        print(f"Scored {len(rows)} checkpoints in {self.reports}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("plan", "audit", "fit", "score", "all"), default="plan")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    study = Study(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    if args.stage in ("fit", "score", "all") and study.neural and not torch.cuda.is_available():
        parser.error("The configured Mamba3 runs require CUDA. Use plan/audit on this host.")
    study.runs.mkdir(parents=True, exist_ok=True)
    study.reports.mkdir(parents=True, exist_ok=True)
    with (study.runs / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = study.runs / "plan.json"
        if path.exists() and read_json(path)["plan_id"] != study.plan_id:
            raise ValueError("Use new run and report directories for changed configuration, code or data")
        snapshot = dict(study.plan, plan_id=study.plan_id)
        save_json(path, snapshot)
        save_json(study.reports / "plan.json", snapshot)
        if args.stage == "audit":
            study.audit_windows(study.load("development.pt"))
        if args.stage in ("fit", "all"):
            for name, expected in study.plan["source_files"].items():
                destination = study.runs / "frozen_project" / name
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ROOT / name, destination)
                if file_sha256(destination) != expected:
                    raise ValueError(f"Frozen source hash mismatch: {name}")
            study.fit(study.load("development.pt"))
        if args.stage in ("score", "all"):
            study.score()
    print(f"Plan {study.plan_id[:12]}: {study.plan['neural_fits']} neural and {study.plan['etas_fits']} ETAS fits")


if __name__ == "__main__":
    main()
