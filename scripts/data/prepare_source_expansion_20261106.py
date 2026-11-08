"""Build and verify standard caches from separately normalized source snapshots."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.catalogs.external_catalog import file_sha256
from src.data import Catalog
from src.data.preparation_modules.tpp import prepare_data_tpp
from normalize_source_expansion_20261106 import FIELDS, OUT, STORE, save_json


def link(path, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        assert path.resolve() == target.resolve(), path
    else:
        path.symlink_to(target, target_is_directory=target.is_dir())


def overlap(frames):
    spec = importlib.util.spec_from_file_location(
        "_regional_overlap",
        ROOT / "experiments/revision/catalog_additions_20261105/audit_and_report_regional.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    batches = [
        "catalog_standardization_20261105", "catalog_additions_20261105", "catalog_more_20261105",
        "catalog_research_20261106", "catalog_benchmarks_20261106",
    ]
    earlier = {}
    for batch in batches:
        report = ROOT / "experiments/revision" / batch / "reports/summary.json"
        for record in json.loads(report.read_text())["datasets"]:
            earlier[record["dataset"]] = pd.read_parquet(Path(record["cache"]) / "catalog.parquet")
    pairs = []
    matches = []
    compared = dict(earlier)
    for name, frame in frames.items():
        for other, reference in compared.items():
            if frame.latitude.max() + 0.5 < reference.latitude.min() or frame.latitude.min() - 0.5 > reference.latitude.max():
                continue
            if frame.longitude.max() + 1 < reference.longitude.min() or frame.longitude.min() - 1 > reference.longitude.max():
                continue
            candidates = module.overlap_candidates(frame, reference)
            for record in candidates:
                record.update(new_dataset=name, old_dataset=other)
            matches.extend(candidates)
            pairs.append({
                "datasets": [name, other], "candidate_pairs": len(candidates),
                "distinct_left_events": len({r["new_event_id"] for r in candidates}),
                "distinct_right_events": len({r["old_event_id"] for r in candidates}),
            })
        compared[name] = frame
    pd.DataFrame(matches).to_csv(OUT / "reports/overlap_candidates.csv", index=False)
    return pairs


def build(name, row):
    cls = Catalog.by_name(name + "-Standard")
    dataset = ROOT / "data" / name
    path = Path(row["artifacts"]["quality_screened"]["path"])
    assert file_sha256(path) == row["artifacts"]["quality_screened"]["sha256"]
    assert file_sha256(row["raw"]["path"]) == row["raw"]["sha256"]
    link(dataset / "processed/standard_source_20261106.parquet", path)
    cache_root = STORE / name / "catalogs"
    cache_root.mkdir(exist_ok=True)
    link(dataset / "catalogs/standard_20261106", cache_root)
    catalog = cls()
    cache = catalog.root_dir
    source = pd.read_parquet(path)
    actual = pd.read_parquet(cache / "catalog.parquet")
    assert np.array_equal(actual.event_id.to_numpy(), source.event_id.to_numpy())
    assert np.array_equal(actual[FIELDS].to_numpy(), source[FIELDS].to_numpy())
    cached_time = pd.to_datetime(actual.ts, utc=True).dt.as_unit("ns")
    source_time = source.time.dt.as_unit("ns")
    assert np.array_equal(cached_time.astype("int64"), source_time.astype("int64"))
    sequence = catalog.full_sequence
    assert sequence.inter_times.dtype == torch.float64
    assert torch.isfinite(sequence.inter_times).all() and (sequence.inter_times[:-1] > 0).all()
    ids = {}
    for label, part, start, end in [
        ("train", catalog.train[0], catalog.train_start, catalog.val_start),
        ("val", catalog.val[0], catalog.val_start, catalog.test_start),
        ("test", catalog.test[0], catalog.test_start, catalog.end),
    ]:
        scored = part.event_index[part.arrival_times >= part.t_nll_start].long().tolist()
        expected = actual.loc[actual.ts.ge(start) & actual.ts.lt(end), "event_index"].tolist()
        assert scored == expected and len(scored) >= 20, (name, label)
        ids[label] = len(scored)
    assert sum(ids.values()) == len(actual)

    settings = dict(
        root_dir=str(dataset / "catalogs/standard_20261106"), catalog_file=str(path),
        start_ts=cls.default_start, end_ts=cls.default_end, mag_completeness=cls.default_mc,
        normalize=True, val_start_ts=cls.default_validation, test_start_ts=cls.default_test,
    )
    cfg = OmegaConf.load(ROOT / "config/mixer_tpp_mamba3_rotary.yaml")
    cfg.dataset = name
    cfg.catalog_cfg = settings
    cfg.use_double_precision = True
    cfg.use_all_data = False
    cfg.use_b_updater = False
    cfg.minibatch_training = True
    cfg.max_seq_len = 512
    cfg.mean_nll_events = 128
    cfg.batch_size = 2
    _, train, val, test, _ = prepare_data_tpp(cfg, dataset)
    for loader in [train, val, test]:
        assert torch.isfinite(next(iter(loader)).inter_times).all()
        for part in loader.dataset.sequences:
            assert not ((part.inter_times[:-1] <= 0) & (part.arrival_times >= part.t_nll_start)).any()
    OmegaConf.save(cfg, OUT / "configs" / f"{name}_mixer_tpp.yaml")
    return actual, {
        "dataset": name, "cache": str(cache), "source": str(path),
        "source_sha256": file_sha256(path), "records": len(actual), **ids,
        "magnitude_cut": cls.default_mc, "start": cls.default_start, "end": cls.default_end,
        "validation_start": cls.default_validation, "test_start": cls.default_test,
        "quality_protocol": cls.quality_protocol, "dataloader_verified": True,
        "artifact_sha256": {p.name: file_sha256(p) for p in cache.iterdir() if p.is_file()},
    }


def main():
    (OUT / "configs").mkdir(parents=True, exist_ok=True)
    reports = OUT / "reports"
    sources = json.loads((reports / "sources.json").read_text(encoding="utf-8"))
    guards = {}
    for manifest in [
        ROOT / "experiments/revision/catalog_optimization_20261105/protected_14_before.json",
        ROOT / "experiments/revision/catalog_benchmarks_20261106/legacy_common_protected_before.json",
    ]:
        guards.update(json.loads(manifest.read_text()))
    for name in ["catalog_index.json", "dataset.json"]:
        path = ROOT / "data/MultiRegion/catalogs/expanded_20261106" / name
        if path.is_file():
            guards[str(path)] = file_sha256(path)
    assert all(file_sha256(path) == sha for path, sha in guards.items())

    frames = {}
    results = []
    for name, row in sources.items():
        frames[name], result = build(name, row)
        results.append(result)
        print("READY", name, result["records"], {k: result[k] for k in ["train", "val", "test"]}, flush=True)
    pairs = overlap(frames)
    assert all(file_sha256(path) == sha for path, sha in guards.items())
    save_json(reports / "summary.json", {
        "datasets": results, "overlap_2s_15km_1m": pairs, "protected_hashes_verified": len(guards),
        "taiwan_catalogs_are_alternative_solutions_for_overlapping_years": True,
    })
    print("VERIFIED", json.dumps(pairs), flush=True)


if __name__ == "__main__":
    main()
