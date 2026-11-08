"""Build and re-audit the registered remediation views through the TPP entry point."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analyze import audit_catalog_integrity as audit
from scripts.data.prepare_remediation_sources import AUDIT, REPORTS, STORE, write_json
from src.catalogs.external_catalog import file_sha256
from src.data import Catalog
from src.data.preparation_modules.tpp import prepare_data_tpp

PROFILES = {
    "NCEDC_Repaired20261106": "NCEDC",
    "USGS_MontanaComCat20261106": "USGS_Montana",
    "Taiwan_CWA1993Plus20261106": "Taiwan_CWA",
    "Chile_Potin2001NovPlus20261106": "Chile_Potin",
    "Turkiye_EAFZ_GFT90_20261106": "Turkiye_EAFZ",
    "Turkiye_EAFZ_GFT95_20261106": "Turkiye_EAFZ",
    "Greece_Santorini_GFT95_20261106": "Greece_SantoriniSwarm",
}
QUALITY_FIELDS = ["station_count", "azimuthal_gap", "rms", "horizontal_error_km", "depth_error_km"]


def require_link(path, target):
    target.mkdir(parents=True, exist_ok=True)
    if not path.exists() and not path.is_symlink():
        path.symlink_to(target, target_is_directory=True)
    if path.resolve() != target.resolve():
        raise ValueError(f"Existing path {path} does not resolve to {target}")


def verify_views():
    previous = pd.read_csv(AUDIT / "overview.csv").set_index("dataset")
    old_caches = {r["dataset"]: Path(r["cache"])
                  for r in json.loads((AUDIT / "cache_integrity.json").read_text())}
    protected = {}
    for path in set(old_caches.values()):
        candidates = [path] if path.is_file() else [path / key for key in
                      ("catalog.parquet", "full_sequence.pt", "metadata.pt", "norm_stats.pt")]
        protected.update((str(p), file_sha256(p)) for p in candidates if p.is_file())

    summary, coverage, longest, mc, fmd, depths, missing = [], [], [], [], [], [], []
    for name, baseline in PROFILES.items():
        cls = Catalog.by_name(name + "-Standard")
        base = ROOT / "data" / name
        require_link(base, STORE / name)
        args = SimpleNamespace(dataset=name, catalog_cfg={}, batch_size=1, use_double_precision=True)
        _, train, val, test, obj = prepare_data_tpp(args, base)
        for loader in (train, val, test):
            assert next(iter(loader)) is not None
        cfg = json.loads((obj.root_dir / "catalog_cfg.json").read_text())
        frame = audit.canonical(pd.read_parquet(obj.root_dir / "catalog.parquet"))
        quality = audit.read_source(obj.catalog_file)
        start, end = (pd.Timestamp(cfg[k], tz="UTC") for k in ("start_ts", "end_ts"))
        cut = obj.mc
        old = audit.read_source(old_caches[baseline] / "catalog.parquet")
        common = old.set_index("event_id").join(frame.set_index("event_id"), how="inner", lsuffix="_old", rsuffix="_new")
        diffs = {key: int(common[key + "_old"].ne(common[key + "_new"]).sum())
                 for key in ("time", "magnitude", "latitude", "longitude", "depth_km")}
        if baseline != "USGS_Montana":
            assert not any(diffs.values())
            expected = old[old.time.ge(start) & old.time.lt(end) & old.magnitude.ge(cut)]
            assert set(expected.event_id).issubset(frame.event_id)
            if baseline == "NCEDC":
                ids = pd.read_csv(AUDIT / "ncedc_2024_type_recovery_candidates.csv", dtype={"id": str})
                new_ids = set(frame.event_id) - set(expected.event_id)
                assert new_ids == set(ids.loc[ids.passes_existing_numeric_quality, "id"])
            else:
                assert set(expected.event_id) == set(frame.event_id)
        if baseline == "NCEDC":
            raw = audit.read_source(STORE / "NCEDC/processed/earthquakes.parquet")
        elif baseline == "USGS_Montana":
            raw = audit.read_source(STORE / "USGS_MontanaComCat/processed/normalised_20261105.parquet")
            full_window = audit.cover(name, raw, quality, quality, pd.Timestamp("2014-01-01", tz="UTC"), end, cut)
            full_window.to_csv(REPORTS / "montana_full_source_coverage.csv", index=False)
        else:
            raw = audit.read_source(Path(previous.loc[baseline, "source_path"]))
            quality = raw[audit.recent_quality(baseline, raw)]
        calendar = audit.cover(name, raw, quality, frame, start, end, cut)
        gaps = audit.gaps(name, raw, frame, start, end, cut)
        audit.diagnostics(name, raw, quality, frame, cfg, mc, fmd, depths)
        fields = list(audit.missing_rows(frame, name, "standard"))
        assert not any(r["missing"] or r["invalid"] for r in fields if r["required"])
        integrity = audit.cache_integrity(ROOT, name, obj.root_dir, cfg, obj.catalog_file, frame)
        assert not integrity["exact_duplicate_physical_rows"]
        assert obj.full_sequence.inter_times.dtype == torch.float64
        empty = calendar.loc[calendar.standard.eq(0), "start"]
        if baseline in ("NCEDC", "USGS_Montana", "Taiwan_CWA", "Chile_Potin") or "_GFT90_" in name:
            assert empty.empty, (name, list(empty))
        summaries = dict(baseline=baseline, records=len(frame), old_records=len(old),
                         start=start, end=end, cutoff=cut, empty_calendar_bins=len(empty),
                         old_empty_calendar_bins=int(previous.loc[baseline, "standard_empty_bins"]),
                         longest_gap_days=gaps[0]["days"],
                         old_longest_gap_days=float(previous.loc[baseline, "longest_standard_gap_days"]),
                         same_id_field_changes=diffs, **integrity)
        summary.append(summaries)
        coverage.append(calendar)
        longest.extend(gaps)
        missing.extend(fields)
        print(name, len(frame), summaries["split_counts"], "empty", len(empty), "maxgap", gaps[0]["days"], flush=True)

    assert all(file_sha256(path) == value for path, value in protected.items())
    write_json(REPORTS / "verification.json", {"views": summary, "protected_files_unchanged": protected})
    pd.concat(coverage, ignore_index=True).to_csv(REPORTS / "coverage_calendar.csv", index=False)
    for filename, rows in (("coverage_gaps", longest), ("mc_temporal", mc), ("fmd", fmd),
                           ("mc_depth", depths), ("missing_fields", missing)):
        pd.DataFrame(rows).to_csv(REPORTS / (filename + ".csv"), index=False)


def write_feature_contract():
    """Mark model inputs as usable only when every row has that field."""
    sources = [AUDIT / "missing_fields.csv", REPORTS / "missing_fields.csv"]
    noa = REPORTS / "noa_integrity/missing_fields.csv"
    if noa.is_file():
        sources.append(noa)
    fields = pd.concat([pd.read_csv(path) for path in sources], ignore_index=True)
    data = fields[fields.stage.eq("standard")].copy()
    data["available"] = (data.status.eq("present") & data.missing.eq(0) & data.invalid.eq(0))
    availability = data.pivot(index="dataset", columns="field", values="available")
    groups = {"time_magnitude_2d": ["event_id", "time", "magnitude", "latitude", "longitude"],
              "physical_depth": ["depth_km"], "strict_location_metadata": QUALITY_FIELDS}
    for group, names in groups.items():
        availability[group] = availability.reindex(columns=names).eq(True).all(axis=1)
    availability.to_csv(REPORTS / "feature_availability.csv")


if __name__ == "__main__":
    verify_views()
    write_feature_contract()
