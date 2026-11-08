"""Normalize DOI-pinned event data and verify the existing em_eqf loaders."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.catalogs.external_catalog import file_sha256
from src.data import Catalog
from src.data.preparation_modules.tpp import prepare_data_tpp

STORE = Path("/root/autodl-fs/em_eqf_catalogs/doi_expansion_20261106")
REPORTS = ROOT / "experiments/revision/catalog_doi_expansion_20261106/reports"
RAW = STORE / "raw"
KEY = ["time", "latitude", "longitude", "depth_km", "magnitude"]


def md5(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "md5").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def west_greece(path):
    assert md5(path) == "ae469a924b6d05e07e03ed9ea5a99150"
    frame = pd.read_csv(path, dtype={"id": "string"})
    assert len(frame) == 21331 and frame.origin.str.endswith("Z").all()
    # The file labels both axes identically. The first is always >= the second,
    # and both column averages match the larger and smaller axes in the paper.
    first = frame["semi_major_ellipse_error_axis"]
    second = frame["semi_major_ellipse_error_axis.1"]
    assert (first >= second).all() and abs(first.mean() - 2.24) < .02 and abs(second.mean() - .84) < .02
    frame = frame.rename(columns={
        "id": "event_id", "depth": "depth_km", "ml": "magnitude",
        "semi_major_ellipse_error_axis": "ellipse_larger_km",
        "semi_major_ellipse_error_axis.1": "ellipse_smaller_km",
    })
    frame["time"] = pd.to_datetime(frame.origin, utc=True, format="ISO8601").dt.as_unit("ns")
    frame["mag_type"] = "ML"
    keep = (frame.N_obs.ge(16) & frame.GAP.between(0, 240) & frame.RMS.between(0, .35)
            & frame.ellipse_larger_km.between(0, 5) & frame.z_error.between(0, 5)
            & frame.magnitude.ge(1.3))
    policy = "NLL-SSST-coherence; 16+ phase observations, gap<=240°, RMS<=0.35 s, larger ellipse axis and vertical uncertainty <=5 km; ML>=1.3 research cutoff."
    return frame, keep, {"policy": policy, "time_basis": "Explicit UTC Z in each origin timestamp."}


def chile(path):
    assert md5(path) == "49d78f37784956e362de82d159083165"
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        member, = [name for name in archive.namelist() if Path(name).name == "CHILE_SEISMICITY_RELOCATED.csv"]
        with archive.open(member) as stream:
            frame = pd.read_csv(stream)
    frame.columns = frame.columns.str.strip().str.lower()
    frame = frame.rename(columns={"seconds": "second"})
    assert len(frame) == 118004 and list(frame.columns[1:]) == [
        "year", "month", "day", "hour", "minute", "second", "longitude",
        "latitude", "depth", "rms", "magnitude", "magnitude_type",
    ], frame.columns.tolist()
    frame = frame.rename(columns={frame.columns[0]: "event_id", "depth": "depth_km"})
    frame["event_id"] = frame.event_id.astype("string")
    times = pd.to_datetime(frame[["year", "month", "day", "hour", "minute"]], utc=True)
    seconds = frame.second.to_numpy() * 100
    assert np.allclose(seconds, np.rint(seconds), rtol=0, atol=1e-7)
    frame["time"] = (times + pd.to_timedelta(np.rint(seconds).astype("int64") * 10, unit="ms")).dt.as_unit("ns")
    anchors = []
    for when, latitude, longitude in [
        ("2010-02-27T06:34:11Z", -36.122, -72.898),
        ("2014-04-01T23:46:47Z", -19.642, -70.817),
    ]:
        distance = (frame.time - pd.Timestamp(when)).abs().dt.total_seconds()
        match = frame.loc[(distance <= 20) & frame.magnitude.ge(7)
                          & (frame.latitude - latitude).abs().le(1)
                          & (frame.longitude - longitude).abs().le(1)]
        assert len(match) == 1, (when, match)
        anchors.append({"reference_utc": when, "source_utc": str(match.time.iloc[0])})
    frame["mag_type"] = frame.magnitude_type.str.strip()
    keep = (frame.mag_type.eq("l") & frame.magnitude.ge(3)
            & frame.rms.between(0, .5) & frame.depth_km.between(-5, 700)
            & frame.time.ge("2000-01-01") & frame.time.lt("2020-07-01"))
    policy = "Author relocated main catalog only; published local magnitude >=3 and relocation RMS<=0.5 s; source depth preserved; 2000–mid-2020."
    return frame, keep, {"policy": policy, "time_basis": "Inferred UTC from matching Maule and Iquique mainshocks. The Illapel mainshock anchor was absent in this file.", "utc_anchors": anchors, "archive_member": member}


def atacama(path):
    assert md5(path) == "f1447bc063f34999cb0590eecc64f4e3"
    frame = pd.read_csv(path, dtype={"idx": "string"})
    assert frame.time.str.endswith("+00:00").all()
    frame["time"] = pd.to_datetime(frame.time, format="ISO8601", utc=True).dt.as_unit("ns")
    frame = frame.rename(columns={"idx": "event_id", "depth": "depth_km", "ML": "magnitude"})
    frame["mag_type"] = "ML"
    assert frame["class"].notna().all()
    classes = frame["class"].value_counts().to_dict()
    natural = frame["class"].isin(["interface", "intraslab", "upper_plate", "outer_rise"])
    keep = (
        natural & frame.airgun.eq(False) & frame.mislocated.eq(False)
        & frame.growclust_relocated.eq(True) & frame.number_picks.ge(12)
        & frame.number_p_and_s_picks.ge(5) & frame.azimuthal_gap.between(0, 180)
        & frame.max_horizontal_uncertainty.between(0, 2)
        & frame.vertical_uncertainty.between(0, 5) & frame.magnitude.ge(1.5)
        & frame.time.ge("2020-11-01") & frame.time.lt("2024-02-15")
    )
    policy = "Author natural classes, no airgun or mislocated flag, GrowClust relocation, 12+ picks, 5+ paired P/S stations, gap<=180°, horizontal/vertical uncertainty <=2/5 km, ML>=1.5."
    return frame, keep, {"policy": policy, "time_basis": "Explicit UTC +00:00.", "source_classes": classes}


def santorini(path):
    assert md5(path) == "69468f61c006cccd5bcd5e2189fd4c24"
    frame = pd.read_csv(path, skipinitialspace=True, dtype={"publicId": "string"})
    assert len(frame) == 25484
    frame = frame.rename(columns={"publicId": "event_id", "depth": "depth_km", "Mamp": "magnitude"})
    frame["time"] = pd.to_datetime(frame["date-time"], format="ISO8601", utc=True).dt.as_unit("ns")
    frame["mag_type"] = "Mamp"
    reference = pd.read_csv(RAW / "santorini_usgs_anchors.csv")
    matches = {}
    for hours in [-3, -2, 0, 2, 3]:
        matched = {}
        for row in reference.itertuples():
            seconds = (frame.time + pd.Timedelta(hours=hours) - pd.Timestamp(row.time)).abs().dt.total_seconds()
            nearby = ((frame.latitude - row.latitude).abs().le(.2)
                      & (frame.longitude - row.longitude).abs().le(.25)
                      & frame.magnitude.ge(4) & seconds.le(10))
            if nearby.any():
                index = seconds[nearby].idxmin()
                matched[row.id] = {"event_id": frame.event_id.loc[index], "seconds": float(seconds.loc[index])}
        matches[hours] = matched
    assert len(matches[0]) >= 20 and all(len(matches[h]) <= 1 for h in matches if h != 0)
    keep = (
        frame.Nphs.ge(12) & frame.Gap.between(0, 180) & frame.RMS.between(0, .35)
        & frame.errH.between(0, 2) & frame.errZ.between(0, 5) & frame.magnitude.ge(1.5)
        & frame.time.ge("2025-02-05") & frame.time.lt("2025-03-01")
    )
    policy = "Author filtered swarm relocations; Mamp>=1.5, 12+ phases, gap<=180°, RMS<=0.35 s, horizontal/vertical uncertainty <=2/5 km; score the dense 5–28 February 2025 interval."
    return frame, keep, {
        "policy": policy, "time_basis": "Inferred UTC from USGS spatial/time anchors.",
        "clock_shift_candidate_counts": {str(k): len(v) for k, v in matches.items()},
        "utc_anchors": matches[0], "anchor_sha256": file_sha256(RAW / "santorini_usgs_anchors.csv"),
    }


def build(name, path, doi, url, parse):
    frame, keep, notes = parse(path)
    assert frame.event_id.is_unique and frame.event_id.notna().all()
    frame["provider"] = name
    frame["absolute_time_ready"] = True
    frame["quality_screen_pass"] = keep & np.isfinite(frame[KEY[1:]]).all(axis=1)
    selected = frame.loc[frame.quality_screen_pass].sort_values(["time", "event_id"], kind="stable")
    assert not selected.duplicated(KEY).any(), name
    store = STORE / name
    data = ROOT / "data" / name
    (store / "processed").mkdir(parents=True, exist_ok=True)
    (store / "catalogs").mkdir(exist_ok=True)
    source = store / "processed/quality_screened.parquet"
    for table, target in [(frame, store / "processed/normalized_all.parquet"), (selected, source)]:
        temporary = target.with_suffix(".parquet.part")
        table.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(target)
    for relative, target in [
        ("processed/standard_source_20261106.parquet", source),
        ("catalogs/standard_20261106", store / "catalogs"),
    ]:
        link = data / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            assert link.resolve() == target.resolve()
        else:
            link.symlink_to(target)
    cls = Catalog.by_name(name + "-Standard")
    catalog = cls()
    cached = pd.read_parquet(catalog.root_dir / "catalog.parquet")
    assert np.array_equal(cached.event_id.to_numpy(), selected.event_id.to_numpy())
    assert np.array_equal(cached[KEY[1:]].to_numpy(), selected[KEY[1:]].to_numpy())
    assert np.array_equal(pd.to_datetime(cached.ts, utc=True).dt.as_unit("ns"), selected.time)
    sequence = catalog.full_sequence
    assert sequence.inter_times.dtype == torch.float64 and (sequence.inter_times[:-1] > 0).all()
    cfg = OmegaConf.create(dict(
        dataset=name, catalog_cfg=dict(
            root_dir=str(data / "catalogs/standard_20261106"), catalog_file=str(source),
            start_ts=cls.default_start, end_ts=cls.default_end, mag_completeness=cls.default_mc,
            normalize=True, val_start_ts=cls.default_validation, test_start_ts=cls.default_test),
        batch_size=2, use_double_precision=True, use_all_data=False, minibatch_training=True,
        max_seq_len=512, mean_nll_events=128, use_b_updater=False,
    ))
    _, train, val, test, _ = prepare_data_tpp(cfg, data)
    for loader in (train, val, test):
        assert torch.isfinite(next(iter(loader)).inter_times).all()
        for part in loader.dataset.sequences:
            scored = part.arrival_times >= part.t_nll_start
            assert not ((part.inter_times[:-1] <= 0) & scored).any()
    report = json.loads((catalog.root_dir / "processing_report.json").read_text())
    counts = report["scored_events"]
    assert sum(counts.values()) == len(selected) and min(counts.values()) >= 20
    result = {
        "catalog": name + "-Standard", "doi": doi, "url": url, "raw_path": str(path),
        "raw_sha256": file_sha256(path), "raw_records": len(frame), "records": len(cached),
        "scored_events": counts, "source_sha256": file_sha256(source), "cache": str(catalog.root_dir),
        "dataloader_verified": True, **notes,
    }
    write_json(REPORTS / f"{name}.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    REPORTS.mkdir(parents=True, exist_ok=True)
    specs = {
        "Greece_WestHellenic": ("west_greece.csv", "10.5281/zenodo.18833544",
            "https://zenodo.org/api/records/18833544/files/catalog_NLL_SSST_coh.csv/content", west_greece),
        "Chile_Potin": ("chile.zip", "10.5281/zenodo.13146436",
            "https://zenodo.org/api/records/13146436/files/chilean_seismic_catalogue-1982-2020-publication_release.zip/content", chile),
        "Chile_Atacama": ("atacama.csv", "10.5281/zenodo.15083298",
            "https://zenodo.org/records/15083298/files/catalog_temp.csv?download=1", atacama),
        "Greece_SantoriniSwarm": ("santorini.csv", "10.5281/zenodo.23059305",
            "https://zenodo.org/api/records/23059305/files/Filtered_relocated_catalog.csv/content", santorini),
    }
    for name in sys.argv[1:] or specs:
        filename, doi, url, parser = specs[name]
        build(name, RAW / filename, doi, url, parser)
