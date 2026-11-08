"""Check the on-disk expanded and unified model input packs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def audit(path, group):
    manifest = json.loads((path/"dataset.json").read_text())
    for file in ("development.pt","test.pt"):
        assert sha(path/file) == manifest["artifacts"][file]
    load = lambda name: torch.load(path/name, map_location="cpu", weights_only=False)["regions"]
    dev, full = load("development.pt"), load("test.pt")
    assert set(dev) == set(full) == set(manifest["regions"])
    for name, r in full.items():
        physical, model = np.asarray(r["physical_ns"]), np.asarray(r["ns"])
        delta = model - physical
        raw, mag = np.asarray(r["raw_magnitude"]), np.asarray(r["magnitude"])
        ids = r["event_id"]
        end = len(dev[name]["ns"])
        for field in ("ns","physical_ns","raw_magnitude","magnitude","event_id"):
            assert np.array_equal(dev[name][field],r[field][:end]), (name,field)
        dates = r.get("time_bounds_ns")
        if dates is not None:
            boundaries = sorted(set(dates.values()))
            assert np.array_equal(np.searchsorted(physical,boundaries),np.searchsorted(model,boundaries))
            assert end == np.searchsorted(physical, dates["val_end"])
        assert len(ids) == len(set(ids)) == len(raw) == len(physical)
        assert np.all(np.diff(model) > 0) and np.all(delta >= 0)
        assert np.isfinite(raw).all() and np.isfinite(mag).all()
        difference = raw-mag
        cut = r["metadata"][0]
        expected = cut + .1*np.floor((raw-cut)/.1+1e-8)
        assert np.allclose(mag, expected, atol=1e-8,rtol=0)
        result = dict(group=group,dataset=name,records=len(ids),model_shifted_rows=int(np.count_nonzero(delta)),
            max_clock_shift_ns=int(delta.max()),magnitude_binned_rows=int(np.count_nonzero(abs(difference)>1e-8)),
            max_magnitude_binning_delta=float(abs(difference).max()))
        for field in ("latitude","longitude","depth","depth_km"):
            result[field+"_missing"] = (int((~np.isfinite(r[field])).sum()) if field in r else None)
        yield result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("/root/autodl-tmp/em_eqf"))
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    unified=Path("/root/autodl-fs/em_eqf_training/unified_tpp_20261106/data")
    paths=[(unified/"benchmark","unified_benchmark"),(unified/"multiregion","unified_multiregion"),
           (args.root/"data/MultiRegion/catalogs/expanded_20261106","expanded")]
    rows=[row for path,group in paths for row in audit(path,group)]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output,index=False)
    print(pd.DataFrame(rows).groupby("group").agg(
        regions=("dataset","size"),max_shift_ns=("max_clock_shift_ns","max"),
        max_magnitude_delta=("max_magnitude_binning_delta","max")).to_string())
