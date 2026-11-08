"""Audited nine-domain data for paired local/global/capacity comparisons."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch

ROOT = Path('/root/autodl-tmp/em_eqf')
STUDY = ROOT/'experiments/revision/multiregion_pretraining'
STORE = Path('/root/autodl-fs/em_eqf_training/region_capacity_20261105')
REPORTS = STUDY/'reports/region_capacity_20261105'
DATA = ROOT/'data/MultiRegion/catalogs/region_capacity_20261105.pt'
SPECS = {
    'japan_kyushu': ('JMA_Kyushu', 3., 2, (30,34,129,133)),
    'japan_west_honshu': ('JMA_WestHonshu', 3., 2, (34,37,130,137)),
    'japan_east_honshu': ('JMA_EastHonshu', 3., 2, (34,41,137,145)),
    'japan_hokkaido': ('JMA_Hokkaido', 3., 2, (41,46,139,146)),
    'new_zealand': ('GeoNet', 3., 0, (-48,-33,165,180)),
    'italy': ('INGV', 2.5, 0, (35,48,6,19)),
    'northern_california': ('NCEDC', 2.5, 0, (32,43,-126,-114)),
    'hawaii': ('USGS_Hawaii', 2.5, 0, (18,23,-161,-154)),
    'alaska': ('USGS_Alaska', 3., 0, (50,72,-180,-128)),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    STORE.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    assert not DATA.exists(), 'Prepared data are immutable; refusing overwrite'
    original = ROOT/'data/MultiRegion/catalogs/long_sequence_20261105.pt'
    protected = {str(original):sha(original)}
    tables = pd.concat([pd.read_csv(ROOT/'experiments/revision'/name/'reports/catalogs.csv') for name in [
        'catalog_standardization_20261105', 'catalog_additions_20261105']], ignore_index=True).set_index('dataset')
    regions, audit = {}, {}
    cutoff = [pd.Timestamp(s,tz='UTC').value for s in ['2019-01-01','2022-01-01','2025-01-01']]
    source_ids = set()
    for region,(name,mc,kind,box) in SPECS.items():
        source = Path(tables.loc[name,'source'])
        expected_sha = tables.loc[name,'source_sha256']
        assert sha(source) == expected_sha
        frame = pd.read_parquet(source)
        assert frame.absolute_time_ready.all() and frame.quality_screen_pass.all()
        frame['time'] = pd.to_datetime(frame.time,utc=True).dt.as_unit('ns')
        frame = frame.dropna(subset=['time','magnitude','latitude','longitude']).copy()
        lat0,lat1,lon0,lon1 = box
        frame = frame[(frame.latitude>=lat0)&(frame.latitude<lat1)&(frame.longitude>=lon0)&(frame.longitude<lon1)
                      &(frame.time<'2025-01-01')&(frame.magnitude>=mc-1e-9)].copy()
        # Lower-edge bins [Mc + .1*k, Mc + .1*(k+1)) avoid cutting the first
        # bin in half for providers reporting magnitudes to two decimals.
        raw_mag = frame.magnitude.to_numpy(np.float64)
        frame['raw_magnitude'] = raw_mag
        frame['magnitude'] = mc+.1*np.floor((raw_mag-mc)/.1+1e-8)
        frame.event_id = frame.event_id.astype(str)
        frame = frame.drop_duplicates('event_id').sort_values(['time','event_id']).reset_index(drop=True)
        physical_ns = frame.time.astype('int64').to_numpy().copy()
        model_ns = physical_ns.copy()
        for i in range(1,len(model_ns)):
            model_ns[i] = max(model_ns[i], model_ns[i-1]+1000)
        max_shift = int(np.max(model_ns-physical_ns))
        assert max_shift<=500000, 'Jitter exceeds half the coarsest 1ms precision'
        assert np.all(np.diff(model_ns)>0)
        split = np.searchsorted(physical_ns,cutoff,side='left').astype(int)
        train_end,val_end,test_end = split.tolist()
        assert train_end>=256+2048 and val_end>train_end and test_end>val_end
        targets = train_end-256
        train_used_end = train_end
        block_starts=list(range(256,train_end,2048))
        gap = np.diff(model_ns[:256])/86400e9
        mean_index = np.mean((frame.magnitude.to_numpy()[:256]-mc)/.1)
        b = np.log1p(1/max(mean_index,1e-9))/(.1*np.log(10))
        metadata = [mc,float(b),float(gap.mean()),float(gap.min()),float(gap.max()),.1,kind]
        assert np.all(np.isfinite(metadata)) and metadata[2]>0
        ids = [f'{name}:{x}' for x in frame.event_id]
        assert not source_ids & set(ids)
        source_ids.update(ids)
        regions[region] = dict(metadata=metadata,ns=model_ns,physical_ns=physical_ns,
            magnitude=frame.magnitude.to_numpy(np.float64),raw_magnitude=frame.raw_magnitude.to_numpy(),
            event_id=ids,train_end=train_end,train_used_end=train_used_end,
            val_end=val_end,test_end=test_end,train_targets=targets,
            block_starts=block_starts,block_targets=[min(2048,train_end-x) for x in block_starts],
            source=str(source),source_sha256=expected_sha,box=box)
        audit[region] = dict(provider=name,mc=mc,metadata=metadata,records=len(frame),
            train_calendar_events=train_end,calibration_events=256,train_scored_events=targets,
            unused_train_tail_events=train_end-train_used_end,val_scored_events=val_end-train_end,
            test_scored_events=test_end-val_end,paired_blocks=len(block_starts),
            last_block_targets=min(2048,train_end-block_starts[-1]),
            calibration_end=frame.time.iloc[255].isoformat(),max_model_time_shift_ns=max_shift,
            source=str(source),source_sha256=expected_sha,box=box,
            training_start=frame.time.iloc[256].isoformat(),training_last_scored=frame.time.iloc[train_used_end-1].isoformat())
        print(region,json.dumps(audit[region]),flush=True)
    # Region boxes are disjoint, including catalogs with potentially global IDs.
    boxes=list(SPECS.values())
    for i,(_,_,_,a) in enumerate(boxes):
        for _,_,_,b in boxes[i+1:]:
            assert a[1]<=b[0] or b[1]<=a[0] or a[3]<=b[2] or b[3]<=a[2]
    vocabulary={'<unknown>':0,**{name:i+1 for i,name in enumerate(sorted(regions))}}
    payload=dict(schema='nine_region_capacity_20261105_v1',regions=regions,vocabulary=vocabulary,
                 cutoff_ns=cutoff,calibration_events=256,block_targets=2048)
    path=STORE/'region_capacity_20261105.pt'
    assert not path.exists()
    torch.save(payload,path);DATA.symlink_to(path)
    report=dict(dataset=str(DATA),storage=str(path),dataset_sha256=sha(path),regions=audit,
        unique_train_scored_events=sum(v['train_scored_events'] for v in audit.values()),
        validation_scored_events=sum(v['val_scored_events'] for v in audit.values()),
        test_scored_events=sum(v['test_scored_events'] for v in audit.values()),
        calendar_split='training before 2019; validation 2019-2021; test 2022-2024, shared across all regions',
        magnitude='Fixed audited catalog thresholds; 0.1 bins use lower edges; no test-based threshold tuning.',
        past_context='Validation/test may observe past training/validation/test events; weights never update on validation/test.',
        metadata='Earliest 256 training events per region only; calibration excluded from scored training events.',
        training='Nonoverlapping paired target blocks, maximum 2048; partial last blocks retained and sampled proportional to target count. Short: up to sixteen 64+128 windows. Long: one up-to-1024 + up-to-2048 window. Identical targets per update.',
        eligibility='Puerto Rico excluded: only 400 pre-2019 events. SED excluded: catalog overlaps INGV.',
        caveats=['Fixed Mc is not a completeness certificate.', 'Catalog quality screening changes the observed process.',
                 'Tiny exact time ties are unfolded at 1 microsecond; physical times retained.'],
        protected_before=protected,protected_after={p:sha(p) for p in protected})
    assert report['protected_before']==report['protected_after']
    (REPORTS/'data.json').write_text(json.dumps(report,indent=2))
    print('PREPARED',report['unique_train_scored_events'],flush=True)


if __name__=='__main__':main()
