"""Immutable target TPP catalogs and temporally/geographically held-out sources."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch

ROOT = Path('/root/autodl-tmp/em_eqf')
TAG = 'chuandian_scedc_tpp_20261105'
BASE = ROOT / 'experiments/revision/multiregion_pretraining'
STORE = Path('/root/autodl-fs/em_eqf_training') / TAG
REPORTS = BASE / 'shared/reports' / TAG
TARGETS = {
    'chuandian': dict(dataset='ChuanDian', source='data/ChuanDian/raw/chuandian_2021.csv', mc=3.,
                     dates=['2000-01-01', '2015-01-01', '2018-01-01', '2021-05-24']),
    'scedc': dict(dataset='SCEDC', source='data/SCEDC/raw/scsn_eq_only_snapshot_20261102.csv', mc=2.5,
                  dates=['1985-01-01', '2005-01-01', '2014-01-01', '2020-01-01']),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))


def finish_spec(spec, train_end, val_end, test_end):
    spec.update(train_end=train_end, train_used_end=train_end, val_end=val_end, test_end=test_end,
                train_targets=train_end - 256, block_starts=list(range(256, train_end, 2048)))
    spec['block_targets'] = [min(2048, train_end - x) for x in spec['block_starts']]
    assert train_end > 256 and val_end > train_end and test_end >= val_end
    return spec


def main():
    STORE.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    output = STORE / 'targets.pt'
    if output.exists():
        assert sha(output) == json.loads((REPORTS / 'data.json').read_text())['target_sha256']
        print('REUSE_TARGET_DATA', flush=True)
        return
    decisions = json.loads((ROOT / 'experiments/revision/main/reports/completeness/training_mc_decision.json').read_text())
    assert decisions['main_cutoffs']['SWC'] == 3. and decisions['main_cutoffs']['SCSN'] == 2.5
    original_path = ROOT / 'data/MultiRegion/catalogs/region_capacity_20261105.pt'
    original = torch.load(original_path, map_location='cpu', weights_only=False)
    protected = {str(original_path): sha(original_path)}
    regions, audit, source_audit = {}, {}, {}
    for name, cfg in TARGETS.items():
        path = ROOT / cfg['source']
        protected[str(path)] = sha(path)
        if name == 'scedc':
            assert sha(path) == decisions['scsn_source_sha256']
        frame = pd.read_csv(path)
        if name == 'scedc':
            assert frame.ET.eq('eq').all()
        frame['timestamp'] = pd.to_datetime(frame.ts, format='mixed').dt.as_unit('ns')
        frame['event_id'] = (frame.EVID.astype(str) if 'EVID' in frame else frame.index.astype(str))
        frame['latitude'] = frame['latitude'] if 'latitude' in frame else frame.Latitude
        frame['longitude'] = frame['longitude'] if 'longitude' in frame else frame.Longitude
        mc = cfg['mc']
        frame = frame.loc[(frame.timestamp >= cfg['dates'][0]) & (frame.timestamp < cfg['dates'][-1]) &
                          (frame.Magnitude >= mc - 1e-9)].copy()
        assert frame[['Magnitude', 'latitude', 'longitude']].notna().all().all()
        before_dedup = len(frame)
        frame = frame.drop_duplicates('event_id').sort_values(['timestamp', 'event_id'], kind='stable').reset_index(drop=True)
        physical = frame.timestamp.astype('int64').to_numpy()
        model_ns = physical.copy()
        for i in range(1, len(model_ns)):
            model_ns[i] = max(model_ns[i], model_ns[i-1] + 1000)
        assert np.all(np.diff(model_ns) > 0)
        assert int((model_ns - physical).max()) <= 5_000_000
        raw = frame.Magnitude.to_numpy(np.float64)
        magnitude = mc + .1 * np.floor((raw - mc) / .1 + 1e-8)
        gaps = np.diff(model_ns[:256]) / 86400e9
        mean_bin = float(np.mean((magnitude[:256] - mc) / .1))
        b = float(np.log1p(1 / max(mean_bin, 1e-9)) / (.1 * np.log(10)))
        # Both legacy targets mix reported magnitude types: category 0 is
        # unknown in the actual model mapping, not Mw.
        metadata = [mc, b, float(gaps.mean()), float(gaps.min()), float(gaps.max()), .1, 0]
        cuts = [int(np.datetime64(x, 'ns').astype(np.int64)) for x in cfg['dates'][1:]]
        split = np.searchsorted(physical, cuts).tolist()
        spec = dict(metadata=metadata, ns=model_ns, physical_ns=physical, magnitude=magnitude,
            raw_magnitude=raw, event_id=[f'{cfg["dataset"]}:{x}' for x in frame.event_id],
            latitude=frame.latitude.to_numpy(), longitude=frame.longitude.to_numpy(), source=str(path), source_sha256=sha(path))
        finish_spec(spec, *split)
        regions[name] = spec
        audit[name] = dict(mc=mc, dates=cfg['dates'], records=len(frame), dropped_duplicate_ids=before_dedup - len(frame),
            calibration_events=256, train_scored=split[0]-256, val_events=split[1]-split[0], test_events=split[2]-split[1],
            tied_rows=int(pd.Series(physical).duplicated().sum()), max_shift_ns=int((model_ns-physical).max()),
            metadata=metadata, source=str(path), source_sha256=sha(path), calibration_end=str(frame.timestamp.iloc[255]),
            test_event_ids_sha256=hashlib.sha256('\n'.join(spec['event_id'][split[1]:split[2]]).encode()).hexdigest())

        # Exclude NCEDC completely: its source box covers part of SCEDC.
        # A two-day boundary buffer also avoids legacy target timezone ambiguity.
        end_cut = int((pd.Timestamp(cfg['dates'][1]) - pd.Timedelta(days=2)).value)
        train_cut = int((pd.Timestamp(cfg['dates'][1]) - pd.DateOffset(years=2)).value)
        source_regions = {}
        for region, old in original['regions'].items():
            if region == 'northern_california':
                continue
            train_end, high = np.searchsorted(old['physical_ns'], [train_cut, end_cut]).tolist()
            assert train_end > 256 and high > train_end
            value = {k: v for k, v in old.items() if k not in ['ns', 'physical_ns', 'magnitude', 'raw_magnitude', 'event_id']}
            for key in ['ns', 'physical_ns', 'magnitude', 'raw_magnitude', 'event_id']:
                value[key] = old[key][:high]
            finish_spec(value, train_end, high, high)
            assert value['physical_ns'][-1] < end_cut and value['physical_ns'][255] < train_cut
            # Disjoint bounding boxes establish no target-source geographic overlap.
            lat0, lat1, lon0, lon1 = old['box']
            overlap = ((frame.latitude >= lat0) & (frame.latitude < lat1) &
                       (frame.longitude >= lon0) & (frame.longitude < lon1))
            assert not overlap.any(), (name, region)
            source_regions[region] = value
        assert len(source_regions) == 8
        vocabulary = {'<unknown>': 0, **{r: i+1 for i, r in enumerate(sorted(source_regions))}}
        payload = dict(schema=TAG+'_source', regions=source_regions, vocabulary=vocabulary, calibration_events=256, block_targets=2048)
        source_path = STORE / f'sources_before_{name}_training_end.pt'
        torch.save(payload, source_path)
        source_audit[name] = dict(path=str(source_path), sha256=sha(source_path),
            source_train_end=str(pd.Timestamp(train_cut)), source_end_exclusive=str(pd.Timestamp(end_cut)),
            excluded=['northern_california'], regions={r: dict(train=v['train_targets'], validation=v['val_end']-v['train_end'],
                last_time=str(pd.Timestamp(int(v['physical_ns'][-1])))) for r,v in source_regions.items()},
            target_in_sources=False, geographic_overlap=False)
        print('TARGET_PREPARED', name, json.dumps(audit[name]), flush=True)
    payload = dict(schema=TAG, regions=regions, vocabulary={'<unknown>':0, 'chuandian':1, 'scedc':2}, calibration_events=256, block_targets=2048)
    torch.save(payload, output)
    link = ROOT / 'data/MultiRegion/catalogs' / f'{TAG}.pt'
    assert not link.exists()
    link.symlink_to(output)
    assert all(sha(path)==digest for path,digest in protected.items())
    save(REPORTS / 'data.json', dict(target_path=str(output), target_sha256=sha(output), targets=audit, sources=source_audit,
        physical_times='Legacy target timestamps as provided; absolute timezone not newly inferred. Source cutoff has a conservative2-day buffer.',
        marks='0.1 lower-edge bins, >= frozen Mc, upper truncation10; original magnitude retained.',
        calibration='Earliest256 target training events only; excluded from scored training targets; used also by zero-shot normalization.',
        scoring='Event/inter-event likelihood in days; target-event windows128, history1024; original target date partitions.',
        protected_files=protected))
    print('TPP_DATA_READY', flush=True)


if __name__ == '__main__':
    main()
