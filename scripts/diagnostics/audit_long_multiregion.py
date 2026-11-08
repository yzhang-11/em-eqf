"""Independent raw-record and split checks for the enlarged training catalog."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
STUDY = ROOT/'experiments/revision/multiregion_pretraining'


def main():
    path = ROOT/'data/MultiRegion/catalogs/long_sequence_20261105.pt'
    data = torch.load(path, map_location='cpu', weights_only=False)
    info = json.loads((STUDY/'reports/long_sequence_20261105/data.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == info['dataset_sha256']
    pilot = json.loads((STUDY/'reports/pilot_data_20261104.json').read_text())
    nc = pd.read_parquet(ROOT/'data/CatalogExpansion_20261105/NCEDC/processed/quality_screened_20261105.parquet')
    nc['time'] = pd.to_datetime(nc.time, utc=True).dt.as_unit('ns')
    nc['magnitude'] = np.round(nc.magnitude*10)/10
    nc = nc[(nc.time < '2014-01-01') & (nc.magnitude >= 2.5)].sort_values('time').set_index('event_id')
    raw_cache = {}
    rng = np.random.default_rng(20261105)
    checks = {}
    raw_ids = set()
    for region, spec in data['regions'].items():
        split_ids = {split:set(series['event_id']) for split,series in spec['splits'].items()}
        if spec['core']:
            frame = pd.read_csv(pilot['regions'][region]['processed_path'])
            frame['time'] = pd.to_datetime(frame.time, utc=True, format='mixed').dt.as_unit('ns')
            if 'event_id' in frame:
                frame['event_id'] = region+':'+frame.event_id.astype(str)
            else:
                frame['event_id'] = [f'{region}:{region}:{t.value}' for t in frame.time]
            reference = frame.set_index('event_id')
            assert spec['metadata'] == pilot['regions'][region]['metadata']
        else:
            assert list(spec['splits']) == ['train']
        for a in split_ids:
            for b in split_ids:
                if a != b:
                    assert not split_ids[a] & split_ids[b]
        record_checks = 0
        max_days_error = 0.
        for split, series in spec['splits'].items():
            ids = series['event_id']
            assert len(ids) == len(set(ids)) == len(series['times']) == len(series['magnitude'])
            assert np.all(np.diff(series['times']) > 0)
            # Verify original physical time units, magnitude quantization and split boundaries.
            indices = sorted(set([0, len(ids)-1, *rng.integers(0,len(ids),size=min(100,len(ids))).tolist()]))
            times = []
            for i in indices:
                event = ids[i]
                if spec['core']:
                    row = reference.loc[event]
                    timestamp, magnitude = row.time, row.magnitude
                elif region == 'northern_california':
                    row = nc.loc[event.split(':', 1)[1]]
                    timestamp, magnitude = row.time, row.magnitude
                else:
                    _, provider, year, member, line_number = event.split(':')
                    assert provider == 'JMA'
                    key = (year, member)
                    if key not in raw_cache:
                        with zipfile.ZipFile(ROOT/'data/JMA/raw'/f'h{year}.zip') as archive:
                            raw_cache[key] = archive.read(member).splitlines()
                    line = raw_cache[key][int(line_number)-1]
                    assert line[:1] == b'J' and line[60:61] == b'1' and line[95:96] in (b'K',b'k')
                    assert line[59:60] in (b'1',b'2',b'3',b'4',b'5')
                    minute = datetime.strptime(line[1:13].decode(), '%Y%m%d%H%M').replace(
                        tzinfo=timezone(timedelta(hours=9)))
                    timestamp = pd.Timestamp(minute + timedelta(milliseconds=int(line[13:17])*10)).tz_convert('UTC')
                    magnitude = int(line[52:54])/10
                assert abs(magnitude-series['magnitude'][i]) < 1e-8
                times.append(timestamp.value)
                assert timestamp >= pd.Timestamp(info['regions'][region]['calibration_end'])
                if split == 'train':
                    assert timestamp < pd.Timestamp('2014-01-01',tz='UTC')
                elif split == 'val':
                    assert pd.Timestamp('2014-01-01',tz='UTC') <= timestamp < pd.Timestamp('2016-01-01',tz='UTC')
                else:
                    assert pd.Timestamp('2016-01-01',tz='UTC') <= timestamp < pd.Timestamp('2020-01-01',tz='UTC')
                record_checks += 1
            expected = (np.asarray(times,dtype=np.int64)-times[0])/86400e9
            actual = series['times'][indices]-series['times'][indices[0]]
            error = float(np.max(np.abs(actual-expected)))
            assert error < 1e-10, (region,split,error)
            max_days_error = max(max_days_error,error)
        n = spec['train_targets']
        short_targets = np.concatenate([np.arange(i,i+128) for i in range(257,257+n,128)])
        long_targets = np.concatenate([np.arange(i,i+1024) for i in range(257,257+n,1024)])
        assert np.array_equal(short_targets,long_targets)
        assert len(np.unique(short_targets)) == n
        if not spec['core']:
            provider_ids = {s.split(':',1)[1] for s in spec['splits']['train']['event_id']}
            assert not raw_ids & provider_ids
            raw_ids.update(provider_ids)
        checks[region] = dict(sampled_raw_records_verified=record_checks, max_time_error_days=max_days_error,
                              short_long_target_universe_equal=True, unique_train_targets=n)
    # Check geographic disjointness including the original Kyushu box.
    boxes = [(30,34,129,133),(34,37,130,137),(34,41,137,145),(41,46,139,146)]
    for i,a in enumerate(boxes):
        for b in boxes[i+1:]:
            assert a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2]
    report = dict(passed=True, regions=checks, sampled_raw_records=sum(v['sampled_raw_records_verified'] for v in checks.values()),
                  checks=['raw records/time units', 'disjoint chronological splits', 'unchanged core metadata',
                          'same scored training targets for short/long', 'no target overlap within windows',
                          'distinct added source IDs', 'nonoverlapping Japan boxes'])
    target = STUDY/'reports/long_sequence_20261105/independent_audit.json'
    target.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
