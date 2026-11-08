"""Build a larger, chronological, separately stored pretraining catalog.

Existing three pilot catalogs and metadata remain byte-for-byte unchanged.
New training-only regions use cached JMA bulletins and screened NCEDC records.
"""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT/'experiments/revision/multiregion_pretraining'
STORE = Path('/root/autodl-fs/em_eqf_training/long_sequence_20261105')
OUTPUT = ROOT/'data/MultiRegion/catalogs/long_sequence_20261105.pt'
BOXES = {
    'japan_west_honshu': (34., 37., 130., 137.),
    'japan_east_honshu': (34., 41., 137., 145.),
    'japan_hokkaido': (41., 46., 139., 146.),
}
PILOT_SHA = 'cfe6ef55a35e12a0b0f0e914401d982b27ebe05fe3844b7b8d142ba3592cd4d5'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new_jma():
    rows = {r: [] for r in BOXES}
    manifests = {}
    for year in range(1998, 2014):
        path = ROOT/'data/JMA/raw'/f'h{year}.zip'
        manifests[str(path)] = sha(path)
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None, path
            for member in archive.infolist():
                if member.is_dir():
                    continue
                for number, line in enumerate(archive.read(member).splitlines(), 1):
                    assert len(line) == 96
                    if (line[:1] != b'J' or line[60:61] != b'1'
                            or line[95:96] not in (b'K', b'k')
                            or line[59:60] not in (b'1', b'2', b'3', b'4', b'5')):
                        continue
                    try:
                        magnitude = int(line[52:54]) / 10
                        if magnitude < 3.:
                            continue
                        latitude = int(line[21:24]) + int(line[24:28])/6000
                        longitude = int(line[32:36]) + int(line[36:40])/6000
                        for region, (lat0, lat1, lon0, lon1) in BOXES.items():
                            if lat0 <= latitude < lat1 and lon0 <= longitude < lon1:
                                rows[region].append((f'JMA:{year}:{member.filename}:{number}',
                                    line[1:13].decode(), int(line[13:17])/100, magnitude,
                                    latitude, longitude, line[54:55].decode()))
                    except ValueError:
                        continue
        print('JMA', year, {r: len(v) for r, v in rows.items()}, flush=True)
    frames = {}
    for region, records in rows.items():
        frame = pd.DataFrame(records, columns=['event_id', 'minute', 'seconds', 'magnitude',
                                               'latitude', 'longitude', 'mag_type'])
        frame['time'] = (pd.to_datetime(frame.pop('minute'), format='%Y%m%d%H%M', utc=True)
                         + pd.to_timedelta(frame.pop('seconds'), unit='s') - pd.Timedelta(hours=9))
        frames[region] = frame
    return frames, manifests


def main():
    sys.path.insert(0, str(ROOT))
    from src.data import Batch, Sequence
    from src.models.tpp.region_mixer_tpp import attach_region
    STORE.mkdir(parents=True, exist_ok=True)
    pilot_path = ROOT/'data/MultiRegion/catalogs/pilot_2007_2019.pt'
    assert sha(pilot_path) == PILOT_SHA
    pilot = json.loads((STUDY/'reports/pilot_data_20261104.json').read_text())
    frames, manifest = new_jma()
    nc_path = ROOT/'data/CatalogExpansion_20261105/NCEDC/processed/quality_screened_20261105.parquet'
    nc = pd.read_parquet(nc_path)
    assert nc.absolute_time_ready.all() and nc.quality_screen_pass.all()
    frames['northern_california'] = nc[nc.time < '2014-01-01'].copy()
    manifest[str(nc_path)] = sha(nc_path)
    for region, spec in pilot['regions'].items():
        path = Path(spec['processed_path'])
        frames[region] = pd.read_csv(path)
        manifest[str(path)] = sha(path)
    vocabulary = {'<unknown>': 0, **{r: i+1 for i, r in enumerate(sorted(frames))}}
    dataset, report = {}, {}
    all_new_ids = set()
    for region, frame in frames.items():
        core = region in pilot['regions']
        frame['time'] = pd.to_datetime(frame.time, utc=True, format='mixed').dt.as_unit('ns')
        if 'event_id' not in frame:
            frame['event_id'] = [f'{region}:{t.value}' for t in frame.time]
        frame.event_id = region + ':' + frame.event_id.astype(str)
        mc = pilot['regions'][region]['mc'] if core else 2.5 if region == 'northern_california' else 3.
        before = len(frame)
        frame = frame.dropna(subset=['time', 'magnitude'])
        frame.magnitude = np.round(frame.magnitude*10)/10
        frame = frame[(frame.magnitude >= mc-1e-8) & (frame.time < '2020-01-01')].copy()
        frame = frame.drop_duplicates('event_id').sort_values(['time', 'magnitude'], ascending=[True, False])
        frame = frame.drop_duplicates('time').sort_values('time').reset_index(drop=True)
        calibration = frame[frame.time < '2014-01-01'].iloc[:256]
        assert len(calibration) == 256
        if core:
            meta = pilot['regions'][region]['metadata']
            assert calibration.time.iloc[-1].isoformat() == pilot['regions'][region]['calibration_end']
            assert len(frame) == pilot['regions'][region]['records']
        else:
            gaps = np.diff(calibration.time.astype('int64').to_numpy())/86400e9
            index_mean = np.mean((calibration.magnitude-mc)/.1)
            b = np.log1p(1/index_mean)/(.1*np.log(10))
            meta = [mc, float(b), float(gaps.mean()), float(gaps.min()), float(gaps.max()), .1,
                    2 if region.startswith('japan') else 0]
        splits = {}
        split_ids = {}
        for split, low, high in [
            ('train', calibration.time.iloc[-1]+pd.Timedelta(nanoseconds=1), pd.Timestamp('2014-01-01', tz='UTC')),
            ('val', pd.Timestamp('2014-01-01', tz='UTC'), pd.Timestamp('2016-01-01', tz='UTC')),
            ('test', pd.Timestamp('2016-01-01', tz='UTC'), pd.Timestamp('2020-01-01', tz='UTC')),
        ]:
            part = frame[(frame.time >= low) & (frame.time < high)]
            if not core and split != 'train':
                assert len(part) == 0
                continue
            timestamps = part.time.astype('int64').to_numpy()
            assert np.all(np.diff(timestamps) > 0)
            times = (timestamps-timestamps[0])/86400e9
            splits[split] = dict(times=times, magnitude=part.magnitude.to_numpy(np.float64),
                                  event_id=part.event_id.to_list())
            split_ids[split] = set(part.event_id)
            assert not (set(calibration.event_id) & split_ids[split])
        for a in split_ids:
            for b in split_ids:
                if a != b:
                    assert not split_ids[a] & split_ids[b]
        if not core:
            assert not all_new_ids & split_ids['train']
            all_new_ids.update(split_ids['train'])
        n = len(splits['train']['times'])
        # Both sequence lengths score exactly this same event universe.
        scored = (n-257)//1024*1024
        assert scored >= 1024
        dataset[region] = dict(metadata=meta, core=core, splits=splits, train_targets=scored)
        histogram = frame[frame.time < '2014-01-01'].magnitude.value_counts().sort_index()
        report[region] = dict(core=core, mc=mc, metadata=meta, calibration_end=calibration.time.iloc[-1].isoformat(),
            records=len(frame), removed_or_tied=before-len(frame), train_events=n, train_scored_events=scored,
            short_windows=scored//128, long_windows=scored//1024,
            val_windows=(len(splits.get('val', {}).get('times', []))-257)//128 if core else 0,
            test_windows=(len(splits.get('test', {}).get('times', []))-257)//128 if core else 0,
            training_magnitude_histogram={str(k): int(v) for k, v in histogram.items()},
            mag_types=frame.mag_type.value_counts().to_dict())
        print(region, json.dumps(report[region]), flush=True)
    # Independent check that reconstruction retains the previous short windows.
    original = torch.load(pilot_path, map_location='cpu', weights_only=False)
    for region in pilot['regions']:
        for split in ['train', 'val', 'test']:
            source = dataset[region]['splits'][split]
            arrival = source['times'][1:193]-source['times'][0]
            inter = np.diff(np.r_[0., arrival, arrival[-1]])
            sequence = Sequence(torch.from_numpy(inter), t_nll_start=float(arrival[63]),
                                mag=torch.from_numpy(source['magnitude'][1:193].copy()))
            expected = original[region][split][0]
            # Original float64 epoch-day subtraction has ~1e-12 day cancellation.
            torch.testing.assert_close(sequence.inter_times, expected.inter_times, rtol=0, atol=1e-11)
            torch.testing.assert_close(sequence.mag, expected.mag, rtol=0, atol=0)
    content = dict(regions=dataset, vocabulary=vocabulary, pilot_sha256=PILOT_SHA,
                   schema='long_sequence_20261105_v1')
    path = STORE/'long_sequence_20261105.pt'
    assert not path.exists(), 'Refuse to overwrite a prepared catalog'
    torch.save(content, path)
    assert not OUTPUT.exists()
    OUTPUT.symlink_to(path)
    summary = dict(dataset=str(OUTPUT), storage=str(path), dataset_sha256=sha(path),
        regions=report, source_sha256=manifest, core_regions=sorted(pilot['regions']),
        split='train before 2014, validation 2014-2015, test 2016-2019; added regions training only',
        selection='Round to 0.1 first, then fixed thresholds: JMA 3.0, NCEDC 2.5; no validation/test threshold selection',
        completeness='Conservative research thresholds; a fixed threshold does not certify catalog completeness.',
        quality='Core unchanged. Added JMA: J, natural 1, analyst K/k, location precision 1-5. NCEDC: expansion quality_screened.',
        common_train_targets='Indices 257 through 257+floor((n-257)/1024)*1024, per region',
        reconstruction_check='First short window of all 9 original region/split combinations matches within 1e-11 days (<1 microsecond).',
        calibration='Earliest 256 events per region; disjoint from all scored and context training windows.',
        core_train_scored_events=sum(v['train_scored_events'] for v in report.values() if v['core']),
        expanded_train_scored_events=sum(v['train_scored_events'] for v in report.values()),
        data_audits='Strict positive gaps; unique IDs; disjoint calibration/train/validation/test; disjoint added Japan boxes')
    target = STUDY/'reports/long_sequence_20261105/data.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2))
    assert sha(pilot_path) == PILOT_SHA
    print('SAVED', target, flush=True)


if __name__ == '__main__':
    main()
