"""Prepare a separate three-catalog transfer pilot, with past-only metadata.

This pilot does not alter the paper's SWC/SCSN splits. Thresholds are conservative
pilot choices, not newly estimated completeness magnitudes.
"""
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import requests
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import Sequence
from src.models.tpp.region_mixer_tpp import attach_region, REGION_FIELDS

OUT = ROOT / 'experiments/revision/multiregion_pretraining'
SOURCES = {
    'japan_kyushu': dict(provider='JMA', mc=2., kind=2, box=(30., 34., 129., 133.)),
    # Preferred-magnitude catalogs mix several measurement methods: mark as mixed.
    'italy': dict(provider='INGV', mc=2.5, kind=0, box=(35., 48., 6., 19.),
                  endpoint='https://webservices.ingv.it/fdsnws/event/1/query'),
    'new_zealand': dict(provider='GeoNet', mc=3., kind=0, box=(-48., -33., 165., 180.),
                        endpoint='https://service.geonet.org.nz/fdsnws/event/1/query'),
}


def fetch_period(source, start, end):
    config = SOURCES[source]
    lat0, lat1, lon0, lon1 = config['box']
    # INGV accepts UTC clock strings but rejects an explicit +00:00 suffix.
    params = dict(starttime=start.strftime('%Y-%m-%dT%H:%M:%S'), endtime=end.strftime('%Y-%m-%dT%H:%M:%S'),
                  minlatitude=lat0, maxlatitude=lat1, minlongitude=lon0, maxlongitude=lon1,
                  minmagnitude=config['mc']-.001, eventtype='earthquake', format='text')
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    path = ROOT / 'data' / config['provider'] / 'raw/fdsn' / f'{key}.txt'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        text = path.read_text()
    else:
        response = requests.get(config['endpoint'], params=params, timeout=(15, 90))
        if response.status_code == 413:
            middle = start + (end-start)/2
            return fetch_period(source, start, middle) + fetch_period(source, middle, end)
        if response.status_code == 204:
            return []
        if response.status_code == 400:
            raise ValueError(f'{response.url}: {response.text[:600]}')
        response.raise_for_status()
        text = response.text
        if not text.startswith('#EventID'):
            raise ValueError(f'Unexpected event API response: {text[:150]}')
        path.write_text(text)
        path.with_suffix('.json').write_text(json.dumps(dict(url=response.url, bytes=len(response.content)), indent=2))
    frame = pd.read_csv(io.StringIO(text), sep='|')
    frame.columns = frame.columns.str.strip()
    return [frame.rename(columns={'Time':'time', 'Magnitude':'magnitude',
                                  'Latitude':'latitude', 'Longitude':'longitude',
                                  'MagType':'mag_type', '#EventID':'event_id'})]


def read_jma():
    rows = []
    for year in range(2007, 2021):
        path = ROOT / 'data/JMA/raw' / f'h{year}.zip'
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                for record in archive.read(member).splitlines():
                    if len(record) != 96 or record[:1] != b'J' or record[60:61] != b'1':
                        continue
                    # Keep analyst-reviewed hypocenters for this initial pilot.
                    if record[95:96] not in (b'K', b'k', b'S', b's'):
                        continue
                    try:
                        mag = int(record[52:54]) / 10
                        if mag < 2.:
                            continue
                        lat = int(record[21:24]) + int(record[24:28])/6000
                        lon = int(record[32:36]) + int(record[36:40])/6000
                        if not (30 <= lat < 34 and 129 <= lon < 133):
                            continue
                        rows.append((record[1:13].decode(), int(record[13:17])/100,
                                     mag, lat, lon, record[54:55].decode()))
                    except ValueError:
                        continue
    frame = pd.DataFrame(rows, columns=['minute', 'seconds', 'magnitude', 'latitude', 'longitude', 'mag_type'])
    frame['time'] = (pd.to_datetime(frame.pop('minute'), format='%Y%m%d%H%M', utc=True)
                     + pd.to_timedelta(frame.pop('seconds'), unit='s') - pd.Timedelta(hours=9))
    return frame


def make_windows(frame, start, end, metadata, context=64, targets=128):
    selected = frame[(frame.time >= start) & (frame.time < end)]
    if len(selected) < context + targets:
        return []
    times = pd.DatetimeIndex(selected.time).as_unit('ns').asi8.astype(np.float64) / 86400e9
    magnitudes = selected.magnitude.to_numpy(np.float64)
    sequences = []
    for left in range(1, len(times)-context-targets+1, targets):
        right = left + context + targets
        arrival = times[left:right] - times[left-1]
        inter = np.diff(np.r_[0., arrival, arrival[-1]])
        seq = Sequence(torch.from_numpy(inter), t_nll_start=float(arrival[context-1]),
                       mag=torch.from_numpy(magnitudes[left:right].copy()))
        sequences.append(attach_region(seq, metadata))
    return sequences


def main():
    (OUT/'shared/reports').mkdir(parents=True, exist_ok=True)
    raw_frames = {'japan_kyushu': read_jma()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {(name, year): pool.submit(fetch_period, name, pd.Timestamp(f'{year}-01-01', tz='UTC'),
                                         pd.Timestamp(f'{year+1}-01-01', tz='UTC'))
                for name in ['italy', 'new_zealand'] for year in range(2007, 2020)}
        for name in ['italy', 'new_zealand']:
            raw_frames[name] = pd.concat([f for (n,y),job in jobs.items() if n == name for f in job.result()], ignore_index=True)
    data, report = {}, {'region_fields': REGION_FIELDS, 'regions': {},
                        'scope':'external-region pilot only; SWC/SCSN paper data unchanged',
                        'calibration_events':256, 'context_events':64, 'scored_events_per_window':128,
                        'threshold_note':'fixed conservative pilot thresholds, completeness audit pending',
                        'ipoc_note':'not used until its absolute-time convention is confirmed'}
    for name, frame in raw_frames.items():
        cfg = SOURCES[name]
        frame.time = pd.to_datetime(frame.time, utc=True)
        frame.magnitude = pd.to_numeric(frame.magnitude, errors='coerce')
        frame = frame.dropna(subset=['time', 'magnitude'])
        frame = frame[(frame.time >= '2007-01-01') & (frame.time < '2020-01-01') & (frame.magnitude >= cfg['mc']-1e-6)]
        # Common 0.1 reporting grid for a controlled initial likelihood comparison.
        frame['magnitude'] = np.round(frame.magnitude * 10) / 10
        if 'event_id' in frame:
            frame = frame.drop_duplicates('event_id')
        before = len(frame)
        frame = frame.sort_values(['time', 'magnitude'], ascending=[True, False]).drop_duplicates('time').sort_values('time').reset_index(drop=True)
        calibration = frame[frame.time < '2014-01-01'].iloc[:256]
        if len(calibration) != 256:
            raise ValueError(f'{name}: insufficient past calibration events')
        gaps = np.diff(pd.DatetimeIndex(calibration.time).as_unit('ns').asi8.astype(np.float64)) / 86400e9
        mean_index = np.mean((calibration.magnitude - cfg['mc'])/.1)
        b = np.log1p(1/max(mean_index, 1e-9)) / (.1 * np.log(10))
        metadata = [cfg['mc'], float(b), float(gaps.mean()), float(gaps.min()), float(gaps.max()), .1, cfg['kind']]
        train_start = calibration.time.iloc[-1] + pd.Timedelta(nanoseconds=1)
        data[name] = {split: make_windows(frame, start, end, metadata) for split,start,end in [
            ('train', train_start, pd.Timestamp('2014-01-01', tz='UTC')),
            ('val', pd.Timestamp('2014-01-01', tz='UTC'), pd.Timestamp('2016-01-01', tz='UTC')),
            ('test', pd.Timestamp('2016-01-01', tz='UTC'), pd.Timestamp('2020-01-01', tz='UTC'))]}
        path = ROOT/'data'/cfg['provider']/'processed'/f'multiregion_{name}_mc{cfg["mc"]}_2007_2019.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        report['regions'][name] = dict(records=len(frame), removed_tied_times=before-len(frame),
            mc=cfg['mc'], metadata=metadata, calibration_end=calibration.time.iloc[-1].isoformat(),
            windows={s:len(v) for s,v in data[name].items()}, processed_path=str(path),
            mag_types=frame.mag_type.value_counts().to_dict())
        print(json.dumps({name:report['regions'][name]}), flush=True)
    path = ROOT/'data/MultiRegion/catalogs/pilot_2007_2019.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    report['dataset_path'] = str(path)
    report['dataset_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/'reports/pilot_data_20261104.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('saved', path, flush=True)


if __name__ == '__main__':
    main()
