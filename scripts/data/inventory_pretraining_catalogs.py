"""Download JMA annual event archives and audit public pretraining catalog sizes.

Raw files stay in data/<provider>/raw; reports stay in experiments/revision.
This does not build training splits, estimate completeness, or change models.
"""
import argparse
import collections
import concurrent.futures
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import time
import zipfile
import zlib

import requests

JMA = 'https://www.data.jma.go.jp/eqev/data/bulletin/data/hypo/h{year}.zip'
THRESHOLDS = (1.0, 1.5, 2.0, 2.5, 3.0)
STAMP = '20261104'


def get(url, **kwargs):
    for attempt in range(3):
        try:
            return requests.get(url, timeout=(15, 90), **kwargs)
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def magnitude(raw):
    value = raw.decode('ascii').strip()
    if not value:
        return None
    # JMA fixed-width negative magnitudes: -1 -> -0.1, A9 -> -1.9.
    if value[0] in 'ABC':
        return -(ord(value[0]) - ord('A') + 1 + int(value[1]) / 10)
    return int(value) / 10


def jma_year(root, year):
    url = JMA.format(year=year)
    dest = root / 'data/JMA/raw' / f'h{year}.zip'
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        response = get(url)
        response.raise_for_status()
        # Validate before publishing a completed raw file.
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            if archive.testzip() is not None:
                raise ValueError('corrupt JMA zip')
        temp = dest.with_suffix('.zip.part')
        temp.write_bytes(response.content)
        temp.replace(dest)
    result = dict(provider='JMA', year=year, url=url, path=str(dest),
                  bytes=dest.stat().st_size,
                  sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                  raw_records=0, jma_natural_records=0,
                  natural_with_magnitude=0, pre2014_natural_with_magnitude=0)
    agencies, types, magtypes, lengths = (collections.Counter() for _ in range(4))
    thresholds = collections.Counter()
    pre_thresholds = collections.Counter()
    with zipfile.ZipFile(dest) as archive:
        for name in archive.namelist():
            if name.endswith('/'):
                continue
            for line in archive.read(name).splitlines():
                result['raw_records'] += 1
                lengths[len(line)] += 1
                if len(line) != 96:
                    raise ValueError(f'JMA record length {len(line)} in {dest}')
                agency, event_type = chr(line[0]), chr(line[60])
                agencies[agency] += 1
                types[event_type] += 1
                if agency != 'J' or event_type != '1':
                    continue
                result['jma_natural_records'] += 1
                mag = magnitude(line[52:54])
                if mag is None:
                    continue
                magtypes[chr(line[54])] += 1
                result['natural_with_magnitude'] += 1
                # JMA timestamps are JST. 2014-01-01 00:00 UTC = 09:00 JST.
                before = line[1:13] < b'201401010900'
                result['pre2014_natural_with_magnitude'] += int(before)
                for mc in THRESHOLDS:
                    if mag >= mc:
                        thresholds[str(mc)] += 1
                        pre_thresholds[str(mc)] += int(before)
    result.update(agencies=dict(agencies), event_types=dict(types),
                  magnitude_types=dict(magtypes), record_lengths=dict(lengths),
                  counts_ge_magnitude=dict(thresholds),
                  pre2014_counts_ge_magnitude=dict(pre_thresholds))
    print(json.dumps({'JMA': year, 'raw': result['raw_records'],
                      'natural_mag': result['natural_with_magnitude']}), flush=True)
    return result


def query_count(provider, endpoint, start, end, extra=None):
    params = dict(starttime=start, endtime=end, format='text')
    params.update(extra or {})
    response = get(endpoint, params=params)
    result = dict(provider=provider, url=response.url, status=response.status_code,
                  starttime=start, endtime=end)
    if response.status_code == 413:
        match = re.search(r'found\s+(\d+)\s+events|requested events is\s+"(\d+)"', response.text)
        if match:
            result.update(raw_records=int(next(x for x in match.groups() if x)),
                          count_basis='server-declared result size; event records not downloaded')
    elif response.ok and response.text.strip().isdigit():
        result.update(raw_records=int(response.text.strip()), count_basis='count API')
    elif response.ok and response.text.startswith('#EventID'):
        result.update(raw_records=len(response.text.splitlines()) - 1,
                      count_basis='downloaded text response')
    result['response_excerpt'] = response.text[:1200]
    print(json.dumps({provider: result.get('raw_records'), 'status': response.status_code}), flush=True)
    return result


class RemoteZip(io.BytesIO):
    """Seek only the public ZIP metadata, without fetching phase arrivals."""
    def __init__(self, url):
        super().__init__()
        self.url = url
        response = requests.head(url, timeout=30)
        response.raise_for_status()
        self.length = int(response.headers['Content-Length'])
        self.position = 0

    def seek(self, offset, whence=0):
        self.position = (offset if whence == 0 else self.position + offset
                         if whence == 1 else self.length + offset)
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = self.length - self.position
        if size <= 0:
            return b''
        response = get(self.url, headers={'Range':
                       f'bytes={self.position}-{min(self.length - 1, self.position + size - 1)}'})
        if response.status_code != 206:
            raise ValueError('ZIP server did not honor byte range')
        self.position += len(response.content)
        return response.content


def fetch_ipoc(root):
    import pandas as pd
    url = ('https://datapub.gfz.de/download/10.5880.FIDGEO.2026.050-Earsgdfk/'
           '2026-050_Najafipour-et-al_data.zip')
    remote = RemoteZip(url)
    with zipfile.ZipFile(remote) as archive:
        info = next(i for i in archive.infolist() if i.filename.endswith('hypocenters.csv'))
    if info.compress_type != zipfile.ZIP_DEFLATED:
        raise ValueError('unexpected ZIP compression')
    dest = root / 'data/IPOCDeep/raw' / Path(info.filename).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        remote.seek(info.header_offset)
        header = remote.read(30)
        if header[:4] != b'PK\x03\x04':
            raise ValueError('invalid local ZIP header')
        begin = info.header_offset + 30 + int.from_bytes(header[26:28], 'little') + int.from_bytes(header[28:30], 'little')
        temp = dest.with_suffix('.csv.part')
        decoder = zlib.decompressobj(-15)
        checksum = size = compressed = 0
        # Fetch a bounded number of large ranges concurrently; never fetch phases.
        block = (info.compress_size + 7) // 8

        def read_range(offset):
            end = min(begin + info.compress_size, offset + block)
            response = get(url, headers={'Range': f'bytes={offset}-{end - 1}'})
            if response.status_code != 206 or len(response.content) != end - offset:
                raise ValueError('incomplete CSV byte range')
            print(json.dumps({'IPOC_compressed_block': offset - begin,
                              'bytes': len(response.content)}), flush=True)
            return response.content

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool, temp.open('wb') as output:
            jobs = [pool.submit(read_range, offset)
                    for offset in range(begin, begin + info.compress_size, block)]
            for job in jobs:
                chunk = job.result()
                compressed += len(chunk)
                data = decoder.decompress(chunk)
                output.write(data)
                checksum = zlib.crc32(data, checksum)
                size += len(data)
            tail = decoder.flush()
            output.write(tail)
            checksum = zlib.crc32(tail, checksum)
            size += len(tail)
        if not decoder.eof or size != info.file_size or compressed != info.compress_size or checksum != info.CRC:
            raise ValueError('IPOC CSV ZIP integrity check failed')
        temp.replace(dest)
    result = dict(provider='IPOCDeep', url=url, zip_member=info.filename,
                  path=str(dest), bytes=dest.stat().st_size, raw_records=0,
                  valid_magnitude=0, pre2014_valid_magnitude=0,
                  counts_ge_magnitude={str(mc): 0 for mc in THRESHOLDS},
                  pre2014_counts_ge_magnitude={str(mc): 0 for mc in THRESHOLDS},
                  sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    for frame in pd.read_csv(dest, skiprows=3, chunksize=200000):
        mag = pd.to_numeric(frame['magnitude'], errors='coerce')
        before = frame['time'] < '2014-01-01'
        result['raw_records'] += len(frame)
        result['valid_magnitude'] += int(mag.notna().sum())
        result['pre2014_valid_magnitude'] += int((before & mag.notna()).sum())
        for mc in THRESHOLDS:
            result['counts_ge_magnitude'][str(mc)] += int((mag >= mc).sum())
            result['pre2014_counts_ge_magnitude'][str(mc)] += int((before & (mag >= mc)).sum())
    print(json.dumps({'IPOCDeep': result['raw_records']}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/em_eqf'))
    parser.add_argument('--start-year', type=int, default=1998)
    parser.add_argument('--end-year', type=int, default=2024)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--jma-only', action='store_true')
    parser.add_argument('--fetch-ipoc', action='store_true')
    args = parser.parse_args()
    reports = args.root / 'experiments/revision/multiregion_pretraining/reports'
    reports.mkdir(parents=True, exist_ok=True)
    rows, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(jma_year, args.root, y): y
                for y in range(args.start_year, args.end_year + 1)}
        for future in concurrent.futures.as_completed(jobs):
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append({'provider': 'JMA', 'year': jobs[future], 'error': repr(exc)})
    rows.sort(key=lambda r: r['year'])
    total = {k: sum(r[k] for r in rows) for k in
             ['raw_records', 'jma_natural_records', 'natural_with_magnitude',
              'pre2014_natural_with_magnitude', 'bytes']}
    for key in ['counts_ge_magnitude', 'pre2014_counts_ge_magnitude']:
        total[key] = {str(mc): sum(r[key].get(str(mc), 0) for r in rows) for mc in THRESHOLDS}
    audit = dict(snapshot_date='2026-11-04', jma_years=rows, jma_total=total,
                 errors=errors, api_counts=[])
    if not args.jma_only:
        endpoints = [
            ('GeoNet', 'https://service.geonet.org.nz/fdsnws/event/1/query', {}),
            ('INGV_ISIDe', 'https://webservices.ingv.it/fdsnws/event/1/query',
             {'eventtype': 'earthquake'}),
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            jobs = [pool.submit(query_count, name, url, '1985-01-01T00:00:00', end, extra)
                    for name, url, extra in endpoints
                    for end in ['2014-01-01T00:00:00', '2026-01-01T00:00:00']]
            for future in concurrent.futures.as_completed(jobs):
                try:
                    audit['api_counts'].append(future.result())
                except Exception as exc:
                    errors.append({'provider': 'api_count', 'error': repr(exc)})
    if args.fetch_ipoc:
        audit['ipoc_deep'] = fetch_ipoc(args.root)
    dest = reports / f'data_inventory_{STAMP}.json'
    dest.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    flat = [{k: r[k] for k in ['year', 'raw_records', 'jma_natural_records',
                              'natural_with_magnitude', 'pre2014_natural_with_magnitude',
                              'bytes', 'url', 'sha256']} for r in rows]
    with (reports / f'jma_annual_inventory_{STAMP}.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]) if flat else ['year'])
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({'jma_total': total, 'errors': errors, 'report': str(dest)}), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
