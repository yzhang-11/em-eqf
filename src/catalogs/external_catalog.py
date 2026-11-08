"""Standard Catalog adapters for the versioned public-earthquake snapshots.

The source Parquet retains physical timestamps and provenance. Cached sequences
use the project's deterministic, bounded numerical treatment of reported ties.
Only training events fit normalization and reference statistics.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import Catalog, Sequence, TppDataset, default_catalogs_dir
from src.data.preprocessing import resolve_tied_times
from src.utils.catalog_pathing import build_hashed_catalog_root
from src.utils.catalog_utils import train_val_test_split_sequence


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def utc_naive(value):
    ts = pd.Timestamp(value)
    return ts.tz_localize('UTC').tz_localize(None) if ts.tz is None else ts.tz_convert('UTC').tz_localize(None)


class ExternalCatalog(Catalog):
    dataset_name = None
    provider = None
    default_start = '1985-01-01'
    default_end = '2026-01-01'
    default_mc = 2.5
    source_floor = 2.5
    magnitude_resolution = 0.1
    time_resolution_seconds = 0.000001
    schema = 'public-standard-20261105-v1'

    def __init__(self, root_dir=None, data_dir=None, catalog_file=None,
                 mag_completeness=None, normalize=True, start_ts=None, end_ts=None,
                 train_start_ts=None, val_start_ts='2014-01-01', test_start_ts='2016-01-01'):
        if self.dataset_name is None:
            raise TypeError('Use a registered concrete external catalog')
        ds = Path(data_dir) if data_dir is not None else default_catalogs_dir/self.dataset_name
        self.catalog_file = (Path(catalog_file) if catalog_file is not None else
                             ds/'processed/standard_source_20261105.parquet').resolve()
        if not self.catalog_file.exists():
            raise FileNotFoundError(f'{self.catalog_file}: run scripts/data/prepare_external_standard.py first')
        self.normalize_enabled = bool(normalize)
        self.mc = float(self.default_mc if mag_completeness is None else mag_completeness)
        if not np.isfinite(self.mc) or self.mc < self.source_floor:
            raise ValueError(f'Mc must be finite and >= the source screening floor {self.source_floor}')
        self.start = utc_naive(start_ts or self.default_start)
        self.end = utc_naive(end_ts or self.default_end)
        self.train_start = utc_naive(train_start_ts) if train_start_ts is not None else self.start
        self.val_start, self.test_start = utc_naive(val_start_ts), utc_naive(test_start_ts)
        if not self.start <= self.train_start < self.val_start < self.test_start < self.end:
            raise ValueError('Expected start <= train_start < validation < test < end')
        cfg = dict(dataset=self.dataset_name, schema=self.schema, source_sha256=file_sha256(self.catalog_file),
                   normalize=self.normalize_enabled, mag_completeness=self.mc, threshold_operator='>=',
                   start_ts=self.start.isoformat(), end_ts=self.end.isoformat(),
                   train_start_ts=self.train_start.isoformat(), val_start_ts=self.val_start.isoformat(),
                   test_start_ts=self.test_start.isoformat(), time_processing='stable-half-resolution-f64-v1',
                   timestamp_resolution_seconds=self.time_resolution_seconds,
                   normalization_scope='training-scored-events-only')
        cache = build_hashed_catalog_root(root_dir or ds/'catalogs/standard_20261105', cfg, migrate_legacy=False)
        metadata = dict(name=self.dataset_name, provider=self.provider, freq='1D', time_zone='UTC',
                        schema=self.schema, mag_completeness=self.mc, mag_roundoff_error=self.magnitude_resolution,
                        start_ts=self.start, end_ts=self.end, train_start_ts=self.train_start,
                        val_start_ts=self.val_start, test_start_ts=self.test_start,
                        source_sha256=cfg['source_sha256'], time_processing=cfg['time_processing'],
                        normalization_scope=cfg['normalization_scope'],
                        completeness_note='Fixed research cutoff; not a certified completeness magnitude.')
        super().__init__(root_dir=cache, metadata=metadata)
        self.full_sequence = TppDataset.load_from_disk(self.root_dir/'full_sequence.pt')[0]
        self.norm_stats = torch.load(self.root_dir/'norm_stats.pt', weights_only=False)
        self._split_datasets()
        self.training_statistics = json.loads((self.root_dir/'training_statistics.json').read_text())

    @property
    def required_files(self):
        return ['full_sequence.pt','metadata.pt','norm_stats.pt','catalog.csv','catalog.parquet',
                'training_statistics.json','processing_report.json']

    def _split_datasets(self):
        parts = train_val_test_split_sequence(self.full_sequence, self.start, self.train_start,
                                              self.val_start, self.test_start, half_open=True)
        self.train, self.val, self.test = [TppDataset([part]) for part in parts]

    def _load_frame(self):
        f = pd.read_parquet(self.catalog_file)
        required = {'time','magnitude','latitude','longitude','depth_km','event_id',
                    'provider','absolute_time_ready','quality_screen_pass'}
        if required-set(f):
            raise ValueError(f'Missing columns: {sorted(required-set(f))}')
        if not f.absolute_time_ready.eq(True).all():
            raise ValueError('Source absolute time is unconfirmed; IPOCDeep remains staged')
        if not f.quality_screen_pass.eq(True).all() or not f.provider.eq(self.provider).all():
            raise ValueError('Expected a single-provider quality-screened source')
        f['ts'] = pd.to_datetime(f.time, format='mixed', utc=True, errors='raise').dt.tz_convert(None).dt.as_unit('ns')
        for key in ['magnitude','latitude','longitude','depth_km']:
            f[key] = pd.to_numeric(f[key],errors='raise')
        f = f[(f.ts >= self.start)&(f.ts < self.end)&(f.magnitude >= self.mc)].copy()
        if not len(f) or not np.isfinite(f[['magnitude','latitude','longitude','depth_km']]).all().all():
            raise ValueError('Empty catalog or nonfinite physical fields')
        if f.event_id.isna().any() or f.event_id.astype(str).str.len().eq(0).any():
            raise ValueError('Event IDs must be nonempty')
        if f.event_id.duplicated().any():
            # Conflicting repeated IDs must be investigated, not silently deduplicated.
            raise ValueError('Duplicate event IDs in screened snapshot')
        f = f.sort_values(['ts','event_id'],kind='stable').reset_index(drop=True)
        for lo,hi in [(self.train_start,self.val_start),(self.val_start,self.test_start),(self.test_start,self.end)]:
            if not ((f.ts >= lo)&(f.ts < hi)).any():
                raise ValueError(f'No scored events in chronological interval [{lo}, {hi})')
        return f

    def generate_catalog(self):
        f = self._load_frame()
        raw_times = (f.ts.astype('int64').to_numpy()-self.start.value)/86400e9
        adjusted = raw_times.copy()
        if raw_times[0] <= 0:
            raise ValueError('Event exactly on observation origin; choose start_ts earlier than the first event')
        # Resolve each physical split separately so dequantization cannot move an
        # event into the next scored period. The CSV keeps its actual timestamp.
        for lo,hi in [(self.start,self.val_start),(self.val_start,self.test_start),(self.test_start,self.end)]:
            mask=((f.ts >= lo)&(f.ts < hi)).to_numpy()
            if mask.any():
                adjusted[mask]=resolve_tied_times(raw_times[mask],self.time_resolution_seconds/86400,
                                                 (hi-self.start)/pd.Timedelta(days=1))
        train=((f.ts >= self.train_start)&(f.ts < self.val_start)).to_numpy()
        if int(train.sum()) < 2:
            raise ValueError('Need at least two scored training events')
        norm_stats, fields = {}, {}
        for key,source in [('latitude','latitude'),('longitude','longitude'),('depth','depth_km')]:
            values=f[source].to_numpy(np.float64)
            low,high=float(values[train].min()),float(values[train].max())
            norm_stats[key]=dict(min=low,max=high)
            fields[key]=(values-low)/max(high-low,1e-8) if self.normalize_enabled else values
        torch.save(norm_stats,self.root_dir/'norm_stats.pt')
        inter=np.diff(adjusted,prepend=0.,append=(self.end-self.start)/pd.Timedelta(days=1))
        seq=Sequence(inter_times=torch.tensor(inter,dtype=torch.float64),
                     mag=torch.tensor(f.magnitude.to_numpy(),dtype=torch.float32),
                     loc=torch.tensor(np.stack([fields['latitude'],fields['longitude']],axis=-1),dtype=torch.float32),
                     depth=torch.tensor(fields['depth'],dtype=torch.float32),
                     event_index=torch.arange(len(f),dtype=torch.int64))
        TppDataset([seq]).save_to_disk(self.root_dir/'full_sequence.pt')
        f['depth']=f.depth_km
        f['event_index']=np.arange(len(f))
        f['t']=adjusted
        f['dt']=np.diff(adjusted,prepend=0.)
        f['time_adjustment_seconds']=(adjusted-raw_times)*86400
        f.to_parquet(self.root_dir/'catalog.parquet',index=False,compression='zstd')
        # This is also consumable by the existing recast/supervised CSV reader.
        f.to_csv(self.root_dir/'catalog.csv',index=False)
        gaps=np.diff(raw_times[train])
        mean_mag=float(f.loc[train,'magnitude'].mean())
        denominator=mean_mag-self.mc+.5*self.magnitude_resolution
        stats=dict(scope='training-scored-events-only',events=int(train.sum()),
                   first_time=str(f.loc[train,'ts'].iloc[0]),last_time=str(f.loc[train,'ts'].iloc[-1]),
                   magnitude_mean=mean_mag,richter_b_reference=math.log10(math.e)/denominator if denominator>0 else None,
                   positive_gap_mean_days=float(gaps[gaps>0].mean()),
                   magnitude_types=f.loc[train,'mag_type'].value_counts().to_dict() if 'mag_type' in f else {},
                   magnitude_histogram=f.loc[train,'magnitude'].round(1).value_counts().sort_index().to_dict(),
                   caveat='Reference b and histogram reflect the selected source/magnitude mix; not a validated Mc.')
        report=dict(dataset=self.dataset_name,source=str(self.catalog_file),source_sha256=self.metadata['source_sha256'],
                    records=len(f),scored_events={name:int(((f.ts>=lo)&(f.ts<hi)).sum()) for name,lo,hi in
                        [('train',self.train_start,self.val_start),('val',self.val_start,self.test_start),('test',self.test_start,self.end)]},
                    tied_rows_retained=int(f.ts.duplicated().sum()),
                    adjusted_rows=int(np.count_nonzero(adjusted!=raw_times)),
                    max_adjustment_seconds=float(f.time_adjustment_seconds.abs().max()),
                    physical_time_preserved=True, normalisation='train only',
                    split_note='Validation/test sequences include history; only their disjoint NLL periods are scored.')
        for name,value in [('training_statistics.json',stats),('processing_report.json',report)]:
            (self.root_dir/name).write_text(json.dumps(value,indent=2,default=str),encoding='utf-8')


class JMARegion(ExternalCatalog):
    provider='JMA'
    default_start='1997-12-31T15:00:00'
    default_end='2024-12-31T15:00:00'
    default_mc=3.0
    source_floor=2.0
    time_resolution_seconds=.01


@Catalog.register('JMA_Kyushu-Standard')
class JMAKyushu(JMARegion):
    dataset_name='JMA_Kyushu'


@Catalog.register('JMA_WestHonshu-Standard')
class JMAWestHonshu(JMARegion):
    dataset_name='JMA_WestHonshu'


@Catalog.register('JMA_EastHonshu-Standard')
class JMAEastHonshu(JMARegion):
    dataset_name='JMA_EastHonshu'


@Catalog.register('JMA_Hokkaido-Standard')
class JMAHokkaido(JMARegion):
    dataset_name='JMA_Hokkaido'


@Catalog.register('GeoNet-Standard')
class GeoNetStandard(ExternalCatalog):
    dataset_name='GeoNet'
    provider='GeoNet'
    default_mc=3.0
    source_floor=3.0
    magnitude_resolution=0.0
    time_resolution_seconds=.001


@Catalog.register('INGV-Standard')
class INGVStandard(ExternalCatalog):
    dataset_name='INGV'
    provider='INGV'


@Catalog.register('NCEDC-Standard')
class NCEDCStandard(ExternalCatalog):
    dataset_name='NCEDC'
    provider='NCEDC'
    source_floor=2.0
    magnitude_resolution=.01
    time_resolution_seconds=.001

