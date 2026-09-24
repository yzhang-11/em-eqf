# em_eqf

PyTorch framework for earthquake-catalog forecasting. It supports window-based
classification/regression, event-sequence temporal point processes (TPPs),
checkpoint workflows, Optuna search, sampling, and forecast evaluation.

## Installation

Python 3.10+ is required. CUDA is recommended for training.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
~~~

Install causal_conv1d, mamba_ssm, and/or flash_attn only when the selected
configuration uses them, with versions compatible with PyTorch and CUDA.

## Quick start

main.py is the common entry point. The CLI model name must match the model key
in YAML. Without --config, config/<model>.yaml is used.

~~~bash
python main.py --model mixer_tpp --mode train --config config/mixer_tpp.yaml \
  --checkpoint_dir checkpoints/mixer_tpp_demo
python main.py --model mixer_tpp --mode test \
  --checkpoint_dir checkpoints/mixer_tpp_demo --ckpt_select best
python main.py --model mixer_tpp --mode optuna \
  --config config/mixer_tpp.yaml --optuna_trials 20
~~~

Training creates a timestamped directory under checkpoints when no output
directory is supplied. Test accepts best, last, or epoch checkpoints; epoch also
requires --ckpt_epoch. Omitting --checkpoint_dir in test mode selects the latest
matching directory.

## Induced-seismicity TPP forecasting

See [the induced-seismicity guide](docs/guides/induced_seismicity_forecasting_guide.md)
and [the catalog guide](docs/guides/add_induced_triplet_catalog_guide.md) for
the long-form workflow.

### Processed triplet

For dataset MyField, prepare:

~~~text
data/MyField/processed/
├── MyField_eq_processed.csv
├── MyField_inj_60min_processed.csv
└── MyField_summary.json
~~~

The earthquake CSV needs time and magnitude columns. Common aliases are
time_iso/ts/timestamp and magnitude/Magnitude/Mw; latitude, longitude, and
depth are optional. The injection CSV needs time and inj_rate_m3_min or
inj_rate. Times can be absolute or relative; summary.time_column_units can
declare units.

Required summary keys:

~~~json
{
  "resample_freq_min": 60,
  "mc": -0.5,
  "inj_fill_policy": "interpolate",
  "is_upsample": false,
  "start_time_iso": "2018-06-04T05:27:26.351",
  "end_time_iso": "2018-08-21T23:59:59"
}
~~~

The loader uses these fields for the time range, magnitude completeness (Mc),
and injection resampling. It filters/sorts events, removes events below Mc, and
jitters exact duplicate times so inter-event times remain positive.

### Register and validate a catalog

Subclass InducedTripletBase in src/catalogs/, register <Dataset>-Standard, and
import the module from src/catalogs/__init__.py. See src/catalogs/st1.py and
src/catalogs/pnr.py.

~~~bash
python - <<'PY'
import src.catalogs
from src.data.catalog import Catalog
print(sorted(Catalog.list_available()))
PY
~~~

dataset: MyField resolves to MyField-Standard. Generated Sequences are cached
under data/MyField/catalogs/<hash>/. The hash includes frequency, Mc,
normalization, and split boundaries.

### Models and time units

| Model | Typical use |
| --- | --- |
| oracle | Injection-aware Oracle marks and autoregressive sampling |
| rtpp_v2 | Recurrent TPP with configurable background |
| rtpp | Recurrent TPP compatible with RECAST |
| etas, etas_zhuang | ETAS triggering plus background rate |
| mixer_tpp | TPP-Mixer event-sequence modeling |

catalog_cfg.freq defines one model time unit. With freq "1D", duration 7.0
means seven days; with "1h", it means seven hours.

Models with a background component accept `bg_model` and `bg_model_cfg`. Common
choices are `kernel`, `mamba`, `rnn`, `conv_mlp`, and `proportional`. For
example:

~~~yaml
bg_model: mamba
bg_model_cfg:
  d_feature: 1
  d_state: 64
  d_model: 16
  model_type: mamba
  scale_init: 1000.0
  smooth_kernel_size: 3
~~~

When comparing background models, keep the dataset, split boundaries, model
seed, and forecast evaluation settings fixed. `scale_init` controls the initial
background-rate scale; values around 300--1000 are common starting points, not
universal defaults.

Example Oracle configuration:

~~~yaml
model: oracle
task_type: tpp
dataset: St1-2018
catalog_cfg:
  freq: "1D"
  mag_completeness: -0.5
  event_feature_builder: oracle
  event_feature_cfg:
    smoothing_window: 10
    activity_epsilon: 0.0
    log_floor: -10.0
~~~

The Oracle builder creates event-aligned marks: vm (log10 absolute interpolated
injection rate), dVc (log10 absolute cumulative-volume change), sv (its sign),
dTS (log10 time since the last non-zero injection sample), and aRs (smoothed
seismicity rate in log10(1/min)). These are stored in the cached Sequence and
consumed using oracle_base_mark_order and
oracle_future_feature_names.

Because the implementation aligns values by interpolation, inspect the sampling
boundary and any missing-value policy when making strict no-future-information
claims for a particular experiment.

### Train, test, and sample

~~~bash
python main.py --model oracle --mode train --config config/oracle.yaml \
  --checkpoint_dir checkpoints/oracle_st1
python main.py --model oracle --mode test \
  --checkpoint_dir checkpoints/oracle_st1 --ckpt_select best
~~~

Test writes metrics_test_<selector>_<trial>.json and reports held-out TPP NLL.
Sampling and calibration are separate; use the forecasting_induced_eq notebooks.

~~~python
from pathlib import Path
from src.utils.tpp_experiments import load_tpp_catalog, sample_tpp_forecasts
catalog, name, kwargs = load_tpp_catalog(
    "St1-2018", base_dir=Path("data/St1-2018"),
    catalog_cfg={"freq": "1D", "mag_completeness": -0.5},
)
past = catalog.test[0]
forecasts = sample_tpp_forecasts(
    model, past, duration=7.0, num_samples=1000,
    samples_per_batch=32, bg_cache_seq=past, seed=0,
)
~~~

Report held-out NLL, Number Test/event counts, magnitude checks, sampled count
distributions, sliding-window error, interval coverage, CRPS, and results by
split start time and physical dataset. Relevant code is in
src/utils/catalog_tests.py, src/utils/forecast_eval.py, and
scripts/run_sliding_window_forecast.py.

### Add a dataset

1. Produce and validate the three processed files.
2. Add/register an InducedTripletBase subclass.
3. Import it from src/catalogs/__init__.py.
4. Set dataset and catalog_cfg in a model YAML.
5. Instantiate the catalog before a long run.

For multiple physical catalogs, use InducedTripletGroupedCatalog with non-empty,
mutually exclusive train, val, and test split_groups.

## Data and splitting

TPP preparation resolves the catalog registry, creates or loads a cached
Sequence, and creates chronological train/validation/test subsequences.
Induced-triplet catalogs default to 70%/15%/15% calendar-time boundaries unless
train_start_ts, val_start_ts, and test_start_ts are set. Validation and test
retain preceding history but start NLL at their split boundary, giving causal
warm-up without scoring that history.

## Configuration, outputs, and development

YAML files are under config/ and loaded with OmegaConf. Common keys are model,
dataset, task_type, optimizer/scheduler settings, architecture parameters, and
catalog_cfg. Transfer supports resume_path, load_specific_parts, freeze_parts,
freeze_loaded_only, and exclude_freeze_parts.

Typical checkpoint files are config.yaml, run.log, best_model_1.pth,
last_model_1.pth, epoch_<N>_model_1.pth, tensorboard/, and
metrics_test_<selector>_1.json.

~~~bash
tensorboard --logdir checkpoints/<experiment>/tensorboard --port 6006
pytest -q
export CUBLAS_WORKSPACE_CONFIG=:4096:8
~~~

Repository guides: [data layout](docs/guides/data_layout.md), [induced
forecasting](docs/guides/induced_seismicity_forecasting_guide.md), and
[catalog integration](docs/guides/add_induced_triplet_catalog_guide.md).

The project is released under the MIT License; see LICENSE.
