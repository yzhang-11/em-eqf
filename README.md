# em_eqf

`em_eqf` is a PyTorch research framework for earthquake-catalog forecasting. It
contains two data/model workflows: fixed-window classification and regression,
and event-sequence temporal point processes (TPPs). The repository also includes
catalog readers, induced-seismicity catalogs that combine earthquakes with
injection time series, experiment runners, analysis tools, and forecast
evaluation utilities.

This README describes the supported entry points and data conventions. The
linked guides contain implementation details for specialized workflows.
中文说明见 [README_CN.md](README_CN.md)。

## Workflows at a Glance

| Workflow | Input and target | Example models | Main data location |
| --- | --- | --- | --- |
| Window classification | Earthquake history window to a binary or count target | `clf_rnn`, `clf_mixer_attnpl_t` | `data/<dataset>/raw/` |
| Window regression | Earthquake history window to a future magnitude target | `lstm`, `reg_mixer_attnpl_t` | `data/<dataset>/raw/` |
| Event-sequence TPP | Ordered event times and marks to event likelihoods and samples | `rtpp_v2`, `etas`, `oracle`, `mixer_tpp` | Catalog-specific; see below |

Model names are registered in code. To see the names accepted by the command
line, run `python main.py --help`. A model's YAML `model` value must match the
`--model` argument. The default YAML path is `config/<model>.yaml`.

## Installation

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The requirements file installs the common project dependencies. Some model
configurations select optional CUDA extensions such as `mamba_ssm`,
`causal_conv1d`, or `flash_attn`; install only the extensions needed by the
chosen configuration, using versions compatible with the installed PyTorch,
CUDA, compiler, and Python versions. For a CPU-only run, choose a model/config
that does not require those extensions. CUDA is recommended for training.

## Data Layout

The project has separate input conventions for window models and TPP catalogs.
Do not assume that every TPP dataset uses the induced-triplet format.

### Window classification and regression

The event-window loader reads earthquake CSV input from `data/<dataset>/raw/`.
The CSV is expected to contain these columns:

| Column | Meaning |
| --- | --- |
| `t` | Event time as a numeric value in the dataset's time coordinate |
| `Magnitude` | Event magnitude |
| `Latitude`, `Longitude`, `Depth` | Event location/depth fields used by the sequence loader |
| `dt` | Inter-event time; the loader recomputes it after sorting |

If `raw/` contains one CSV, it is selected. If it contains multiple CSV files,
exactly one must have a `processed_` filename prefix; that file is selected.
Check the selected file and its units before training. Window length, forecast
horizon, stride, magnitude cutoffs, feature settings, and chronological split
behavior are configured in the model YAML (for example `Mc`, `Mf`, `Twindow`,
`Tfore`, `dt`, and `split_by_time`).

### TPP catalogs

TPP datasets are resolved through the `Catalog` registry. Their input format
depends on the registered catalog class: some readers consume catalog files in
`raw/`, while induced-triplet readers consume prepared files in `processed/`.
Catalog configuration is supplied by `catalog_cfg` in the model YAML. To list
registered catalogs:

```bash
python - <<'PY'
import src.catalogs
from src.data.catalog import Catalog
print("\n".join(sorted(Catalog.list_available())))
PY
```

The raw-catalog directory conventions and induced-triplet input layout are
described below. The migration utility is available as
`scripts/maintenance/reorganize_data_layout.py`.

### Induced-seismicity triplet files

An `InducedTripletBase` dataset named `MyField` uses this layout:

```text
data/MyField/
├── processed/
│   ├── MyField_eq_processed.csv
│   ├── MyField_inj_60min_processed.csv
│   └── MyField_summary.json
└── catalogs/                 # generated, hashed Sequence cache
```

The earthquake CSV requires a time column and magnitude column. The default
aliases include `time_iso`, `ts`, `time`, `timestamp`, `origin_time`,
`origin_timestamp`, and `Date` for time, and `magnitude`, `mag`, `Magnitude`,
`ML`, and `Mw` for magnitude. Latitude, longitude, and depth are optional. The
injection CSV requires a time column and `inj_rate_m3_min` or `inj_rate`. Time columns may be
absolute timestamps or numeric relative times. Numeric relative units are
inferred from common column names or can be declared by actual CSV column name
in `summary.time_column_units`. Accepted values are `unit`/`model`/`freq`, `s`,
`m`, `h`, `d`, and `abs`; numeric `t` or `time` columns default to model units.
Column aliases can also be customized through `summary.column_aliases`.

The summary JSON must include:

```json
{
  "resample_freq_min": 60,
  "mc": -0.5,
  "inj_fill_policy": "interpolate",
  "is_upsample": false,
  "start_time_iso": "2018-06-04T05:27:26.351",
  "end_time_iso": "2018-08-21T23:59:59"
}
```

`resample_freq_min` describes the prepared injection file's sampling interval;
`mc` is the `InducedTripletBase` fallback when the selected catalog class and
`catalog_cfg.mag_completeness` do not provide a value; a registered subclass
may define its own default. The remaining fields describe injection
filling/upsampling and the absolute data bounds. The selected `catalog_cfg.freq`
is the model's time unit. The loader filters
earthquakes at the configured Mc, sorts event times, and adds a tiny jitter to
duplicate times so inter-event times remain positive. Inspect preprocessing,
interpolation, and feature construction before making claims about strict
causal availability of auxiliary time-series features.

The triplet loader writes hashed catalog artifacts under
`data/<dataset>/catalogs/<hash>/`. The hash includes the time unit, Mc,
normalization, split boundaries, and event-feature settings. Preserve this cache
with its metadata; do not reuse a cache after changing source data without
rebuilding or validating it.

## First Run Checklist

Before starting a long run:

1. Run `python main.py --help` and confirm that the requested model is registered.
2. Open the selected YAML and verify `model`, `task_type`, `dataset`, the time
   unit, magnitude thresholds, and split settings. The YAML `model` must equal
   the CLI `--model` value.
3. For a window model, inspect `data/<dataset>/raw/` and confirm the selected
   CSV contains `t`, `Magnitude`, `Latitude`, `Longitude`, `Depth`, and `dt`.
   For a TPP model, confirm the catalog-specific files and `catalog_cfg`.
4. For a triplet catalog, validate registration and construction before training
   by replacing `MyField` with the dataset name:

   ```bash
   python - <<'PY'
   import src.catalogs
   from src.utils.tpp_experiments import load_tpp_catalog

   catalog, registry_name, init_kwargs = load_tpp_catalog(
       "MyField",
       base_dir="data/MyField",
       catalog_cfg={"freq": "1h"},
   )
   print("registry:", registry_name)
   print("train/val/test:", len(catalog.train[0]), len(catalog.val[0]), len(catalog.test[0]))
   print("init kwargs:", init_kwargs)
   PY
   ```

5. Ensure the output directory is writable and that `resume_path`, if set, points
   to a compatible checkpoint. Keep the final YAML and checkpoint together.

The triplet validation command checks registry loading, source-file discovery,
summary fields, cache construction, and chronological splits. It requires the
dataset files to exist; it is not a synthetic smoke test.

## Train, Test, and Tune

`main.py` is the common CLI for registered models:

```bash
# Train using config/clf_rnn.yaml and a known output directory.
python main.py --model clf_rnn --mode train \
  --config config/clf_rnn.yaml \
  --checkpoint_dir checkpoints/clf_rnn_demo

# Evaluate the best checkpoint from that run.
python main.py --model clf_rnn --mode test \
  --checkpoint_dir checkpoints/clf_rnn_demo \
  --ckpt_select best
```

The example requires a valid `data/ChuanDian/raw/` input. Change `dataset` and
the data-related YAML fields to match the catalog available on your machine.
Before training, inspect `resume_path`: a valid path loads a checkpoint, while
an unset path starts from scratch. The command-line `--checkpoint_dir` controls
where the run is saved; without it, training creates a timestamped directory
under `checkpoints/`.

Testing supports `--ckpt_select best`, `last`, or `epoch`; `epoch` also requires
`--ckpt_epoch N`. `--trial_index` selects the numbered checkpoint (default 1).
If `--checkpoint_dir` is omitted in test mode, the CLI searches for the latest
timestamped checkpoint directory for that model. Classification tests can use
`--threshold VALUE` to override the validation threshold or
`--no_val_threshold` to ignore the threshold stored in the checkpoint.

Optuna is available through `--mode optuna` when the selected YAML defines an
Optuna search space. The repository's model YAML files do not currently include
a default search space. For example, add this block to a copy of
`config/clf_rnn.yaml`:

```yaml
optuna:
  objective: val_loss
  direction: minimize
  search_space:
    learning_rate:
      type: float
      low: 0.0001
      high: 0.01
      log: true
```

Set the objective explicitly when possible. The implementation defaults to
`weighted_metrics`; that objective also requires `optuna.metric_weights`.
Search-space keys may target nested configuration values with dotted paths, for
example `bg_model_cfg.scale_init`. Supported `type` values are `float`, `int`,
and `categorical`; `float` and `int` accept `low`, `high`, optional `step`, and
optional `log`, while `categorical` requires a non-empty `choices` list. The
other supported objective is `weighted_metrics`; when using it, also define
`optuna.metric_weights` with metric names and signed weights, such as
`{auc: -1.0}` for minimizing AUC or `{RMSE: 1.0}` for minimizing RMSE.

Then pass that YAML to the CLI:

```bash
python main.py --model clf_rnn --mode optuna \
  --config config/clf_rnn_optuna.yaml --optuna_trials 20
```

Use `python main.py --help` for the full CLI, including profile selection,
storage, trial parallelism, and automatic checkpoint testing options.

## TPP Training and Forecast Evaluation

Common TPP models include `rtpp`, `rtpp_v2`, `etas`, `etas_zhuang`, `nhpp`,
`oracle`, and `mixer_tpp`. They differ in the event representation, mark
handling, background-rate support, and required configuration. Check each
model's YAML and implementation before transferring settings between models.
Background models such as `kernel`, `mamba`, `rnn`, `conv_mlp`, and
`proportional` are configuration-dependent; registered choices can be listed
with:

```bash
python - <<'PY'
from src.models.bg import BGModel
print("\n".join(sorted(BGModel.list_available())))
PY
```

For induced-triplet data, `catalog_cfg.freq` defines the model time unit. For
example, with `freq: "1D"`, a forecast duration of `7.0` is seven days; with
`freq: "1h"`, it is seven hours. The default split for
`InducedTripletBase` is chronological by calendar time at 70%/15%/15%; set
`train_start_ts`, `val_start_ts`, and `test_start_ts` to define explicit
boundaries. Registered subclasses may provide their own boundary defaults, so
check the selected catalog class and saved metadata. Validation and test
sequences retain earlier events for model history, while their NLL begins at
the corresponding split boundary. Other catalog classes may use different
split rules as well.

Example induced-seismicity configuration:

```yaml
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
```

With `event_feature_builder: oracle`, the loader adds event-aligned marks to the
cached sequence: `vm` is log10 absolute interpolated injection rate; `dVc` is
log10 absolute sequential cumulative-volume change; `sv` is the sign of that
change; `dTS` is log10 time since the last non-zero injection sample; and `aRs`
is a smoothed seismicity rate in log10(1/min). Their ordering and future feature
selection are controlled by `oracle_base_mark_order` and
`oracle_future_feature_names`. These marks use interpolation; inspect their
alignment and missing-data handling when evaluating causal availability.

Train and test it with the same CLI pattern:

```bash
python main.py --model oracle --mode train --config config/oracle.yaml \
  --checkpoint_dir checkpoints/oracle_st1
python main.py --model oracle --mode test \
  --checkpoint_dir checkpoints/oracle_st1 --ckpt_select best
```

The test command reports held-out likelihood metrics and writes a JSON file,
for example `metrics_test_best_1.json`. NLL evaluates event likelihood; it does
not replace forecast sampling or calibration. The repository's induced
forecasting notebooks demonstrate sampling and Number/Magnitude tests. Sliding
window metrics and helpers are in `src/utils/forecast_sliding.py`,
`src/utils/forecast_eval.py`, and the scripts under `scripts/`.

Related notebooks:

- [Induced forecasting](notebooks/forecasting_induced_eq.ipynb)
- [Induced triplet forecasting](notebooks/forecasting_induced_eq_triplet.ipynb)
- [Induced forecasting diagnostics](notebooks/forecasting_induced_eq_diagnostics.ipynb)
- [Background-model comparison](notebooks/induced_eq_bg_comparison.ipynb)
- [TPP evaluation](notebooks/tpp_evaluating.ipynb)

For a programmatic forecast, the public sampling helper can be used after
loading a model and catalog. The snippet assumes `model` is already the trained
TPP model on the intended device; `load_tpp_catalog` creates the catalog but
does not load a model checkpoint:

```python
from src.utils.tpp_experiments import load_tpp_catalog, sample_tpp_forecasts

# Load the trained model from its checkpoint before sampling.
catalog, catalog_name, init_kwargs = load_tpp_catalog(
    "St1-2018",
    base_dir="data/St1-2018",
    catalog_cfg={"freq": "1D", "mag_completeness": -0.5},
)
past = catalog.test[0]
forecasts = sample_tpp_forecasts(
    model,
    past,
    duration=7.0,
    num_samples=1000,
    samples_per_batch=32,
    bg_cache_seq=past,
    seed=0,
)
```

For reproducible comparisons, report the split boundaries, Mc, time unit,
model/config, random seed, checkpoint selector, and forecast sampling/evaluation
settings. For multi-catalog evaluations, retain per-catalog results as well as
any aggregate.

## Outputs and Reproducibility

Each run directory is self-contained and commonly contains:

```text
config.yaml
run.log
best_model_1.pth
last_model_1.pth
epoch_<N>_model_1.pth       # when configured/saved
tensorboard/
metrics_test_<selector>_1.json
```

Checkpoints use trial index 1 for the standard training flow. Keep the saved
`config.yaml` with the checkpoint because testing restores model settings from
the checkpoint/configuration. Test metrics are written beside the checkpoint:
`best` produces `metrics_test_best_1.json`, `last` produces
`metrics_test_last_1.json`, and `epoch` produces
`metrics_test_epoch_<N>_1.json`. To view TensorBoard logs:

```bash
tensorboard --logdir checkpoints/<run>/tensorboard --port 6006
```

The YAML files under `config/` are the source of model, optimizer, data, split,
and optional-feature settings. Deterministic training is configured in YAML;
for CUDA runs that require deterministic cuBLAS behavior, set this before
starting Python:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Optuna runs add `optuna_trials/trial_<number>/` directories containing the
trial configuration, checkpoint files, validation metrics, and optional
post-hoc test results. Study summaries are exported as
`optuna_summary_<profile>.json` and `optuna_summary_all.json` when profiles are
used. Transfer-learning settings such as `resume_path`, `load_specific_parts`,
`freeze_parts`, `freeze_loaded_only`, and `exclude_freeze_parts` are read from
the YAML and should be kept with the run configuration.

## Repository Map

| Path | Purpose |
| --- | --- |
| `main.py` | Train, test, and Optuna CLI entry point |
| `config/` | Model and experiment YAML files |
| `src/catalogs/` | Catalog implementations and registration |
| `src/data/` | Data loading, sequences, event windows, and splits |
| `src/models/` | Classifier, regressor, TPP, and background models |
| `src/train/` | Training steps, optimization, and checkpointing |
| `src/cli/` | CLI parsing and train/test/Optuna orchestration |
| `src/utils/` | Metrics, plotting, sampling, and evaluation helpers |
| `scripts/run/` | Experiment runners |
| `scripts/summarize/` | Experiment result aggregation |
| `scripts/analyze/` | Analysis tools |
| `notebooks/` | Preprocessing, forecasting, and analysis examples |

Script naming and folders are described in [Scripts README](scripts/README.md).

## Common Errors

| Symptom | Likely cause and correction |
| --- | --- |
| `Model name in config must match command line argument` | `--model` and the YAML `model` differ. Use the same registered name in both places. |
| `No CSV files found` or an ambiguous processed-file error | The window dataset has no CSV in `data/<dataset>/raw/`, or multiple CSVs without exactly one `processed_*.csv`. Remove ambiguity or rename the intended file. |
| `checkpoint ... not found` | Check `--checkpoint_dir`, `--trial_index`, and the selector. `epoch` also requires `--ckpt_epoch`; expected names are `best_model_<trial>.pth`, `last_model_<trial>.pth`, and `epoch_<N>_model_<trial>.pth`. |
| Missing triplet summary key | Add all six required keys to `<Dataset>_summary.json`, then rebuild the catalog cache. |
| Cache metadata conflicts with the current configuration | Do not reuse the old hashed cache after changing `freq`, Mc, normalization, split boundaries, or event features. Rebuild it or use the newly generated hash directory. |
| TPP forecasts have the wrong duration | `catalog_cfg.freq` is the model time unit. A duration of `7.0` means seven units of that frequency, not always seven days. |
| Optuna reports an empty search space | Add `optuna.search_space` or `optuna.profiles.<name>.params`; use `--optuna_profile` when multiple profiles exist. |
| Optional-module import failure | The selected YAML uses a CUDA extension that is not installed or is incompatible with the current PyTorch/CUDA/Python environment. Choose a compatible configuration or install the matching extension. |

## Tests and License

Run the test suite from the repository root:

```bash
python -m pytest -q
```

The project is released under the MIT License; see [LICENSE](LICENSE).
