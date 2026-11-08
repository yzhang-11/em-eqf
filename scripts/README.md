# Scripts

Use one entrypoint per operation. Root-level forwarding wrappers have been removed.

| Directory | Purpose |
| --- | --- |
| `run/` | Experiment runners |
| `summarize/` | Metric summaries |
| `analyze/` | Analyses and comparisons |
| `data/` | Source acquisition and preparation |
| `diagnostics/` | Focused validation |
| `tuning/` | Hyperparameter studies |
| `maintenance/` | Storage and repository maintenance |
| `automation/` | Shared runner implementation |

```bash
python scripts/run/run_clf_mf_tf_grid.py --help
python scripts/summarize/summarize_etas_grid.py --help
python scripts/maintenance/index_experiments.py
```

Data, experiment names, and result files are indexed in `experiments/registry/`.
