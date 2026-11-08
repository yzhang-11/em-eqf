"""Matched RF and causal regression references, reusing current preparation."""
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier

ROOT = Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0, str(ROOT))
from src.data.preparation import prepare_data, prepare_data_rf
from src.utils.metrics import classification_metrics

OUT = ROOT / os.environ.get('REVIEW_EXPERIMENT', 'experiments/revision/main')


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2))


def main():
    summary = {'classification': [], 'regression': []}
    for horizon, mf in [(10, 4.), (20, 4.5), (30, 4.5), (60, 5.), (90, 5.5)]:
        c = OmegaConf.load(OUT / 'configs/rf.yaml')
        c.Tfore, c.Mf = horizon, mf
        c.save_dir = str(OUT / 'runs/references' / f'rf_h{horizon}')
        _, _, xtr, xva, xte, ytr, yva, yte = prepare_data_rf(c, str(ROOT / 'data/ChuanDian'))
        split = np.genfromtxt(Path(c.save_dir) / 'forecast_origins.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
        origins = split['origin_day'][split['split'] == 'test']
        for seed in range(5):
            folder = OUT / 'runs/references' / f'rf_h{horizon}_seed_{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            c.seed = seed
            OmegaConf.save(c, folder / 'config.yaml')
            model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=seed, n_jobs=2)
            model.fit(xtr, ytr)
            val_score = model.predict_proba(xva)[:, 1]
            vm = classification_metrics(yva, val_score, apply_sigmoid=False)
            score = model.predict_proba(xte)[:, 1]
            tm = classification_metrics(yte, score, threshold=vm['threshold'], apply_sigmoid=False)
            np.savez_compressed(folder / 'predictions_test.npz', target=yte, score=score,
                                origin_day=origins, threshold=vm['threshold'])
            dump(folder / 'metrics_validation.json', vm)
            dump(folder / 'metrics_test.json', tm)
            joblib.dump(model, folder / 'rf.joblib')
            summary['classification'].append(dict(horizon=horizon, seed=seed, n_test=len(yte), **tm))
        print(f'RF horizon {horizon}: five seeds completed', flush=True)

    c = OmegaConf.load(OUT / 'configs/reg_azdx_seed_0.yaml')
    c.save_dir = str(OUT / 'runs/references/regression')
    df, tr, va, te, ds = prepare_data(c, str(ROOT / 'data/ChuanDian'))
    y = np.asarray(ds.inverse_normalize_label(np.asarray(ds.labels))).reshape(-1)
    ytr, yte = y[tr.dataset.indices], y[te.dataset.indices]
    origins = ds.forecast_origins[te.dataset.indices]
    times, mags = df.t.to_numpy(), df.Magnitude.to_numpy()
    persistent, empty = [], 0
    for origin in origins:
        past = mags[(times >= origin - 180) & (times < origin)]
        empty += not len(past)
        persistent.append(float(past.max()) if len(past) else float(np.median(ytr)))
    refs = {'training_mean': np.full_like(yte, ytr.mean()),
            'training_median': np.full_like(yte, np.median(ytr)),
            'past_180_day_maximum': np.asarray(persistent)}
    keep, next_origin = [], -np.inf
    for i, origin in enumerate(origins):
        if origin >= next_origin:
            keep.append(i)
            next_origin = origin + float(c.Tfore)
    for name, pred in refs.items():
        row = {'name': name, 'n_test': len(yte), 'empty_history_fallbacks': empty if name.startswith('past') else 0}
        for label, idx in [('all_windows', np.arange(len(yte))), ('nonoverlap', np.asarray(keep))]:
            err = yte[idx] - pred[idx]
            row[label] = dict(n=len(idx), RMSE=float(np.sqrt(np.mean(err ** 2))), MAE=float(np.mean(abs(err))))
        summary['regression'].append(row)
        np.savez_compressed(Path(c.save_dir) / f'{name}.npz', target=yte, prediction=pred,
                            origin_day=origins, nonoverlap_indices=keep)
    dump(OUT / 'baseline_summary.json', summary)
    print(json.dumps(summary['regression']), flush=True)


if __name__ == '__main__':
    main()
