"""Run the frozen review_v4 configurations through the existing CLI."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import threading

import numpy as np
import yaml

ROOT = Path('/root/autodl-tmp/em_eqf')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    plan = json.loads((ROOT / 'experiments/revision/controls/plan.json').read_text())
    if not args.smoke and plan.get('catalog_protocol_status') != 'frozen':
        raise RuntimeError('Freeze raw catalogs, Mc and preprocessing before formal training')
    out = ROOT / ('experiments/revision/archive/review_v4_controls_smoke' if args.smoke else 'experiments/revision/controls')
    out.mkdir(parents=True, exist_ok=True)
    if not args.smoke:
        primary = ROOT / 'experiments/revision/main/runner_summary.json'
        summary = json.loads(primary.read_text())
        if summary['failed'] or summary['completed'] != 65:
            raise RuntimeError('Primary comparisons must complete successfully before controls')
    jobs = plan['jobs']
    def order(j):
        return (j['group'] != 'loss', j['seed'], j['model'], j['variant'], j['name'])
    jobs = sorted(jobs, key=order)
    halt = threading.Event()
    if args.smoke:
        names = ['clf_azdx_h30_seed_0', 'reg_azdx_seed_0', 'lstm_seed_0']
        jobs = [next(j for j in jobs if j['name'] == n) for n in names]
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg')

    def run(job):
        if halt.is_set():
            return dict(name=job['name'], state='held_after_failure')
        name = job['name']
        run_dir = out / 'runs' / name
        run_dir.mkdir(parents=True, exist_ok=True)
        status_path = run_dir / 'job_status.json'
        config = Path(job['config'])
        if args.smoke:
            c = yaml.safe_load(config.read_text())
            c.update(epochs=1, save_epochs=[])
            config = run_dir / 'smoke_config.yaml'
            config.write_text(yaml.safe_dump(c, sort_keys=False))
        digest = hashlib.sha256(config.read_bytes()).hexdigest()
        if status_path.exists():
            old = json.loads(status_path.read_text())
            if old.get('state') == 'completed' and old.get('config_sha256') == digest:
                if all((run_dir / f).is_file() for f in ['best_model_1.pth', 'metrics_test_best_1.json', 'predictions_test.npz']):
                    return old
            raise RuntimeError(f'Refusing to overwrite existing run: {run_dir}')
        status = dict(name=name, state='running', config_sha256=digest,
                      started=datetime.now(timezone.utc).isoformat(), smoke=args.smoke)

        def save():
            tmp = status_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(status, indent=2))
            tmp.replace(status_path)

        raw_config = yaml.safe_load(config.read_text())
        source_checkpoint = raw_config.get('resume_path')
        if source_checkpoint:
            status['source_checkpoint_sha256'] = hashlib.sha256(Path(source_checkpoint).read_bytes()).hexdigest()
        save()
        start = time.monotonic()
        try:
            for mode in ['train', 'test']:
                status['phase'] = mode
                save()
                cmd = [sys.executable, '-u', 'main.py', '--model', job['model'], '--mode', mode,
                       '--config', str(config), '--checkpoint_dir', str(run_dir)]
                with (run_dir / f'{mode}_stdout.log').open('w') as log:
                    subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            metrics = json.loads((run_dir / 'metrics_test_best_1.json').read_text())
            with np.load(run_dir / 'predictions_test.npz') as pred:
                assert len(pred['target']) > 0
                for key in pred.files:
                    assert np.isfinite(pred[key]).all(), (name, key)
            status.update(state='completed', metrics=metrics)
        except Exception as error:
            status.update(state='failed', error=str(error))
            halt.set()
        status['elapsed_seconds'] = round(time.monotonic() - start, 2)
        save()
        print(json.dumps(status), flush=True)
        return status

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, jobs))
    summary = {'smoke': args.smoke, 'completed': sum(j['state'] == 'completed' for j in results),
               'failed': sum(j['state'] != 'completed' for j in results), 'jobs': results}
    (out / 'runner_summary.json').write_text(json.dumps(summary, indent=2))
    return 0 if not summary['failed'] else 1


if __name__ == '__main__':
    sys.exit(main())
