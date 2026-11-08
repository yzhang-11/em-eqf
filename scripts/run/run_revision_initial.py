"""Formal AZDX source fit and five independent target-generation fits per catalog."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import torch

ROOT = Path('/root/autodl-tmp/em_eqf')
OUT = ROOT / 'experiments/revision/main'


def main():
    plan = json.loads((OUT / 'plan.json').read_text())
    if plan.get('catalog_protocol_status') != 'frozen':
        raise RuntimeError('Catalog and GPU checks must pass before formal training')
    jobs = ['pretrain_azdx_main_seed_0']
    for seed in range(5):
        jobs += [f'generation_swc_main_seed_{seed}', f'generation_scsn_main_seed_{seed}']
    halt = threading.Event()
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg')

    def run(name):
        if halt.is_set():
            return dict(name=name, state='held_after_failure')
        folder = OUT / 'runs' / name
        folder.mkdir(parents=True, exist_ok=True)
        state_file = folder / 'job_status.json'
        if state_file.exists():
            old = json.loads(state_file.read_text())
            if old.get('state') == 'completed':
                return old
            raise RuntimeError(f'Existing run requires inspection: {folder}')
        config = OUT / 'configs' / f'{name}.yaml'
        source = name.startswith('pretrain_')
        state = dict(name=name, state='running', phase='train',
                     started_utc=datetime.now(timezone.utc).isoformat(),
                     config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())

        def save():
            temporary = state_file.with_suffix('.tmp')
            temporary.write_text(json.dumps(state, indent=2))
            temporary.replace(state_file)

        save()
        started = time.monotonic()
        try:
            cmd = [sys.executable, '-u', 'main.py', '--model', 'mixer_tpp', '--config', str(config), '--checkpoint_dir', str(folder)]
            with (folder / 'train_stdout.log').open('w') as log:
                process = subprocess.Popen(cmd + ['--mode', 'train'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                state['pid'] = process.pid
                save()
                code = process.wait()
                if code:
                    raise RuntimeError(f'Training exit code {code}; inspect train_stdout.log')
            checkpoint_path = folder / ('last_model_1.pth' if source else 'best_model_1.pth')
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            for key, value in checkpoint['model_state_dict'].items():
                if torch.is_tensor(value) and value.is_floating_point():
                    assert torch.isfinite(value).all(), key
            expected_epochs = int(checkpoint['hyperparameters']['epochs'])
            assert f'Last model at epoch {expected_epochs} saved' in (folder / 'run.log').read_text()
            if not source:
                state['phase'] = 'test'
                save()
                with (folder / 'test_stdout.log').open('w') as log:
                    subprocess.run(cmd + ['--mode', 'test'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                state['metrics'] = json.loads((folder / 'metrics_test_best_1.json').read_text())
                assert all(torch.isfinite(torch.tensor(v)) for v in state['metrics'].values())
            state.update(state='completed', checkpoint=str(checkpoint_path), checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())
        except Exception as error:
            state.update(state='failed', error=str(error))
            halt.set()
        state['elapsed_seconds'] = round(time.monotonic() - started, 2)
        save()
        print(json.dumps(state), flush=True)
        return state

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, jobs))
    (OUT / 'initial_training_summary.json').write_text(json.dumps(results, indent=2))
    return 0 if all(x['state'] == 'completed' for x in results) else 1


if __name__ == '__main__':
    sys.exit(main())
