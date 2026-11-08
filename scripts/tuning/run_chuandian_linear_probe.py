"""Matched affine-head probes and fine-tuning on immutable ChuanDian targets."""
import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path('/root/autodl-tmp/em_eqf')
STORE = Path('/root/autodl-fs/em_eqf_training/mamba3_chuandian_transfer_20261105')
BASE = ROOT / 'experiments/revision/controls'
TAG = 'chuandian_linear_probe_20261105'
REPORTS, CONFIGS, RUNS = [BASE / name / TAG for name in ['reports', 'configs', 'runs']]
WORK = STORE / 'linear_probe'
READOUT = 'last'
FAMILIES = ['mamba2_legacy', 'mamba3_oracle_residual']
SEEDS = [0, 1, 2]
FINE_MODES = ['scratch', 'full', 'lpft']
LRS = [1e-5, 1e-4, 4e-4]
CORE_PATH = STORE / 'frozen_downstream/run_chuandian_encoder_transfer.py'
spec = importlib.util.spec_from_file_location('original_transfer_core', CORE_PATH)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
c = core.c


def configure_readout(readout):
    global READOUT, TAG, REPORTS, CONFIGS, RUNS, WORK
    assert readout in ['last', 'mean']
    READOUT = readout
    suffix = '_mean' if readout == 'mean' else ''
    TAG = f'chuandian_linear_probe{suffix}_20261105'
    REPORTS, CONFIGS, RUNS = [BASE / name / TAG for name in ['reports', 'configs', 'runs']]
    WORK = STORE / f'linear_probe{suffix}'


def pool_hidden(hidden, batch, readout):
    if readout == 'mean':
        mask = batch['mask'][:, :, None]
        return hidden.masked_fill(~mask, 0).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    rows = torch.arange(len(hidden), device=hidden.device)
    return hidden[rows, batch['lengths'].to(hidden.device) - 1]


def read(path):
    return json.loads(Path(path).read_text())


def setup():
    c.bootstrap()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.load(core.DATA, map_location='cpu', weights_only=False)


def initialize():
    for p in [REPORTS, CONFIGS, WORK / 'runs', WORK / 'features']:
        p.mkdir(parents=True, exist_ok=True)
    if not RUNS.exists():
        RUNS.symlink_to(WORK / 'runs', target_is_directory=True)
    plan_path = CONFIGS / 'study_plan.json'
    if plan_path.exists():
        plan = read(plan_path)
        assert c.sha(WORK / 'frozen_runner.py') == plan['runner_sha256']
        return plan
    data = setup()
    source = torch.load(STORE / 'transfer_sources_before_20110212.pt', map_location='cpu', weights_only=False)
    cutoff = int(np.datetime64('2011-02-12', 'ns').astype(np.int64))
    assert len(source['regions']) == 9
    assert not any('chuan' in name.lower() for name in source['regions'])
    for region in source['regions'].values():
        assert np.asarray(region['physical_ns']).max() < cutoff
    for task in data['tasks'].values():
        tr, va, te = [task['splits'][n] for n in ['train', 'validation', 'test']]
        origin = task['origins']
        assert origin[tr].max() + task['Tfore'] <= origin[va].min()
        assert origin[va].max() + task['Tfore'] <= origin[te].min()
        assert all(np.max(h['relative_time']) < 1 for h in task['histories'])
    shutil.copy2(Path(__file__), WORK / 'frozen_runner.py')
    plan = dict(protocol=TAG, layers=2, width=64, families=FAMILIES, seeds=SEEDS,
        data_sha256=c.sha(core.DATA), core_sha256=c.sha(CORE_PATH), runner_sha256=c.sha(WORK / 'frozen_runner.py'),
        source_checkpoints={f'{family}/{seed}': dict(path=str(core.source_path(family, seed)), sha256=c.sha(core.source_path(family, seed)))
                            for family, seed in itertools.product(FAMILIES, SEEDS)},
        source_cutoff_exclusive='2011-02-12', source_regions=9, target_in_source=False,
        split_dates=data['split_dates'],
        pooling=READOUT, readout=f'{READOUT} output of encoder layer 2; mean excludes padding; no trainable pooling.',
        primary_features='64 frozen/finetuned encoder features + original log elapsed gap, 65 inputs; affine head has 66 parameters.',
        diagnostics=['64 encoder features without gap', 'gap-only affine baseline'],
        feature_scaling='Train-only fixed StandardScaler, computed at the initial encoder state; reused unchanged during fine-tuning.',
        probe=dict(classification='L2 logistic regression, C grid [.01,.1,1,10,100]; no class weighting.',
                   regression='Median linear quantile regression, L1 alpha grid [.1,.01,.001,.0001,0]; quantile=.5.',
                   selection='Validation AUPRC / magnitude MAE only; grid order prefers stronger regularization on exact ties.'),
        fine_modes=FINE_MODES, fine_grid=LRS, head_lr=.001, weight_decay=.001,
        fine_loss='BCEWithLogits / median pinball (.5 * absolute error), matching the probe task losses.',
        fine_selection='Same three encoder learning rates per mode, family, task on seed0 validation only; selected rate fixed for seeds1/2.',
        max_epochs=150, minimum_epochs=25, warmup_epochs=15, patience=10, batch_size=128,
        schedule='Original 15-epoch warmup and factor0.5 each30 epochs; gradient clip3.',
        checkpoint='Minimum validation negative AUPRC / magnitude MAE; no test scores until all fits finish.',
        regression_reporting='Original fixed magnitude normalization [3,9] and original clipping at evaluation; also report unclipped MAE.',
        limitations='Exploratory follow-up: this historical test period was already inspected in the previous study; one target region and overlapping windows.',
        readout_comparison='Reuse completed Last controls. Mean has identical width, seeds, data, heads and validation tuning budget; pool selection uses validation only.',
        random_head_initialization='Identical head initialization within seed across scratch/full; LP-FT loads its own pretrained linear probe.',
        probe_encoder_frozen=True, input_projection_transferred=True)
    c.save(plan_path, plan)
    c.save(REPORTS / 'plan.json', plan)
    c.save(REPORTS / 'runtime_environment.json', dict(python=sys.executable,
        versions={name: importlib.metadata.version(name) for name in ['torch', 'triton', 'tilelang', 'mamba-ssm', 'scikit-learn']},
        gpu=torch.cuda.get_device_name(), precision=torch.get_float32_matmul_precision(), tf32=torch.backends.cuda.matmul.allow_tf32))
    return plan


class AffineTransfer(nn.Module):
    def __init__(self, family, seed, pretrained, frozen):
        super().__init__()
        from src.models.mamba.mixer_seq import MixerModel
        settings = read(c.CONFIGS / 'study_plan.json')['backbones'][family]
        torch.manual_seed(seed)
        self.encoder = MixerModel(**copy.deepcopy(settings['mixer_model_config']), device='cuda', dtype=torch.float32)
        self.random_encoder_sha256 = c.state_hash(self.encoder)
        if pretrained:
            payload = torch.load(core.source_path(family, seed), map_location='cpu', weights_only=False)
            prefix = 'base_model.encoder.'
            state = {key[len(prefix):]: value for key, value in payload['model_state_dict'].items() if key.startswith(prefix)}
            self.encoder.load_state_dict(state, strict=True)
        self.loaded_encoder_sha256 = c.state_hash(self.encoder)
        self.frozen = frozen
        self.encoder.requires_grad_(not frozen)
        self.register_buffer('feature_mean', torch.zeros(65, device='cuda'))
        self.register_buffer('feature_scale', torch.ones(65, device='cuda'))
        torch.manual_seed(seed + 4242)
        self.head = nn.Linear(65, 1, device='cuda')
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def features(self, batch):
        with torch.set_grad_enabled(self.training and not self.frozen):
            hidden = self.encoder(features=batch['features'], times=batch['times'], input_mask=batch['mask'])
            pooled = pool_hidden(hidden, batch, READOUT)
        return torch.cat([pooled, batch['gap'][:, None]], dim=-1)

    def forward(self, batch):
        return self.head((self.features(batch) - self.feature_mean) / self.feature_scale).squeeze(-1)


def feature_path(task, family, seed, pretrained):
    return WORK / 'features' / f'{task}_{family}_{"pretrained" if pretrained else "random"}_seed_{seed}.npz'


@torch.no_grad()
def extract_features(data, task_name, family, seed, pretrained):
    path = feature_path(task_name, family, seed, pretrained)
    if path.exists():
        with np.load(path) as archive:
            return archive['features'].copy(), str(archive['encoder_sha256'].item()), str(archive['random_encoder_sha256'].item())
    task = data['tasks'][task_name]
    model = AffineTransfer(family, seed, pretrained, True).eval()
    before = c.state_hash(model.encoder)
    x = np.empty((len(task['labels']), 65), dtype=np.float32)
    for indices in core.batches(task, np.arange(len(x))):
        values = model.features(core.collate(data, task, indices))
        assert torch.isfinite(values).all()
        x[indices] = values.cpu().numpy()
    assert before == c.state_hash(model.encoder)
    np.savez_compressed(path, features=x, encoder_sha256=before, random_encoder_sha256=model.random_encoder_sha256,
                        origins_sha256=hashlib.sha256(task['origins'].tobytes()).hexdigest())
    del model
    torch.cuda.empty_cache()
    return x, before, str(np.load(path)['random_encoder_sha256'].item())


def probe_fit(data, task_name, family, seed, pretrained, columns='encoder_gap'):
    import joblib
    mode = 'pretrained_probe' if pretrained else 'random_probe'
    name = f'{task_name}_{family}_{mode}_{columns}_seed_{seed}'
    folder = RUNS / name
    folder.mkdir(exist_ok=True)
    if (folder / 'training_complete.json').exists():
        return
    x, encoder_hash, random_hash = extract_features(data, task_name, family, seed, pretrained)
    indices = {'encoder_gap': np.arange(65), 'encoder_only': np.arange(64), 'gap_only': np.array([64])}[columns]
    x = x[:, indices]
    task = data['tasks'][task_name]
    tr, va = task['splits']['train'], task['splits']['validation']
    scaler = StandardScaler().fit(x[tr])
    z = scaler.transform(x)
    grid = [.01, .1, 1., 10., 100.] if task_name == 'classification' else [.1, .01, .001, .0001, 0.]
    trials = []
    best = None
    for regularization in grid:
        if task_name == 'classification':
            estimator = LogisticRegression(C=regularization, max_iter=3000, tol=1e-9, solver='lbfgs')
        else:
            estimator = QuantileRegressor(quantile=.5, alpha=regularization, solver='highs')
        estimator.fit(z[tr], task['labels'][tr])
        prediction = estimator.decision_function(z[va]) if task_name == 'classification' else estimator.predict(z[va])
        value = core.metrics(task_name, task['labels'][va], prediction)
        score = -value['auprc'] if task_name == 'classification' else value['mae']
        trials.append(dict(regularization=regularization, score=score, validation=value))
        if best is None or score < best['score'] - 1e-10:
            best = dict(score=score, validation=value, regularization=regularization,
                        weight=np.ravel(estimator.coef_).astype(np.float32), bias=float(np.ravel(estimator.intercept_)[0]), estimator=estimator)
    np.savez_compressed(folder / 'linear_head.npz', weight=best['weight'], bias=np.float32(best['bias']),
                        mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32), columns=indices)
    joblib.dump(dict(scaler=scaler, estimator=best['estimator'], columns=indices,
                     data_sha256=c.sha(core.DATA), features_sha256=c.sha(feature_path(task_name, family, seed, pretrained))),
                folder / 'linear_head_reference.joblib')
    result = dict(task=task_name, family=family, mode=mode, seed=seed, features=columns,
                  trainable_parameters=len(indices) + 1, encoder_sha256=encoder_hash, random_encoder_sha256=random_hash,
                  regularization=best['regularization'], validation=best['validation'], trials=trials,
                  encoder_unchanged=True, test_used=False, head_sha256=c.sha(folder / 'linear_head.npz'))
    c.save(folder / 'training_complete.json', result)
    print('PROBE_COMPLETE', name, 'VAL', json.dumps(best['validation']), flush=True)


def load_affine(model, path):
    with np.load(path) as head:
        assert len(head['weight']) == 65
        model.head.weight.copy_(torch.as_tensor(head['weight'][None], device='cuda'))
        model.head.bias.fill_(float(head['bias']))
        model.feature_mean.copy_(torch.as_tensor(head['mean'], device='cuda'))
        model.feature_scale.copy_(torch.as_tensor(head['scale'], device='cuda'))


def fine_fit(data, task_name, family, seed, mode, lr):
    name = f'{task_name}_{family}_{mode}_lr_{lr:g}_seed_{seed}'
    folder = RUNS / name
    folder.mkdir(exist_ok=True)
    if (folder / 'training_complete.json').exists():
        return read(folder / 'training_complete.json')
    pretrained = mode != 'scratch'
    model = AffineTransfer(family, seed, pretrained, False)
    x, expected_hash, _ = extract_features(data, task_name, family, seed, pretrained)
    assert model.loaded_encoder_sha256 == expected_hash
    task = data['tasks'][task_name]
    tr, va = task['splits']['train'], task['splits']['validation']
    scale = StandardScaler().fit(x[tr])
    with torch.no_grad():
        model.feature_mean.copy_(torch.as_tensor(scale.mean_, dtype=torch.float32, device='cuda'))
        model.feature_scale.copy_(torch.as_tensor(scale.scale_, dtype=torch.float32, device='cuda'))
        if mode == 'lpft':
            load_affine(model, RUNS / f'{task_name}_{family}_pretrained_probe_encoder_gap_seed_{seed}/linear_head.npz')
    head_hash = c.state_hash(model.head)
    optimizer = torch.optim.AdamW([dict(params=model.head.parameters(), lr=.001, initial_lr=.001),
        dict(params=model.encoder.parameters(), lr=lr, initial_lr=lr)], weight_decay=.001)
    best, best_epoch, history, started = float('inf'), 0, [], time.time()
    for epoch in range(1, 151):
        model.train()
        factor = min(1., epoch / 15) * max(.001, .5 ** (epoch // 30))
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] * factor
        total, count = 0., 0
        for ids in core.batches(task, tr, seed, epoch):
            batch = core.collate(data, task, ids)
            output = model(batch)
            target = batch['labels']
            loss = (nn.functional.binary_cross_entropy_with_logits(output, target) if task_name == 'classification'
                    else .5 * torch.abs(output - target).mean())
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 3.)
            assert torch.isfinite(norm)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            count += len(ids)
        prediction = core.predict(model, data, task, va)
        value = core.metrics(task_name, task['labels'][va], prediction)
        score = -value['auprc'] if task_name == 'classification' else value['mae']
        history.append(dict(epoch=epoch, training_loss=total / count, validation=value, seconds=time.time() - started))
        if score < best:
            best, best_epoch = score, epoch
            torch.save(dict(model_state_dict=model.state_dict(), epoch=epoch, validation=value), folder / 'best_model.pth')
        if epoch == 1 or epoch % 10 == 0:
            c.save(folder / 'progress.json', dict(history=history, best_epoch=best_epoch))
            print('FINE_PROGRESS', name, epoch, round(score, 6), round(time.time() - started, 1), flush=True)
        if epoch >= 25 and epoch - best_epoch >= 10:
            break
    state = torch.load(folder / 'best_model.pth', map_location='cpu', weights_only=False)
    result = dict(task=task_name, family=family, mode=mode, seed=seed, name=name, encoder_lr=lr,
        features='encoder_gap', score=best, validation=state['validation'], best_epoch=best_epoch,
        epochs_completed=epoch, seconds=time.time() - started, head_initial_sha256=head_hash,
        random_encoder_sha256=model.random_encoder_sha256, loaded_encoder_sha256=model.loaded_encoder_sha256,
        checkpoint_sha256=c.sha(folder / 'best_model.pth'), history=history, test_used=False)
    c.save(folder / 'training_complete.json', result)
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def preflight():
    data = setup()
    checks = []
    synthetic = torch.tensor([[[1., 2.], [3., 4.], [float('nan'), float('nan')]]], device='cuda')
    mask_batch = dict(mask=torch.tensor([[True, True, False]], device='cuda'), lengths=torch.tensor([2]))
    assert torch.equal(pool_hidden(synthetic, mask_batch, 'mean'), torch.tensor([[2., 3.]], device='cuda'))
    assert torch.equal(pool_hidden(synthetic, mask_batch, 'last'), torch.tensor([[3., 4.]], device='cuda'))
    for task_name, family in itertools.product(data['tasks'], FAMILIES):
        task = data['tasks'][task_name]
        lengths = np.asarray([len(h['times']) for h in task['histories']])
        tr = task['splits']['train']
        ids = tr[np.argsort(lengths[tr])[-4:]]
        batch = core.collate(data, task, ids)
        for frozen in [True, False]:
            model = AffineTransfer(family, 0, True, frozen)
            before = c.state_hash(model.encoder)
            model.train()
            output = model(batch)
            assert torch.isfinite(output).all()
            (output.square().mean()).backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if frozen:
                assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 66
                assert all(p.grad is None for p in model.encoder.parameters())
                assert before == c.state_hash(model.encoder)
            model.eval()
            with torch.no_grad():
                features = model.features(batch)
                one = core.collate(data, task, ids[:1])
                single = model.features(one)
                error = float((features[0] - single[0]).abs().max())
                assert torch.allclose(features[0], single[0], atol=.003, rtol=.003), error
                hidden = model.encoder(features=batch['features'], times=batch['times'], input_mask=batch['mask'])
                manual = torch.stack([hidden[i, :int(n)].mean(dim=0) for i, n in enumerate(batch['lengths'])])
                assert torch.allclose(pool_hidden(hidden, batch, 'mean'), manual, atol=3e-6, rtol=3e-6)
                if READOUT == 'mean':
                    assert torch.allclose(features[:, :64], manual, atol=3e-6, rtol=3e-6)
            checks.append(dict(task=task_name, family=family, frozen=frozen, finite_gradient=True,
                               padding_batch_difference=error, strict_encoder_loading=True, hidden_shape=list(features.shape)))
            del model
            torch.cuda.empty_cache()
    c.save(REPORTS / 'preflight.json', dict(passed=True, pooling=READOUT, padding_nan_ignored=True, manual_masked_mean_verified=True, checks=checks))
    print('PREFLIGHT_COMPLETE', flush=True)


def probe_worker(family, seed):
    data = setup()
    for task_name, pretrained in itertools.product(data['tasks'], [False, True]):
        for columns in ['encoder_gap', 'encoder_only']:
            probe_fit(data, task_name, family, seed, pretrained, columns)
    if family == FAMILIES[0] and seed == 0:
        for task_name in data['tasks']:
            probe_fit(data, task_name, family, seed, False, 'gap_only')


def fine_worker(family, seed):
    data = setup()
    for task_name, mode in itertools.product(data['tasks'], FINE_MODES):
        selection_path = CONFIGS / f'lr_{task_name}_{family}_{mode}.json'
        if seed == 0:
            trials = [fine_fit(data, task_name, family, seed, mode, lr) for lr in LRS]
            selected = min(trials, key=lambda r: r['score'])
            c.save(selection_path, dict(selected_lr=selected['encoder_lr'],
                trials=[{k: r[k] for k in ['encoder_lr', 'score', 'validation', 'name']} for r in trials], test_used=False))
        else:
            fine_fit(data, task_name, family, seed, mode, read(selection_path)['selected_lr'])


def execute_jobs(jobs, workers):
    active = []
    while jobs or active:
        while jobs and len(active) < workers:
            phase, family, seed = jobs.pop(0)
            log = (REPORTS / f'{phase}_{family}_seed_{seed}.log').open('a')
            pool_args = ['--pool', READOUT] if READOUT != 'last' else []
            command = [sys.executable, '-u', str(WORK / 'frozen_runner.py'), *pool_args, f'--{phase}', family, '--seed', str(seed)]
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            active.append((process, log, command))
            print('START', phase, family, seed, process.pid, flush=True)
        for entry in active[:]:
            process, log, command = entry
            status = process.poll()
            if status is not None:
                log.close()
                active.remove(entry)
                if status:
                    for other, _, _ in active:
                        other.terminate()
                    raise RuntimeError(f'Worker failed: {command}, exit={status}')
        time.sleep(2)


def master(workers):
    initialize()
    if not (REPORTS / 'preflight.json').exists():
        preflight()
    execute_jobs([('probe', family, seed) for family, seed in itertools.product(FAMILIES, SEEDS)], workers)
    execute_jobs([('fine', family, 0) for family in FAMILIES], workers)
    execute_jobs([('fine', family, seed) for family, seed in itertools.product(FAMILIES, [1, 2])], workers)
    c.save(REPORTS / 'training_complete.json', dict(complete=True, test_scored=False))
    print('ALL_TRAINING_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initialize', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--probe', choices=FAMILIES)
    parser.add_argument('--fine', choices=FAMILIES)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--pool', choices=['last', 'mean'], default='last')
    args = parser.parse_args()
    configure_readout(args.pool)
    if args.initialize:
        initialize()
    elif args.preflight:
        preflight()
    elif args.probe:
        probe_worker(args.probe, args.seed)
    elif args.fine:
        fine_worker(args.fine, args.seed)
    elif args.run:
        master(args.workers)
    else:
        parser.error('Choose an action')
