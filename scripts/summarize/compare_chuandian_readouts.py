"""Paired Last/Mean comparisons; choose pooling using validation scores only."""
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path('/root/autodl-tmp/em_eqf')
BASE = ROOT / 'experiments/revision/controls'
STORE = Path('/root/autodl-fs/em_eqf_training/mamba3_chuandian_transfer_20261105')
REPORTS = BASE / 'reports/chuandian_linear_probe_mean_20261105'
spec = importlib.util.spec_from_file_location('affine_summary', STORE / 'linear_probe_mean/frozen_summary.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)
FAMILIES = ['mamba2_legacy', 'mamba3_oracle_residual']
MODES = ['random_probe', 'pretrained_probe', 'scratch', 'full', 'lpft']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def folder(pool, task, family, mode, seed):
    suffix = '_mean' if pool == 'mean' else ''
    work = STORE / f'linear_probe{suffix}'
    if mode.endswith('_probe'):
        return work / f'runs/{task}_{family}_{mode}_encoder_gap_seed_{seed}'
    config = BASE / f'configs/chuandian_linear_probe{suffix}_20261105/lr_{task}_{family}_{mode}.json'
    lr = read(config)['selected_lr']
    return work / f'runs/{task}_{family}_{mode}_lr_{lr:g}_seed_{seed}'


def main():
    data = torch.load(STORE / 'chuandian_transfer.pt', map_location='cpu', weights_only=False)
    plans = {pool: read(BASE / f'configs/chuandian_linear_probe{suffix}_20261105/study_plan.json')
             for pool, suffix in [('last', ''), ('mean', '_mean')]}
    for key in ['data_sha256', 'source_checkpoints', 'seeds', 'fine_grid', 'head_lr', 'weight_decay', 'fine_loss', 'schedule']:
        assert plans['last'][key] == plans['mean'][key], key
    audits = []
    for task, family, seed, pretrained in itertools.product(data['tasks'], FAMILIES, range(3), [False, True]):
        name = f'{task}_{family}_{"pretrained" if pretrained else "random"}_seed_{seed}.npz'
        with np.load(STORE / 'linear_probe/features' / name) as a, np.load(STORE / 'linear_probe_mean/features' / name) as b:
            for key in ['encoder_sha256', 'random_encoder_sha256', 'origins_sha256']:
                assert a[key].item() == b[key].item(), (name, key)
            assert a['features'].shape == b['features'].shape
            assert np.array_equal(a['features'][:, -1], b['features'][:, -1])
        audits.append(dict(feature_cache=name, identical_encoder_weights=True, identical_gap_and_targets=True))
    frames = []
    for pool, suffix in [('last', ''), ('mean', '_mean')]:
        report = BASE / f'reports/chuandian_linear_probe{suffix}_20261105'
        assert read(report / 'final_audit.json')['passed']
        frame = pd.read_csv(report / 'task_metrics.csv')
        frames.append(frame.assign(pool=pool))
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(REPORTS / 'readout_metrics.csv', index=False)
    decisions, comparisons, records = [], [], []
    for task_name, family, mode in itertools.product(data['tasks'], FAMILIES, MODES):
        task = data['tasks'][task_name]
        ids = task['splits']['test']
        y = task['labels'][ids]
        key = 'auprc' if task_name == 'classification' else 'mae'
        values, validation, weights, heads = {}, {}, {}, {}
        for pool in ['last', 'mean']:
            prediction, val, initial, head_initial = [], [], [], []
            for seed in range(3):
                directory = folder(pool, task_name, family, mode, seed)
                complete = read(directory / 'training_complete.json')
                with np.load(directory / 'predictions_test.npz') as result:
                    assert np.array_equal(result['origin'], task['origins'][ids])
                    assert np.array_equal(result['target'], y)
                    prediction.append(result['prediction'])
                val.append(complete['validation'][key])
                initial.append(complete['random_encoder_sha256'])
                head_initial.append(complete.get('head_initial_sha256'))
                if mode.endswith('_probe'):
                    assert complete['trainable_parameters'] == 66
            values[pool] = np.stack(prediction)
            validation[pool] = float(np.mean(val))
            weights[pool] = initial
            heads[pool] = head_initial
        assert weights['last'] == weights['mean']
        if mode in ['scratch', 'full']:
            assert heads['last'] == heads['mean']
        selected = ('mean' if validation['mean'] > validation['last'] else 'last') if task_name == 'classification' else ('mean' if validation['mean'] < validation['last'] else 'last')
        decisions.append(dict(task=task_name, family=family, mode=mode, selected=selected,
                              validation=validation, test_used=False))
        if task_name == 'classification':
            delta = np.mean([summary.average_precision_score(y, b) - summary.average_precision_score(y, a)
                             for a, b in zip(values['last'], values['mean'])])
            blocks = [3, 21]
        else:
            delta = np.mean(np.abs(6 * (np.clip(values['last'], 0, 1) - y)) -
                            np.abs(6 * (np.clip(values['mean'], 0, 1) - y)))
            blocks = [18, 54]
        comparison = dict(task=task_name, family=family, mode=mode, mean_improvement=float(delta),
            positive_favors_mean=True, intervals=[summary.paired_interval(y, values['last'], values['mean'],
                task_name == 'classification', block, draws=2000) for block in blocks])
        comparisons.append(comparison)
        group = metrics.loc[(metrics.task == task_name) & (metrics.family == family) &
                            (metrics['mode'] == mode) & (metrics.features == 'encoder_gap')]
        row = dict(task=task_name, family=family, mode=mode, selected_by_validation=selected)
        for pool in ['last', 'mean']:
            subset = group.loc[group.pool == pool]
            row[pool] = float(subset[key].mean())
            row[pool + '_std'] = float(subset[key].std(ddof=1))
        row['mean_improvement'] = float(delta)
        records.append(row)
        print('READOUT_COMPARISON', task_name, family, mode, 'validation_selected', selected, flush=True)
    pd.DataFrame(records).to_csv(REPORTS / 'readout_comparison.csv', index=False)
    output = dict(comparisons=comparisons, validation_selected=decisions, controls=audits,
        notes='Same 64-D readout, gap and 66 affine-head parameters. Pool selection uses mean validation metric across3seeds; ties prefer Last. Historical test set already inspected: exploratory follow-up.')
    (REPORTS / 'readout_comparison.json').write_text(json.dumps(output, indent=2))
    lines = ['# Last 与 masked Mean 的等维对照', '',
        '模型均为两层64维；只改变最后一层隐藏序列的读出方式。Mean仅对有效事件求均值。拼接相同gap后，预测头均为65→1，共66个参数。',
        'Last复用已完成实验；Mean使用相同数据、源权重、随机初始化、损失、学习率网格、训练预算和三个种子。标准化仅拟合训练集。',
        '每个任务/模型/训练方式按三个种子的平均验证指标选择池化，平局保留Last。以下同时报告两种固定读出，测试指标不参与选择。', '',
        '|任务|模型|方式|Last|Mean|验证集选择|', '|---|---|---|---|---|---|']
    for row in records:
        lines.append(f"|{row['task']}|{row['family']}|{row['mode']}|{row['last']:.5f} ± {row['last_std']:.5f}|{row['mean']:.5f} ± {row['mean_std']:.5f}|{row['selected_by_validation']}|")
    lines += ['', '分类为AUPRC（越高越好）；回归为MAE，单位为震级（越低越好）。', '',
        '## 冻结探针下的预训练收益', '', '|任务|模型|Last：随机→预训练|Mean：随机→预训练|', '|---|---|---|---|']
    lookup = {(r['task'], r['family'], r['mode']): r for r in records}
    for task_name, family in itertools.product(data['tasks'], FAMILIES):
        a, b = [lookup[(task_name, family, mode)] for mode in ['random_probe', 'pretrained_probe']]
        lines.append(f"|{task_name}|{family}|{a['last']:.5f} → {b['last']:.5f}|{a['mean']:.5f} → {b['mean']:.5f}|")
    lines += ['', '## 不确定性和边界', '',
        'readout_comparison.json给出成对种子及循环时间块的2000次重采样区间；正改善量表示Mean优于Last。分类块长30/210天，回归180/540天。',
        '该历史测试集此前已看过，本轮为探索性补充。验证集选择某一池化不保证其测试指标更好，报告中保留这种差异。',
        '两种池化的24份特征缓存均验证编码器权重、随机初始权重、预测起点及gap相同；两组数据和源模型校验值一致。']
    (REPORTS / 'readout_comparison_zh.md').write_text('\n'.join(lines) + '\n')
    (REPORTS / 'readout_audit.json').write_text(json.dumps(dict(passed=True, compared_groups=20,
        feature_caches=24, controls=audits, same_data=True, same_source_checkpoints=True,
        same_affine_parameters=66, validation_only_pool_selection=True), indent=2))
    print('READOUT_COMPARISON_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
