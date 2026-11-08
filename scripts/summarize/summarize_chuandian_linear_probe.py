"""Score precommitted affine-head comparisons after all validation-only fits."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

STORE = Path('/root/autodl-fs/em_eqf_training/mamba3_chuandian_transfer_20261105')
def load_study(pool='last'):
    global study, c
    folder = 'linear_probe_mean' if pool == 'mean' else 'linear_probe'
    spec = importlib.util.spec_from_file_location('affine_experiment_' + pool, STORE / folder / 'frozen_runner.py')
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    if hasattr(study, 'configure_readout'):
        study.configure_readout(pool)
    c = study.c


load_study()


def read(path):
    return json.loads(Path(path).read_text())


def probe_predictions(data, folder, result):
    path = study.feature_path(result['task'], result['family'], result['seed'], result['mode'] == 'pretrained_probe')
    with np.load(path) as features, np.load(folder / 'linear_head.npz') as head:
        x = features['features'][:, head['columns']]
        float32_prediction = ((x - head['mean']) / head['scale']) @ head['weight'] + head['bias']
        assert features['encoder_sha256'].item() == result['encoder_sha256']
        columns = head['columns'].copy()
    # Preserve the solver's original precision. Large cancelling coefficients
    # in unpenalized quantile fits cannot be safely exported as FP32 weights.
    reference = folder / 'linear_head_reference.joblib'
    if reference.exists():
        payload = joblib.load(reference)
        assert payload['data_sha256'] == c.sha(study.core.DATA)
        assert payload['features_sha256'] == c.sha(path)
        scaler, estimator = payload['scaler'], payload['estimator']
    else:
        task = data['tasks'][result['task']]
        tr = task['splits']['train']
        scaler = StandardScaler().fit(x[tr])
        if result['task'] == 'classification':
            estimator = LogisticRegression(C=result['regularization'], max_iter=3000, tol=1e-9, solver='lbfgs')
        else:
            estimator = QuantileRegressor(quantile=.5, alpha=result['regularization'], solver='highs')
        estimator.fit(scaler.transform(x)[tr], task['labels'][tr])
        joblib.dump(dict(scaler=scaler, estimator=estimator, columns=columns,
                         data_sha256=c.sha(study.core.DATA), features_sha256=c.sha(path)), reference)
    z = scaler.transform(x)
    prediction = estimator.decision_function(z) if result['task'] == 'classification' else estimator.predict(z)
    validation = data['tasks'][result['task']]['splits']['validation']
    drift = float(np.max(np.abs(prediction[validation] - float32_prediction[validation])))
    c.save(folder / 'probe_precision_audit.json', dict(reference_sha256=c.sha(reference),
        coefficient_max_abs=float(np.abs(estimator.coef_).max()), validation_prediction_max_float32_export_error=drift,
        original_regularization_unchanged=True, reference_refit_training_only=True,
        original_float32_head_retained_for_finetuning_provenance=True))
    return prediction


def paired_interval(y, a, b, classification, block, seed=819, draws=2000):
    """Resample paired seeds and circular time blocks, never individual windows."""
    rng = np.random.default_rng(seed)
    n, seeds = a.shape[1], a.shape[0]
    values = []
    for _ in range(draws):
        starts = rng.integers(n, size=int(np.ceil(n / block)))
        indices = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        selected_seeds = rng.integers(seeds, size=seeds)
        if classification:
            if len(np.unique(y[indices])) < 2:
                continue
            score = np.mean([average_precision_score(y[indices], b[s, indices]) -
                             average_precision_score(y[indices], a[s, indices]) for s in selected_seeds])
        else:
            actual = 3 + 6 * y[indices]
            pa = 3 + 6 * np.clip(a[selected_seeds][:, indices], 0, 1)
            pb = 3 + 6 * np.clip(b[selected_seeds][:, indices], 0, 1)
            score = np.mean(np.abs(pa - actual) - np.abs(pb - actual))
        values.append(score)
    return dict(block_windows=block, block_days=block * 10, draws=len(values),
                low=float(np.quantile(values, .025)), high=float(np.quantile(values, .975)))


def baseline_intervals(draws):
    data = torch.load(study.core.DATA, map_location='cpu', weights_only=False)
    comparisons = []
    for task_name, family in itertools.product(data['tasks'], study.FAMILIES):
        task = data['tasks'][task_name]
        te = task['splits']['test']
        y = task['labels'][te]
        b = np.stack([np.load(study.RUNS / f'{task_name}_{family}_pretrained_probe_encoder_gap_seed_{seed}/predictions_test.npz')['prediction'] for seed in study.SEEDS])
        if task_name == 'classification':
            gap_path = study.RUNS / f'{task_name}_mamba2_legacy_random_probe_gap_only_seed_0/predictions_test.npz'
            gap = np.load(gap_path)['prediction']
            rf = []
            for seed in study.SEEDS:
                p = study.BASE / f'runs/chuandian_warmup_guard_20261105/classification_rf_seed_{seed}/predictions_test.npz'
                with np.load(p) as prediction:
                    assert np.array_equal(prediction['origin'], task['origins'][te])
                    assert np.array_equal(prediction['target'], y)
                    rf.append(prediction['prediction'])
            baselines = [('gap_only', np.stack([gap] * 3)), ('rf', np.stack(rf))]
            blocks = [3, 21]
        else:
            constant = np.median(task['labels'][task['splits']['train']])
            baselines = [('training_median', np.full_like(b, constant))]
            blocks = [18, 54]
        for name, a in baselines:
            if task_name == 'classification':
                delta = np.mean([average_precision_score(y, pb) - average_precision_score(y, pa) for pa, pb in zip(a, b)])
            else:
                delta = np.mean(np.abs(6 * (np.clip(a, 0, 1) - y)) - np.abs(6 * (np.clip(b, 0, 1) - y)))
            comparisons.append(dict(task=task_name, family=family, baseline=name, transferred='pretrained_probe',
                improvement=float(delta), positive_is_better=True,
                intervals=[paired_interval(y, a, b, task_name == 'classification', block, draws=draws) for block in blocks]))
    c.save(study.REPORTS / 'baseline_uncertainty.json', dict(comparisons=comparisons,
        interpretation='Descriptive follow-up contrasts against existing simple/strong baselines; no model or hyperparameter selection.'))
    path = study.REPORTS / 'results_zh.md'
    marker = '\n## 与基线的配对不确定性\n'
    text = path.read_text().split(marker)[0]
    lines = [marker, '以下为主线性探针相对基线的描述性补充；正值表示改善。报告更长时间块（分类210天、回归540天）的95%区间。', '',
             '|任务|编码器|基线|改善量|95%区间|', '|---|---|---|---|---|']
    for item in comparisons:
        interval = item['intervals'][-1]
        lines.append(f"|{item['task']}|{item['family']}|{item['baseline']}|{item['improvement']:.5f}|[{interval['low']:.5f}, {interval['high']:.5f}]|")
    path.write_text(text + '\n'.join(lines) + '\n')
    print('BASELINE_INTERVALS_COMPLETE', flush=True)


def main(draws):
    assert (study.REPORTS / 'training_complete.json').exists()
    plan = read(study.CONFIGS / 'study_plan.json')
    assert c.sha(study.WORK / 'frozen_runner.py') == plan['runner_sha256']
    assert c.sha(study.CORE_PATH) == plan['core_sha256']
    assert c.sha(study.core.DATA) == plan['data_sha256']
    data = study.setup()
    rows, predictions, audits = [], {}, []
    folders = sorted(study.RUNS.glob('*probe*/training_complete.json'))
    for task, family, seed, mode in itertools.product(data['tasks'], study.FAMILIES, study.SEEDS, study.FINE_MODES):
        lr = read(study.CONFIGS / f'lr_{task}_{family}_{mode}.json')['selected_lr']
        folders.append(study.RUNS / f'{task}_{family}_{mode}_lr_{lr:g}_seed_{seed}/training_complete.json')
    assert len(folders) == 86, len(folders)
    for complete in folders:
        result = read(complete)
        task_name, family, mode, seed = [result[k] for k in ['task', 'family', 'mode', 'seed']]
        task = data['tasks'][task_name]
        folder = complete.parent
        va, te = task['splits']['validation'], task['splits']['test']
        features = result['features']
        if mode.endswith('_probe'):
            assert result['encoder_unchanged']
            assert result['head_sha256'] == c.sha(folder / 'linear_head.npz')
            all_predictions = probe_predictions(data, folder, result)
            val_prediction, prediction = all_predictions[va], all_predictions[te]
        else:
            model = study.AffineTransfer(family, seed, mode != 'scratch', False)
            state = torch.load(folder / 'best_model.pth', map_location='cuda:0', weights_only=False)
            model.load_state_dict(state['model_state_dict'], strict=True)
            val_prediction = study.core.predict(model, data, task, va)
            prediction = study.core.predict(model, data, task, te)
            assert c.sha(folder / 'best_model.pth') == result['checkpoint_sha256']
            del model
            torch.cuda.empty_cache()
        validation = study.core.metrics(task_name, task['labels'][va], val_prediction, result['validation'].get('threshold'))
        key = 'auprc' if task_name == 'classification' else 'mae'
        error = abs(validation[key] - result['validation'][key])
        assert error < 2e-5, (str(folder), validation, result['validation'])
        value = study.core.metrics(task_name, task['labels'][te], prediction, result['validation'].get('threshold'))
        if task_name == 'regression':
            value['mae_unclipped'] = float(np.mean(np.abs(6 * (prediction - task['labels'][te]))))
        np.savez_compressed(folder / 'predictions_test.npz', origin=task['origins'][te], target=task['labels'][te], prediction=prediction)
        c.save(folder / 'metrics_test.json', value)
        rows.append(dict(task=task_name, family=family, mode=mode, features=features, seed=seed, n=len(te), **value))
        predictions[(task_name, family, mode, features, seed)] = prediction
        audits.append(dict(run=folder.name, validation_metric_replay_error=error, origin_sha256=hashlib.sha256(task['origins'][te].tobytes()).hexdigest()))
    frame = pd.DataFrame(rows)
    frame.to_csv(study.REPORTS / 'task_metrics.csv', index=False)
    summary = []
    for (task, family, mode, features), group in frame.groupby(['task', 'family', 'mode', 'features'], sort=False):
        columns = ['auprc', 'auroc', 'f1'] if task == 'classification' else ['mae', 'rmse', 'r2']
        summary.append(dict(task=task, family=family, mode=mode, features=features, seeds=len(group),
            metrics={column: dict(mean=float(group[column].mean()), std=float(group[column].std(ddof=1)) if len(group) > 1 else 0.) for column in columns}))
    c.save(study.REPORTS / 'summary.json', dict(groups=summary, rows=rows))
    comparisons = []
    for task_name, family in itertools.product(data['tasks'], study.FAMILIES):
        task = data['tasks'][task_name]
        y = task['labels'][task['splits']['test']]
        for baseline, transferred, features in [('random_probe', 'pretrained_probe', 'encoder_gap'),
                ('random_probe', 'pretrained_probe', 'encoder_only'), ('scratch', 'full', 'encoder_gap'), ('scratch', 'lpft', 'encoder_gap')]:
            a = np.stack([predictions[(task_name, family, baseline, features, seed)] for seed in study.SEEDS])
            b = np.stack([predictions[(task_name, family, transferred, features, seed)] for seed in study.SEEDS])
            if task_name == 'classification':
                delta = np.mean([average_precision_score(y, pb) - average_precision_score(y, pa) for pa, pb in zip(a, b)])
                blocks = [3, 21]
            else:
                delta = np.mean(np.abs(6 * (np.clip(a, 0, 1) - y)) - np.abs(6 * (np.clip(b, 0, 1) - y)))
                blocks = [18, 54]
            comparisons.append(dict(task=task_name, family=family, baseline=baseline, transferred=transferred, features=features,
                improvement=float(delta), positive_is_better=True,
                intervals=[paired_interval(y, a, b, task_name == 'classification', block, draws=draws) for block in blocks]))
            print('COMPARISON_COMPLETE', task_name, family, baseline, transferred, features, float(delta), flush=True)
    c.save(study.REPORTS / 'paired_uncertainty.json', dict(comparisons=comparisons,
        method='Paired resampling of three seeds and circular time blocks; block-size sensitivity, overlapping windows.',
        limitations='Exploratory intervals; one target region, three pretrained seeds, dependent historical test windows.'))
    # Verify that pretrained initialization is the sole initialization change in matched scratch/full fits.
    initialization_checks = []
    for task, family, seed in itertools.product(data['tasks'], study.FAMILIES, study.SEEDS):
        records = {}
        for mode in ['scratch', 'full']:
            lr = read(study.CONFIGS / f'lr_{task}_{family}_{mode}.json')['selected_lr']
            records[mode] = read(study.RUNS / f'{task}_{family}_{mode}_lr_{lr:g}_seed_{seed}/training_complete.json')
        assert records['scratch']['head_initial_sha256'] == records['full']['head_initial_sha256']
        assert records['scratch']['random_encoder_sha256'] == records['full']['random_encoder_sha256']
        initialization_checks.append(dict(task=task, family=family, seed=seed, matched=True))
    c.save(study.REPORTS / 'final_audit.json', dict(passed=True, data_sha256=plan['data_sha256'], runs=len(rows),
        preflight=read(study.REPORTS / 'preflight.json'), validation_replay=audits, initialization=initialization_checks,
        test_only_after_training=True, original_source_unchanged=True))
    old = study.BASE / 'reports/chuandian_warmup_guard_20261105'
    assert read(old / 'data.json')['sha256'] == plan['data_sha256']
    previous = pd.read_csv(old / 'task_metrics.csv')
    previous.to_csv(study.REPORTS / 'previous_task_metrics.csv', index=False)
    readout = getattr(study, 'READOUT', 'last')
    lines = [f'# 川滇严格线性探针及微调验证（{readout}）', '',
        f'两层64维编码器；固定{readout}读出（mean排除padding）；主对照拼接原有gap，仅65→1单层线性头，共66个参数。',
        '冻结编码器时缓存特征；特征标准化仅拟合目标训练集。附64维纯编码器探针及gap-only对照。',
        '冻结线性求解器以原精度joblib保存并重放；神经网络使用原FP32参数。FP32导出头仅供LP-FT初始化，求解器精度与神经网络精度分别重放核验。',
        '来源不含川滇，原始源事件均早于2011-02-12；复用原来经过未来窗口清除的数据划分、预测起点和标签。',
        '分类探针为逻辑回归；回归探针为0.5分位数线性回归，以MAE目标评估条件中位数。',
        '全量训练和LP-FT使用BCE/中位数pinball损失。每组在seed0验证集比较相同三个学习率，随后固定用于seed1/2；至少25轮再允许早停。',
        '本轮损失、预测头和调参协议与旧MLP结果不同；预训练收益只在本轮匹配对照内判定。旧RF/LSTM结果作为外部参照。', '',
        'random_probe：随机编码器冻结；pretrained_probe：预训练编码器冻结；scratch：随机初始化全量训练；full：预训练全量微调；lpft：先训练线性探针再全量微调。', '',
        '## 主对照：相同65维输入和单层预测头', '',
        '|任务|编码器|方法|主指标均值±种子标准差|', '|---|---|---|---|']
    for item in summary:
        if item['features'] != 'encoder_gap':
            continue
        metric = item['metrics']['auprc' if item['task'] == 'classification' else 'mae']
        lines.append(f"|{item['task']}|{item['family']}|{item['mode']}|{metric['mean']:.5f} ± {metric['std']:.5f}|")
    lines += ['', '## 探针诊断', '', '|任务|编码器|方法|输入|主指标均值±种子标准差|', '|---|---|---|---|---|']
    for item in summary:
        if item['features'] == 'encoder_gap':
            continue
        metric = item['metrics']['auprc' if item['task'] == 'classification' else 'mae']
        family = '无编码器' if item['features'] == 'gap_only' else item['family']
        mode = 'gap_only' if item['features'] == 'gap_only' else item['mode']
        lines.append(f"|{item['task']}|{family}|{mode}|{item['features']}|{metric['mean']:.5f} ± {metric['std']:.5f}|")
    lines += ['', '## 原有RF和LSTM参照（相同预测目标）', '', '|任务|模型|主指标均值±种子标准差|', '|---|---|---|']
    for (task_name, family), group in previous.loc[previous.family.isin(['rf', 'lstm_features', 'lstm_raw'])].groupby(['task', 'family']):
        key = 'auprc' if task_name == 'classification' else 'mae'
        lines.append(f"|{task_name}|{family}|{group[key].mean():.5f} ± {group[key].std(ddof=1):.5f}|")
    lines += ['', '## 训练常量基线', '', '|任务|方法|测试指标|', '|---|---|---|']
    for task_name, task in data['tasks'].items():
        tr, te = task['splits']['train'], task['splits']['test']
        for method in ['mean', 'median']:
            constant = float(getattr(np, method)(task['labels'][tr]))
            if task_name == 'classification':
                constant = np.log(np.clip(constant, 1e-8, 1-1e-8) / (1-np.clip(constant, 1e-8, 1-1e-8)))
            values = study.core.metrics(task_name, task['labels'][te], np.full(len(te), constant), threshold=.5)
            lines.append(f"|{task_name}|training_{method}|{values['auprc' if task_name == 'classification' else 'mae']:.5f}|")
    lines += ['', '## 判定边界', '',
        '预训练表征：pretrained_probe 对 random_probe；预训练初始化：full 对 scratch；分阶段微调：lpft 对 scratch。',
        '统计区间见paired_uncertainty.json，正值表示预训练改善。按时间块与种子成对重采样，并报告两种块长；不把重叠窗口当成独立样本。',
        '该历史测试集此前已经看过，因此本轮属于探索性补充，不能当成全新留出集上的确认实验。仍只有一个迁移目标地区。', '',
        '数据、编码器来源、环境与冻结代码见plan.json/runtime_environment.json/final_audit.json；逐种子指标见task_metrics.csv。']
    (study.REPORTS / 'results_zh.md').write_text('\n'.join(lines) + '\n')
    baseline_intervals(draws)
    print('LINEAR_TRANSFER_FINALIZED', str(study.REPORTS), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap-draws', type=int, default=2000)
    parser.add_argument('--baselines-only', action='store_true')
    parser.add_argument('--pool', choices=['last', 'mean'], default='last')
    args = parser.parse_args()
    if args.pool != 'last':
        load_study(args.pool)
    if args.baselines_only:
        baseline_intervals(args.bootstrap_draws)
    else:
        main(args.bootstrap_draws)
