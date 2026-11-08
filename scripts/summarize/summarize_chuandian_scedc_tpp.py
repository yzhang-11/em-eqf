"""Event-weighted conditional time/mark NLL and matched temporal ETAS audit."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

STORE = Path('/root/autodl-fs/em_eqf_training/chuandian_scedc_tpp_20261105')
spec = importlib.util.spec_from_file_location('target_tpp_runner', STORE / 'frozen_runner.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
c = study.c
REPORTS = study.REPORTS


def read(path):
    return json.loads(Path(path).read_text())


def etas_module():
    spec = importlib.util.spec_from_file_location('_chuandian_scedc_etas', STORE / 'frozen_etas.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def score_etas():
    assert (REPORTS / 'training_complete.json').exists()
    etas = etas_module()
    data = torch.load(study.DATA, map_location='cpu', weights_only=False)
    rows, audits = [], []
    for target, s in data['regions'].items():
        low, high = s['val_end'], s['test_end']
        for mode in ['window', 'full']:
            candidates = {classic: read(REPORTS / f'etas/{target}_{mode}{"_classic" if classic else ""}.json') for classic in [False, True]}
            selected = min(candidates, key=lambda k: candidates[k]['validation']['total'])
            for classic, result in candidates.items():
                model = etas.FastETAS(s, mode, classic)
                theta = np.array(result['selected']['theta'])
                context = 1024 if mode == 'window' else None
                value, time_loss, _ = model.evaluate(theta, 'test', context, return_points=True)
                queries = np.unique(np.linspace(low, high-1, min(32, high-low)).astype(int))
                # direct_points anchors windows at train_end; for test, the
                # scoring-block origin must instead be val_end.
                exact = etas.direct_points(dict(s, train_end=low), model.describe(theta), queries, context)
                error = float(np.max(np.abs(time_loss[queries-low] - exact)))
                assert error < 1e-4, (target, mode, classic, error)
                mark_loss = etas.mark_nll(s['magnitude'][low:high], model.mc, model.b)
                windows = [dict(events=min(128, high-first),
                    time=float(time_loss[first-low:first-low+128].mean()),
                    mag=float(mark_loss[first-low:first-low+128].mean()),
                    total=float((time_loss[first-low:first-low+128] + mark_loss[first-low:first-low+128]).mean())) for first in range(low, high, 128)]
                label = f'{target}_etas_{mode}_{"classic" if classic else "subcritical"}'
                c.save(REPORTS / f'etas/{label}_test.json', dict(metrics=value, windows=windows,
                    event_ids=s['event_id'][low:high], exact_test_queries=len(queries), max_absolute_error=error))
                rows.append(dict(target=target, method='etas_'+mode, family='classic' if classic else 'subcritical',
                                 validation_selected=classic==selected, **value))
                audits.append(dict(target=target, fit_mode=mode, family=classic, max_test_nll_error=error,
                    validation_selected=classic==selected, train_optimizer_success=result['selected']['success']))
                del model
    c.save(REPORTS / 'etas/test_results.json', dict(rows=rows, audits=audits, unit='nats/event'))
    print('TARGET_ETAS_SCORED', flush=True)


def target_folder(target, family, mode, seed, updates=800):
    lr = read(study.CONFIGS / f'lr_{target}_{family}_{mode}.json')['lr']
    return study.RUNS / f'{target}_{family}_{mode}_lr_{lr:g}_steps_{updates}_seed_{seed}'


def block_interval(difference, weights, block, draws=2000):
    rng = np.random.default_rng(9127)
    seeds, n = difference.shape
    values = []
    for _ in range(draws):
        start = rng.integers(n, size=int(np.ceil(n/block)))
        ix = ((start[:,None]+np.arange(block)) % n).ravel()[:n]
        si = rng.integers(seeds, size=seeds)
        values.append(float(np.mean(np.sum(difference[si][:,ix]*weights[ix], axis=1)/weights[ix].sum())))
    return dict(block_windows=block, target_events_per_block=block*128,
                low=float(np.quantile(values,.025)), high=float(np.quantile(values,.975)), draws=draws)


def main():
    assert (REPORTS / 'training_complete.json').exists()
    plan = read(study.CONFIGS / 'study_plan.json')
    assert c.sha(study.DATA) == plan['data_sha256']
    assert c.sha(STORE / 'frozen_runner.py') == plan['runner_sha256']
    assert c.sha(STORE / 'frozen_etas.py') == plan['etas_sha256']
    assert c.sha(STORE / 'frozen_summary.py') == read(REPORTS / 'scoring_code.json')['sha256']
    data_audit = read(REPORTS / 'data.json')
    assert all(c.sha(path) == digest for path, digest in data_audit['protected_files'].items())
    for target, source in plan['source_data'].items():
        assert c.sha(source['path']) == source['sha256']
    subprocess.run([plan['etas_environment'], str(STORE/'frozen_summary.py'), '--etas'], check=True)
    study.setup()
    data = torch.load(study.DATA, map_location='cpu', weights_only=False)
    rows, audits, window_scores = [], [], {}
    model_checks = []
    guarded = read(study.CONFIGS / 'budget_guard.json')['triggered']
    for target, family, seed in itertools.product(study.TARGETS, study.FAMILIES, study.SEEDS):
        source_dir = study.source_run(target, family, seed)
        source = read(source_dir/'training_complete.json')
        assert source['backbone'] == plan['backbones'][family]
        assert source['dataset_sha256'] == plan['source_data'][target]['sha256']
        assert c.sha(source_dir/'best_model.pth') == source['checkpoint_sha256']
        source_data = torch.load(study.source_data(target), map_location='cpu', weights_only=False)
        source_model, _ = study.build(family, seed, source_data['vocabulary'])
        source_model.load_state_dict(torch.load(source_dir/'best_model.pth',map_location='cuda:0',weights_only=False)['model_state_dict'], strict=True)
        source_state_sha = c.state_hash(source_model)
        replay, _ = c.evaluate(source_model,c.evaluation_pools(source_data,'val',1024))
        error=max(abs(replay['regions'][r][k]-source['validation']['regions'][r][k]) for r in replay['regions'] for k in ['time','mag','total'])
        assert error<5e-5,(target,family,seed,'source replay',error)
        del source_model
        torch.cuda.empty_cache()
        source_ref=dict(source_validation_replay_error=error)
        modes = ['local','mixed_zero','fine'] + [job['mode']+'_1600' for job in guarded if job['target']==target and job['family']==family]
        for mode in modes:
            train_mode = mode.removesuffix('_1600')
            updates = 1600 if mode.endswith('_1600') else 800
            directory = source_dir if mode=='mixed_zero' else target_folder(target,family,train_mode,seed,updates)
            complete = read(directory/'training_complete.json')
            state = torch.load(directory/'best_model.pth',map_location='cuda:0',weights_only=False)
            assert state['backbone'] == complete['backbone'] == plan['backbones'][family]
            if mode != 'mixed_zero':
                assert complete['dataset_sha256'] == plan['data_sha256']
                if train_mode == 'fine':
                    assert complete['loaded_sha256'] == source_state_sha
            model,_=study.build(family,seed,data['vocabulary'])
            model.load_state_dict(state['model_state_dict'],strict=True)
            assert c.sha(directory/'best_model.pth')==complete['checkpoint_sha256']
            validation,_=c.evaluate(model,{target:c.evaluation_pools(data,'val',1024)[target]})
            if mode!='mixed_zero':
                error=max(abs(validation['regions'][target][k]-complete['validation']['regions'][target][k]) for k in ['time','mag','total'])
                assert error<5e-5,(directory,error)
            metrics,windows=c.evaluate(model,{target:c.evaluation_pools(data,'test',1024)[target]})
            value=metrics['regions'][target]
            expected=data['regions'][target]['test_end']-data['regions'][target]['val_end']
            spec = data['regions'][target]
            event_hash = hashlib.sha256('\n'.join(spec['event_id'][spec['val_end']:spec['test_end']]).encode()).hexdigest()
            assert event_hash == data_audit['targets'][target]['test_event_ids_sha256']
            assert sum(v['events'] for v in windows[target])==expected
            assert abs(value['time']+value['mag']-value['total'])<2e-5
            label=f'{target}_{family}_{mode}_seed_{seed}'
            c.save(REPORTS / f'neural/{label}.json',dict(validation=validation['regions'][target],test=value,
                windows=windows[target],checkpoint=str(directory/'best_model.pth'),checkpoint_sha256=complete['checkpoint_sha256']))
            window_scores[(target,family,mode,seed)]=windows[target]
            rows.append(dict(target=target,family=family,method=mode,seed=seed,events=expected,best_step=state['step'],**value))
            audits.append(dict(target=target,family=family,mode=mode,seed=seed,validation_replay_error=error if mode!='mixed_zero' else None,
                               strict_loading=True,**source_ref))
            del model
            torch.cuda.empty_cache()
        paired_budgets = [800] + ([1600] if 'local_1600' in modes and 'fine_1600' in modes else [])
        for updates in paired_budgets:
            local=read(target_folder(target,family,'local',seed,updates)/'training_complete.json')
            fine=read(target_folder(target,family,'fine',seed,updates)/'training_complete.json')
            assert local['initial_random_sha256']==fine['initial_random_sha256']==source['initial_random_sha256']
            assert local['draw_sha256']==fine['draw_sha256'] and local['exposures']==fine['exposures']
            model_checks.append(dict(target=target,family=family,seed=seed,updates=updates,paired_initialization=True,paired_target_exposures=True))
    frame=pd.DataFrame(rows)
    frame.to_csv(REPORTS/'neural_metrics.csv',index=False)
    summary=[]
    for (target,family,mode),group in frame.groupby(['target','family','method'],sort=False):
        summary.append(dict(target=target,family=family,method=mode,
            metrics={k:dict(mean=float(group[k].mean()),std=float(group[k].std(ddof=1))) for k in ['time','mag','total']}))
    etas=read(REPORTS/'etas/test_results.json')
    intervals=[]
    for target,family in itertools.product(study.TARGETS,study.FAMILIES):
        selected=next(row for row in etas['rows'] if row['target']==target and row['method']=='etas_window' and row['validation_selected'])
        etas_windows=read(REPORTS/f"etas/{target}_etas_window_{selected['family']}_test.json")['windows']
        weights=np.asarray([r['events'] for r in etas_windows])
        modes = ['local','mixed_zero','fine'] + [job['mode']+'_1600' for job in guarded if job['target']==target and job['family']==family]
        matrix={mode:np.stack([[r['total'] for r in window_scores[(target,family,mode,seed)]] for seed in study.SEEDS]) for mode in modes}
        matrix['etas']=np.tile([r['total'] for r in etas_windows],(3,1))
        pairs = [('local','fine'),('local','mixed_zero'),('etas','fine'),('etas','mixed_zero'),('etas','local')]
        for mode in ['local','fine']:
            if mode+'_1600' in matrix:
                pairs.extend([(mode,mode+'_1600'),('etas',mode+'_1600')])
        if 'local_1600' in matrix and 'fine_1600' in matrix:
            pairs.append(('local_1600','fine_1600'))
        for baseline,method in pairs:
            difference=matrix[baseline]-matrix[method]
            intervals.append(dict(target=target,family=family,baseline=baseline,method=method,
                improvement=float(np.mean(np.sum(difference*weights,axis=1)/weights.sum())),positive_is_better=True,
                intervals=[block_interval(difference,weights,b) for b in [1,4,8]]))
    c.save(REPORTS/'summary.json',dict(neural=summary,etas=etas['rows']))
    c.save(REPORTS/'paired_uncertainty.json',dict(comparisons=intervals,method='Paired seeds and circular128-event-window blocks; event-weighted likelihood;1/4/8window sensitivity.'))
    c.save(REPORTS/'final_audit.json',dict(passed=True,neural_runs=len(rows),primary_neural_runs=36,source_runs=12,etas_fits=8,
        all_training_fits=len(list(study.RUNS.glob('*/training_complete.json'))),
        validation_replay=audits,paired_initialization_and_exposure=model_checks,source_excluded_targets=True,
        etas_direct_test_queries=etas['audits'],dataset_sha256=plan['data_sha256'],
        protected_files_unchanged=True,source_data_hashes_verified=True,
        checkpoint_backbones_verified=True,full_tpp_source_weights_verified=True,test_event_ids_verified=True))
    lines=['# 川滇与SCEDC：混合区域TPP、单区域TPP和ETAS', '',
        '本表仅为TPP条件事件时间与震级似然；没有分类/回归预测头或Last/Mean池化。单位nats/事件，越低越好。时间单位为天，震级采用0.1下边界概率质量。',
        '两种骨干均为两层64维。每地使用按相应截止日截取的8个外区目录：川滇源训练<2013、验证至2014-12-30；SCEDC源训练<2003、验证至2004-12-30。完全排除NCEDC以避免SCEDC跨目录重叠；源目录与两地边界均不交叠。目录采用已整理的回溯快照，时间截断不代表已核实当年的发布版本。',
        '混合源每模型6400步。本区从头训练和混合源全量微调均为800个目标区更新，使用相同目标抽样和相同3学习率验证搜索预算。预训练增加了额外计算，未声称总计算量匹配。',
        '统一使用验证最佳检查点，其步数可能早于6400/800；相同目标抽样与曝光的核验针对完整训练预算。具体所选步数见逐种子CSV。',
        '零样本指没有目标区权重更新；使用最早256个目标训练事件进行无监督b和时间尺度校准，不能解释为完全不接触目标历史。',
        '训练历史最多1024事件、每步最多2048目标；测试在相同128目标窗口及1024历史下因果评分。ETAS同窗为主对照，全历史为额外信息对照。', '',
        '|地区|方法|时间NLL|震级NLL|联合NLL|','|---|---|---|---|---|']
    for item in summary:
        m=item['metrics'];fmt=lambda k:f"{m[k]['mean']:.5f} ± {m[k]['std']:.5f}"
        lines.append(f"|{item['target']}|{item['family']}/{item['method']}|{fmt('time')}|{fmt('mag')}|{fmt('total')}|")
    for item in etas['rows']:
        if item['validation_selected']:
            lines.append(f"|{item['target']}|{item['method']}/{item['family']}|{item['time']:.5f}|{item['mag']:.5f}|{item['total']:.5f}|")
    lines+=['', '## 数据和解释', '',
        '|目标地区|外区训练事件|外区验证事件|目标训练事件|目标验证事件|目标测试事件|',
        '|---|---|---|---|---|---|']
    for target in study.TARGETS:
        source_regions = data_audit['sources'][target]['regions']
        source_train = sum(v['train'] for v in source_regions.values())
        source_val = sum(v['validation'] for v in source_regions.values())
        d = data_audit['targets'][target]
        lines.append(f"|{target}|{source_train}|{source_val}|{d['train_scored']}|{d['val_events']}|{d['test_events']}|")
    lines+=['',
        '川滇Mc=3.0：2000开始训练，2015开始验证，2018至2021-05-24测试（1261事件）。SCEDC Mc=2.5：1985开始训练，2005开始验证，2014至2020测试（5005事件）。沿用返修已冻结的主Mc及SCEDC eq-only官方快照。',
        '每地最早256个训练事件只做标定，不作为训练目标；元数据均为训练早期已知信息。测试允许观察过去的真实事件，模型权重固定。',
        '固定Mc不是所有主震后时段完整性的证明。旧目录时间戳按提供值使用；混合源截止日留2天边界余量。仅1个SCEDC同刻事件展开1微秒，物理时间保留。',
        'ETAS为原有Gamma求积递归实现，避免事件两两展开；每种历史拟合经典和次临界两族，每族2个训练似然起点，验证集选族。验证和测试均与直接解析计算核对。未计算空间定位似然。',
        'local/fine预算是否需要更长训练的规则在测试前固定；触发的1600步结果应另列，不替换800步主表。',
        'paired_uncertainty.json给出成对种子及事件块的区间。测试历史此前已存在，本轮为探索性补充；两个地区测试年代不同，优先看逐区结果。',
        '配置和数据来源见plan.json/data.json；逐种子指标（主对照36行，另附触发的1600步敏感性）见neural_metrics.csv，源代码/权重/逐窗似然与最终审计完整保留。']
    (REPORTS/'results_zh.md').write_text('\n'.join(lines)+'\n')
    print('TARGET_TPP_FINALIZED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--etas',action='store_true')
    args=parser.parse_args()
    if args.etas:score_etas()
    else:main()
