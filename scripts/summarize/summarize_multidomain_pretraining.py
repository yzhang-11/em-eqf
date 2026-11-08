"""Audit and summarize the fixed multi-domain pilot without selecting on test."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

import torch

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT/'experiments/revision/multiregion_pretraining'
RUNS = STUDY/'runs/multidomain_id'
REPORTS = STUDY/'reports/multidomain_id'
METHODS = ['stats_film', 'stats_concat', 'id_concat', 'stats_id_concat', 'stats_id_film', 'id_input']
TRANSFER = ['stats_film', 'stats_id_concat', 'stats_id_film', 'id_input']
REGIONS = ['italy', 'japan_kyushu', 'new_zealand']
LABELS = dict(stats_film='统计条件 FiLM', stats_concat='统计条件拼接', id_concat='ID 在预测头拼接',
              stats_id_concat='统计条件 + ID 拼接', stats_id_film='统计条件 + ID FiLM',
              id_input='每个事件输入 ID', local='每区单独训练')


def read(path):
    return json.loads(path.read_text())


def save(name, obj):
    (REPORTS/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_sd(values):
    values = list(values)
    return dict(mean=statistics.mean(values), sd=statistics.stdev(values) if len(values)>1 else 0., n=len(values))


def write_csv(name, rows):
    with (REPORTS/name).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    torch.set_num_threads(1)
    data_path = ROOT/'data/MultiRegion/catalogs/pilot_2007_2019.pt'
    data_sha = digest(data_path)
    vocabulary = read(REPORTS/'region_vocabulary.json')
    rows = [read(p) for p in sorted(RUNS.glob('*/metrics.json'))]
    by_name = {r['config']['name']:r for r in rows}
    assert len(rows)==len(by_name), 'Duplicate run names'
    audit=[]
    for row in rows:
        cfg=row['config']; path=Path(row['checkpoint'])
        assert cfg['dataset_sha256']==data_sha
        assert cfg['region_id_map']==vocabulary
        assert path.parent.name==cfg['name']
        assert digest(path)==row['checkpoint_sha256']
        payload=torch.load(path, map_location='cpu', weights_only=False)
        assert payload['config']==cfg
        assert all(torch.isfinite(v).all().item() for v in payload['model_state_dict'].values())
        assert 0 < row['best_step'] <= cfg['optimizer_steps']
        assert abs(row['test']['macro']['total']-sum(row['test']['macro'][k] for k in ['time','mag']))<1e-10
        if cfg['name'].startswith('heldout_'):
            assert not set(cfg['sources']) & set(cfg['targets'])
            assert row['test_uses_unknown_id']
        if cfg['initialization']:
            assert digest(Path(cfg['initialization']))==cfg['initialization_sha256']
        audit.append(dict(run=cfg['name'], checkpoint_sha256=row['checkpoint_sha256'], finite_weights=True))
    expected={f'known_{m}_seed_{s}' for m in METHODS for s in range(3)}
    expected|={f'local_{r}_seed_{s}' for r in REGIONS for s in range(3)}
    expected|={f'heldout_{r}_{m}_seed_{s}' for r in REGIONS for m in TRANSFER for s in range(3)}
    assert expected<=set(by_name), f'Missing jobs: {expected-set(by_name)}'
    known=[]; region_rows=[]; seed_rows=[]
    for method in METHODS+['local']:
        seed_values=[]
        for seed in range(3):
            if method=='local':
                runs=[by_name[f'local_{r}_seed_{seed}'] for r in REGIONS]
                metrics={r:x['test']['regions'][r] for r,x in zip(REGIONS,runs)}
                val=statistics.mean(x['validation']['macro']['total'] for x in runs)
            else:
                run=by_name[f'known_{method}_seed_{seed}'];runs=[run]
                metrics=run['test']['regions'];val=run['validation']['macro']['total']
            total={k:statistics.mean(metrics[r][k] for r in REGIONS) for k in ['time','mag','total']}
            seed_values.append(dict(validation=val, **total))
            seed_rows.append(dict(method=method,seed=seed,validation=val,**total))
            for r in REGIONS:
                region_rows.append(dict(method=method,seed=seed,region=r,**metrics[r]))
        known.append(dict(method=method, **{k:mean_sd(x[k] for x in seed_values) for k in seed_values[0]},
                          parameters=runs[0]['total_parameters']))
    known_by_method={x['method']:x for x in known}
    selected=min(METHODS,key=lambda m:known_by_method[m]['validation']['mean'])
    default=read(REPORTS/'default_model.json')
    assert default['method']==selected
    assert default['checkpoint']==by_name[f'known_{selected}_seed_0']['checkpoint']
    for method in METHODS:
        assert abs(default['candidate_validation'][method]-known_by_method[method]['validation']['mean'])<1e-10
    heldout=[]; heldout_regions=[]
    for method in TRANSFER:
        per_seed=[]
        for seed in range(3):
            rr=[by_name[f'heldout_{r}_{method}_seed_{seed}'] for r in REGIONS]
            per_seed.append({k:statistics.mean(x['test']['macro'][k] for x in rr) for k in ['time','mag','total']})
        heldout.append(dict(method=method, **{k:mean_sd(x[k] for x in per_seed) for k in ['time','mag','total']}))
        for region in REGIONS:
            values=[by_name[f'heldout_{region}_{method}_seed_{s}']['test']['macro']['total'] for s in range(3)]
            heldout_regions.append(dict(method=method,region=region,**mean_sd(values)))
    qa=read(REPORTS/'runtime_qa.json')
    assert qa['default_loader_reproduced'] and qa['legacy_checkpoint_reproduced']
    assert all(x['encoder_frozen'] and x['other_region_embeddings_unchanged'] for x in qa['freeze_checks'])
    adaptation=[]
    for baseline in qa['mean_initialization']:
        target=baseline['target']; method=baseline['method']
        selection=read(REPORTS/f'transfer_selection_{target}.json')
        candidates={m:statistics.mean(by_name[f'heldout_{target}_{m}_seed_{s}']['validation']['macro']['total']
                                     for s in range(3)) for m in TRANSFER if m!='stats_film'}
        assert method==selection['method']==min(candidates,key=candidates.get)
        parent=by_name[f'heldout_{target}_{method}_seed_0']
        options=[dict(target=target,method=method,mode='zero_id',
                      validation=baseline['zero_id_validation']['total'],test=parent['test']['macro']['total'],updates=0),
                 dict(target=target,method=method,mode='mean_initialization',
                      validation=baseline['validation']['total'],test=baseline['test']['total'],updates=0)]
        for mode in ['embedding','heads','scratch']:
            matched=[x for x in rows if x['config']['targets']==[target] and
                     x['config']['method']==method and x['config']['adaptation']==mode]
            assert len(matched)==1,(target,method,mode)
            x=matched[0]
            options.append(dict(target=target,method=method,mode=mode,validation=x['validation']['macro']['total'],
                                test=x['test']['macro']['total'],updates=x['config']['optimizer_steps'],
                                run=x['config']['name']))
        winner=min(options,key=lambda x:x['validation'])['mode']
        for option in options:
            option['selected_by_target_validation']=option['mode']==winner
        adaptation.extend(options)
    save('summary.json',dict(dataset_sha256=data_sha,completed_fits=len(rows),seeds=[0,1,2],known=known,
                             heldout=heldout,adaptation=adaptation,default=default))
    save('checkpoint_audit.json',dict(checked=len(audit),passed=True,runs=audit))
    write_csv('known_by_seed.csv',seed_rows)
    write_csv('known_by_region.csv',region_rows)
    write_csv('heldout_by_region.csv',heldout_regions)
    write_csv('adaptation.csv',[{**dict(run=''),**x} for x in adaptation])
    write_csv('all_runs.csv',[dict(run=x['config']['name'],method=x['config']['method'],seed=x['config']['seed'],
                                  sources='|'.join(x['config']['sources']),targets='|'.join(x['config']['targets']),
                                  adaptation=x['config']['adaptation'] or '',steps=x['config']['optimizer_steps'],
                                  best_step=x['best_step'],validation=x['validation']['macro']['total'],
                                  time=x['test']['macro']['time'],mag=x['test']['macro']['mag'],
                                  total=x['test']['macro']['total'],checkpoint=x['checkpoint']) for x in rows])
    lines=['# 多地区联合训练与地区 ID 实验（2026-11-04）','',
           '**结论：已实现地区 ID embedding、统计条件及 DeepAR 式逐事件输入 ID；当前默认采用验证集选出的统计条件 FiLM。ID 未稳定超过统计条件方案。**','',
           f'本轮完成 {len(rows)} 次拟合：已知地区 27 次、留区预训练 36 次、适配和从头训练对照 12 次。原先 30 次试验另存，未混入本表。',
           '本轮适配中 3 次为输入 ID 扩展后新西兰重新选源模型的结果；旧新西兰适配文件保留，但不混入最终适配比较。','',
           '## 数据和设置','',
           '- 实际使用日本九州 24,926、意大利 14,273、新西兰 36,542 个事件，共 75,741。已下载的全部百万级目录尚未用于训练。',
           '- 数据范围 2007—2019；各区最早 256 个事件仅校准统计量，其后至 2014 年训练；2014—2015 验证，2016—2019 测试。',
           '- 每窗 64 个历史事件、128 个评分事件；步长 128，评分事件不重复。训练窗数日本/意大利/新西兰为 64/48/194；验证 20/10/20；测试 106/48/66。',
           '- Mc 分别为 2.0/2.5/3.0，离散震级间隔 0.1。阈值是原型设置，尚待正式目录完备性检验。Italy/NZ 的混合震级类型编码为 unknown，未伪装为统一 ML。',
           '- 所有组使用相同逐序列时间尺度、Mc/参考 b 及离散震级概率计算。ID-only 仅表示不额外将统计量送入条件层，仍使用这些必需的尺度和概率信息。',
           '- 共享主干为 2 层 Mamba、隐藏维数 64、32 分量时间分布。统计条件 FiLM 有 4,512 个条件层参数；ID 维数 8。预测头拼接为 Linear→SiLU；input-ID 将 ID 拼到每个观测事件。',
           '- 不加 b 水平或 b 平滑辅助损失；AdamW 的常规 weight decay=0.01。旧时间编码、原论文 SWC/SCSN 配置及数据均保留。',
           '- 已知地区联合模型训练 450 步、每区单独模型 150 步，batch=16；等总步数、每区期望抽样次数相同。各区均匀抽样；学习率 3e-4，warmup/cosine、梯度裁剪 1。',
           '- 注意：backbone 配置里的 learning_rate 是构建占位值；实际优化器使用每个任务 config 顶层 learning_rate 和 optimizer_steps。',
           '- 留区实验仅在其余两区训练 300 步；目标区 256 个历史事件用于尺度校准，模型参数不更新。目标区 ID 强制为 0，不使用随机未训练的目标 ID。',
           '- 骨干/检查点选择只看相应训练来源的验证集；三种子 0/1/2。以下 ± 为种子间样本标准差，不是置信区间。NLL 单位 nats/事件，越低越好。连续时间密度项可以为负。','',
           '## 已知地区的未来预测','',
           '先对每个种子的三个地区等权平均，再计算三种子均值和标准差。每区单独训练对应三个独立模型，联合训练对应一个共享模型。','',
           '| 方法 | 验证联合 NLL | 测试联合 NLL | 测试时间项 | 测试震级项 |','|---|---:|---:|---:|---:|']
    for row in known:
        lines.append('| '+LABELS[row['method']]+' | '+' | '.join(f"{row[k]['mean']:.6f} ± {row[k]['sd']:.6f}" for k in ['validation','total','time','mag'])+' |')
    lines+=['','**默认选型为 stats_film，依据是验证集；展示的测试集未参与选型。联合训练的主要收益来自时间项，不能把联合 NLL 的改善当作震级建模已经明显改善。**','',
            '## 未见地区的直接预测','',
            '下面是独立的留一地区实验。它与上表训练来源不同，不能直接相减作为某种方法的提升。ID=0 是固定零向量，训练时并未专门学习未知地区分支；其退化不能证明所有 ID 方法都无法迁移。','',
            '| 方法 | 三地区三种子联合 NLL |','|---|---:|']
    for row in heldout:
        lines.append(f"| {LABELS[row['method']]} | {row['total']['mean']:.6f} ± {row['total']['sd']:.6f} |")
    lines+=['','## 新地区少样本适配（仅种子 0）','',
            '为检验 ID 的适配能力，先只在三种 ID 预训练家族中按源地区验证均值选模型；这一过程不使用目标区验证/测试。该表属于可选 ID 路径，并不替代默认统计条件路径。',
            '目标区早期训练段的前 4 窗（512 个评分事件）适配，随后 2 窗（256 个事件）选检查点和适配策略。每种适配 100 步；embedding LR=1e-3，heads/scratch LR=1e-4；适配 weight decay=0。',
            'mean_initialization 是源地区 ID 向量取均值、零次梯度更新；embedding 仅允许目标 ID 的 8 个坐标变化（完整表有 32 个可训练坐标，其余行梯度屏蔽）；heads 冻结主干、更新预测头/条件层和目标 ID。scratch 用同一少样本训练/验证和 100 步预算从头训练。',
            '下面标记的策略仅依据目标区的历史验证集选择，包含不更新、均值初始化和从头训练候选；不按测试结果挑选。','',
            '| 目标地区 | 源验证选定的 ID 模型 | 策略 | 历史验证 NLL | 测试 NLL | 验证集选择 |','|---|---|---|---:|---:|---|']
    for x in adaptation:
        lines.append(f"| {x['target']} | {x['method']} | {x['mode']} | {x['validation']:.6f} | {x['test']:.6f} | {'是' if x['selected_by_target_validation'] else ''} |")
    lines+=['','**均值初始化本身解释了 ID 适配的大部分改善，进一步只训练目标 embedding 的增益很小；当前不能宣称 ID 小样本学习有稳定优势。**','',
            '## 检查、文件和复现','',
            f'- {len(audit)} 个检查点逐个核对 SHA-256、数据摘要、词表、配置、有限权重及初始化来源；见 checkpoint_audit.json。',
            '- 16 项模型检查通过（区域模型 14、输入 ID 模型 2）。运行时复算默认检查点与旧检查点指标；核对 ID-only 适配只改目标行、预测头适配冻结编码器；见 runtime_qa.json。',
            '- region_vocabulary.json 固定 ID：unknown=0，italy=1，japan_kyushu=2，new_zealand=3。ID 是类别查表，不当作有大小顺序的数值。',
            '- default_model.json / configs/multidomain_id/default.yaml 指向已验证的 seed=0 默认检查点；模型家族按三种子验证均值选定，seed=0 固定作为可加载实例。',
            '- all_runs.csv 汇总全部 75 个任务；known_by_region.csv、known_by_seed.csv、heldout_by_region.csv、adaptation.csv 分别给出各协议明细。','',
            '服务器原有目录：','',
            '```text',str(ROOT/'src/models/tpp/region_mixer_tpp.py'),str(ROOT/'src/models/tpp/global_mixer_tpp.py'),
            str(ROOT/'scripts/run/run_multidomain_pretraining.py'),str(ROOT/'scripts/summarize/summarize_multidomain_pretraining.py'),
            str(STUDY/'configs/multidomain_id'),str(RUNS),str(REPORTS),'```','',
            '```bash','cd /root/autodl-tmp/em_eqf',
            '/root/miniconda3/bin/python scripts/run/run_multidomain_pretraining.py',
            '/root/miniconda3/bin/python scripts/summarize/summarize_multidomain_pretraining.py','```','',
            '已完成任务会先核对配置并复用，不重复训练。默认加载：','',
            '```python','import sys',"sys.path.insert(0, '/root/autodl-tmp/em_eqf/scripts/run')",
            'from run_multidomain_pretraining import load_default_multidomain_model',
            'model, config = load_default_multidomain_model()','```','',
            f'数据 SHA-256：`{data_sha}`。默认检查点 SHA-256：`{default["checkpoint_sha256"]}`。','',
            '当前是 3 地区、短预算原型，少样本只做 1 个种子；尚未训练全部新增目录，尚未验证计数 CRPS、N/M-test 或原论文目标区迁移效果。正式实验需要增加地理分区、审查 Mc 并保持预训练时间截断。','',
            '## 方法依据','',
            '- DeepAR 原论文的类别特征使用可学习 embedding，并随协变量输入自回归网络；小数据实验使用序列身份。https://arxiv.org/abs/1704.04110',
            '- 原版 TimesFM 面向未见序列的零样本预测，预训练不依赖数据集专属协变量；论文也明确允许微调时加入类别 embedding。https://arxiv.org/html/2310.10688v4',
            '- Moirai 的同变量/异变量注意力偏置表达身份关系，支持任意变量数，并非为每个全球实际地区建立独立向量。https://arxiv.org/html/2402.02592v2','']
    (REPORTS/'results.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(completed=len(rows),default=selected,
                          local_nll=known_by_method['local']['total']['mean'],
                          default_nll=known_by_method[selected]['total']['mean'],
                          checkpoint_audit=True),ensure_ascii=False))


if __name__=='__main__':
    main()
