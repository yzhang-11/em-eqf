"""Report architecture ablations and ChuanDian encoder-transfer controls."""
from pathlib import Path
import csv
import json
import statistics

ROOT=Path('/root/autodl-tmp/em_eqf')
REG=ROOT/'experiments/revision/multiregion_pretraining'
TARGET=ROOT/'experiments/revision/controls'
UP=REG/'reports/mamba3_oracle_residual_20261105'
DOWN=TARGET/'reports/chuandian_multiregion_transfer_20261105'
RUNS=TARGET/'runs/chuandian_multiregion_transfer_20261105'
STORE=Path('/root/autodl-fs/em_eqf_training/mamba3_chuandian_transfer_20261105')


def read(path):return json.loads(Path(path).read_text())


def stats(values):
    values=list(values)
    return dict(mean=statistics.mean(values),sd=statistics.stdev(values) if len(values)>1 else None,n=len(values))


def fmt(value):return f"{value['mean']:.5f}"+(f" ± {value['sd']:.5f}" if value['sd'] is not None else '')


def target_groups(rows,runs,methods):
    target={}
    for task,metrics in [('classification',['auprc','auroc','f1','balanced_accuracy','brier']),('regression',['mae','rmse','r2','bias'])]:
        result=[]
        for family,mode in methods:
            subset=[r for r in rows if r['task']==task and r['family']==family and r['mode']==mode]
            assert len(subset)==3,(task,family,mode)
            validation=[]
            for seed in [0,1,2]:
                name=f'{task}_rf_seed_{seed}' if family=='rf' else f'{task}_{family}_{mode}_seed_{seed}'
                complete=read(runs/name/'training_complete.json')
                validation.append(complete['validation']['auprc' if task=='classification' else 'mae'])
            result.append(dict(family=family,mode=mode,validation=stats(validation),
                **{k:stats(r[k] for r in subset) for k in metrics}))
        target[task]=result
    return target


def main():
    new=read(UP/'neural_results.json');old=read(REG/'reports/region_capacity_20261105/neural_results.json')
    downstream=read(DOWN/'test_results.json');regions=new['plan']['regions']
    reference=[dict(r,variant='mamba2_legacy') for r in old['rows'] if r['size']=='small' and r['length']=='long' and r['eval_context']==1024]
    rows=reference+new['rows'];dr=downstream['rows']
    def ntpp(variant,scope,seeds=(0,1,2),region=None,metric='total'):
        result=[]
        for seed in seeds:
            values=[r[metric] for r in rows if r['variant']==variant and r['scope']==scope and r['seed']==seed and (region is None or r['region']==region)]
            assert len(values)==(len(regions) if region is None else 1),(variant,scope,seed,region)
            result.append(statistics.mean(values))
        return stats(result)
    table=[]
    for region in regions:
        table.append(dict(region=region,baseline_local=ntpp('mamba2_legacy','local',region=region),
            baseline_global=ntpp('mamba2_legacy','global',region=region),
            mamba3_local=ntpp('mamba3_oracle_residual','local',region=region),
            mamba3_global=ntpp('mamba3_oracle_residual','global',region=region)))
    macro={v:{s:{k:ntpp(v,s,metric=k) for k in ['time','mag','total']}
        for s in ['local','global_compute','global']} for v in ['mamba2_legacy','mamba3_oracle_residual']}
    ablations=[dict(variant=v,**{s:ntpp(v,s,seeds=(0,)) for s in ['local','global_compute','global']})
               for v in ['mamba2_legacy']+new['plan']['variants']]
    methods=[(v,m) for v in ['mamba2_legacy','mamba3_oracle_residual'] for m in ['scratch','frozen','last_block','full']]
    methods += [('rf','scratch'),('lstm_features','scratch'),('lstm_raw','scratch')]
    target=target_groups(dr,RUNS,methods)
    guard_root=TARGET/'reports/chuandian_warmup_guard_20261105'
    guard=None
    if (guard_root/'test_results.json').exists():
        guarded=read(guard_root/'test_results.json')
        guard=target_groups(guarded['rows'],TARGET/'runs/chuandian_warmup_guard_20261105',methods)
        with (guard_root/'task_metrics.csv').open('w',newline='') as stream:
            keys=list(dict.fromkeys(k for row in guarded['rows'] for k in row))
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(guarded['rows'])
    data=read(DOWN/'data.json');source=read(UP/'transfer_source_data.json')
    dummy=read(DOWN/'dummy_baselines.json')
    wins=sum(r['mamba3_global']['mean']<r['mamba3_local']['mean'] for r in table)
    summary=dict(regional_macro=macro,regional_paired=table,seed0_ablations=ablations,
        mamba3_global_beats_own_local_regions=wins,chuandian=target,data=data,
        transfer_source_data=source,new_regional_fits=new['completed_new_fits'],transfer_source_fits=6,
        downstream_selected_fits=len(dr),strict_encoder_loading=downstream['all_encoder_weights_loaded_strictly'],
        regression_dummy_baselines=dummy,
        chuandian_warmup_guard=guard,
        frozen_and_initialization_verified=downstream['frozen_weights_and_paired_heads_verified'])
    (UP/'combined_summary.json').write_text(json.dumps(summary,indent=2))
    (DOWN/'summary.json').write_text(json.dumps(summary,indent=2))
    for destination,values in [(UP/'regional_metrics.csv',rows),(DOWN/'task_metrics.csv',dr)]:
        keys=list(dict.fromkeys(k for row in values for k in row))
        with destination.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(values)
    etas=read(REG/'reports/region_capacity_20261105/summary.json')['etas_macro']
    lines=['# Mamba3、Oracle 时间头、残差与跨区域编码器迁移','',
        f"新增区域对照 {new['completed_new_fits']} 次、严格按时间截止的迁移源预训练6次；chuandian 最终对照66组（60次神经网络训练、6个选定 RF，另有仅用验证集选择的 RF 参数试验）。",'',
        '## 区域联合训练与单区训练','',
        '沿用九地区230,640个训练目标、2019—2021验证、2022—2024测试；Mc、归一化、每次更新的计分目标、学习率、优化器和预算均与上一轮一致。全部模型2层、64维、32分量 Weibull 时间头和历史条件 GR 震级头，未加入 b 正则项。','',
        '跨区7200步等于九个单区各800步的总预算；跨区800步是另训的等单模型步数对照。完整预算内各地区初始权重、抽样目标及曝光次数已逐项核验。按验证集选择最优检查点，检查点可能早于预算终点。','',
        'Mamba3 为 SISO Mamba3Rotary，157,577参数；Mamba2为151,693参数。Mamba3不支持原 Mamba2 的 xPos，因此用其实际事件时间 RoPE；状态维128、头维64、扩展倍数2。骨干实现所需的数值路径差异随代码快照保留。','',
        'Oracle 参数化预测分量 log-mean 与 shape，再换算 Weibull rate，分布族保持相同。偏置按各自公式初始化到相同单位指数分布。残差使用现有 use_residual 开关，启用 Mamba 与前馈支路的加法残差及已有归一化，不增加模型层数。','',
        '### 三种子主比较：联合 NLL（nats/事件，地区等权，越低越好）','',
        '| 模型 | 单区800步 | 跨区800步 | 跨区7200步 |','|---|---:|---:|---:|']
    for variant in ['mamba2_legacy','mamba3_oracle_residual']:
        lines.append('| '+variant+' | '+' | '.join(fmt(macro[variant][s]['total']) for s in ['local','global_compute','global'])+' |')
    lines+=['',f"同窗 ETAS：{etas['window']['total']:.5f}；全历史 ETAS：{etas['full']['total']:.5f}，沿用同一数据上的已核验结果。",'',
        f"组合模型跨区7200步相对其单区800步，在 {wins}/9 地区的三种子平均 NLL 更低。",'',
        '| 地区 | Mamba2单区 | Mamba2跨区 | Mamba3组合单区 | Mamba3组合跨区 |','|---|---:|---:|---:|---:|']
    for row in table:lines.append('| '+row['region']+' | '+' | '.join(fmt(row[k]) for k in ['baseline_local','baseline_global','mamba3_local','mamba3_global'])+' |')
    lines+=['','### 时间和震级分项（三种子）','','| 模型 | 预算 | 时间 NLL | 震级 NLL | 联合 NLL |','|---|---|---:|---:|---:|']
    for v in ['mamba2_legacy','mamba3_oracle_residual']:
        for scope in ['local','global_compute','global']:
            lines.append(f'| {v} | {scope} | '+' | '.join(fmt(macro[v][scope][k]) for k in ['time','mag','total'])+' |')
    lines+=['','### 逐步消融（仅种子0）','','| 变体 | 单区800步 | 跨区800步 | 跨区7200步 |','|---|---:|---:|---:|']
    for row in ablations:lines.append('| '+row['variant']+' | '+' | '.join(fmt(row[s]) for s in ['local','global_compute','global'])+' |')
    lines+=['','中间消融及 Mamba2+Oracle+残差控制只用种子0；Mamba3完整组合与原 Mamba2 主比较使用三个种子。不能把单种子消融当作三种子结论。','',
        '## chuandian 编码器迁移','',
        '原先训练至2018年的跨区模型没有直接用作本节迁移源。迁移源的所有外区记录严格早于2011-02-12：2009年前训练，2009至2011-02-12验证；九地区不含 chuandian。每种骨干、每个种子另训一个7200步跨区模型，按外区验证均值选检查点。','',
        '目标地区沿用原切分：训练目标结束不晚于2011-02-12，验证目标在该日至2016-04-03，测试目标在2016-04-03至2021-05-24。剔除跨分界的目标窗口。验证/测试可使用已经发生的过去事件；所有归一化、缺失值填补只由目标训练期拟合。','',
        '分类：Mc=3，历史180天，预测未来30天内是否出现 M≥4.5，步长10天。回归：Mc=3，历史360天，预测未来180天最大震级，步长10天。context=0，分类负样本不按未来周边事件筛选；回归仅保留未来非空、最大震级已定义的目标。','',
        '| 任务 | 训练窗口 | 验证窗口 | 测试窗口 | 最长历史事件数 |','|---|---:|---:|---:|---:|']
    for task,spec in data['tasks'].items():lines.append(f"| {task} | {spec['counts']['train']} | {spec['counts']['validation']} | {spec['counts']['test']} | {spec['history_max']} |")
    lines+=['','### 迁移方法和基线','',
        '- scratch：相同结构从随机编码器训练；编码器与任务头学习率和 full 对照相同。',
        '- frozen：完整加载并冻结编码器，关闭其 dropout，只训练池化层与任务头。',
        '- last_block：仅解冻最后一个编码层和最终归一化，前面保持冻结与 eval。',
        '- full：编码器全量微调，编码器学习率4e-4，池化/任务头1e-3。',
        '- RF：18个已有的过去窗口地震统计特征，另加距最后事件的已知时间间隔；12组参数仅在种子0验证集选择，固定设置后训练三个种子。',
        '- lstm_features：两层64维 LSTM，原有18个统计特征在5个内部滑窗上的序列。',
        '- lstm_raw：两层64维 LSTM，输入与 Mamba 完全相同的标准化事件特征。','',
        '所有方法使用完全相同的目标和历史窗口，并提供已知的距最后事件时间。迁移包含输入投影、两层骨干及最后归一化的全部参数，严格加载，无缺失或多余键；不转移地区 FiLM 或 TPP 输出头。每个骨干、任务、种子在四种适配方法间具有相同的随机任务头初始化。','',
        '神经网络统一最多150轮、batch128、AdamW、weight decay0.001、warmup与阶梯衰减、验证耐心10轮；源模型及目标模型均保持两层64维。分类用 focal(alpha0.5,gamma2)，按验证AUPRC选检查点；阈值只按验证F1选择。回归用固定[3,9]缩放下的 SmoothL1(beta0.3)，按验证MAE选检查点，报告还原后的震级误差。','',
        '### 分类测试结果（三种子均值 ± 样本标准差）','',
        '| 方法 | AUPRC↑ | AUROC↑ | F1↑ | 平衡准确率↑ |','|---|---:|---:|---:|---:|']
    for row in target['classification']:
        label=row['family']+'/'+row['mode'];lines.append('| '+label+' | '+' | '.join(fmt(row[k]) for k in ['auprc','auroc','f1','balanced_accuracy'])+' |')
    lines+=['',f"分类测试阳性率为 {data['tasks']['classification']['label_means']['test']:.5f}，常量概率预测的 AUPRC 等于该阳性率。分类主指标为AUPRC，各指标的最优方法可能不同。"]
    lines+=['','### 回归测试结果（三种子均值 ± 样本标准差）','','| 方法 | MAE↓ | RMSE↓ | R²↑ |','|---|---:|---:|---:|']
    for row in target['regression']:
        label=row['family']+'/'+row['mode'];lines.append('| '+label+' | '+' | '.join(fmt(row[k]) for k in ['mae','rmse','r2'])+' |')
    for row in dummy['rows']:
        lines.append('| '+row['method']+' | '+' | '.join(f"{row['test'][k]:.5f}" for k in ['mae','rmse','r2'])+' |')
    lines+=['','在发现回归测试 R² 为负后，补充训练标签均值/中位数的常量预测，用于检查结果的实际意义；常量仅由训练标签估计，没有重新训练或调整原神经模型及 RF。训练中位数常量预测 MAE 为0.63471，优于本轮所有学习模型，说明本轮回归尚未验证出超过简单基线的有效收益。','']
    lines+=['','测试窗口在同一数据划分内可能相互重叠，不能视为独立样本。±只反映随机种子波动，不是时间序列置信区间。地区与年份分布有变化，结果只支持本轮设置下的比较。','',
    ]
    if guard is not None:
        guard_plan=read(guard_root/'plan.json')
        lines+=['## 下游早停保护补充对照','',
            '原程序有21/60个神经模型在15轮warmup结束前触发早停。补充核查统一要求至少训练25轮（warmup15轮+耐心10轮）再允许早停，其余数据、模型、随机种子、优化器和验证准则不变。','',
            f"按原训练日志中的停止轮数触发，重跑 {len(guard_plan['rerun'])} 组停止于25轮之前的模型；另 {len(guard_plan['reused'])} 组原停止不早于25轮或RF的结果与新规则等价，复用原权重。触发规则不使用测试指标，原结果完整保留。此项为完成原测试后发现训练设置问题的补充实验。",'']
        for task,keys in [('classification',['auprc','auroc','f1']),('regression',['mae','rmse','r2'])]:
            lines+=['### '+task+'：至少25轮后早停','','| 方法 | '+' | '.join(keys)+' |','|---|---:|---:|---:|']
            for row in guard[task]:
                lines.append('| '+row['family']+'/'+row['mode']+' | '+' | '.join(fmt(row[k]) for k in keys)+' |')
            if task=='regression':
                for row in dummy['rows']:lines.append('| '+row['method']+' | '+' | '.join(f"{row['test'][k]:.5f}" for k in keys)+' |')
            lines.append('')
        best=min(v['mae']['mean'] for v in guard['regression'])
        naive=min(v['test']['mae'] for v in dummy['rows'])
        lines += [f'保护早停后，学习模型的最低平均 MAE 为 {best:.5f}；训练常量基线的最低 MAE 为 {naive:.5f}。'+
                  ('本轮回归仍未超过该简单基线。' if best>=naive else '本补充对照中已有模型超过该简单基线。'),'']
        (guard_root/'summary.json').write_text(json.dumps(dict(results=guard,plan=guard_plan),indent=2))
    lines += [
        '## 代码与核验','',
        '共享模型代码仅修正 RegionalMixerTPP 包装时保留原时间头参数化；修复前已复现所请求均值失真，修复后与已有测试共39项通过。Oracle和残差本来已有实现，本轮复用。所有训练模型代码均有冻结快照，Mamba2参照保留上一轮前向实现。','',
        '旧 Mamba2 的三个种子初始权重完全匹配；旧检查点验证 NLL 重放最大差约6.94e-6，处于数值核验容差内，因此其参照结果可复用。Mamba3 的3072观测事件长窗已验证前向和有限梯度。目标任务另做完整编码器加载、两种任务的有限梯度及冻结参数不变检查。','',
        '数值环境随各程序实际设置核对：区域 TPP 构建器会启用 high/TF32；下游直接构造编码器，使用 highest/禁用TF32。参数均FP32，Mamba3原生扫描Q/K/V为BF16、相位累计FP32。最终审计按对应设置重放，区域代表模型NLL及目标任务代表模型预测差异均为0；详见final_audit.json。','',
        '主要配置：'+str(REG/'configs/mamba3_oracle_residual_20261105'),
        '区域结果与检查点：'+str(UP)+'；'+str(REG/'runs/mamba3_oracle_residual_20261105'),
        '下游配置：'+str(TARGET/'configs/chuandian_multiregion_transfer_20261105'),
        '下游结果与检查点：'+str(DOWN)+'；'+str(RUNS),
        '实际存储：'+str(STORE),
        '逐行数据：regional_metrics.csv、task_metrics.csv、各运行目录 predictions_test.npz。分类预测存为 logits；回归预测和标签存为[3,9]的标准化值。','']
    (UP/'results_zh.md').write_text('\n'.join(lines));(DOWN/'results_zh.md').write_text('\n'.join(lines))
    if guard is not None:
        start=lines.index('## 下游早停保护补充对照')
        intro=['# chuandian 早停保护后的编码器迁移对照','',
            '均为两层64维，三个随机种子；完整迁移输入投影、两层编码器和最终归一化。迁移源不含chuandian且全部早于2011-02-12。目标训练/验证/测试分界为2011-02-12、2016-04-03、2021-05-24。','',
            '分类：180天历史预测未来30天M≥4.5；回归：360天历史预测未来180天最大震级；Mc=3、步长10天、context=0。所有方法使用相同预测起点和目标。','']
        (guard_root/'results_zh.md').write_text('\n'.join(intro+lines[start:]))
    print(json.dumps(dict(regional=macro,chuandian=target)),flush=True)


if __name__=='__main__':main()
