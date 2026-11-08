"""Summarize matched local/global fits, model size, long histories and ETAS."""
from pathlib import Path
import csv
import json
import math
import statistics

ROOT=Path('/root/autodl-tmp/em_eqf')
STUDY=ROOT/'experiments/revision/multiregion_pretraining'
REPORTS=STUDY/'reports/region_capacity_20261105'
RUNS=STUDY/'runs/region_capacity_20261105'


def read(path):return json.loads(Path(path).read_text())


def stats(values):
    values=list(values)
    return dict(mean=statistics.mean(values),sd=statistics.stdev(values) if len(values)>1 else None,n=len(values))


def fmt(value):
    return f"{value['mean']:.5f}"+(f" ± {value['sd']:.5f}" if value['sd'] is not None else '')


def main():
    neural=read(REPORTS/'neural_results.json');data=read(REPORTS/'data.json')
    rows=neural['rows'];plan=neural['plan'];chosen=neural['selected_capacity'];regions=plan['regions']
    etas_rows=read(REPORTS/'etas/test_results.json')['rows']
    etas_test_audit=read(REPORTS/'etas/direct_test_audit.json')
    assert etas_test_audit['passed']
    def group(scope,size,length,context,region=None,seeds=(0,1,2),metric='total'):
        found=[r for r in rows if r['scope']==scope and r['size']==size and r['length']==length
               and r['eval_context']==context and r['seed'] in seeds and (region is None or r['region']==region)]
        per_seed={s:[r[metric] for r in found if r['seed']==s] for s in seeds}
        assert all(len(v)==(1 if region else len(regions)) for v in per_seed.values()),(scope,size,length,context,region)
        return stats(statistics.mean(v) for v in per_seed.values())
    def etas(region,mode,context):
        found=[r for r in etas_rows if r['region']==region and r['mode']==mode and r['eval_context']==context and r['validation_selected']]
        assert len(found)==1
        return found[0]
    paired=[]
    for region in regions:
        local=group('local',chosen,'long',1024,region)
        compute=group('global_compute',chosen,'long',1024,region)
        pooled=group('global',chosen,'long',1024,region)
        differences=[]
        for seed in [0,1,2]:
            l=next(r['total'] for r in rows if r['scope']=='local' and r['size']==chosen and r['length']=='long' and r['eval_context']==1024 and r['region']==region and r['seed']==seed)
            g=next(r['total'] for r in rows if r['scope']=='global' and r['size']==chosen and r['length']=='long' and r['eval_context']==1024 and r['region']==region and r['seed']==seed)
            differences.append(g-l)
        paired.append(dict(region=region,local=local,global_compute=compute,**{'global':pooled},
            global_minus_local=stats(differences),etas_full=etas(region,'full','full'),etas_matched=etas(region,'window',1024)))
    macro={scope:{k:group(scope,chosen,'long',1024,metric=k) for k in ['time','mag','total']}
           for scope in ['local','global_compute','global']}
    etas_macro={mode:{k:statistics.mean(etas(r,mode,'full' if mode=='full' else 1024)[k] for r in regions)
                      for k in ['time','mag','total']} for mode in ['full','window']}
    test_counts={r:data['regions'][r]['test_scored_events'] for r in regions}
    total_events=sum(test_counts.values())
    pooled={scope:stats(sum(r['total']*test_counts[r['region']] for r in rows
        if r['scope']==scope and r['size']==chosen and r['length']=='long' and r['eval_context']==1024 and r['seed']==seed)/total_events
        for seed in [0,1,2]) for scope in ['local','global_compute','global']}
    pooled_etas={mode:sum(etas(r,mode,'full' if mode=='full' else 1024)['total']*test_counts[r] for r in regions)/total_events
                 for mode in ['full','window']}
    capacity=[]
    for scope in ['global_compute','global']:
        for size in ['small','large']:
            capacity.append(dict(scope=scope,size=size,parameters=plan['parameters'][size],
                **{k:group(scope,size,'long',1024,metric=k) for k in ['time','mag','total']}))
    screen=[]
    for size in ['small','large']:
        for scope in ['local','global_compute','global']:
            for length in ['short','long']:
                screen.append(dict(size=size,scope=scope,length=length,total=group(scope,size,length,1024,seeds=(0,))))
    contexts=[]
    for scope in ['local','global_compute','global']:
        for context in [64,1024,4096]:
            contexts.append(dict(scope=scope,context=context,total=group(scope,chosen,'long',context)))
    fitfiles=list((REPORTS/'etas').glob('*.json'))
    fits=[read(f) for f in fitfiles if '_full' in f.name or '_window' in f.name]
    max_error=max(v['max_absolute_nll_quadrature_error'] for v in fits)
    successful=sum(v['selected']['success'] for v in fits)
    poisson_checks=[dict(region=v['region'],mode=v['mode'],family=v.get('family','subcritical'),
        etas=v['selected']['train_time_nll'],poisson=1-math.log(v['base_rate'])) for v in fits]
    controls=read(REPORTS/'convergence_controls/test_results.json')
    assert controls['initialization_and_training_prefix_verified']
    control_comparisons=[]
    for region in controls['plan']['regions']:
        extended=stats(v['total'] for v in controls['rows'] if v['region']==region)
        primary=next(v for v in paired if v['region']==region)
        control_comparisons.append(dict(region=region,local_1600=extended,
            local_800=primary['local'],global_7200=primary['global']))
    diagnostic=read(REPORTS/'training_validation_diagnostics.json')
    audit=read(REPORTS/'final_audit.json');assert audit['passed']
    training_cost=[]
    for size in ['small','large']:
        records=[read(RUNS/f'global_all_regions_{size}_long_seed_{seed}/training_complete.json') for seed in [0,1,2]]
        training_cost.append(dict(size=size,seconds=stats(v['elapsed_seconds'] for v in records),
            peak_allocated_mib=stats(v['peak_allocated_bytes']/1024**2 for v in records)))
    summary=dict(plan=plan,selected_capacity=chosen,completed_neural_fits=neural['completed_fits'],
        completed_etas_fits=len(fits),regions=paired,macro=macro,etas_macro=etas_macro,
        all_events_weighted_joint_nll=dict(neural=pooled,etas=pooled_etas),
        capacity=capacity,screen_seed0=screen,history_contexts=contexts,
        macro_test_wins=sum(v['global']['mean']<v['local']['mean'] for v in paired),
        paired_initialization_and_exposures_verified=neural['paired_initialization_and_exposures_verified'],
        etas_optimizer_successful=successful,etas_max_absolute_direct_nll_error=max_error,
        etas_test_numerical_audit=etas_test_audit,
        etas_poisson_training_sanity=poisson_checks,
        higher_budget_local_controls=dict(plan=controls['plan'],comparisons=control_comparisons),
        seed0_train_validation_diagnostic=diagnostic,
        global_training_cost=training_cost,
        final_audit=audit,
        units='nats/event',lower_is_better=True)
    (REPORTS/'summary.json').write_text(json.dumps(summary,indent=2))
    with (REPORTS/'paired_region_results.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=['region','local','local_sd','global_compute','global_compute_sd','global','global_sd','difference','etas_full','etas_matched'])
        writer.writeheader()
        for v in paired:writer.writerow(dict(region=v['region'],local=v['local']['mean'],local_sd=v['local']['sd'],
            global_compute=v['global_compute']['mean'],global_compute_sd=v['global_compute']['sd'],
            **{'global':v['global']['mean']},global_sd=v['global']['sd'],difference=v['global_minus_local']['mean'],
            etas_full=v['etas_full']['total'],etas_matched=v['etas_matched']['total']))
    lines=['# 单区域、跨区域、模型容量与 ETAS 对照','',
        f"完成 {neural['completed_fits']} 次神经模型主实验、{len(controls['rows'])} 次单区更长预算核查及 {len(fits)} 次 ETAS 拟合。主要对照覆盖 9 个地区，采用三个种子；容量与序列长度的完整初筛使用种子 0。",'',
        f"训练目标事件 {data['unique_train_scored_events']:,}，验证 {data['validation_scored_events']:,}，测试 {data['test_scored_events']:,}。全部地区统一：训练 <2019，验证 2019—2021，测试 2022—2024。",'',
        '本轮九地区均参与跨区联合训练，比较它们各自未来时段的预测。未见地区的零样本或少样本泛化需要另设留区实验。','',
        '## 主要比较','',
        f"主表使用按验证集选定的 {chosen} 容量，长序列训练。所有模型对相同测试事件评分，神经模型及同窗 ETAS 使用 1024 个历史事件。数值为联合 NLL，nats/事件，越低越好；± 为三种子样本标准差。",'',
        '比较对象均为时间与震级的联合预测；ETAS 为时间 ETAS，未计算空间定位似然。时间单位为天，震级为0.1格点上的概率质量。NLL 可为负，不应解释为准确率或直接换算为准确率提升百分比。','',
        '| 地区 | 单区 800步 | 跨区 800步 | 跨区 7200步 | ETAS 同窗 | ETAS 全历史 |',
        '|---|---:|---:|---:|---:|---:|']
    for v in paired:
        lines.append(f"| {v['region']} | {fmt(v['local'])} | {fmt(v['global_compute'])} | {fmt(v['global'])} | {v['etas_matched']['total']:.5f} | {v['etas_full']['total']:.5f} |")
    lines.append('| 等地区平均 | '+' | '.join([fmt(macro[s]['total']) for s in ['local','global_compute','global']])+f" | {etas_macro['window']['total']:.5f} | {etas_macro['full']['total']:.5f} |")
    lines+=['',f"跨区 7200 步相对单区 800 步，平均 NLL 更低的地区为 {summary['macro_test_wins']}/9。应逐区查看，不能据一个总均值宣称所有地区都提升。",'',
        '每个单区模型最多 800 步。一个跨区主模型最多 7200 步，与九个单区模型的总预算相同；各地区最大预算内看到的目标事件序列与曝光次数精确一致。跨区 800 步是独立训练的对照，有自己的 800 步学习率调度。两类预算回答不同的问题，不能混为一谈。','',
        '抽样与曝光一致性的核验针对完整训练预算。各模型最终使用验证最优检查点，其步数可能早于预算终点，保存在每个训练目录的 training_complete.json 中。','',
        '单区按本区验证选检查点；跨区按九地区验证均值选择一个共享检查点。没有按测试结果选地区、容量、检查点或 ETAS 版本。','',
        '## 时间与震级分项','',
        '| 方法 | 时间 NLL | 震级 NLL | 联合 NLL |','|---|---:|---:|---:|']
    for scope in ['local','global_compute','global']:
        lines.append('| '+scope+' | '+' | '.join(fmt(macro[scope][k]) for k in ['time','mag','total'])+' |')
    for mode in ['window','full']:
        lines.append('| ETAS '+mode+' | '+' | '.join(f"{etas_macro[mode][k]:.5f}" for k in ['time','mag','total'])+' |')
    lines+=['','上表先按地区汇总再等权平均。将全部19,979个测试事件直接加权汇总，联合 NLL 如下，用于检查总均值对地区规模权重的依赖：','',
        '| 方法 | 全测试事件加权联合 NLL |','|---|---:|']
    for scope in ['local','global_compute','global']:lines.append(f'| {scope} | {fmt(pooled[scope])} |')
    for mode in ['window','full']:lines.append(f"| ETAS {mode} | {pooled_etas[mode]:.5f} |")
    lines+=['','## 扩充数据后的容量效果','',
        '模型为 RegionalMixerTPP：Mamba2Rotary 骨干、32分量 Weibull 时间头、历史条件 GR 震级头。输入为按地区参考统计缩放的震级与 log(间隔时间)，输出时间分布还原为天。FiLM 使用 Mc、log(b_ref)、log(tau_mean)、log(震级步长)及四维震级类型 one-hot；本轮不含地区 ID embedding。b_weight 与 b_smooth_weight 均为0，未启用 b 滤波器。','',
        'small 为两层、宽度 64，共 151,693 参数；large 为四层、宽度 128，共 977,841 参数。其余模型结构、标定方式、训练样本与优化设置一致。下表的跨区长序列结果覆盖三个种子。','',
        '| 预算 | 容量 | 时间 NLL | 震级 NLL | 联合 NLL |','|---|---|---:|---:|---:|']
    for v in capacity:lines.append(f"| {v['scope']} | {v['size']} | "+' | '.join(fmt(v[k]) for k in ['time','mag','total'])+' |')
    lines+=['','训练资源记录（跨区长序列7200步，三个种子）：','',
        '| 容量 | 训练耗时秒 | 峰值已分配显存 MiB |','|---|---:|---:|']
    for value in training_cost:
        lines.append(f"| {value['size']} | {fmt(value['seconds'])} | {fmt(value['peak_allocated_mib'])} |")
    lines+=['','耗时包含验证和保存，且与其他 GPU 任务并行；这里只记录本轮资源占用，不作为独占 GPU 的吞吐基准。显存为 PyTorch 已分配峰值，未含 CUDA 驱动与保留缓存。']
    lines+=['','种子0另用相同1024历史对全部训练目标和验证目标评分，诊断容量差异：','',
        '| 容量 | 训练联合 NLL | 验证联合 NLL |','|---|---:|---:|']
    for size in ['small','large']:
        value=diagnostic['results'][size]
        lines.append(f"| {size} | {value['train']['macro']['total']:.5f} | {value['validation']['macro']['total']:.5f} |")
    lines+=['','这项诊断未使用测试集，数值对应验证选中的检查点；训练与验证来自不同年代，不能将两者差值全部归因为过拟合。','',
        '## 单区训练预算补充核查','',
        '在查看测试结果前，用种子0验证曲线检查单区训练是否仍在改善：若最优点为最后800步，且600到800步改善超过0.002，则补跑1600步、三个种子。触发地区为 '+', '.join(controls['plan']['regions'])+'。补跑学习率按1600步调度，前800个训练区块抽样和初始权重与主实验一致。','',
        '| 地区 | 单区800步 | 单区1600步 | 跨区7200步 |','|---|---:|---:|---:|']
    for v in control_comparisons:
        lines.append(f"| {v['region']} | {fmt(v['local_800'])} | {fmt(v['local_1600'])} | {fmt(v['global_7200'])} |")
    lines+=['','1600步对照增加了单区预算，用来检查主结论对训练不足是否敏感；它单独列出，不替换主表中预算匹配的800步基线。']
    lines+=['','## 容量与训练序列初筛（种子0）','',
        '短序列为 64 历史 + 最多128目标，batch最多16；长序列为最多1024历史 + 最多2048目标，batch1，最长3072事件。每次更新两种长度的计分目标完全相同。末尾不足2048个目标的区块保留，按目标数比例抽样，损失按事件加权，避免舍弃最新训练数据。','',
        '| 容量 | 方法 | 短序列训练 | 长序列训练 |','|---|---|---:|---:|']
    for size in ['small','large']:
        for scope in ['local','global_compute','global']:
            cells=[next(v for v in screen if v['size']==size and v['scope']==scope and v['length']==length)['total'] for length in ['short','long']]
            lines.append(f'| {size} | {scope} | '+' | '.join(fmt(v) for v in cells)+' |')
    lines+=['','## 更长测试历史','',
        '相同已训练模型、相同测试目标，分别提供64、1024、4096个过去事件；最多4224个事件/测试窗口。历史不足的地区使用已有全部过去事件。此表改变的是评测历史，不重新训练，也不根据这些测试结果选默认配置。','',
        '| 方法 | 历史64 | 历史1024 | 历史4096 |','|---|---:|---:|---:|']
    for scope in ['local','global_compute','global']:
        lines.append('| '+scope+' | '+' | '.join(fmt(next(v for v in contexts if v['scope']==scope and v['context']==context)['total']) for context in [64,1024,4096])+' |')
    lines+=['','## 各地区数据','',
        '| 地区 | 来源 | Mc | 训练目标数 | 验证数 | 测试数 |','|---|---|---:|---:|---:|---:|']
    for region in regions:
        spec=data['regions'][region]
        lines.append(f"| {region} | {spec['provider']} | {spec['mc']:.1f} | {spec['train_scored_events']:,} | {spec['val_scored_events']:,} | {spec['test_scored_events']:,} |")
    lines+=['','## ETAS 与数据核验','',
        'ETAS 同时拟合经典五参数版本（允许 p<1、不限制分支率）与亚临界归一化版本，各有全历史及同训练历史窗两个对照。每次使用两个初值进行训练期最大似然优化，震级为仅用训练目标估计的离散截断 GR。每地区、每种历史条件下，版本仅由验证集选择。全历史 ETAS 可利用比神经模型更多的过去事件，另给同窗 ETAS 以便检查信息量影响。','',
        f"ETAS 使用 Gamma 积分的指数递推加速，不做粒子滤波；42 项数学/梯度/既有实现对照测试通过。实际目录的直接计算核验，单事件 NLL 最大误差为 {max_error:.3g}。优化器报告成功的最优拟合为 {successful}/{len(fits)}；各次重启及边界参数均保留。",'',
        f"最终选定 ETAS 的测试预测另取 {etas_test_audit['queries']} 个查询点，用精确核与解析积分独立重算，时间 NLL 最大误差为 {etas_test_audit['maximum_absolute_time_nll_error']:.3g}。这些查询只用于核验计算，不重新拟合或选择模型。",'',
        '27组地区×种子的初始权重、完整预算内抽样序列和曝光次数核验一致，三个代表检查点重放的 NLL 差异为0。扰动未来震级时，先前预测上下文变化为0。原 xPos 实现对未来时间扰动存在有限精度敏感性：前缀对数密度最大变化约1.61e-4；诊断时固定中心后变化为0，三个代表模型逐区平均 NLL 最大变化约1.57e-5。成对缩放在精确算术下抵消，此项作为数值敏感性记录，未修改时间编码或主表评分。','',
        '方法依据：[Ogata, Matsu’ura & Katsura, 1993, Fast likelihood computation of ETAS](https://doi.org/10.1029/93GL02142)。经典与归一化公式分别与仓库原 ETAS、ETASZhuang 实现核对。','',
        'Mc 使用预先固定的数据门槛，未证明各年代目录均完备；模型比较的门槛、震级格点、时间单位、切分和测试事件相同。每地区参考统计仅由最早256个训练事件计算，这些事件不计入训练损失。所有训练目标区块及三个评测历史的目标编号已逐项核对。','',
        '九个区域的地理框互不重叠。未使用与意大利目录重叠的瑞士记录，也未把训练前事件过少的波多黎各混入长序列主对照。测试时可以用已经发生的测试事件更新历史，不更新权重或重新估计参考统计。','',
        '本轮与此前三地区实验的地区、Mc及测试年份不同，不能直接将旧实验的绝对 NLL 与本表相减。结果属于此数据与预算下的条件似然比较，800步不等于已证明所有单区模型完全收敛。','',
        '## 文件位置','',str(REPORTS),str(RUNS),str(STUDY/'configs/region_capacity_20261105'),
        '实际存储：/root/autodl-fs/em_eqf_training/region_capacity_20261105/',
        '主表 CSV：paired_region_results.csv；全量指标：neural_results.csv、etas/test_results.json；方案与配置见 configs 目录。',
        '模型代码及训练驱动使用 frozen_project 快照，实验未改写共享模型实现。','']
    (REPORTS/'results_zh.md').write_text('\n'.join(lines))
    print(json.dumps(dict(completed_fits=neural['completed_fits'],selected_capacity=chosen,macro=macro,etas=etas_macro)),flush=True)


if __name__=='__main__':main()
