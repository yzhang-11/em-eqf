"""Verify and summarize all completed multi-region transfer pilot runs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'experiments/revision/multiregion_pretraining'


def main():
    report = OUT/'reports'
    dataset = json.loads((report/'pilot_data_20261104.json').read_text())
    results = []
    for path in sorted((OUT/'runs').glob('*/metrics.json')):
        result = json.loads(path.read_text())
        assert result['config']['dataset_sha256'] == dataset['dataset_sha256']
        assert result['name'] == path.parent.name
        cfg = result['config']
        if not cfg['fewshot']:
            assert cfg['heldout'] not in cfg['sources']
        checkpoint = path.parent/'best_model.pth'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == result['checkpoint_sha256']
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state_dict']
        assert all(torch.isfinite(value).all() for value in state.values() if torch.is_floating_point(value))
        expected = json.loads((path.parent/'test_window_nll.json').read_text())
        for metric in ['time', 'mag', 'total']:
            assert len(expected[metric]) == result['test_windows']
            assert abs(np.mean(expected[metric])-result['metrics'][metric]) < 1e-10
        results.append(result)
    assert len(results) == 30, len(results)
    rows = [dict(name=x['name'], region=x['config']['heldout'], seed=x['config']['seed'],
                 variant=x['name'].split('__')[1], test_windows=x['test_windows'],
                 parameters=x['trainable_parameters'], seconds=x['elapsed_seconds'],
                 **x['metrics']) for x in results]
    frame = pd.DataFrame(rows)
    core = frame[frame.variant.isin(['pooled','conditioned'])]
    assert len(core) == 18
    comparisons = []
    for (region, seed), values in core.groupby(['region', 'seed']):
        plain = values[values.variant == 'pooled'].iloc[0]
        conditional = values[values.variant == 'conditioned'].iloc[0]
        comparisons.append(dict(region=region, seed=int(seed), pooled=float(plain.total),
                                conditioned=float(conditional.total),
                                total_gain=float(plain.total-conditional.total),
                                time_gain=float(plain.time-conditional.time),
                                magnitude_gain=float(plain.mag-conditional.mag)))
    gains = pd.DataFrame(comparisons)
    summary = core.groupby(['region','variant']).agg(
        total_mean=('total','mean'), total_sd=('total','std'),
        time_mean=('time','mean'), mag_mean=('mag','mean')).reset_index()
    macro = core.groupby(['seed','variant'])[['time','mag','total']].mean().reset_index()
    values = dict(fitting_jobs=len(frame), paired_region_seed_comparisons=len(gains),
                  improvements=int((gains.total_gain > 0).sum()),
                  mean_gain=float(gains.total_gain.mean()),
                  mean_time_gain=float(gains.time_gain.mean()),
                  mean_magnitude_gain=float(gains.magnitude_gain.mean()),
                  regions=summary.to_dict('records'),
                  macro=macro.groupby('variant')[['time','mag','total']].mean().to_dict('index'),
                  adapter_parameters=int(core[core.variant=='conditioned'].parameters.iloc[0]
                                         -core[core.variant=='pooled'].parameters.iloc[0]),
                  protocol='retrospective leave-one-catalog-out; 256 past calibration events; no target-region parameter updates in core comparisons',
                  limitation='three source regions, short 300-update pilot; thresholds need a formal completeness audit; no full new-catalog training or count-forecast evaluation')
    frame.to_csv(report/'pilot_all_runs_20261104.csv', index=False)
    summary.to_csv(report/'pilot_region_summary_20261104.csv', index=False)
    gains.to_csv(report/'pilot_paired_gains_20261104.csv', index=False)
    (report/'pilot_summary_20261104.json').write_text(json.dumps(values, indent=2), encoding='utf-8')
    text = ['# 跨区域预训练首轮验证', '',
            '已完成 30 次拟合或微调；核心比较为 3 个留出地区 × 3 个随机种子，共 9 对。9 对均改善。', '',
            '以下为联合 NLL（nats/事件，越低越好）的三种子均值和标准差，先在每个地区内比较。', '',
            '| 留出地区 | 多区域混合 | 加条件适配 | 改善 |', '|---|---:|---:|---:|']
    for region in ['japan_kyushu','italy','new_zealand']:
        a=summary[(summary.region==region)&(summary.variant=='pooled')].iloc[0]
        b=summary[(summary.region==region)&(summary.variant=='conditioned')].iloc[0]
        text.append(f'| {region} | {a.total_mean:.6f} ± {a.total_sd:.6f} | {b.total_mean:.6f} ± {b.total_sd:.6f} | {a.total_mean-b.total_mean:.6f} |')
    text += ['', f'跨地区等权平均 NLL：{values["macro"]["pooled"]["total"]:.6f} → {values["macro"]["conditioned"]["total"]:.6f}。',
             f'平均改善 {values["mean_gain"]:.6f}，其中时间项 {values["mean_time_gain"]:.6f}、震级项 {values["mean_magnitude_gain"]:.6f}。主要收益来自时间预测，震级收益很小。', '',
             '## 固定设置', '',
             '- 原始筛选后事件：日本九州 24,926；意大利 14,273；新西兰 36,542，合计 75,741。',
             '- 各地区最早 256 事件仅用于历史校准，估计参考 b 和时间尺度；这些校准事件不进入梯度训练。',
             '- 梯度训练：2007 年校准前缀之后至 2014-01-01；验证：2014—2015；测试：2016—2019。',
             '- 每条序列含 64 个历史事件和 128 个评分事件；测试地区整体不参与预训练或验证选模。',
             '- 2 层 Mamba，d_model=64，32 个 Weibull 时间混合分量；保留原时间编码。',
             '- 两组都使用按序列的 Mc、b 和时间归一化，时间密度按物理单位还原，震级均用 0.1 网格概率。',
             '- 主干输入为震级及归一化 log 间隔；输入维数从旧预训练的 1 改为本试验的 2，所有比较模型完全相同。',
             '- 新增条件层仅 4,512 参数。输入为 Mc、log b_ref、log tau、log 震级精度和测量类型；混合口径目录标为 unknown/mixed。没有地区 ID。',
             '- 300 次优化更新，batch_size=8，AdamW，学习率 3e-4，权重衰减 0.01，梯度裁剪 1，源地区等概率采样。',
             '- 每 100 次更新在源地区验证，按源地区等权 NLL 选择检查点。核心比较种子为 0、1、2。',
             '- b 辅助损失和时间平滑损失均为 0；未引入粒子滤波或高斯过程。', '',
             '## 少量目标区适配（仅 seed 0）', '',
             '| 地区 | 条件模型直接预测 | 用 512 个目标区评分事件适配 | 同样少量数据从头训练 |', '|---|---:|---:|---:|']
    for region in ['japan_kyushu','italy','new_zealand']:
        part=frame[(frame.region==region)&(frame.seed==0)].set_index('variant')
        text.append(f'| {region} | {part.loc["conditioned","total"]:.6f} | {part.loc["conditioned_fewshot","total"]:.6f} | {part.loc["scratch_fewshot","total"]:.6f} |')
    text += ['', '适配使用目标区训练期最早 4 个窗口、共 512 个评分事件，100 次更新、学习率 1e-4，不使用目标区验证或测试标签选模。它优于同预算从头训练，但相对直接预测只在意大利进一步改善，不作为默认步骤。', '',
             '## 实现和复现', '',
             '```bash', 'cd /root/autodl-tmp/em_eqf',
             'python scripts/data/prepare_multiregion_pretraining.py',
             'python scripts/run/run_multiregion_pretraining.py --seed 0',
             'python scripts/run/run_multiregion_pretraining.py --mode core --seed 1',
             'python scripts/run/run_multiregion_pretraining.py --mode core --seed 2',
             'python scripts/summarize/summarize_multiregion_pretraining.py', '```', '',
             '- 新模型：`src/models/tpp/region_mixer_tpp.py`；通过 `RegionalMixerTPP(base_model, args, condition_regions=True)` 显式启用。',
             '- `condition_regions=False` 关闭 FiLM，保留各地区必需的归一化、概率下限和单位换算。',
             '- 内部预测状态由隐藏向量和区域参数组成；抽取纯历史特征时取 `[..., :model.context_size]`。',
             '- 每个任务的 YAML 位于 `configs/`；最佳权重、训练过程和逐窗测试值位于 `runs/<任务名>/`。',
             '- 汇总表：`reports/pilot_all_runs_20261104.csv`；逐地区三种子结果：`reports/pilot_region_summary_20261104.csv`。',
             '- 新模型要求带区域信息的非空历史。采样期间区域参数保存在每次调用独立的推理缓存中，并广播到全部模拟轨迹。',
             '- 9 项针对性检查通过，包括旧模式对照、混合批次、概率质量归一化、时间密度 Jacobian、尺度等变性、历史因果性和实际 CUDA 采样。',
             '- 额外核实两组共享初始权重完全相同、零初始化条件层输出一致；原有 17 个相关源文件与改动前快照一致。', '',
             '## 结论边界', '',
             '结果支持继续使用共享主干和轻量条件适配层研究跨区域预训练。它是约 7.6 万事件、三种子的短程试验，不是已用全部 721.6 万原始目录训练的模型。',
             'Mc=2.0/2.5/3.0 是本轮固定阈值，正式大规模实验前仍须做区域和时间分段的完备性核查。尚未验证计数 CRPS、N/M-test 或长期生成性能，原论文模型和结果没有替换。',
             'IPOC-Deep 原始数据已完整下载，但因配套说明未明确绝对时间口径，本轮未接入；当前新西兰与意大利仅下载了试验所需的时间、空间和震级子集。', '']
    (report/'multiregion_pilot_results_20261104.md').write_text('\n'.join(text), encoding='utf-8')
    print(json.dumps(values, indent=2))


if __name__ == '__main__':
    main()
