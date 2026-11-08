# Mixer 骨干检查与接入说明（2026-11-05）

## 当前 Mamba3 默认配置（用户指定，2026-11-05）

使用 `config/mixer_tpp_mamba3_event_time.yaml`：Mamba3EventTime、MIMO rank=2、
仅衰减调制、仅原生相位、屏蔽直接log(delta_t)、legacy时间头、use_residual=false。
区域适配器按每条序列的训练tau生成physical_dt，屏蔽操作在归一化后按特征名执行。
下文Mamba3Rotary配置与残差开启实验为历史配置，读取旧检查点时沿用其原设置。

## 当前入口与扩展

| 入口 | 作用 | 骨干切换方式 |
|---|---|---|
| MixerModel / MixerModelWrapper | 公用序列编码器及输入适配 | mixer_model_config.ssm_cfg.layer |
| MixerTPP / model: mixer_tpp | 基础时间点过程，时间与震级预测头 | 同上 |
| RegionalMixerTPP | 区域统计条件化、区域时间尺度与离散震级似然 | 先构建 MixerTPP，再包装；编码器共用 |
| GlobalMixerTPP | 在输入端加入区域 ID embedding | 输入维度随 embedding 增加，其余骨干选项相同 |
| reg_mixer_attnpl_t / clf_mixer_attnpl_t | 回归或分类的 Mixer + 时间池化任务头 | 同上 |

没有另外注册 mixer_tpp_v2。当前新增的时间头选项为 mixer_time_parametrization: legacy/oracle，
分别表示 Weibull 的率/形状与对数均值/形状参数化；这与 Mamba 骨干版本是两个独立选项。
本次骨干对照统一使用 legacy，不把时间分布输出参数化变化混入结果。

## Mamba2Rotary 检查

常用 FP32、BF16、分组与 xPos 的完整前向/缓存推理已经验证。新增回归测试覆盖：
原生及回退状态更新、dt_limit、逐通道 D、无输出 RMSNorm、训练梯度与无卷积缓存重置。
修复了回退漏旋转、有限 dt_limit 单步未应用、D_has_hdim 形状错误和无卷积缓存重置问题。
默认训练 forward 和参数布局保持不变，已有 Mamba2Rotary 检查点可继续加载。

## 新模块配置

```yaml
mixer_model_config:
  ssm_cfg:
    layer: Mamba3Rotary
    d_state: 64
    expand: 2
    headdim: 32
    rope_fraction: 0.5
    rotary_emb_base: 10000.0
    is_mimo: true
    mimo_rank: 2
    chunk_size: 16
```

SISO 使用 is_mimo: false、chunk_size: 64。两者均保留原生学习相位并叠加真实事件时间相位。
MIMO 的训练使用原生 TileLang 前后向，推理使用原生 Triton 旋转与 CuTe 单步核。
从 Mamba2 配置迁移时，移除 use_conv/d_conv/rotary_emb_scale_base 等专属参数；Mamba3 当前没有 xPos。

提供的完整模板：

- config/mixer_tpp_mamba2_rotary.yaml
- config/mixer_tpp_mamba3_rotary.yaml
- config/mixer_tpp_mamba3_rotary_mimo.yaml
- config/reg_mixer_attnpl_t_mamba3_rotary.yaml
- config/reg_mixer_attnpl_t_mamba3_rotary_mimo.yaml
- config/clf_mixer_attnpl_t_mamba3_rotary.yaml
- config/clf_mixer_attnpl_t_mamba3_rotary_mimo.yaml

CLI 的 --model 仍用原任务名，例如：

```bash
source /root/autodl-tmp/venvs/em_eqf-cu128/env.sh
python main.py --model mixer_tpp --config config/mixer_tpp_mamba3_rotary_mimo.yaml
python main.py --model reg_mixer_attnpl_t --config config/reg_mixer_attnpl_t_mamba3_rotary_mimo.yaml
```

## 时间精度与权重迁移

编码器保留 FP64 事件时间，仅把神经网络 features 转成编码器权重类型。
分类/回归池化头收到的浮点附加特征会转成 enc_out 的类型；原始输入和编码器时间不被改写。
四份分类/回归配置均已验证完整前后向与学习角度梯度。

新分类/回归配置关闭了原 Mamba2 checkpoint 的 resume_path。
用新 TPP 检查点迁移编码器时，须匹配 input_dim、features_input_keys（包括适配器排序）、
d_model、d_state、headdim、层编号及 SISO/MIMO 配置。
当前 TPP 使用两项输入特征，原分类/回归模板只用 mag；输入投影不能直接整块复制。
可以保持 mag-only 并只加载匹配的编码层，或把下游输入也改为相同两项特征并调整 input_dim。
分类模板保留原有额外注意力层；回归模板保留其原层数，所以不能默认整体严格加载全部 TPP 编码器。
旧 Mamba2 与 Mamba3 的 mixer 参数形状不同；两者之间不应直接严格加载整个 mixer。

## 统一效果对照

脚本：scripts/run/run_mamba_backbone_comparison.py。
协议与结果位于 experiments/revision/mamba_backbones_20261105。
固定 7 区域训练；意大利、九州、新西兰按 2014 前训练、2014–2015 验证、2016–2019 测试。
4 个版本 × 3 种子 × 1500 更新；训练每窗 256 历史 + 1024 计分事件，batch=2。
共同的 stats-FiLM、TPP 头、输入尺度和非 mixer 初始化，以及每种子的采样顺序均配对。
主指标是测试集区域宏平均联合 NLL（nats/event，越低越好），并分开报告时间和震级 NLL。
全部拟合完成后才评估测试集；检查点按验证集选择。参数量不同会在结果中列出。


## 当前工作区与冻结对照的差别

本轮 12 组实验使用训练开始前的代码快照，残差连接为开启状态。
运行期间其他任务给 Block/MixerModel 增加了 use_residual 开关，并把基础 TPP builder 和
Mamba3 TPP 模板改成默认 use_residual: false。这些修改已保留，未被本任务覆盖。
因此，在当前工作区加载本轮检查点时，必须在 mixer_model_config 中显式设置
use_residual: true；否则权重虽然可以加载，前向计算与本轮实验不同。
reports/benchmark_configs 下的四份配置已显式记录 true；原始 study_plan 与代码归档保持不变。
冻结快照对全部 12 个检查点复算，宏平均和逐区域 time/mag/total 指标均与原结果完全一致。

mixer_time_parametrization: oracle 当前已验证的是基础 MixerTPP。
RegionalMixerTPP 会重建时间解码器，现有实现没有传递该开关；不能据基础模型测试声称区域包装器也已支持 oracle。
本轮 RegionalMixerTPP 对照统一使用 legacy，因此该限制不影响报告中的骨干比较。

吞吐量字段仅作为实际运行记录。各次训练的编译缓存及并发负载不同，
不能从这些记录推导严格的速度提升。
