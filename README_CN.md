# em_eqf（中文说明）

em_eqf 是一个基于 PyTorch 的地震目录预测框架，统一支持滑动窗口分类/回归、事件序列时间点过程（TPP）、checkpoint 训练与测试、Optuna 搜索、预测采样和评估。仓库同时提供基于“地震目录 + 注水时间序列”triplet 数据的诱发地震预测流程。

## 安装

要求 Python 3.10 或更高版本。训练建议使用 CUDA，也支持 CPU。

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
~~~

只有所选配置实际使用时才安装 causal_conv1d、mamba_ssm、flash_attn，并确保它们与 PyTorch/CUDA 版本匹配。

## 快速开始

main.py 是统一入口；命令行中的 model 必须与 YAML 中的 model 一致。未指定
--config 时默认读取 config/<model>.yaml。

~~~bash
python main.py --model mixer_tpp --mode train --config config/mixer_tpp.yaml \
  --checkpoint_dir checkpoints/mixer_tpp_demo

python main.py --model mixer_tpp --mode test \
  --checkpoint_dir checkpoints/mixer_tpp_demo --ckpt_select best

python main.py --model mixer_tpp --mode optuna \
  --config config/mixer_tpp.yaml --optuna_trials 20
~~~

训练未指定输出目录时会在 checkpoints 下创建时间戳目录。测试支持 best、
last、epoch；选择 epoch 时还要提供 --ckpt_epoch。测试未指定
--checkpoint_dir 时会自动选择最新匹配目录。

## 诱发地震点过程预测

完整说明见[诱发地震预测指南](docs/guides/induced_seismicity_forecasting_guide.md)
和 [catalog 接入指南](docs/guides/add_induced_triplet_catalog_guide.md)。

### 1. 准备 triplet 处理文件

以 MyField 为例：

~~~text
data/MyField/processed/
├── MyField_eq_processed.csv
├── MyField_inj_60min_processed.csv
└── MyField_summary.json
~~~

地震 CSV 必须包含时间列和震级列。常用别名包括
time_iso/ts/timestamp，以及 magnitude/Magnitude/Mw；纬度、经度、深度是可选列。
注水 CSV 必须包含时间列和 inj_rate_m3_min 或 inj_rate。时间可以是绝对时间戳，
也可以是相对数值；需要时在 summary.time_column_units 中声明单位。

summary.json 至少包含：

~~~json
{
  "resample_freq_min": 60,
  "mc": -0.5,
  "inj_fill_policy": "interpolate",
  "is_upsample": false,
  "start_time_iso": "2018-06-04T05:27:26.351",
  "end_time_iso": "2018-08-21T23:59:59"
}
~~~

这些字段分别用于注水源频率、震级完备阈值 Mc、缺失值填充/插值策略、
是否上采样以及数据起止时间。加载器会过滤和排序地震事件，删除低于 Mc
的事件，并对完全相同的事件时间加入极小扰动，使 inter-event time 保持正值。
当 catalog_cfg.freq 与源注水频率不一致时，会在加载阶段重采样注水序列。

### 2. 注册 catalog 并验证

在 src/catalogs/ 中继承 InducedTripletBase，实现并注册
<Dataset>-Standard，然后在 src/catalogs/__init__.py 导入该模块。可参考
src/catalogs/st1.py、src/catalogs/pnr.py 和
src/catalogs/induced_triplet_base.py。

~~~bash
python - <<'PY'
import src.catalogs
from src.data.catalog import Catalog
print(sorted(Catalog.list_available()))
PY
~~~

配置中的 dataset: MyField 会解析为 MyField-Standard。生成的 Sequence 缓存位于
data/MyField/catalogs/<hash>/；hash 包含 freq、Mc、归一化和切分边界等参数。
如果缓存元信息与当前配置冲突，应删除旧缓存或修改配置生成新缓存。

### 3. 模型、时间单位与背景率

常用诱发地震 TPP 模型：

| 模型 | 典型用途 |
| --- | --- |
| oracle | 注水感知的 Oracle 事件特征与自回归采样 |
| rtpp_v2 | 带可配置背景模型的循环 TPP |
| rtpp | 兼容 RECAST 形式的循环 TPP |
| etas、etas_zhuang | ETAS 触发项与背景率 |
| mixer_tpp | TPP-Mixer 事件序列建模 |

catalog_cfg.freq 定义模型时间单位。freq: "1D" 时 duration: 7.0 表示 7 天；
freq: "1h" 时表示 7 小时。它必须和注水重采样频率、预测时长保持一致。

带背景率的模型可通过 bg_model 和 bg_model_cfg 切换背景模块。常见选项包括
kernel、mamba、rnn、conv_mlp 和 proportional。例如：

~~~yaml
bg_model: mamba
bg_model_cfg:
  d_feature: 1
  d_state: 64
  d_model: 16
  model_type: mamba
  scale_init: 1000.0
  smooth_kernel_size: 3
~~~

比较不同背景模型时，应固定数据集、时间切分、随机种子和预测评估参数。
scale_init 控制背景率初始尺度，300--1000 可作为起始范围，实际仍需按数据集检查。

Oracle 配置示例：

~~~yaml
model: oracle
task_type: tpp
dataset: St1-2018
catalog_cfg:
  freq: "1D"
  mag_completeness: -0.5
  event_feature_builder: oracle
  event_feature_cfg:
    smoothing_window: 10
    activity_epsilon: 0.0
    log_floor: -10.0
~~~

设置 event_feature_builder: oracle 后，系统会在每个地震事件时刻构造事件级对齐特征：

- vm：插值注水速率绝对值的 log10；
- dVc：累计注水体积变化绝对值的 log10；
- sv：累计体积变化的符号；
- dTS：距最近一次非零注水采样的时间的 log10；
- aRs：平滑地震率，单位为 log10(1/min)。

这些字段会写入缓存的 Sequence，并依据 oracle_base_mark_order 和
oracle_future_feature_names 供 Oracle 模型使用。当前实现对注水序列进行插值对齐，
因此如果实验要求严格的无未来信息设定，应额外检查采样边界、插值方向和缺失值填充策略，
避免把未来注水或未来地震信息泄漏到输入中。

### 4. 训练、测试和预测采样

~~~bash
python main.py --model oracle --mode train --config config/oracle.yaml \
  --checkpoint_dir checkpoints/oracle_st1

python main.py --model oracle --mode test \
  --checkpoint_dir checkpoints/oracle_st1 --ckpt_select best
~~~

测试会计算留出集 TPP 负对数似然，并写入
metrics_test_<selector>_<trial>.json。训练/测试 NLL 不等于完整的预测能力评估；
采样、校准和可视化应单独执行。推荐使用：

- notebooks/forecasting_induced_eq.ipynb：单目录预测；
- notebooks/forecasting_induced_eq_triplet.ipynb：triplet 目录预测；
- notebooks/forecasting_induced_eq_diagnostics.ipynb：诊断；
- notebooks/induced_eq_bg_comparison.ipynb：背景模型对比。

代码方式可使用 src/utils/tpp_experiments.py：

~~~python
from pathlib import Path
from src.utils.tpp_experiments import load_tpp_catalog, sample_tpp_forecasts

catalog, name, kwargs = load_tpp_catalog(
    "St1-2018",
    base_dir=Path("data/St1-2018"),
    catalog_cfg={"freq": "1D", "mag_completeness": -0.5},
)
past = catalog.test[0]
forecasts = sample_tpp_forecasts(
    model,
    past,
    duration=7.0,
    num_samples=1000,
    samples_per_batch=32,
    bg_cache_seq=past,
    seed=0,
)
~~~

建议至少报告以下结果：

1. 留出集 NLL；
2. Number Test/事件数分布检验；
3. Magnitude Test/震级分布检验；
4. 预测轨迹、事件数直方图与分位数区间；
5. 滑动窗口的误差、区间覆盖率和 CRPS；
6. 按 train/validation/test 起始时间和物理目录分别汇总，避免不同时间段或不同场地相互掩盖问题。

相关实现位于 src/utils/catalog_tests.py、src/utils/forecast_eval.py、
src/utils/forecast_sliding.py 和 scripts/run_sliding_window_forecast.py。

### 5. 新增诱发地震目录

1. 生成并检查 eq、inj、summary 三个处理文件；
2. 新建 InducedTripletBase 子类并注册；
3. 在 src/catalogs/__init__.py 导入新模块；
4. 在模型 YAML 设置 dataset 和 catalog_cfg；
5. 长时间训练前先实例化 catalog：

~~~bash
python - <<'PY'
import src.catalogs
from src.utils.tpp_experiments import load_tpp_catalog
catalog, name, init_kwargs = load_tpp_catalog(
    "MyField", base_dir="data/MyField", catalog_cfg={"freq": "1h"}
)
print(name, init_kwargs)
print(len(catalog.train[0]), len(catalog.val[0]), len(catalog.test[0]))
PY
~~~

如果要把多个物理目录组合为一个 family，使用 InducedTripletGroupedCatalog。
split_groups 必须包含非空且互斥的 train、val、test 分组。字段约束和报错排查
见新增 catalog 指南。

## 数据切分与防泄漏

TPP 管线先通过 registry 找到 catalog，生成或读取缓存 Sequence，再创建按时间排序的
train/validation/test 子序列。诱发 triplet 默认按日历时间 70%/15%/15% 切分；也可在
catalog_cfg 中显式设置 train_start_ts、val_start_ts、test_start_ts。

验证集和测试集会保留边界之前的历史作为因果 warm-up，但 NLL 从各自边界开始计算。
因此不能把整个序列随机打乱后再切分，也不要用测试区间的注水数据构造训练阶段可见的
统计量。需要改变时间单位、Mc、归一化或切分边界时，应让缓存 hash 发生变化并重新检查
元数据。

## 配置、产物与开发

YAML 位于 config/，由 OmegaConf 读取。常见字段包括 model、dataset、task_type、
优化器/调度器、网络结构参数和 catalog_cfg。迁移学习相关字段包括 resume_path、
load_specific_parts、freeze_parts、freeze_loaded_only 和 exclude_freeze_parts。

checkpoint 目录通常包含：

~~~text
config.yaml
run.log
best_model_1.pth
last_model_1.pth
epoch_<N>_model_1.pth
tensorboard/
metrics_test_<selector>_1.json
~~~

~~~bash
tensorboard --logdir checkpoints/<experiment>/tensorboard --port 6006
pytest -q
export CUBLAS_WORKSPACE_CONFIG=:4096:8
~~~

推荐阅读：[数据布局指南](docs/guides/data_layout.md)、[诱发地震预测指南](docs/guides/induced_seismicity_forecasting_guide.md)、
[catalog 接入指南](docs/guides/add_induced_triplet_catalog_guide.md)。

项目采用 MIT License，详见 LICENSE。
