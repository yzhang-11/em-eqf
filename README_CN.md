# em_eqf（中文说明）

`em_eqf` 是一个基于 PyTorch 的地震目录预测研究框架，包含两类数据与模型流程：
固定时间窗分类/回归，以及事件序列时间点过程（TPP）。仓库还提供目录读取器、
结合地震与注水时间序列的诱发地震目录、实验运行脚本、分析工具和预测评估工具。

本文说明主要入口和数据约定；专项流程的实现细节请参阅文末链接的指南。
英文说明见 [README.md](README.md)。

## 工作流概览

| 工作流 | 输入与目标 | 模型示例 | 主要数据位置 |
| --- | --- | --- | --- |
| 时间窗分类 | 地震历史窗口预测二分类或计数目标 | `clf_rnn`、`clf_mixer_attnpl_t` | `data/<dataset>/raw/` |
| 时间窗回归 | 地震历史窗口预测未来震级 | `lstm`、`reg_mixer_attnpl_t` | `data/<dataset>/raw/` |
| 事件序列 TPP | 根据有序事件时间和标记计算似然并生成样本 | `rtpp_v2`、`etas`、`oracle`、`mixer_tpp` | 由目录类型决定，见下文 |

模型名通过代码注册。运行 `python main.py --help` 可查看 CLI 接受的模型名。
YAML 中的 `model` 必须与 `--model` 参数一致；未传入 `--config` 时默认读取
`config/<model>.yaml`。

## 安装

要求 Python 3.10 或更高版本。在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

`requirements.txt` 安装项目通用依赖。部分模型配置会使用 `mamba_ssm`、
`causal_conv1d` 或 `flash_attn` 等可选 CUDA 扩展；只安装所选配置需要的扩展，
并确保其与 PyTorch、CUDA、编译器及 Python 版本兼容。CPU 运行时请选择不依赖这些
扩展的模型和配置。训练建议使用 CUDA。

## 数据目录与格式

窗口模型和 TPP 目录使用不同的数据约定；并非所有 TPP 数据集都采用诱发地震
triplet 格式。

### 时间窗分类与回归

事件窗口加载器从 `data/<dataset>/raw/` 读取地震目录 CSV，文件应包含以下列：

| 列名 | 含义 |
| --- | --- |
| `t` | 数据集时间坐标中的数值型事件时间 |
| `Magnitude` | 事件震级 |
| `Latitude`、`Longitude`、`Depth` | 序列加载器使用的位置和深度字段 |
| `dt` | 相邻事件的时间差；加载时会排序并重新计算 |

`raw/` 中只有一个 CSV 时会读取该文件；若有多个 CSV，则必须恰好有一个文件名以
`processed_` 开头，加载器会选择该文件。训练前应核对实际读取的文件和时间单位。
窗口长度、预测跨度、步长、震级阈值、特征和按时间切分方式在模型 YAML 中配置，
例如 `Mc`、`Mf`、`Twindow`、`Tfore`、`dt` 和 `split_by_time`。

### TPP 目录

TPP 数据集通过 `Catalog` 注册表解析。输入格式由具体目录类决定：有的读取
`raw/` 中的目录文件，诱发地震 triplet 加载器则读取 `processed/` 中的预处理文件。
目录参数由模型 YAML 中的 `catalog_cfg` 提供。查看已注册目录：

```bash
python - <<'PY'
import src.catalogs
from src.data.catalog import Catalog
print("\n".join(sorted(Catalog.list_available())))
PY
```

通用 raw 目录布局和诱发地震 triplet 输入布局见下文；目录迁移工具位于
`scripts/maintenance/reorganize_data_layout.py`。

### 诱发地震 triplet 文件

名为 `MyField` 的 `InducedTripletBase` 数据集使用以下布局：

```text
data/MyField/
├── processed/
│   ├── MyField_eq_processed.csv
│   ├── MyField_inj_60min_processed.csv
│   └── MyField_summary.json
└── catalogs/                 # 自动生成的哈希 Sequence 缓存
```

地震 CSV 必须包含时间列和震级列。默认时间列别名包括 `time_iso`、`ts`、`time`、
`timestamp`、`origin_time`、`origin_timestamp`、`Date`；震级列别名包括
`magnitude`、`mag`、`Magnitude`、`ML`、`Mw`。纬度、经度和深度可选。注水 CSV
必须包含时间列，以及 `inj_rate_m3_min` 或 `inj_rate` 速率列。时间列可以是绝对
时间戳或相对数值。相对数值单位会根据常见列名推断，也可以在
`summary.time_column_units` 中按 CSV 实际列名声明。可用值为
`unit`/`model`/`freq`、`s`、`m`、`h`、`d` 和 `abs`；数值型 `t` 或 `time` 列默认使用
模型时间单位。也可通过 `summary.column_aliases` 自定义列别名。

summary JSON 必须包含：

```json
{
  "resample_freq_min": 60,
  "mc": -0.5,
  "inj_fill_policy": "interpolate",
  "is_upsample": false,
  "start_time_iso": "2018-06-04T05:27:26.351",
  "end_time_iso": "2018-08-21T23:59:59"
}
```

`resample_freq_min` 表示已处理注水文件的采样间隔。若所用目录类和
`catalog_cfg.mag_completeness` 都未提供震级阈值，则 `mc` 是 `InducedTripletBase`
使用的回退值；已注册子类可能定义自己的默认值。其他字段说明注水缺失值处理、
上采样和数据绝对时间范围。`catalog_cfg.freq` 决定模型时间单位。加载器按配置
Mc 筛选地震、排序事件时间，并对重复时间加入极小扰动以保证相邻事件时间差为正。
若要声称辅助时间序列特征严格因果可用，应检查预处理、插值和特征构造过程。

Triplet 加载器把目录产物写入 `data/<dataset>/catalogs/<hash>/`。哈希包含时间单位、
Mc、归一化、切分边界和事件特征配置。保留缓存及其元数据；源数据变更后，不要直接
复用旧缓存，应重新生成或核对缓存。

## 首次运行检查

开始长时间训练前，建议依次检查：

1. 运行 `python main.py --help`，确认所需模型已注册。
2. 打开实际使用的 YAML，核对 `model`、`task_type`、`dataset`、时间单位、震级
   阈值和切分设置。YAML 中的 `model` 必须与 CLI 的 `--model` 相同。
3. 使用窗口模型时，检查 `data/<dataset>/raw/`，确认加载到的 CSV 包含
   `t`、`Magnitude`、`Latitude`、`Longitude`、`Depth` 和 `dt`。使用 TPP 时，
   确认具体目录类所需文件及 `catalog_cfg`。
4. 使用 triplet 目录时，在训练前验证注册和实例化。将 `MyField` 替换为实际数据集名：

   ```bash
   python - <<'PY'
   import src.catalogs
   from src.utils.tpp_experiments import load_tpp_catalog

   catalog, registry_name, init_kwargs = load_tpp_catalog(
       "MyField",
       base_dir="data/MyField",
       catalog_cfg={"freq": "1h"},
   )
   print("registry:", registry_name)
   print("train/val/test:", len(catalog.train[0]), len(catalog.val[0]), len(catalog.test[0]))
   print("init kwargs:", init_kwargs)
   PY
   ```

5. 确认输出目录可写；如果设置了 `resume_path`，确认它指向兼容的 checkpoint。
   训练完成后应将最终 YAML 与 checkpoint 一起保存。

上述 triplet 命令会检查目录注册、源文件、summary 字段、缓存构建和按时间切分，
要求数据文件已经存在，并不是合成数据的空跑测试。

## 训练、测试与调参

`main.py` 是已注册模型的统一 CLI：

```bash
# 使用 config/clf_rnn.yaml 训练，并指定输出目录。
python main.py --model clf_rnn --mode train \
  --config config/clf_rnn.yaml \
  --checkpoint_dir checkpoints/clf_rnn_demo

# 测试该目录中的最佳 checkpoint。
python main.py --model clf_rnn --mode test \
  --checkpoint_dir checkpoints/clf_rnn_demo \
  --ckpt_select best
```

此示例要求 `data/ChuanDian/raw/` 中存在有效输入。请根据本机目录修改 `dataset` 和
YAML 中的数据参数。训练前检查 `resume_path`：有效路径会载入 checkpoint；不设置该
字段则从头训练。命令行 `--checkpoint_dir` 指定运行产物目录；不传时，会在
`checkpoints/` 下创建带时间戳的目录。

测试支持 `--ckpt_select best`、`last` 或 `epoch`；选择 `epoch` 时还必须传
`--ckpt_epoch N`。`--trial_index` 选择编号 checkpoint，默认值为 1。测试未指定
`--checkpoint_dir` 时，CLI 会搜索该模型最新的时间戳 checkpoint 目录。分类测试可用
`--threshold VALUE` 覆盖验证集阈值，也可用 `--no_val_threshold` 忽略 checkpoint 中
保存的阈值。

只有所选 YAML 定义了 Optuna 搜索空间时，才能通过 `--mode optuna` 调参。仓库当前
模型 YAML 没有预置搜索空间。可以复制 `config/clf_rnn.yaml`，并在副本中加入以下配置：

```yaml
optuna:
  objective: val_loss
  direction: minimize
  search_space:
    learning_rate:
      type: float
      low: 0.0001
      high: 0.01
      log: true
```

建议显式设置 `objective`。实现默认目标为 `weighted_metrics`，该目标还必须配置
`optuna.metric_weights`。
搜索参数名可以使用点号路径来修改嵌套配置，例如 `bg_model_cfg.scale_init`。
支持的 `type` 为 `float`、`int` 和 `categorical`：前两者使用 `low`、`high`，
可选 `step` 和 `log`；`categorical` 必须提供非空 `choices`。另一个可用目标是
`weighted_metrics`；使用它时还要配置 `optuna.metric_weights`，填写指标名和带符号
权重，例如最小化 AUC 可写 `{auc: -1.0}`，最小化 RMSE 可写 `{RMSE: 1.0}`。

再将该 YAML 路径传给 CLI：

```bash
python main.py --model clf_rnn --mode optuna \
  --config config/clf_rnn_optuna.yaml --optuna_trials 20
```

运行 `python main.py --help` 可查看 profile、存储、试验并行度和自动测试 checkpoint 等完整选项。

## TPP 训练与预测评估

常用 TPP 模型包括 `rtpp`、`rtpp_v2`、`etas`、`etas_zhuang`、`nhpp`、`oracle` 和
`mixer_tpp`。它们在事件表示、标记处理、背景率支持和必需配置上有所不同；将参数
迁移到另一模型前，应检查该模型的 YAML 和实现。常见背景模块包括 `kernel`、
`mamba`、`rnn`、`conv_mlp` 和 `proportional`，具体支持情况取决于模型。查看已注册
背景模块：

```bash
python - <<'PY'
from src.models.bg import BGModel
print("\n".join(sorted(BGModel.list_available())))
PY
```

对诱发地震 triplet，`catalog_cfg.freq` 定义模型时间单位。例如 `freq: "1D"` 时，
预测时长 `7.0` 表示 7 天；`freq: "1h"` 时表示 7 小时。`InducedTripletBase` 默认按
日历时间以 70%/15%/15% 切分；设置 `train_start_ts`、`val_start_ts` 和
`test_start_ts` 可指定边界。已注册子类也可能提供自己的边界默认值，因此应检查所用
目录类及保存的元数据。验证集和测试集保留更早的事件作为模型历史，但 NLL 从对应
切分边界开始计算。其他目录类也可能采用不同切分规则。

诱发地震配置示例：

```yaml
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
```

设置 `event_feature_builder: oracle` 后，加载器会为每个地震事件生成对齐标记并写入
缓存序列：`vm` 是插值注水速率绝对值的 log10；`dVc` 是事件处累计体积变化绝对值的
log10；`sv` 是该变化的符号；`dTS` 是距最近一次非零注水采样时间的 log10；`aRs` 是
平滑地震率，单位为 log10(1/min)。标记顺序和未来特征由 `oracle_base_mark_order` 与
`oracle_future_feature_names` 控制。这些标记使用插值对齐；评估因果可用性时，应检查
时间对齐和缺失值处理。

使用统一 CLI 训练和测试：

```bash
python main.py --model oracle --mode train --config config/oracle.yaml \
  --checkpoint_dir checkpoints/oracle_st1
python main.py --model oracle --mode test \
  --checkpoint_dir checkpoints/oracle_st1 --ckpt_select best
```

测试会报告留出集似然指标，并写入 JSON 文件，例如 `metrics_test_best_1.json`。NLL
衡量事件似然，不能替代预测采样和校准。诱发地震预测 notebook 演示了采样及
Number/Magnitude 检验。滑窗指标和辅助函数位于 `src/utils/forecast_sliding.py`、
`src/utils/forecast_eval.py` 及 `scripts/` 下的相关脚本。

相关 notebook：

- [诱发地震预测](notebooks/forecasting_induced_eq.ipynb)
- [诱发地震 triplet 预测](notebooks/forecasting_induced_eq_triplet.ipynb)
- [诱发地震预测诊断](notebooks/forecasting_induced_eq_diagnostics.ipynb)
- [背景模型对比](notebooks/induced_eq_bg_comparison.ipynb)
- [TPP 评估](notebooks/tpp_evaluating.ipynb)

加载模型和目录后，也可调用程序化预测接口。以下示例假定 `model` 已是加载到目标
设备上的训练后 TPP 模型；`load_tpp_catalog` 只创建目录，不会载入模型 checkpoint：

```python
from src.utils.tpp_experiments import load_tpp_catalog, sample_tpp_forecasts

# 采样前需先从 checkpoint 加载训练好的模型。
catalog, catalog_name, init_kwargs = load_tpp_catalog(
    "St1-2018",
    base_dir="data/St1-2018",
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
```

为了让对比可复现，应报告切分边界、Mc、时间单位、模型与配置、随机种子、checkpoint
选择方式以及预测采样/评估参数。多目录评估应保留每个物理目录的结果，同时再给汇总指标。

## 产物与可复现性

每次运行的目录通常包含：

```text
config.yaml
run.log
best_model_1.pth
last_model_1.pth
epoch_<N>_model_1.pth       # 按配置保存时生成
tensorboard/
metrics_test_<selector>_1.json
```

标准训练流程使用 trial 编号 1。checkpoint 应与保存的 `config.yaml` 一起保留，因为
测试会从 checkpoint/配置恢复模型设置。测试指标会写在 checkpoint 同目录：选择
`best` 时为 `metrics_test_best_1.json`，选择 `last` 时为
`metrics_test_last_1.json`，选择 `epoch` 时为 `metrics_test_epoch_<N>_1.json`。
查看 TensorBoard 日志：

```bash
tensorboard --logdir checkpoints/<run>/tensorboard --port 6006
```

`config/` 下的 YAML 是模型、优化器、数据、切分和可选功能的配置来源。确定性训练
参数也在 YAML 中设置；CUDA 运行若需要 cuBLAS 确定性行为，请在启动 Python 前设置：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Optuna 运行会新增 `optuna_trials/trial_<number>/` 目录，其中保存 trial 配置、
checkpoint、验证指标和可选的事后测试结果。使用 profile 时还会导出
`optuna_summary_<profile>.json` 与 `optuna_summary_all.json`。迁移学习相关的
`resume_path`、`load_specific_parts`、`freeze_parts`、`freeze_loaded_only` 和
`exclude_freeze_parts` 也由 YAML 控制，应与运行配置一起保存。

## 仓库目录

| 路径 | 用途 |
| --- | --- |
| `main.py` | 训练、测试和 Optuna CLI 入口 |
| `config/` | 模型及实验 YAML |
| `src/catalogs/` | 目录实现与注册 |
| `src/data/` | 数据加载、序列、事件窗口和切分 |
| `src/models/` | 分类器、回归器、TPP 和背景模型 |
| `src/train/` | 训练步骤、优化和 checkpoint 保存 |
| `src/cli/` | CLI 解析及训练/测试/Optuna 流程 |
| `src/utils/` | 指标、绘图、采样和评估辅助函数 |
| `scripts/run/` | 实验运行脚本 |
| `scripts/summarize/` | 实验结果汇总 |
| `scripts/analyze/` | 分析工具 |
| `notebooks/` | 预处理、预测和分析示例 |

脚本目录和命名约定见 [Scripts README](scripts/README.md)。

## 常见错误

| 现象 | 常见原因与处理 |
| --- | --- |
| `Model name in config must match command line argument` | `--model` 与 YAML 的 `model` 不一致；两处都使用同一个已注册模型名。 |
| `No CSV files found` 或 processed 文件不唯一 | `data/<dataset>/raw/` 没有 CSV，或多个 CSV 中没有恰好一个 `processed_*.csv`；请补充文件或消除命名歧义。 |
| `checkpoint ... not found` | 检查 `--checkpoint_dir`、`--trial_index` 和选择器。选择 `epoch` 时还必须传 `--ckpt_epoch`；文件名应为 `best_model_<trial>.pth`、`last_model_<trial>.pth` 或 `epoch_<N>_model_<trial>.pth`。 |
| triplet summary 缺少字段 | 在 `<Dataset>_summary.json` 中补齐 6 个必需字段，然后重新构建目录缓存。 |
| 缓存元数据与当前配置冲突 | 修改 `freq`、Mc、归一化、切分边界或事件特征后不要复用旧哈希缓存；重新生成或使用新哈希目录。 |
| TPP 预测时长不正确 | `catalog_cfg.freq` 是模型时间单位；`7.0` 表示该频率的 7 个单位，不一定是 7 天。 |
| Optuna 报告搜索空间为空 | 添加 `optuna.search_space` 或 `optuna.profiles.<name>.params`；存在多个 profile 时使用 `--optuna_profile`。 |
| 可选模块导入失败 | 当前 YAML 使用了未安装或版本不兼容的 CUDA 扩展；选择兼容配置，或安装与 PyTorch/CUDA/Python 匹配的扩展。 |

## 测试与许可证

在仓库根目录运行测试：

```bash
python -m pytest -q
```

项目采用 MIT License，详见 [LICENSE](LICENSE)。
