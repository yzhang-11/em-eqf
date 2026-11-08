# 数据、实验与结果

## 目录约定

```text
data/<dataset>/raw/                 原始快照，保留来源版本
data/<dataset>/processed/           清洗后的输入
data/<dataset>/catalogs/            点过程 Sequence 缓存
data/<dataset>/cache/windows/       分类、回归、LSTM 共用窗口缓存
experiments/mtpp/                   分类、回归与自然地震点过程实验
experiments/netas/                  ETAS、RECAST、Oracle 多数据集实验
experiments/revision/               后续修订与扩展实验
experiments/registry/               数据、run、指标与权重索引
checkpoints/                       独立训练与预训练 checkpoint
```

SWC 对应代码中的 `ChuanDian`，SCSN 对应 `SCEDC`。不同原始快照、完整性震级和时间划分仍分别登记。

## 查找当前数据对应的实验文件

```bash
python scripts/maintenance/index_experiments.py --dataset ChuanDian
python scripts/maintenance/index_experiments.py --dataset SCEDC
# 数据或实验发生变化后，先刷新再查询
python scripts/maintenance/index_experiments.py --dataset ChuanDian --refresh
```

普通 `--dataset` 查询只读取已保存的索引，不扫描实验目录，也不改写文件；不带查询参数时重建完整索引。嵌套 Optuna trial 与 RF 对照运行也纳入索引。

`experiments/registry/runs.csv` 每行给出数据集、源 CSV、具体 YAML、时间划分、指标文件和 checkpoint。
`datasets.json` 给出各数据集的 raw/processed 文件；`notebook_models.json` 给出每个分析预设对应的权重及其可用状态。

| 用途 | 位置 |
| --- | --- |
| 八种子分类与预训练消融 | `experiments/mtpp/classification/paper_pretraining/{azdx,azdx_minus_b,scsn,scsn_minus_b,scratch}` |
| 回归消融 | `experiments/mtpp/regression/mixer_ablation_5_seeds` |
| LSTM 对照 | `experiments/mtpp/regression/lstm/5_seeds_v6` |
| SWC 点过程消融 | `experiments/mtpp/generation/swc_ablation` |
| SCSN/SWC 点过程比较预设 | `notebooks/config/model_specs/tpp_comparison/model_specs.tpp_comparison.v1.json` |
| ETAS 多数据集主汇总 | `experiments/netas/etas/split_0p7/reports/etas_grid_metrics_group_mean_std.csv` |
| RECAST 多数据集主汇总 | `experiments/netas/recast/split_0p7/reports/rtpp_v2_grid_metrics_group_mean_std.csv` |
| Oracle 多数据集主汇总 | `experiments/netas/oracle/split_0p7/reports/oracle_grid_metrics_group_mean_std.csv` |

每组实验使用 `configs/`、`runs/`、`reports/`。旧名字到新名字的对应在 `experiments/registry/path_renames.json`，不保留旧路径转发层。
分类论文实验从发布前提交 `1d80f7de1` 恢复，仅恢复配置、指标和 400 个 checkpoint，跳过重复图片和日志；恢复清单位于 `paper_recovery.json` 指向的归档目录。

## 缓存规则

事件窗口只有一个当前格式：`cache/windows/<hash>.pkl`，旁边的 JSON 记录来源 SHA-256、窗口参数与训练拟合边界。
模型类别、LSTM 时间步、特征列和 Mag_elaps 不参与原始窗口缓存标识。相同参数的 `3` 与 `3.0` 共用缓存。
源数据或拟合边界改变会生成新缓存；写入采用临时文件和原子替换，避免中断后留下半个 pickle。
命中已有缓存时跳过归一化和窗口重建，完整元数据不重复写入；损坏的元数据会从当前来源信息原子修复。
旧的 classifier/regressor/lstm_regressor/rf_classifier 缓存不再读取，也不再提交到 Git。

## 存储和恢复

当前数据、raw/processed 输入及 revision 实验保留。历史大运行目录的实体位于同账户归档盘，规范路径中的 runs/profiles 链接直接访问实体。
归档位置：`/root/autodl-fs/em_eqf_archive/maintenance_20261109/`。
`manifest.json` 记录迁移清单及逐文件 SHA-256；旧窗口缓存位于 `obsolete_window_caches.tar.gz`，删除本地版本前已完整校验压缩包。

维护脚本默认只打印计划：

```bash
python scripts/maintenance/clean_storage.py --archive-root /path/to/new/archive
```

需要从旧缓存归档取回单个文件时，先用 `tar -tzf` 查看，再用 `tar -xzf <archive> -C <temporary-directory> <relative-file>` 提取，避免覆盖当前数据。
代码备份和原始索引在 `/root/autodl-tmp/em_eqf_maintenance_backups/20261109/`。
旧报告中未实体化的 LFS 文件及原因列于 `experiments/registry/lfs_report_status.json`，不将占位文本当作指标数据。

## 代码与提交

脚本按 run、summarize、data、analyze、diagnostics、tuning、maintenance 分目录，直接调用真实入口。
已移除根目录转发脚本及已完成的一次性迁移代码。旧演示脚本已归档，pytest 目录只保留可自动运行的测试。
`research-main` 提供简洁的主题提交历史；完整旧历史保留在 `archive-development-20261109`，详见 `HISTORY.md`。未推送或改写已发布分支。
