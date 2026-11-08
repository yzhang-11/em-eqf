# Data and experiment index

Regenerate with `python scripts/maintenance/index_experiments.py`.

`datasets.json` lists source and processed inputs. `runs.csv` maps each dataset to its exact run, YAML, metrics and checkpoints.

`notebook_models.json` maps each notebook preset (including `SCEDC_t` and `ChuanDian_t`) to the exact checkpoint and records whether its weights are materialized.

SWC is the paper name for `ChuanDian`; SCSN is the paper name for `SCEDC`. Names remain unchanged in data loading.

| Experiment | Datasets | Runs | First result file |
| --- | --- | ---: | --- |
| `mtpp/classification/azdx` | ChuanDian | 25 | `experiments/mtpp/classification/azdx/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/azdx_minus_b` | ChuanDian | 25 | `experiments/mtpp/classification/azdx_minus_b/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/grid_azdx` | ChuanDian | 50 | `experiments/mtpp/classification/grid_azdx/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/grid_azdx_2` | ChuanDian | 50 | `experiments/mtpp/classification/grid_azdx_2/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/grid_encoder_lr` | ChuanDian | 15 | `experiments/mtpp/classification/grid_encoder_lr/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/grid_r_2_azdx_minus_b` | ChuanDian | 15 | `experiments/mtpp/classification/grid_r_2_azdx_minus_b/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/grid_r_2_layer_1` | ChuanDian | 15 | `experiments/mtpp/classification/grid_r_2_layer_1/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/grid_r_2_scedc` | ChuanDian | 15 | `experiments/mtpp/classification/grid_r_2_scedc/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/grid_r_2_scedc_minus_b` | ChuanDian | 15 | `experiments/mtpp/classification/grid_r_2_scedc_minus_b/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/grid_r_2_scedc_minus_b_bs_64` | ChuanDian | 50 | `experiments/mtpp/classification/grid_r_2_scedc_minus_b_bs_64/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/grid_r_2_scratch` | ChuanDian | 15 | `experiments/mtpp/classification/grid_r_2_scratch/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/grid_r_2_scratch_2` | ChuanDian | 15 | `experiments/mtpp/classification/grid_r_2_scratch_2/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/grid_tf_90_alpha_075` | ChuanDian | 10 | `experiments/mtpp/classification/grid_tf_90_alpha_075/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/grid_tf_90_azdx` | ChuanDian | 150 | `experiments/mtpp/classification/grid_tf_90_azdx/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/grid_tf_90_stability_grid` | ChuanDian | 60 | `experiments/mtpp/classification/grid_tf_90_stability_grid/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/grid_tf_90_stability_grid_v2` | ChuanDian | 36 | `experiments/mtpp/classification/grid_tf_90_stability_grid_v2/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/paper_pretraining/azdx` | ChuanDian | 40 | `experiments/mtpp/classification/paper_pretraining/azdx/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/paper_pretraining/azdx_minus_b` | ChuanDian | 40 | `experiments/mtpp/classification/paper_pretraining/azdx_minus_b/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/paper_pretraining/scratch` | ChuanDian | 40 | `experiments/mtpp/classification/paper_pretraining/scratch/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/paper_pretraining/scsn` | ChuanDian | 40 | `experiments/mtpp/classification/paper_pretraining/scsn/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/paper_pretraining/scsn_minus_b` | ChuanDian | 40 | `experiments/mtpp/classification/paper_pretraining/scsn_minus_b/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/scedc` | ChuanDian | 25 | `experiments/mtpp/classification/scedc/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/scedc_minus_b` | ChuanDian | 25 | `experiments/mtpp/classification/scedc_minus_b/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/scratch` | ChuanDian | 25 | `experiments/mtpp/classification/scratch/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/single_window_pretrain` | ChuanDian | 25 | `experiments/mtpp/classification/single_window_pretrain/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/single_window_pretrain_b64` | ChuanDian | 25 | `experiments/mtpp/classification/single_window_pretrain_b64/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/single_window_pretrain_b64_layer1` | ChuanDian | 25 | `experiments/mtpp/classification/single_window_pretrain_b64_layer1/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/classification/single_window_pretrain_b64_lr3e3` | ChuanDian | 25 | `experiments/mtpp/classification/single_window_pretrain_b64_lr3e3/reports/clf_grid_metrics_group_best.csv` |
| `mtpp/generation/legacy_ablation_v1` | ChuanDian | 5 | `experiments/mtpp/generation/legacy_ablation_v1/reports/mixer_tpp_grid_metrics_group_best.csv` |
| `mtpp/generation/swc_ablation` | ChuanDian | 6 | `experiments/mtpp/generation/swc_ablation/reports/mixer_tpp_grid_metrics_group_best.csv` |
| `mtpp/regression/grid_inter9_excl2_round` | ChuanDian | 18 | `experiments/mtpp/regression/grid_inter9_excl2_round/reports/reg_grid_metrics_group_best.csv` |
| `mtpp/regression/grid_r04_stable8` | ChuanDian | 24 | `experiments/mtpp/regression/grid_r04_stable8/reports/reg_grid_metrics_group_best.csv` |
| `mtpp/regression/lstm/5_seeds` | ChuanDian | 60 | `experiments/mtpp/regression/lstm/5_seeds/reports/lstm_grid_metrics_group_best.csv` |
| `mtpp/regression/lstm/5_seeds_v6` | ChuanDian | 18 | `experiments/mtpp/regression/lstm/5_seeds_v6/reports/lstm_grid_metrics_group_best.csv` |
| `mtpp/regression/lstm/optuna_profiles/run_20x2` |  | 0 | `` |
| `mtpp/regression/lstm/seed` | ChuanDian | 3 | `experiments/mtpp/regression/lstm/seed/reports/lstm_grid_metrics_group_best.csv` |
| `mtpp/regression/lstm/seed_global` | ChuanDian | 3 | `experiments/mtpp/regression/lstm/seed_global/reports/lstm_grid_metrics_group_best.csv` |
| `mtpp/regression/mixer_ablation_5_seeds` | ChuanDian | 30 | `experiments/mtpp/regression/mixer_ablation_5_seeds/reports/reg_grid_metrics_group_best.csv` |
| `mtpp/regression/mixer_attnpl_t_optuna_profiles/20260510-232207` |  | 0 | `` |
| `mtpp/regression/mixer_attnpl_t_optuna_profiles/run_20260511-000926` |  | 0 | `` |
| `mtpp/regression/mixer_attnpl_t_optuna_profiles/run_20260511-001210` |  | 0 | `` |
| `mtpp/regression/mixer_layer_1_grid` | ChuanDian | 24 | `experiments/mtpp/regression/mixer_layer_1_grid/reports/reg_grid_metrics_group_best.csv` |
| `mtpp/regression/mixer_layer_1_grid_2` | ChuanDian | 8 | `experiments/mtpp/regression/mixer_layer_1_grid_2/reports/reg_grid_metrics_group_best.csv` |
| `mtpp/regression/mixer_layer_1_grid_bs_128` | ChuanDian | 25 | `experiments/mtpp/regression/mixer_layer_1_grid_bs_128/reports/reg_grid_metrics_group_best.csv` |
| `mtpp/regression/mixer_layer_1_grid_bs_128_followup` | ChuanDian | 54 | `experiments/mtpp/regression/mixer_layer_1_grid_bs_128_followup/reports/reg_grid_metrics_group_best.csv` |
| `mtpp/regression/mixer_layer_1_grid_max_grad_norm` | ChuanDian | 18 | `experiments/mtpp/regression/mixer_layer_1_grid_max_grad_norm/reports/reg_grid_metrics_group_best.csv` |
| `netas/etas/baseline` | CB_HAB1a, CB_HAB4, PNR_1z, St1-2018 | 12 | `experiments/netas/etas/baseline/reports/etas_grid_metrics_group_best.csv` |
| `netas/etas/norm_0p2` | CB_HAB1a, CB_HAB4, PNR_1z, St1-2018 | 87 | `experiments/netas/etas/norm_0p2/reports/etas_grid_metrics_group_best.csv` |
| `netas/etas/split_0p7` | CB_HAB1a, CB_HAB4, PNR_1z, St1-2018 | 84 | `experiments/netas/etas/split_0p7/reports/etas_grid_metrics_group_best.csv` |
| `netas/oracle/baseline` | CB_HAB1a, CB_HAB4, St1-2018 | 5 | `experiments/netas/oracle/baseline/reports/oracle_grid_metrics_group_best.csv` |
| `netas/oracle/split_0p7` | CB_HAB1a, CB_HAB4, PNR_1z, St1-2018 | 12 | `experiments/netas/oracle/split_0p7/reports/oracle_grid_metrics_group_best.csv` |
| `netas/recast/dual_ablation` | CB_HAB1a | 8 | `` |
| `netas/recast/grid_20260528-102245` | CB_HAB1a | 3 | `` |
| `netas/recast/legacy_0p4` | CB_HAB1a, CB_HAB4, PNR_1z, St1-2018 | 12 | `experiments/netas/recast/legacy_0p4/reports/rtpp_v2_grid_metrics_group_best.csv` |
| `netas/recast/legacy_norm_0p2` | CB_HAB1a, CB_HAB4, PNR_1z, St1-2018 | 12 | `experiments/netas/recast/legacy_norm_0p2/reports/rtpp_v2_grid_metrics_group_best.csv` |
| `netas/recast/norm_0p2` | CB_HAB1a, CB_HAB4, PNR_1z, St1-2018 | 87 | `experiments/netas/recast/norm_0p2/reports/rtpp_v2_grid_metrics_group_best.csv` |
| `netas/recast/split_0p7` | St1-2018 | 0 | `experiments/netas/recast/split_0p7/reports/rtpp_v2_grid_metrics_group_best.csv` |
| `revision/catalog_additions_20261105` | SED_Switzerland, USGS_Alaska, USGS_Hawaii, USGS_PuertoRico | 0 | `experiments/revision/catalog_additions_20261105/reports/SED_INGV_overlap_candidates.csv` |
| `revision/catalog_benchmarks_20261106` | ENPP_ComCat25, ENPP_JapanDeprecated, ENPP_QTMSaltonSea10, ENPP_QTMSanJacinto10, ENPP_SCEDC20, ENPP_SCEDC25, ENPP_SCEDC30, ENPP_White06 | 0 | `experiments/revision/catalog_benchmarks_20261106/reports/benchmark_catalogs.csv` |
| `revision/catalog_more_20261105` | NOA_Greece, USGS_NewMadrid, USGS_PNW, USGS_Utah | 0 | `experiments/revision/catalog_more_20261105/reports/ISOR_Hengill_annual.csv` |
| `revision/catalog_research_20261106` | NORWAY_Bitdalsvatn, NORWAY_SW, USGS_Montana | 0 | `experiments/revision/catalog_research_20261106/reports/USGS_Montana_annual.csv` |
| `revision/catalog_source_expansion_20261106` | Taiwan_CWA, Taiwan_Wu2008, Turkiye_EAFZ | 0 | `experiments/revision/catalog_source_expansion_20261106/reports/eafz_identical_physical_records.csv` |
| `revision/catalog_standardization_20261105` | GeoNet, INGV, JMA_EastHonshu, JMA_Hokkaido, JMA_Kyushu, JMA_WestHonshu, NCEDC | 0 | `experiments/revision/catalog_standardization_20261105/reports/catalogs.csv` |
| `revision/controls` | ChuanDian | 60 | `experiments/revision/controls/reports/large_target_support.csv` |
| `revision/generation_baselines` | ChuanDian, SCEDC | 12 | `experiments/revision/generation_baselines/reports/generation_periods.csv` |
| `revision/magnitude_models` |  | 0 | `experiments/revision/magnitude_models/reports/confirmation_validation_scores.csv` |
| `revision/main` | AZDX, ChuanDian, SCEDC | 86 | `` |
| `revision/mamba3_mimo_no_deltat_20261105` |  | 0 | `experiments/revision/mamba3_mimo_no_deltat_20261105/reports/all_results.csv` |
| `revision/mamba_backbones_20261105` |  | 0 | `experiments/revision/mamba_backbones_20261105/reports/all_results.csv` |
| `revision/mamba_phase_ablation_20261105` |  | 0 | `experiments/revision/mamba_phase_ablation_20261105/reports/all_results.csv` |
| `revision/mamba_time_dynamics_20261105` |  | 0 | `experiments/revision/mamba_time_dynamics_20261105/reports/all_results.csv` |
| `revision/multiregion_pretraining` |  | 0 | `experiments/revision/multiregion_pretraining/reports/jma_annual_inventory_20261104.csv` |
| `revision/scedc_full_mamba2_baseline_20261105` |  | 0 | `experiments/revision/scedc_full_mamba2_baseline_20261105/reports/all_results.csv` |
| `revision/scedc_full_recast_20261106` |  | 0 | `experiments/revision/scedc_full_recast_20261106/reports/all_results.csv` |
| `revision/scedc_full_time_dynamics_20261105` |  | 0 | `experiments/revision/scedc_full_time_dynamics_20261105/reports/all_results.csv` |
| `standalone` | AZDX, Basel, CB_HAB1a, CB_HAB1b, CB_HAB4, CCL, CD2021, ChinaArray, ChuanDian, CooperBasin, FORGE2022, Geysers, PNR, PNR_1z, QTMSaltonSea, QTMSanJacinto, S2, SCEDC, SSFS, SSFS2005, St1-2018, St1-2020, White | 881 | `` |
