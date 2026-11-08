# Mamba3EventTime

Canonical model: src/models/mamba/mamba3_event_time.py, class Mamba3EventTime.
MIMO backend: src/models/mamba/mamba3_mimo_kernels.py.
Mixer dispatch: src/models/mamba/mixer_seq.py.
TPP profile: config/mixer_tpp_mamba3_event_time.yaml.

Current user-selected Mamba3 default: MIMO rank=2, d_model/state=64,
headdim=32, 2 blocks WITHOUT residual additions, decay_only, native phase,
and no direct log(delta_t) content. The TPP time head uses legacy parameterization.
Historical SCEDC full-sequence experiments used residuals; their saved results
do not measure this new no-residual default.

## Independent controls

TPP adapter options, NOT inferred from model name:
    provide_physical_dt: true
    mask_input_features: [log_inter_times]
Set mask_input_features=[] to restore original feature values.

SSM options under mixer_model_config.ssm_cfg:
    layer: Mamba3EventTime
    is_mimo: true
    mimo_rank: 2
    time_update_mode: decay_only
    use_event_time_phase: false

time_update_mode accepts base, full_dt, decay_only.
use_event_time_phase=true adds omega * normalized timestamp.
Masking occurs by actual feature name AFTER normalization. It changes neither
projection shape nor parameter count, magnitude values, score masks, targets,
timestamps nor physical_dt. The column is zero, not removed.
The generic adapter defaults remain mask_input_features=[] and
provide_physical_dt=false; this NEW profile explicitly selects the ablation.
No model-name checks remain in the adapter.

physical_dt is observed inter_times / adapter time_scale_base, padding=1.
The profile uses the fixed TRAINING tau_mean scale; never use min-max gaps or
cumulative t/tau as physical_dt. Custom callers must explicitly supply gaps
when time_update_mode != base. Missing gaps fail explicitly.
Extra phase is controlled in the model; an explicit phase_times override may
still be used for diagnostics.

## Compatibility

Mamba3Rotary remains available with its legacy defaults (base updates, SISO,
extra event-time phase enabled). mamba3_rotary_mimo is a compatibility module
alias for mamba3_mimo_kernels, preserving runtime instrumentation.
No parameter keys or tensor shapes changed. A state_dict can load strictly.
Evaluation still needs the ORIGINAL input/phase/update settings: weight
compatibility alone does not make different ablations predict identically.
Historical experiment snapshots and installed native kernels are unchanged.
