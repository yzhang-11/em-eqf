# Mamba3Rotary: native learned phase plus real event time

This SISO/MIMO Mamba-3 variant implements the recommended additive phase:

```text
native_phase_i = sum(k <= i) pi * tanh(angle_k) * DT_k
phase_i = native_phase_i + omega * event_time_i
```

Both terms rotate the same B/C channel pairs: interleaved for SISO,
split-half (j, j + d_state/2) for MIMO. All MIMO ranks share the phase. The learned angle
projection and its gradients through DT remain active. Input-dependent A,
B/C RMSNorm and biases, exponential-trapezoidal recurrence, skip connection
and gating are retained. The event-time term is added after native phase
accumulation and is not multiplied by learned DT.

## Configuration

Use the complete config/mixer_tpp_mamba3_rotary.yaml through the existing
--config option (the CLI also requires --model mixer_tpp), or select the layer in another mixer task:

```yaml
mixer_model_config:
  ssm_cfg:
    layer: Mamba3Rotary
    d_state: 128
    expand: 2
    headdim: 64
    rope_fraction: 0.5
    rotary_emb_base: 10000.0
    chunk_size: 64
```

To train from the project root, use:

    /root/autodl-tmp/venvs/em_eqf-cu128/bin/python main.py --model mixer_tpp --config config/mixer_tpp_mamba3_rotary.yaml --mode train

The existing adapters pass actual event arrival times in the explicit times
argument. With normalize_time_by_token: true, the timestamps are divided by
the fixed training tau_mean. Set it to false to use the catalog's original
time unit. Use the same unit and training scale for fitting and generation.

## Training, checkpoints and inference

The two phase branches share the combined-phase gradient. The native branch
uses upstream angle_dt_fwd/angle_dt_bwd, preserving angle and DT gradients;
the physical branch differentiates omega * event_time. The upstream Triton/TileLang
scan/backward kernels and the installed mamba_ssm package are unchanged.
Native phase accumulation uses FP32; event timestamps stay FP32 or FP64
until modulo reduction. Q/K/V use the upstream scan's BF16 math.

The parameter names and shapes match native Mamba-3 of the same SISO/MIMO configuration, so its state dict
can initialize this model. Its predictions change when the time term is
active. The earlier event-time-only prototype removed angle projection rows
and has a different projection shape.

Both full and cached inference accept times with shape (B, L), or (L,) for
shared timestamps. A single-event step accepts (B, D) or (B, 1, D). The cache
stores native phase only, FP32 SSM state, previous K (BF16 for MIMO) and V. Each new event
advances native phase once, then adds its current absolute time phase.
The physical term is never fed back into native accumulation. One history
prefix can broadcast into a larger sampling cache.

## Native MIMO kernels

Use config/mixer_tpp_mamba3_rotary_mimo.yaml, or configure:

    ssm_cfg:
      layer: Mamba3Rotary
      is_mimo: true
      mimo_rank: 2
      d_state: 64
      headdim: 32
      chunk_size: 16

Rank 2/4 are covered by GPU tests. MIMO chooses split-half pairs automatically;
do not force rotary_emb_interleaved: true. The default MIMO chunk size is
64 // mimo_rank, at least 8. The example uses the RTX 6000D configuration
validated on this server. Larger state/head/rank sizes can require a smaller
chunk size because the native TileLang kernels use shared memory.

Training calls the native TileLang forward/backward kernels with the combined
phase. Dense backward pads neutral trailing events internally and crops the
gradients, so the real sequence length need not divide chunk_size. Forward
outputs and final cache states retain their actual event endpoints.

Inference calls the native Triton Q/K rotary kernel (rotate_pairwise=False),
then the native CuTe MIMO step kernel. Tensor contiguity and 128-bit
row-stride alignment are adapted. Heads smaller than 64 are zero-extended
for the native 64-wide state/reduction tile, then cropped; the native step
currently supports d_state 32/64/128. No Python state-recurrence fallback is used.
Output RMSNorm follows the native module's per-rank postprocessing. Fused and
unfused output normalization and GQA groups are supported.

The native TileLang scan has no initial-state input, so cached chunk
continuation calls the native single-event kernel once per new event.
History is not replayed and cache memory stays fixed. Packed training/prefill
uses cu_seqlens; packed continuation is supplied as one row per sequence,
without cu_seqlens. MIMO final cache states are non-differentiable, following
the native scan contract. xPos and seq_idx-only packing remain unsupported.

The current server environment is
/root/autodl-tmp/venvs/em_eqf-cu128/bin/python (mamba-ssm 2.3.2.post1).

```bash
python -m pytest -q tests/model/test_mamba3_rotary.py tests/model/test_mamba3_rotary_mimo.py
```


## Controlled external-clock ablations (2026-11-05)

Optional forward/step argument phase_times overrides only the external rotary
clock. Omitting it preserves the existing native-phase + real-time behavior.
Passing zeros selects native learned phases only. Explicit index or surrogate
clocks isolate the effect of event order or incorrect time alignment while
leaving input features, actual event times, targets and learned DT untouched.
Cached continuation must supply the corresponding absolute override clock for
each new event. The override is never accumulated into the learned-phase cache.
Installed native kernels and the state_dict parameter layout are unchanged.

scripts/run/run_mamba_phase_ablation.py --run freezes a separate project and
runs SISO native/index/real/surrogate plus MIMO native/real, seeds 0/1/2.
All fits use identical real magnitude/log-delta-t inputs, validation-selected
checkpoints and the existing seven-region time split. It explicitly enables
residuals, matching the preceding backbone comparison.


## Physical-gap dynamics variants (2026-11-05)

Optional time_update_mode is base (default), full_dt, or decay_only.
For the latter two, pass physical_dt with shape (batch, length), containing
the positive observed inter-event interval divided by the fixed training
calibration tau. Use one for padding/survival slots. The existing min-max
normalized inter_times extra is not this physical interval.

base: ADT=A*learned_DT, DT=learned_DT.
full_dt: DT=learned_DT*physical_dt, ADT=A*DT; learned phase increments
also use this effective DT.
decay_only: DT=learned_DT, ADT=A*DT*physical_dt; learned phase and current
write scale retain learned_DT, while decay and the previous-write term change.

Forward and step accept physical_dt, including cached continuation. The four
native cache tensors and state_dict parameter shapes are unchanged. Omit the
new arguments to recover the existing model. The native kernels are unchanged.

scripts/run/run_mamba_time_dynamics.py --run compares all three update modes
with/without the additive real-time phase. Direct log-delta-t content is masked
in every primary arm. Previous with-delta-t results are historical references,
not matched direct-input controls for the new dynamics variants.
