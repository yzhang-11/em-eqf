"""MIMO scan and recurrent inference for additive event-time Mamba-3.

Autograd wrapper adapted from state-spaces/mamba's mamba3_mimo.py.
Copyright (c) 2026, Dao AI Lab, Goombalab.
The installed Mamba-3 kernels remain unchanged.
"""
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F
from mamba_ssm.modules.mamba3 import mamba3_step_fn

from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import (
    compute_dacs_segsum_triton, compute_dacs_segsum_triton_varlen,
)
from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_fwd import mamba_mimo_forward
from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_fwd_varlen import mamba_mimo_forward_varlen
from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd import mamba_mimo_bwd_combined
from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd_varlen import mamba_mimo_bwd_combined_varlen

class _Mamba3MIMOEventTimeFunction(torch.autograd.Function):
    """Custom autograd function for Mamba-3 with Triton/Tilelang kernels."""

    @staticmethod
    def forward(
        ctx,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        ADT: Tensor,
        DT: Tensor,
        Trap: Tensor,
        Q_bias: Tensor,
        K_bias: Tensor,
        MIMO_V: Tensor,
        MIMO_Z: Tensor,
        MIMO_Out: Union[Tensor, None],
        Out_Norm_Weight: Union[Tensor, None],
        Angles: Tensor,
        D: Tensor,
        Z: Tensor,
        chunk_size: int,
        rotary_dim_divisor: int,
        dtype: torch.dtype,
        return_state: bool,
        cu_seqlens: Optional[Tensor],
        fuse_pregate_headwise_rms_norm: bool,
        outproj_norm_eps: float,
    ) -> Tensor | Tuple[Tensor, Tuple]:
        """Forward pass: call Triton/Tilelang kernel and save tensors for backward."""
        ctx.chunk_size = chunk_size
        ctx.rotary_dim_divisor = rotary_dim_divisor
        ctx.dtype = dtype
        ctx.fuse_pregate_headwise_rms_norm = fuse_pregate_headwise_rms_norm
        ctx.outproj_norm_eps = outproj_norm_eps
        (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, MIMO_V, MIMO_Z, MIMO_Out, Out_Norm_Weight, Angles, D, Z) = tuple(
            t.contiguous() if t is not None else None
            for t in (
                Q, K, V, ADT, DT, Trap, Q_bias, K_bias, MIMO_V, MIMO_Z, MIMO_Out, Out_Norm_Weight, Angles, D, Z,
            )
        )
        # Kernels require cu_seqlens as int32; torch.cumsum promotes int32→int64 on CUDA
        if cu_seqlens is not None and cu_seqlens.dtype != torch.int32:
            cu_seqlens = cu_seqlens.to(torch.int32)

        # The layer already added native learned and event-time phases.
        Angles_Cumsum = Angles

        if cu_seqlens is not None:
            DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton_varlen(ADT, chunk_size, cu_seqlens=cu_seqlens)
            Out, Final_SSM_State, Final_K = mamba_mimo_forward_varlen(
                Q, K, V, Q_bias, K_bias, MIMO_V, MIMO_Out,
                Z, D, MIMO_Z, Angles_Cumsum,
                DA_CS, DA_CS_REV, DT, Trap, Segsum,
                cu_seqlens=cu_seqlens,
                return_state=return_state,
                chunk_size=chunk_size, rotary_dim_divisor=rotary_dim_divisor,
                dtype=dtype,
                fuse_pregate_headwise_rms_norm=fuse_pregate_headwise_rms_norm,
                outproj_norm_weight=Out_Norm_Weight,
                outproj_norm_eps=outproj_norm_eps,
            )

        else:
            DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton(ADT, chunk_size)
            Out, Final_SSM_State, Final_K = mamba_mimo_forward(
                Q, K, V, Q_bias, K_bias, MIMO_V, MIMO_Out,
                Z, D, MIMO_Z, Angles_Cumsum,
                DA_CS, DA_CS_REV, DT, Trap, Segsum,
                return_state=return_state,
                chunk_size=chunk_size, rotary_dim_divisor=rotary_dim_divisor,
                dtype=dtype,
                fuse_pregate_headwise_rms_norm=fuse_pregate_headwise_rms_norm,
                outproj_norm_weight=Out_Norm_Weight,
                outproj_norm_eps=outproj_norm_eps,
            )

        ctx.chunk_size = chunk_size
        ctx.save_for_backward(
            Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles_Cumsum,
            D, Z,
            MIMO_V, MIMO_Out, MIMO_Z, Out_Norm_Weight,
            cu_seqlens,
        )

        if not return_state:
            return Out
        else:
            Final_SSM_State = Final_SSM_State.permute(0, 1, 3, 2).contiguous().detach()
            Final_K = Final_K.contiguous().detach()
            Final_V = (
                V[:, -1] if cu_seqlens is None
                else V[0, cu_seqlens[1:].long() - 1]
            ).contiguous().detach()
            ctx.mark_non_differentiable(Final_SSM_State, Final_K, Final_V)
            return Out, Final_SSM_State, Final_K, Final_V

    @staticmethod
    def backward(ctx, dout, *args) -> tuple:
        """Backward pass: compute gradients using Tilelang backward kernels."""

        if len(ctx.saved_tensors) == 0:
            raise RuntimeError(
                "Backward called but forward ran without gradient tracking. "
                "Ensure inputs require grad or run under torch.enable_grad()."
            )
        dout = dout.contiguous()

        (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles_Cumsum,
            D, Z,
            MIMO_V, MIMO_Out, MIMO_Z, Out_Norm_Weight,
            cu_seqlens
            ) = ctx.saved_tensors

        # Native dense MIMO backward requires whole chunks. Neutral trailing
        # events have DT=ADT=V=0, so they cannot change any real output/state.
        # Pad only backward reconstruction, preserving forward cache endpoints.
        original_length = Q.shape[1]
        padding = (-original_length) % ctx.chunk_size if cu_seqlens is None else 0
        if padding:
            def pad_axis(tensor, axis):
                if tensor is None:
                    return None
                shape = list(tensor.shape)
                shape[axis] = padding
                return torch.cat((tensor, tensor.new_zeros(shape)), dim=axis)
            Q, K, V = (pad_axis(tensor, 1) for tensor in (Q, K, V))
            Z = pad_axis(Z, 1)
            Angles_Cumsum = pad_axis(Angles_Cumsum, 1)
            ADT, DT, Trap = (pad_axis(tensor, 2) for tensor in (ADT, DT, Trap))
            dout = pad_axis(dout, 1)

        if cu_seqlens is not None:
            DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton_varlen(ADT, ctx.chunk_size, cu_seqlens=cu_seqlens)
            (dQ, dK, dV,
                dADT, dDT, dTrap, dQ_bias, dK_bias,
                dMIMO_V, dMIMO_Z, dMIMO_Out, dAngles_Cumsum,
                dD, dZ, dOut_Norm_Weight) = mamba_mimo_bwd_combined_varlen(
                    dout,
                    Q,
                    K,
                    V,
                    Q_bias,
                    K_bias,
                    MIMO_V,
                    MIMO_Out,
                    Z,
                    MIMO_Z,
                    Angles_Cumsum,
                    DA_CS,
                    DA_CS_REV,
                    DT,
                    Trap,
                    D,
                    Segsum,
                    ctx.chunk_size,
                    ctx.rotary_dim_divisor,
                    ctx.dtype,
                    cu_seqlens=cu_seqlens,
                    fuse_pregate_headwise_rms_norm=ctx.fuse_pregate_headwise_rms_norm,
                    outproj_norm_weight=Out_Norm_Weight,
                    outproj_norm_eps=ctx.outproj_norm_eps,
                )
        else:
            DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton(ADT, ctx.chunk_size)
            (dQ, dK, dV,
                dADT, dDT, dTrap, dQ_bias, dK_bias,
                dMIMO_V, dMIMO_Z, dMIMO_Out, dAngles_Cumsum,
                dD, dZ, dOut_Norm_Weight) = mamba_mimo_bwd_combined(
                    dout,
                    Q,
                    K,
                    V,
                    Q_bias,
                    K_bias,
                    MIMO_V,
                    MIMO_Out,
                    Z,
                    MIMO_Z,
                    Angles_Cumsum,
                    DA_CS,
                    DA_CS_REV,
                    DT,
                    Trap,
                    D,
                    Segsum,
                    ctx.chunk_size,
                    ctx.rotary_dim_divisor,
                    ctx.dtype,
                    fuse_pregate_headwise_rms_norm=ctx.fuse_pregate_headwise_rms_norm,
                    outproj_norm_weight=Out_Norm_Weight,
                    outproj_norm_eps=ctx.outproj_norm_eps,
                )

        if padding:
            dQ, dK, dV = (tensor[:, :original_length] for tensor in (dQ, dK, dV))
            dADT, dDT, dTrap = (tensor[..., :original_length] for tensor in (dADT, dDT, dTrap))
            dAngles_Cumsum = dAngles_Cumsum[:, :original_length]
            if dZ is not None:
                dZ = dZ[:, :original_length]
        # Native-angle/DT and physical-time gradients branch in the layer.
        dAngles = dAngles_Cumsum
        if dOut_Norm_Weight is not None and Out_Norm_Weight is not None:
            dOut_Norm_Weight = dOut_Norm_Weight.reshape_as(Out_Norm_Weight)

        return (
            dQ,
            dK,
            dV,
            dADT,
            dDT,
            dTrap,
            dQ_bias,
            dK_bias,
            dMIMO_V,
            dMIMO_Z,
            dMIMO_Out,
            dOut_Norm_Weight,
            dAngles,
            dD,
            dZ,
            None, None, None, None, None, None, None,
        )

def mimo_event_time_scan(model, Q, K, V, adt, dt, trap, phase, z, cu_seqlens, return_state):
    """Native parallel MIMO scan at the supplied combined absolute phases."""
    fused_norm = model.fuse_pregate_headwise_norm
    reduce_output = fused_norm or not model.is_outproj_norm
    return _Mamba3MIMOEventTimeFunction.apply(
        Q.to(torch.bfloat16), K.to(torch.bfloat16), V.to(torch.bfloat16),
        adt.transpose(1, 2).float(), dt.transpose(1, 2).float(),
        trap.transpose(1, 2).to(torch.bfloat16),
        model.C_bias.float(), model.B_bias.float(),
        model.mimo_x.float(), model.mimo_z.float(),
        model.mimo_o.float() if reduce_output else None,
        model.norm.weight.float() if fused_norm else None,
        phase.float(), model.D.float(),
        z.to(torch.bfloat16) if reduce_output else None,
        model.chunk_size, model.rotary_dim_divisor, torch.bfloat16,
        return_state, cu_seqlens, fused_norm,
        model.norm.eps if fused_norm else 1e-5,
    )



def _aligned_rows(tensor):
    """Satisfy the native CuTe wrapper's 128-bit leading-stride contract."""
    elements = 16 // tensor.element_size()
    width = tensor.shape[-1]
    padded_width = (width + elements - 1) // elements * elements
    if width == padded_width:
        return tensor.contiguous()
    storage = torch.empty(
        *tensor.shape[:-1], padded_width, device=tensor.device, dtype=tensor.dtype,
    )
    view = storage[..., :width]
    view.copy_(tensor)
    return view


def mimo_step_values(model, B, C, x, z, adt, dt, trap, phase, ssm_state, k_state, v_state):
    """Native Triton rotation followed by native CuTe MIMO state update.

    Phases are already advanced and combined by the layer. Zero rates tell
    the rotary kernel to use this absolute phase without accumulating twice.
    Contiguous/padded views fix stride assumptions. Heads below 64 are
    zero-extended to the native 64-wide state/reduction tile, then cropped.
    """
    from mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step import apply_rotary_qk_inference_fwd

    if mamba3_step_fn is None:
        raise RuntimeError("Native MIMO inference requires the Mamba-3 CuTe step dependencies")
    q = C[:, 0].to(torch.bfloat16).repeat_interleave(model.nheads // model.num_bc_heads, dim=2)
    k = B[:, 0].to(torch.bfloat16).repeat_interleave(model.nheads // model.num_bc_heads, dim=2)
    q, k, _ = apply_rotary_qk_inference_fwd(
        q=q, k=k, angle_state=phase,
        angle_proj=torch.zeros_like(phase), dt=dt[:, 0].contiguous(),
        bias_q=model.C_bias.permute(1, 0, 2).float().contiguous(),
        bias_k=model.B_bias.permute(1, 0, 2).float().contiguous(),
        conjugate=False, inplace=False, rotate_pairwise=False,
    )
    v_real = x[:, 0].to(torch.bfloat16).contiguous()
    gate_real = z[:, 0].to(torch.bfloat16).contiguous()
    head_padding = max(0, 64 - model.headdim)
    v = F.pad(v_real, (0, head_padding)) if head_padding else v_real
    gate = F.pad(gate_real, (0, head_padding)) if head_padding else gate_real
    previous_v = v_state.to(v.dtype).contiguous()
    step_state = ssm_state
    if head_padding:
        previous_v = F.pad(previous_v, (0, head_padding))
        step_state = F.pad(ssm_state, (0, 0, 0, head_padding))
    step_dt = _aligned_rows(dt[:, 0].float())
    step_a = _aligned_rows(adt[:, 0] / dt[:, 0].clamp_min(torch.finfo(torch.float32).tiny))
    step_trap = _aligned_rows(trap[:, 0].to(torch.bfloat16).float().sigmoid())
    xproj = model.mimo_x.permute(1, 0, 2).float().contiguous()
    zproj = model.mimo_z.permute(1, 0, 2).float().contiguous()
    outproj = model.mimo_o.permute(1, 0, 2).float().contiguous()
    if head_padding:
        xproj, zproj, outproj = (
            F.pad(projection, (0, head_padding)) for projection in (xproj, zproj, outproj)
        )
    if model.is_outproj_norm:
        y = torch.empty(
            v.shape[0], model.mimo_rank, model.nheads, v.shape[-1],
            device=v.device, dtype=v.dtype,
        )
        mamba3_step_fn(
            step_state, k_state.to(q.dtype).contiguous(), previous_v,
            step_a, k.contiguous(), q.contiguous(), model.D.float().contiguous(),
            v, step_dt, step_trap, xproj,
            outproj=None, state_out=None, out=y, z=None, zproj=None,
            tile_D=64, num_warps=4,
        )
        y = y[..., :model.headdim].contiguous()
        if model.fuse_pregate_headwise_norm:
            # The native module also normalizes each rank after the CuTe step.
            y = model._postprocess(
                y, outproj[..., :model.headdim].contiguous(),
                gate_real, zproj[..., :model.headdim].contiguous(), model.headdim,
            )
    else:
        y = torch.empty_like(v)
        mamba3_step_fn(
            step_state, k_state.to(q.dtype).contiguous(), previous_v,
            step_a, k.contiguous(), q.contiguous(), model.D.float().contiguous(),
            v, step_dt, step_trap, xproj,
            outproj=outproj, state_out=None, out=y, z=gate, zproj=zproj,
            tile_D=64, num_warps=4,
        )
        y = y[..., :model.headdim].contiguous()
    if head_padding:
        ssm_state.copy_(step_state[..., :model.headdim, :])
    k_state.copy_(k)
    v_state.copy_(v_real)
    return y.unsqueeze(1)
