"""Mamba-3 SISO/MIMO with native learned phases plus actual event-time phases.

Scan autograd is adapted from state-spaces/mamba's mamba3_siso_combined.py.
Copyright (c) 2025, Dao AI Lab, Goombalab.
The native CUDA/Triton kernels and installed mamba_ssm package are unchanged.
"""
import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from einops import rearrange
from src.models.mamba.time_update import apply_time_update, TIME_UPDATE_MODES
import triton

from mamba_ssm.modules.mamba3 import Mamba3, heavy_tail_activation
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import _triton_alloc_fn
from mamba_ssm.ops.triton.mamba3.mamba3_siso_fwd import mamba3_siso_fwd
from mamba_ssm.ops.triton.mamba3.mamba3_siso_bwd import (
    compute_dzdo, compute_dqkv, compute_dqktheta, compute_ddt_dtrap_dinput_states,
)
from mamba_ssm.ops.triton.mamba3.mamba3_siso_step import mamba3_siso_step
from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_fwd, angle_dt_bwd

class _Mamba3EventTimeFunction(torch.autograd.Function):
    """Custom autograd function for Mamba-3 with Triton kernels."""

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
        Angles: Tensor,
        D: Optional[Tensor],
        Z: Optional[Tensor],
        Input_SSM_State: Optional[Tensor],
        Input_K_State: Optional[Tensor],
        Input_V_State: Optional[Tensor],
        cu_seqlens: Optional[Tensor],
        chunk_size: int,
        return_final_states: bool,
    ) -> Tensor | Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass: call Triton kernel and save tensors for backward."""

        try:
            triton.set_allocator(_triton_alloc_fn)
        except Exception:
            pass

        needs_backward = any(ctx.needs_input_grad)
        has_varlen = cu_seqlens is not None

        all_states_present = (Input_SSM_State is not None) and (Input_K_State is not None) and (Input_V_State is not None)
        all_states_absent = (Input_SSM_State is None) and (Input_K_State is None) and (Input_V_State is None)

        assert all_states_present or all_states_absent, "Input states must be provided together or all be None."

        # Combined absolute phases arrive from the native accumulator plus
        # the event-time branch. This scan does not accumulate them again.
        Angles_Cumsum = Angles

        Input_States = (
            (Input_SSM_State, Input_K_State, Input_V_State)
            if Input_SSM_State is not None
            else None
        )

        Out, Out_v, SSM_States, DA_CS, DA_CS_SUM, Q_rot, K_scaled, QK_dot, Scale, Gamma, Final_States = mamba3_siso_fwd(
            Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles_Cumsum, D, Z, Input_States,
            chunk_size=chunk_size,
            store_states_adt_outv=needs_backward,
            return_final_states=return_final_states,
            cu_seqlens=cu_seqlens,
        )

        Final_SSM_State = Final_States[0] if Final_States is not None else None
        Final_K_State = Final_States[1] if Final_States is not None else None
        Final_V_State = Final_States[2] if Final_States is not None else None

        if needs_backward:
            ctx.chunk_size = chunk_size
            ctx.has_D = D is not None
            ctx.has_Z = Z is not None
            ctx.has_input_state = Input_SSM_State is not None
            ctx.return_final_states = return_final_states
            ctx.has_varlen = has_varlen

            # Save tensors - use empty tensor placeholders for None values
            D_save = D if D is not None else torch.empty((), device=Q.device)
            Z_save = Z if Z is not None else torch.empty((), device=Q.device)
            Input_SSM_State_save = Input_SSM_State if Input_SSM_State is not None else torch.empty((), device=Q.device)
            Input_K_State_save = Input_K_State if Input_K_State is not None else torch.empty((), device=Q.device)
            Input_V_State_save = Input_V_State if Input_V_State is not None else torch.empty((), device=Q.device)
            Final_SSM_State_save = Final_SSM_State if Final_SSM_State is not None else torch.empty((), device=Q.device)
            cu_seqlens_save = cu_seqlens if cu_seqlens is not None else torch.empty((), device=Q.device, dtype=torch.int32)

            ctx.save_for_backward(
                Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles_Cumsum,
                D_save, Z_save, Input_SSM_State_save, Input_K_State_save, Input_V_State_save,
                Out, Out_v, SSM_States, DA_CS, DA_CS_SUM, Q_rot, K_scaled, QK_dot, Scale, Gamma,
                Final_SSM_State_save, cu_seqlens_save
            )
        else:
            ctx.chunk_size = chunk_size
            ctx.has_D = D is not None
            ctx.has_Z = Z is not None
            ctx.has_input_state = Input_SSM_State is not None
            ctx.return_final_states = return_final_states
            ctx.has_varlen = has_varlen
            ctx.save_for_backward()

        if return_final_states:
            return Out, Final_SSM_State, Final_K_State, Final_V_State
        return Out

    @staticmethod
    def backward(
        ctx,
        grad_out: Optional[Tensor] = None,
        grad_final_ssm_state: Optional[Tensor] = None,
        grad_final_k_state: Optional[Tensor] = None,
        grad_final_v_state: Optional[Tensor] = None
    ) -> tuple:
        """Backward pass: compute gradients using Triton backward kernels."""

        try:
            triton.set_allocator(_triton_alloc_fn)
        except Exception:
            pass

        if len(ctx.saved_tensors) == 0:
            raise RuntimeError(
                "Backward called but forward ran without gradient tracking. "
                "Ensure inputs require grad or run under torch.enable_grad()."
            )
        if grad_out is None and grad_final_ssm_state is None and grad_final_k_state is None and grad_final_v_state is None:
            raise RuntimeError("No gradients provided for backward pass.")

        (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles_Cumsum,
        D_save, Z_save, Input_SSM_State_save, Input_K_State_save, Input_V_State_save,
        Out, Out_v, SSM_States, DA_CS, DA_CS_SUM, Q_rot, K_scaled, QK_dot, Scale, Gamma,
        Final_SSM_State_save, cu_seqlens_save) = ctx.saved_tensors

        D = D_save if ctx.has_D else None
        Z = Z_save if ctx.has_Z else None
        Input_SSM_State = Input_SSM_State_save if ctx.has_input_state else None
        Input_K_State = Input_K_State_save if ctx.has_input_state else None
        Input_V_State = Input_V_State_save if ctx.has_input_state else None
        cu_seqlens = cu_seqlens_save if ctx.has_varlen else None

        if grad_out is None:
            grad_out = torch.zeros_like(Out)

        # Step 1: Compute dZ and scale grad_out if Z gating is present
        if Z is not None:
            dZ, grad_out_scaled = compute_dzdo(
                grad_out, Z, Out_v, chunk_size=ctx.chunk_size
            )
        else:
            dZ = None
            grad_out_scaled = grad_out

        # Step 2: Compute main gradients (dQ_mid, dK_mid, dV, dADT, dQK_dot, dD, dInput_SSM_State)
        dQ_mid, dK_mid, dV, dADT, dQK_dot, dD, dInput_SSM_State = compute_dqkv(
            q=Q_rot,
            k=K_scaled,
            v=V,
            da_cs=DA_CS,
            da_cs_sum=DA_CS_SUM,
            qk_dot=QK_dot,
            SSM_States=SSM_States,
            do=grad_out_scaled,
            d_ossm_state=grad_final_ssm_state,
            d_ov_state=grad_final_v_state,
            D=D,
            chunk_size=ctx.chunk_size,
            has_input_state=ctx.has_input_state,
            Cu_Seqlens=cu_seqlens,
        )

        # Step 3: Compute gradients through rotary embeddings and biases
        dQ, dK, dQ_bias, dK_bias, dAngles_Cumsum, dScale, dGamma = compute_dqktheta(
            q=Q,
            k=K,
            scale=Scale,
            gamma=Gamma,
            q_bias=Q_bias,
            k_bias=K_bias,
            angles=Angles_Cumsum,
            dq_in=dQ_mid,
            dk_in=dK_mid,
            dqk=dQK_dot,
            d_ok_state=grad_final_k_state,
            chunk_size=ctx.chunk_size,
            Cu_Seqlens=cu_seqlens,
        )

        # Step 4: Compute dDT, dTrap, and input state gradients
        dDT, dTrap, dInput_SSM_State_final, dInput_K_State, dInput_V_State = compute_ddt_dtrap_dinput_states(
            dscale=dScale,
            dgamma=dGamma,
            dt=DT,
            trap=Trap.float(),
            d_issm_state=dInput_SSM_State if ctx.has_input_state else None,
            input_k_state=Input_K_State,
            input_v_state=Input_V_State,
            Cu_Seqlens=cu_seqlens,
        )

        # Return the combined-phase gradient. _NativeAngleFunction sends
        # its learned-angle/DT branch backward; the time branch is separate.
        dAngles = dAngles_Cumsum

        if ctx.has_input_state:
            dInput_SSM_State = dInput_SSM_State_final
        else:
            dInput_SSM_State = None
            dInput_K_State = None
            dInput_V_State = None

        return (
            dQ,                     # Q
            dK,                     # K
            dV,                     # V
            dADT,                   # ADT
            dDT,                    # DT
            dTrap,                  # Trap
            dQ_bias,                # Q_bias
            dK_bias,                # K_bias
            dAngles,                # Angles
            dD,                     # D
            dZ,                     # Z
            dInput_SSM_State,       # Input_SSM_State
            dInput_K_State,         # Input_K_State
            dInput_V_State,         # Input_V_State
            None,                   # cu_seqlens (not differentiable)
            None,                   # chunk_size (not differentiable)
            None,                   # return_final_states (not differentiable)
        )

def _event_time_scan(
    Q, K, V, ADT, DT, Trap, Q_bias, K_bias, phase, D, Z,
    states, cu_seqlens, chunk_size, return_final_states,
):
    """Scan using the sum of native learned and actual event-time phases."""
    return _Mamba3EventTimeFunction.apply(
        Q.to(torch.bfloat16), K.to(torch.bfloat16), V.to(torch.bfloat16),
        ADT.float().contiguous(), DT.float().contiguous(), Trap.to(torch.bfloat16),
        Q_bias.float(), K_bias.float(), phase.float(),
        D.float() if D is not None else None,
        Z.to(torch.bfloat16) if Z is not None else None,
        *states, cu_seqlens, chunk_size, return_final_states,
    )


class _NativeAngleFunction(torch.autograd.Function):
    """Preserve Mamba-3's learned-angle/DT accumulation and its gradients."""

    @staticmethod
    def forward(ctx, angles, dt, initial, cu_seqlens, chunk_size):
        ctx.set_materialize_grads(False)
        ctx.has_initial = initial is not None
        ctx.has_varlen = cu_seqlens is not None
        ctx.chunk_size = chunk_size
        placeholder = torch.empty(0, device=angles.device, dtype=torch.int32)
        ctx.save_for_backward(angles, dt, cu_seqlens if cu_seqlens is not None else placeholder)
        return angle_dt_fwd(
            angles, dt, init_state=initial, chunk_size=chunk_size,
            return_output_state=True, cu_seqlens=cu_seqlens,
        )

    @staticmethod
    def backward(ctx, grad_phase, grad_final):
        angles, dt, saved_cu = ctx.saved_tensors
        if grad_phase is None:
            grad_phase = torch.zeros_like(angles)
        dangles, ddt, dinitial = angle_dt_bwd(
            grad_out=grad_phase.contiguous(), angle=angles, dt=dt,
            has_init_state=ctx.has_initial, chunk_size=ctx.chunk_size,
            grad_output_state=grad_final,
            cu_seqlens=saved_cu if ctx.has_varlen else None,
        )
        return dangles, ddt, dinitial, None, None


class _Mamba3TimeBase(Mamba3):
    """Mamba-3 SISO/MIMO with native learned phases plus real event-time phases.

    phase_i = sum_{k <= i} pi * tanh(angle_k) * DT_k + omega * time_i.
    Times are event timestamps (or timestamps divided by the adapter's fixed
    training scale). Optional phase_times overrides only the external rotary
    clock for controlled ablations; features, learned DT and native phases
    remain unchanged. Zero phase_times disables the external clock. Native learned angle projections,
    their DT gradients, and exponential-trapezoidal dynamics are retained.

    Training uses upstream Triton (SISO) or TileLang (MIMO) scan/backward
    kernels. MIMO uses split-half rotary pairs across all ranks, and its
    native Triton rotation and CuTe step are used for fixed-memory inference.
    """

    def __init__(
        self, d_model, rotary_emb_base=10000.0,
        rotary_emb_interleaved=None, rotary_emb_scale_base=None, time_update_mode="base", use_event_time_phase=True, **kwargs,
    ):
        if time_update_mode not in TIME_UPDATE_MODES:
            raise ValueError(f"Unknown time_update_mode: {time_update_mode}")
        self.time_update_mode = time_update_mode
        self.use_event_time_phase = bool(use_event_time_phase)
        is_mimo = bool(kwargs.get("is_mimo", False))
        expected_interleaving = not is_mimo
        if rotary_emb_interleaved is not None and bool(rotary_emb_interleaved) != expected_interleaving:
            raise ValueError("SISO requires interleaved rotary pairs; MIMO requires split-half pairs")
        if is_mimo:
            rank = kwargs.get("mimo_rank", 4)
            if rank not in (1, 2, 4, 8):
                raise ValueError("MIMO rank must be one of 1, 2, 4, 8")
            kwargs.setdefault("chunk_size", max(8, 64 // rank))
            if kwargs["chunk_size"] < 8 or kwargs["chunk_size"] & (kwargs["chunk_size"] - 1):
                raise ValueError("MIMO chunk_size must be a power of two >= 8")
        if rotary_emb_scale_base is not None:
            raise NotImplementedError("Mamba3Rotary implements unscaled time RoPE, without xPos")
        if not math.isfinite(rotary_emb_base) or rotary_emb_base <= 0:
            raise ValueError("rotary_emb_base must be finite and positive")
        super().__init__(d_model, **kwargs)
        if self.nheads % self.num_bc_heads:
            raise ValueError("ngroups must divide the number of SSM heads")
        if any(value < 4 or value & (value - 1) for value in (self.d_state, self.headdim)):
            raise ValueError("Triton kernels require power-of-two d_state and headdim >= 4")
        if self.num_rope_angles % 2:
            raise ValueError("The number of rotary angles must be even")
        self.rotary_emb_base = float(rotary_emb_base)

    def _time_angles(self, times, batch_size, seqlen, device):
        if times is None:
            raise ValueError("Mamba3Rotary requires actual event times")
        if times.ndim == 1 and times.shape[0] == seqlen:
            times = times.unsqueeze(0).expand(batch_size, -1)
        if times.shape != (batch_size, seqlen):
            raise ValueError(
                f"times must have shape ({batch_size}, {seqlen}) or ({seqlen},), "
                f"got {tuple(times.shape)}"
            )
        # Preserve timestamp precision even when the model weights use BF16.
        # Reduce modulo 2pi before the FP32 cast for large FP64 event times.
        time_dtype = torch.float64 if times.dtype == torch.float64 else torch.float32
        times = times.to(device=device, dtype=time_dtype)
        frequencies = self.rotary_emb_base ** (
            -torch.arange(self.num_rope_angles, device=device, dtype=time_dtype)
            / self.num_rope_angles
        )
        phase = torch.remainder(times[..., None] * frequencies, 2 * math.pi)
        return phase.to(torch.float32).unsqueeze(2).expand(
            -1, -1, self.nheads, -1
        ).contiguous()

    def _project(self, u, physical_dt=None):
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            self.in_proj(u),
            [self.d_inner, self.d_inner,
             self.d_state * self.num_bc_heads * self.mimo_rank,
             self.d_state * self.num_bc_heads * self.mimo_rank,
             self.nheads, self.nheads, self.nheads, self.num_rope_angles],
            dim=-1,
        )
        if self.is_mimo:
            B = rearrange(B, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
            C = rearrange(C, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
        else:
            B = rearrange(B, "b l (g n) -> b l g n", g=self.num_bc_heads)
            C = rearrange(C, "b l (g n) -> b l g n", g=self.num_bc_heads)
        B, C = self.B_norm(B), self.C_norm(C)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        dt = F.softplus(dd_dt.float() + self.dt_bias.float())
        A = -heavy_tail_activation(dd_A.float()).clamp_min(self.A_floor)
        adt, dt = apply_time_update(A, dt, physical_dt, self.time_update_mode)
        return B, C, x, z, adt, dt, trap, angles

    def _native_angles(self, angles, dt, initial=None, cu_seqlens=None):
        # FP32 accumulation keeps prefill/chunk/step state continuous. The scan
        # still uses native BF16 Q/K/V, independently of phase precision.
        rates = angles.float().unsqueeze(2).expand(-1, -1, self.nheads, -1).contiguous()
        return _NativeAngleFunction.apply(
            rates, dt.transpose(1, 2).contiguous(), initial,
            cu_seqlens, self.chunk_size,
        )

    def _finish(self, y, z, dtype):
        if self.is_mimo:
            if self.is_outproj_norm and not self.fuse_pregate_headwise_norm:
                projected_z = torch.einsum("blhp,hrp->blrhp", z.float(), self.mimo_z.float())
                y = self.norm(
                    rearrange(y, "b l r h p -> b l r (h p)").float(),
                    rearrange(projected_z, "b l r h p -> b l r (h p)"),
                )
                y = rearrange(y, "b l r (h p) -> b l r h p", p=self.headdim)
                y = torch.einsum("blrhp,hrp->blhp", y, self.mimo_o.float())
            return self.out_proj(rearrange(y, "b l h p -> b l (h p)").to(dtype))
        y = rearrange(y, "b l h p -> b l (h p)")
        if self.is_outproj_norm:
            y = self.norm(y, rearrange(z, "b l h p -> b l (h p)"))
        return self.out_proj(y.to(dtype))

    def forward(self, u, times, seq_idx=None, cu_seqlens=None, inference_params=None, phase_times=None, physical_dt=None):
        if u.ndim != 3 or u.shape[1] == 0:
            raise ValueError("u must have shape (batch, nonempty seqlen, d_model)")
        if seq_idx is not None:
            raise NotImplementedError("Use cu_seqlens for packed Mamba3Rotary sequences")
        batch, length, _ = u.shape
        phase_clock = times if phase_times is None else phase_times
        if phase_times is None and not self.use_event_time_phase:
            phase_clock = torch.zeros(u.shape[:2], device=u.device, dtype=torch.float64)
        phase = self._time_angles(phase_clock, batch, length, u.device)
        cache = None
        if inference_params is not None:
            cache_batch = batch if cu_seqlens is None else cu_seqlens.numel() - 1
            cache = self._get_states_from_cache(inference_params, cache_batch)
            continuing = inference_params.seqlen_offset > 0
            if continuing and cache[1].shape[0] != cache_batch:
                raise ValueError("Cached batch size does not match the continuation batch")
            if continuing and self.is_mimo:
                if cu_seqlens is not None:
                    raise NotImplementedError("MIMO cache continuation uses one row per sequence, without cu_seqlens")
                # TileLang's scan has no initial-state input. Consume only the
                # new events, retaining fixed-size state instead of replaying history.
                return torch.cat([
                    self.step(u[:, i:i+1], *cache, times=times[..., i:i+1],
                              phase_times=None if phase_times is None else phase_times[..., i:i+1],
                              physical_dt=None if physical_dt is None else physical_dt[..., i:i+1])[0]
                    for i in range(length)
                ], dim=1)
            if continuing and length == 1 and cu_seqlens is None:
                return self.step(u, *cache, times=times, phase_times=phase_times, physical_dt=physical_dt)[0]
        else:
            continuing = False

        B, C, x, z, adt, dt, trap, angles = self._project(u, physical_dt)
        native_phase, final_native = self._native_angles(
            angles, dt, cache[0] if continuing else None, cu_seqlens,
        )
        phase = torch.remainder(native_phase + phase, 2 * math.pi).contiguous()
        if self.is_mimo:
            from src.models.mamba.mamba3_mimo_kernels import mimo_event_time_scan
            result = mimo_event_time_scan(
                self, C, B, x, adt, dt, trap, phase, z, cu_seqlens, cache is not None,
            )
        else:
            states = cache[1:] if continuing else (None, None, None)
            result = _event_time_scan(
                C, B, x, adt.transpose(1, 2), dt.transpose(1, 2),
                trap.transpose(1, 2), self.C_bias.squeeze(1), self.B_bias.squeeze(1),
                phase, self.D, z if not self.is_outproj_norm else None,
                states, cu_seqlens, self.chunk_size, cache is not None,
            )
        if cache is not None:
            y, final_ssm, final_k, final_v = result
            with torch.no_grad():
                # One observed prefix may warm up a larger sampling batch.
                for destination, source in zip(
                    cache, (final_native, final_ssm, final_k, final_v)
                ):
                    destination.copy_(source)
        else:
            y = result
        return self._finish(y, z, u.dtype)

    @torch.no_grad()
    def step(self, u, angle_state, ssm_state, k_state, v_state, times, phase_times=None, physical_dt=None):
        """Consume one event and return output plus four updated cache tensors."""
        single_axis = u.ndim == 2
        if single_axis:
            u = u.unsqueeze(1)
        if u.ndim != 3 or u.shape[1] != 1:
            raise ValueError("step expects (batch, d_model) or (batch, 1, d_model)")
        phase_clock = times if phase_times is None else phase_times
        if phase_times is None and not self.use_event_time_phase:
            phase_clock = torch.zeros(u.shape[:2], device=u.device, dtype=torch.float64)
        phase = self._time_angles(phase_clock, u.shape[0], 1, u.device)[:, 0].contiguous()
        B, C, x, z, adt, dt, trap, angles = self._project(u, physical_dt)
        native_phase, final_native = self._native_angles(angles, dt, angle_state)
        phase = torch.remainder(native_phase[:, 0] + phase, 2 * math.pi).contiguous()
        if self.is_mimo:
            from src.models.mamba.mamba3_mimo_kernels import mimo_step_values
            y = mimo_step_values(
                self, B, C, x, z, adt, dt, trap, phase, ssm_state, k_state, v_state,
            )
            angle_state.copy_(final_native)
            out = self._finish(y, z, u.dtype)
            return (out.squeeze(1) if single_axis else out), angle_state, ssm_state, k_state, v_state
        # Native phase was already advanced once above. Give the step kernel
        # this combined absolute phase and zero rates to avoid a second advance.
        y, states = mamba3_siso_step(
            Q=C[:, 0].to(torch.bfloat16), K=B[:, 0].to(torch.bfloat16),
            V=x[:, 0].to(torch.bfloat16), ADT=adt[:, 0], DT=dt[:, 0],
            Trap=trap[:, 0].to(torch.bfloat16),
            Q_bias=self.C_bias.squeeze(1).float(), K_bias=self.B_bias.squeeze(1).float(),
            Angles=torch.zeros_like(phase), D=self.D.float(),
            Z=z[:, 0].to(torch.bfloat16) if not self.is_outproj_norm else None,
            Input_States=(phase, ssm_state, k_state, v_state),
        )
        for destination, source in zip(
            (angle_state, ssm_state, k_state, v_state),
            (final_native, states[1], states[2], states[3]),
        ):
            destination.copy_(source)
        out = self._finish(y.unsqueeze(1), z, u.dtype)
        if single_axis:
            out = out.squeeze(1)
        return out, angle_state, ssm_state, k_state, v_state

    def allocate_inference_cache(self, batch_size, max_seqlen, device=None, dtype=None, **kwargs):
        device = self.in_proj.weight.device if device is None else device
        dtype = self.in_proj.weight.dtype if dtype is None else dtype
        return (
            torch.zeros(batch_size, self.nheads, self.num_rope_angles, device=device, dtype=torch.float32),
            torch.zeros(batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=torch.float32),
            torch.zeros(
                (batch_size, self.mimo_rank, self.nheads, self.d_state)
                if self.is_mimo else (batch_size, self.nheads, self.d_state),
                device=device, dtype=torch.bfloat16 if self.is_mimo else torch.float32,
            ),
            torch.zeros(batch_size, self.nheads, self.headdim, device=device, dtype=dtype),
        )

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        if self.layer_idx is None:
            raise ValueError("Set layer_idx when using Mamba3Rotary inference caching")
        memory = inference_params.key_value_memory_dict
        if self.layer_idx not in memory:
            memory[self.layer_idx] = self.allocate_inference_cache(
                batch_size, inference_params.max_seqlen
            )
        states = memory[self.layer_idx]
        if len(states) != 4 or states[1].shape[0] < batch_size:
            raise ValueError("Mamba3Rotary inference cache has an incompatible shape")
        if states[1].shape[0] != batch_size and batch_size != 1:
            raise ValueError("Only a single prefix can broadcast into a larger inference cache")
        if initialize_states:
            for state in states:
                state.zero_()
        return states


class Mamba3EventTime(_Mamba3TimeBase):
    """Event-time Mamba3; defaults to the SCEDC best time-NLL mechanism.

    Physical gap is observed delta_t / fixed training tau, NOT cumulative time.
    Default MIMO rank 2; decay_only preserves learned DT and native phase.
    Extra event-time phase is opt-in. Feature masking is configured independently in the TPP adapter.
    """
    def __init__(self, d_model, time_update_mode="decay_only",
                 use_event_time_phase=False, **kwargs):
        defaults = dict(is_mimo=True, mimo_rank=2, d_state=64, headdim=32,
                        expand=2, rope_fraction=.5)
        for key, value in defaults.items():
            kwargs.setdefault(key, value)
        if kwargs["is_mimo"] and kwargs["mimo_rank"] == 2:
            kwargs.setdefault("chunk_size", 16)
        super().__init__(d_model, time_update_mode=time_update_mode,
                         use_event_time_phase=use_event_time_phase, **kwargs)
