"""External physical-gap modulation without changing native kernel contracts."""
import torch

TIME_UPDATE_MODES = ('base', 'full_dt', 'decay_only')


def apply_time_update(A, dt, physical_dt=None, mode='base'):
    """Return ADT and DT; physical_dt is a positive observed gap divided by tau.

    Padding gaps should be supplied as one by the adapter. All real gaps must
    be positive. The caller validates values once; this function does not add
    GPU-to-host synchronization inside every recurrent layer.
    """
    if mode not in TIME_UPDATE_MODES:
        raise ValueError(f'Unknown time_update_mode: {mode}')
    if mode == 'base':
        return A * dt, dt
    if physical_dt is None:
        raise ValueError(f'physical_dt is required for {mode}')
    gap = physical_dt.to(device=dt.device, dtype=dt.dtype)
    if gap.ndim == 1 and gap.shape[0] == dt.shape[1]:
        gap = gap[None].expand(dt.shape[0], -1)
    if gap.shape != dt.shape[:2]:
        raise ValueError(f'physical_dt must have shape {tuple(dt.shape[:2])}; got {tuple(gap.shape)}')
    gap = gap.unsqueeze(-1)
    if mode == 'full_dt':
        effective_dt = dt * gap
        return A * effective_dt, effective_dt
    return A * dt * gap, dt
