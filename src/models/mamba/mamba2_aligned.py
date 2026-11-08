"""Keep Mamba2 projections compatible with cached causal-convolution kernels."""
import torch.nn.functional as F
from mamba_ssm.modules.mamba2 import Mamba2


def _align_projection_rows(module, inputs, output):
    # Mamba2 splits this projection into views. Its cached-prefill convolution
    # uses a channel-last view whose row stride must be divisible by eight.
    # Padding the backing storage, then slicing, preserves every value and
    # parameter shape, including when the number of SSM heads is not a multiple
    # of eight. The view remains differentiable and works for both full and step.
    if output.ndim >= 2 and output.stride(-2) % 8:
        padding = (-output.shape[-1]) % 8
        return F.pad(output, (0, padding))[..., :output.shape[-1]]
    return output


class AlignedMamba2(Mamba2):
    """Upstream Mamba2 with aligned projection storage and unchanged weights."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_proj.register_forward_hook(_align_projection_rows)
