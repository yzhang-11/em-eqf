"""Cache offsets must count consumed history tokens, not forward calls."""
from types import SimpleNamespace

import torch

from src.models.tpp.mixer_tpp import MixerTPP


def test_prefill_then_decode_appends_after_the_whole_history():
    model=SimpleNamespace(base_model=lambda batch,inference_params=None:batch)
    params=SimpleNamespace(seqlen_offset=0)
    history=torch.zeros(1,7,4)
    result=MixerTPP.get_current_state(model,history,inference_params=params,return_all=True)
    assert result.shape==(1,7,4)
    assert params.seqlen_offset==7
    result=MixerTPP.get_current_state(model,torch.ones(1,2,4),inference_params=params)
    assert result.shape==(1,1,4)
    assert params.seqlen_offset==9
