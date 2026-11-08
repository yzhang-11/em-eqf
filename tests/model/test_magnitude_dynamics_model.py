from types import SimpleNamespace
import numpy as np
import torch

from src.models.tpp.magnitude_dynamics import filter_gas
from src.models.tpp.magnitude_dynamics_model import GASMagnitudeTPP


def test_streaming_history_matches_causal_filter_and_copies_parallel_states():
    state=SimpleNamespace(gas_theta=np.array([.1,np.log(32.),np.log(.1)]),gas_family='gr_free',
        density_head=SimpleNamespace(mag_min=3.,step=.1),_gas_sample_state=None)
    marks=np.array([[3.,3.2,4.]])
    GASMagnitudeTPP._update_gas_history(state,SimpleNamespace(mag=torch.tensor(marks)))
    expected=filter_gas(state.gas_theta,state.gas_family,np.array([0.,.2,1.]),.1,7.,return_trace=True)
    assert abs(state._gas_sample_state[0]-expected['final_state'])<1e-14
    GASMagnitudeTPP._update_gas_history(state,SimpleNamespace(mag=torch.tensor([[3.],[3.],[5.],[3.]])))
    assert state._gas_sample_state[0]==state._gas_sample_state[1]==state._gas_sample_state[3]
    assert state._gas_sample_state[0]!=state._gas_sample_state[2]
    expected=filter_gas(state.gas_theta,state.gas_family,np.array([0.,.2,1.,2.]),.1,7.,return_trace=True)
    assert abs(state._gas_sample_state[2]-expected['final_state'])<1e-14
