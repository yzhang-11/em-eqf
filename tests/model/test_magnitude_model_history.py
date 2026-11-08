from types import SimpleNamespace
import numpy as np
import torch

from src.models.tpp.magnitude_model import MagnitudeTPP,causal_mark_features


def stub():
    return SimpleNamespace(feature_mode='history',mag_completeness=torch.tensor(3.),
                           _history_sum=None,_history_weight=None)


def test_history_features_exclude_current_and_future_events():
    marks=np.array([[3.,3.2,4.,3.1,5.]])
    intervals=np.array([[1.,.1,.2,5.,1.]])
    original=causal_mark_features(marks,intervals,3.)
    changed=marks.copy();changed[:,2:]+=2.
    revised=causal_mark_features(changed,intervals,3.)
    np.testing.assert_array_equal(original[:,:3],revised[:,:3])
    assert not np.allclose(original[:,3],revised[:,3])


def test_streaming_state_matches_frozen_feature_definition():
    marks=np.array([[3.,3.2,4.,3.1,5.]])
    intervals=np.array([[1.,.1,.2,5.,1.]])
    state=stub()
    MagnitudeTPP._update_mark_history(state,SimpleNamespace(mag=torch.tensor(marks[:,:3]),inter_times=torch.tensor(intervals[:,:3])))
    for i in range(3,6):
        extended_marks=np.concatenate([marks,np.zeros((1,1))],axis=1)
        extended_intervals=np.concatenate([intervals,np.zeros((1,1))],axis=1)
        expected=causal_mark_features(extended_marks,extended_intervals,3.)[:,i]
        actual=(state._history_sum/state._history_weight.unsqueeze(-1)).flatten(1).numpy()
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)
        if i<5:
            MagnitudeTPP._update_mark_history(state,SimpleNamespace(mag=torch.tensor(marks[:,i:i+1]),inter_times=torch.tensor(intervals[:,i:i+1])))


def test_parallel_trajectories_copy_history_and_update_independently():
    state=stub()
    MagnitudeTPP._update_mark_history(state,SimpleNamespace(mag=torch.tensor([[3.,3.2,4.]]),inter_times=torch.tensor([[1.,.1,.2]])))
    MagnitudeTPP._update_mark_history(state,SimpleNamespace(mag=torch.tensor([[3.],[3.],[5.],[3.]]),inter_times=torch.ones(4,1)))
    torch.testing.assert_close(state._history_sum[0],state._history_sum[1])
    torch.testing.assert_close(state._history_sum[0],state._history_sum[3])
    assert not torch.allclose(state._history_sum[0],state._history_sum[2])
