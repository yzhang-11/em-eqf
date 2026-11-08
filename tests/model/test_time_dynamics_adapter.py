"""Observed gaps must be tau-normalized and padding must remain harmless."""
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/run'))
from run_mamba_time_dynamics import PhysicalGapAdapter
from run_mamba3_mimo_no_deltat import NoDeltaTAdapter
from src.data.dot_dict import DotDict


def test_physical_gap_units_padding_and_direct_feature_isolation():
    intervals=torch.tensor([[.2,1.,0.],[2.,0.,0.]],dtype=torch.float64)
    marks=torch.tensor([[3.,4.,0.],[2.,0.,0.]],dtype=torch.float64)
    mask=torch.tensor([[1.,1.,0.],[1.,0.,0.]],dtype=torch.float64)
    params=torch.zeros(2,3,7,dtype=torch.float64)
    params[0,:,2]=.1
    params[1,:,2]=2.
    batch=DotDict(inter_times=intervals,region_params=params)
    times=intervals.cumsum(1)*mask
    features=torch.stack([intervals,marks],-1)
    class Delegate:
        features_input_keys=['log_inter_times','mag']
        def __call__(self,_):
            return dict(features=features,times=times,input_mask=mask)
    adapter=PhysicalGapAdapter(NoDeltaTAdapter(Delegate()))
    result=adapter(batch)
    expected=torch.tensor([[2.,10.,1.],[1.,1.,1.]],dtype=torch.float64)
    torch.testing.assert_close(result['physical_dt'],expected,rtol=0,atol=1e-14)
    assert torch.equal(result['features'][...,1],marks)
    assert torch.count_nonzero(result['features'][...,0])==0
    assert result['times'] is times
    assert torch.equal(batch.inter_times,intervals)
