"""SCEDC full-prefix masks and the single-catalog physical-clock adapter."""
from pathlib import Path
import sys
import pytest
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/run'))
from run_scedc_full_time_dynamics import CatalogTimeAdapter
from src.data.dot_dict import DotDict
from src.data import Batch,Sequence


@pytest.mark.parametrize('phase',['native','real'])
@pytest.mark.parametrize('direct',[False,True])
def test_single_catalog_adapter_preserves_clock_marks_and_targets(phase,direct):
    gaps=torch.tensor([[.5,.25,0.]],dtype=torch.float64)
    clock=gaps.cumsum(1)
    mask=torch.tensor([[1.,1.,0.]])
    features=torch.tensor([[[.3,2.1],[-.7,3.2],[0.,0.]]])
    class Delegate:
        features_input_keys=['log_inter_times','mag']
        time_scale_base=torch.tensor(.25)
        def __call__(self,_):return dict(features=features,times=clock,input_mask=mask)
    batch=DotDict(inter_times=gaps)
    result=CatalogTimeAdapter(Delegate(),phase,direct)(batch)
    assert torch.equal(result['features'][...,1],features[...,1])
    assert result['times'] is clock and result['times'].dtype==torch.float64
    torch.testing.assert_close(result['physical_dt'],torch.tensor([[2.,1.,1.]],dtype=torch.float64))
    if direct:assert result['features'] is features
    else:assert torch.count_nonzero(result['features'][...,0])==0
    if phase=='native':assert torch.count_nonzero(result['phase_times'])==0
    else:assert 'phase_times' not in result
    assert batch.inter_times is gaps


def test_actual_scedc_split_is_full_prefix_with_disjoint_scoring_periods():
    path=Path('/root/autodl-fs/em_eqf_training/scedc_full_time_dynamics_20261105/dataset_snapshot.pt')
    data=torch.load(path,map_location='cpu',weights_only=False)
    seq={k:Sequence(**v) for k,v in data['sequences'].items()}
    assert data['threshold_operator']=='>' and data['metadata']['mag_completeness']=='2.0'
    assert seq['train'].t_start==seq['val'].t_start==seq['test'].t_start==0.
    assert seq['train'].t_nll_start==1461.
    assert seq['val'].t_nll_start==8766.
    assert seq['test'].t_nll_start==12053.
    for earlier,later in [('train','val'),('val','test')]:
        a,b=seq[earlier],seq[later]
        torch.testing.assert_close(a.arrival_times,b.arrival_times[:len(a)],rtol=0,atol=1e-10)
        assert a.t_end<b.t_nll_start
    for name,s in seq.items():
        batch=Batch.from_list([s])
        assert batch.seq_len==len(s)+1==data['counts'][name]['input_tokens']
        assert int(batch.nll_event_mask.sum())==data['counts'][name]['scored_events']
        assert batch.inter_times.dtype==torch.float64
    assert sum(data['counts'][k]['scored_events'] for k in seq)<len(seq['test'])
