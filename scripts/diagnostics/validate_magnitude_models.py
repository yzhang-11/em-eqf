"""Check integrated likelihoods and genuine autoregressive sampling of selected heads."""
from pathlib import Path
import json
import sys
import numpy as np
import torch

ROOT=Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0,str(ROOT))
from src.data import Batch,TppDataset
from src.models.tpp.magnitude_model import load_magnitude_model


def main():
    base=ROOT/'experiments/revision/magnitude_models'
    selection=json.loads((base/'selection.json').read_text());proof=[]
    for region,cache in [('swc','ChuanDian/catalogs/c6afef21'),('scsn','SCEDC/catalogs/25a12469')]:
        recipe=selection['recipes'][region]['selected'];family,k,mode=recipe['family'],recipe['components'],recipe['features']
        for seed in range(5):
            run=base/'runs'/f'confirm_{region}_main_{family}{k}_{mode}_seed_{seed}'
            model,args=load_magnitude_model(run/'best_head.pth',torch.device('cuda:0'))
            sequence=TppDataset.load_from_disk(ROOT/'data'/cache/'full_sequence.pt')[0].double()
            periods=json.loads((base/'features'/f'{region}_main_events.json').read_text())['periods']
            sequence.t_nll_start=periods['test']['start']
            batch=Batch.from_list([sequence]).to('cuda:0')
            with torch.inference_mode():values=model.nll_loss(batch,reduction='per_time')
            metrics=json.loads((ROOT/'experiments/revision/main/runs'/f'generation_{region}_main_seed_{seed}'/'metrics_test_best_1.json').read_text())
            with np.load(base/'reports/comparison'/f'{region}_main_seed_{seed}.npz') as d:target=-d['selected_logp'].sum()/periods['test']['days']
            row=dict(region=region,seed=seed,time_nll_error=abs(float(values['time'])-metrics['nll_test_time']),
                     mark_nll_error=abs(float(values['mag'])-target))
            assert row['time_nll_error']<1e-5,row
            assert row['mark_nll_error']<1e-5,row
            if seed==0:
                origin=periods['validation']['start']
                # A history has no scored target interval; set its metadata before slicing.
                sequence.t_nll_start=sequence.t_start
                past=sequence.get_subsequence(sequence.t_start,origin).to('cuda:0')
                past.t_nll_start=past.t_start
                torch.manual_seed(20261104)
                with torch.inference_mode():
                    samples=model.sample(batch_size=4,duration=30.,past_seq=past,max_sample_len=16384,return_sequences=True)
                row['sample_counts']=[len(x) for x in samples]
                step=float(model.density_head.step);mc=float(model.mag_completeness)
                for trajectory in samples:
                    assert torch.isfinite(trajectory.mag).all()
                    assert not len(trajectory) or float(trajectory.mag.min())>=mc-1e-6
                    if len(trajectory):
                        index=(trajectory.mag.double()-mc)/step
                        assert float((index-index.round()).abs().max())<1e-3
            print(json.dumps(row),flush=True);proof.append(row)
            del model,batch,sequence,values;torch.cuda.empty_cache()
    (base/'reports/integration_validation.json').write_text(json.dumps(dict(state='passed',proof=proof,
        loss='time+mark only; both b penalties disabled'),indent=2)+'\n')


if __name__=='__main__':main()
