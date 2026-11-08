"""Check fitted GAS likelihood, b access, timing and integration with real time models."""
from pathlib import Path
import json
import sys
import time

import numpy as np
import torch

ROOT=Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts/run'))
from run_magnitude_dynamics import OUT,REPORT,data_for
from src.models.tpp.magnitude_dynamics import filter_gas
from src.models.tpp.magnitude_dynamics_model import load_gas_magnitude_model
from src.data import Batch,TppDataset


def main():
    torch.set_num_threads(1)
    selection=json.loads((REPORT/'selection.json').read_text());proof=[]
    for row in selection['best']:
        if row['family']!=selection['choices'][row['region']]:continue
        data=data_for(row['region'],row['variant'])
        meta=json.loads((OUT/'features'/f"{row['region']}_{row['variant']}_events.json").read_text())
        sequence=TppDataset.load_from_disk(meta['sequence_path'])[0].double()
        sequence.t_nll_start=meta['periods']['test']['start']
        batch=Batch.from_list([sequence]).to('cuda:0')
        model,_=load_gas_magnitude_model(row['checkpoint'],torch.device('cuda:0'))
        with torch.inference_mode():values=model.nll_loss(batch,reduction='per_time')
        metrics=json.loads((ROOT/'experiments/revision/main/runs'/f"generation_{row['region']}_{row['variant']}_seed_0"/'metrics_test_best_1.json').read_text())
        mask=data['split']==3
        target=filter_gas(row['theta'],row['family'],data['excess'],data['step'],data['width'],return_trace=True)
        expected=-target['logp'][mask].sum()/meta['periods']['test']['days']
        with torch.inference_mode():predicted_b=model.compute_b_value(sequence.to('cuda:0'),predict_b=True)
        b_error=float(np.max(abs(predicted_b[0,:len(data['excess'])].cpu().numpy()-target['value'])))
        entry=dict(region=row['region'],variant=row['variant'],
            time_nll_error=abs(float(values['time'])-metrics['nll_test_time']),
            mark_nll_error=abs(float(values['mag'])-expected),b_prediction_error=b_error)
        assert entry['time_nll_error']<1e-5 and entry['mark_nll_error']<1e-5 and b_error<2e-6,entry
        timings=[]
        for _ in range(7):
            begin=time.perf_counter()
            filter_gas(row['theta'],row['family'],data['excess'],data['step'],data['width'],return_trace=True)
            timings.append(time.perf_counter()-begin)
        entry.update(events=len(data['excess']),cpu_filter_median_seconds=float(np.median(timings)),state_scalars=1)
        proof.append(entry);print(json.dumps(entry),flush=True)
        del model,batch;torch.cuda.empty_cache()
    (REPORT/'integration_validation.json').write_text(json.dumps(dict(state='passed',proof=proof,
        scope='real frozen original time heads; GAS scoring and compute_b_value match CPU reference'),indent=2)+'\n')


if __name__=='__main__':main()
