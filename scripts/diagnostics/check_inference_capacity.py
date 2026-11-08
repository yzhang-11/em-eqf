"""Check saved likelihoods and the actual GPU sampler after the inference fix."""
import json
from pathlib import Path
import sys

import torch

ROOT=Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0,str(ROOT))
from src.train.config_setup import load_and_prepare_model
from src.data import Catalog,Batch
from src.catalogs.pathing import build_tpp_catalog_init_kwargs


def main():
    proof=[]
    for region in ['swc','scsn']:
        for seed in range(5):
            run=ROOT/'experiments/revision/main/runs'/f'generation_{region}_main_seed_{seed}'
            model,args=load_and_prepare_model(run/'best_model_1.pth',torch.device('cuda:0'),compile=False)
            cls=Catalog.by_name(args.dataset+'-Standard')
            cat=cls(**build_tpp_catalog_init_kwargs(cls,ROOT/'data'/args.dataset,args.catalog_cfg))
            seq=cat.test[0].double();batch=Batch.from_list([seq]).to(torch.device('cuda:0'))
            expected=json.loads((run/'metrics_test_best_1.json').read_text())
            # Only the time/magnitude data likelihood is compared, so the
            # separate Gamma-reference penalty needs no a_t/s_t annotation.
            with torch.no_grad():
                out=model.nll_loss(batch,reduction='per_time',use_b_updater=False)
            row=dict(region=region,seed=seed)
            for name in ['time','mag']:
                actual=out[name].mean().item()
                difference=abs(actual-expected['nll_test_'+name])
                assert difference<1e-5,(region,seed,name,difference)
                row[name+'_nll_abs_difference']=difference
            if seed==0:
                origin=seq.t_nll_start
                past=seq.get_subsequence(seq.t_start,origin).to(torch.device('cuda:0'))
                past.t_nll_start=past.t_start
                values=[]
                for capacity in [1000,16384]:
                    torch.manual_seed(20261102)
                    values.append(model.sample(batch_size=4,duration=30,past_seq=past,
                        max_sample_len=capacity,return_sequences=True))
                assert [len(x) for x in values[0]]==[len(x) for x in values[1]]
                row['capacity_counts']=[len(x) for x in values[0]]
                row['capacity_max_time_difference_days']=max(float((a.arrival_times-b.arrival_times).abs().max())
                    if len(a) else 0. for a,b in zip(*values))
                row['capacity_max_magnitude_difference']=max(float((a.mag-b.mag).abs().max())
                    if len(a) else 0. for a,b in zip(*values))
                print('capacity_check',row,flush=True)
                assert row['capacity_max_time_difference_days']<1e-7 and row['capacity_max_magnitude_difference']<1e-6,row
                del past,values
            proof.append(row)
            print(row,flush=True)
            del model,batch,seq,cat,out
            torch.cuda.empty_cache()
    path=ROOT/'experiments/revision/archive/development/sampling_capacity_20261102/gpu_and_nll_checks.json'
    path.write_text(json.dumps(proof,indent=2))


if __name__=='__main__':
    main()
