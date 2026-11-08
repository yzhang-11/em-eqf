"""Compare final predicted densities after full versus cached true-event replay."""
import json
from pathlib import Path
import sys

import torch
from mamba_ssm.utils.generation import InferenceParams

ROOT=Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0,str(ROOT))
from src.data import Batch,Catalog
from src.catalogs.pathing import build_tpp_catalog_init_kwargs
from src.train.config_setup import load_and_prepare_model


def main():
    status=json.loads((ROOT/'experiments/revision/main/reports/calibration/status.json').read_text())
    if status.get('state')!='running' or status.get('region')!='scsn' or status.get('seed',4)>3:
        print('Deferred GPU replay to avoid overlapping the next training stage',flush=True)
        return
    output=[]
    for region in ['swc','scsn']:
        run=ROOT/'experiments/revision/main/runs'/f'generation_{region}_main_seed_0'
        model,args=load_and_prepare_model(run/'best_model_1.pth',torch.device('cuda:0'),compile=False)
        cls=Catalog.by_name(args.dataset+'-Standard')
        cat=cls(**build_tpp_catalog_init_kwargs(cls,ROOT/'data'/args.dataset,args.catalog_cfg))
        sequence=cat.full_sequence.double()
        origin=cat.test[0].t_nll_start
        past=sequence.get_subsequence(sequence.t_start,origin).cuda();past.t_nll_start=past.t_start
        n=len(past);end=float(sequence.arrival_times[n+3])
        complete=sequence.get_subsequence(sequence.t_start,end).cuda()
        with torch.inference_mode():
            full=model.base_model(Batch.from_list([complete])[:,:-1])
            params=InferenceParams(max_seqlen=n+16384,max_batch_size=1,
                key_value_memory_dict=model.base_model.encoder.allocate_inference_cache(1,n+16384))
            prefix=Batch.from_list([past])
            states=model.get_current_state(prefix[:,:-1],inference_params=params,return_all=True)
            history=Batch.init_sample_batch(past_seq=past.init_sample_sequence(),batch_size=1,max_sample_len=8)
            for index in range(4):
                dt=sequence.inter_times[n+index].reshape(1,1).cuda()
                mag=sequence.mag[n+index].reshape(1,1).cuda()
                history.update_sample_batch(next_inter_times=dt,next_mag=mag)
                state=model.get_current_state(history.get_tmp_batch(),inference_params=params)
                expected=full[:,n+index:n+index+1]
                # Use predictive likelihoods, not equality of internal rotated Q/K.
                probe=state.new_tensor([[1.]])
                time_a=model.get_inter_time_dist(state).log_prob(probe)
                time_b=model.get_inter_time_dist(expected).log_prob(probe)
                probe_mag=state.new_tensor([[float(args.mag_completeness)+.5]])
                mag_a=model.get_magnitude_dist(state).log_prob(probe_mag)
                mag_b=model.get_magnitude_dist(expected).log_prob(probe_mag)
                row=dict(region=region,step=index+1,history_events=n,
                    hidden_max_error=float((state-expected).abs().max()),
                    time_log_density_error=float((time_a-time_b).abs().max()),
                    magnitude_log_density_error=float((mag_a-mag_b).abs().max()))
                output.append(row);print(row,flush=True)
        del model,cat,sequence,past,full,params,prefix,states,history,complete
        torch.cuda.empty_cache()
    path=ROOT/'experiments/revision/archive/development/attention_cache_fix_20261102/cached_density_replay.json'
    path.write_text(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
