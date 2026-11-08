"""Validate native fixed code against legacy evidence and saved checkpoints."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path('/root/autodl-tmp/em_eqf')
sys.path.insert(0,str(ROOT))
from src.train.config_setup import load_and_prepare_model
from src.data import Catalog,Batch
from src.catalogs.pathing import build_tpp_catalog_init_kwargs

OUT=ROOT/'experiments/revision/archive/development/attention_cache_fix_20261102'


class ProbeComplete(Exception):
    pass


def probe(model,past,fill):
    encoder=model.base_model.encoder
    allocate=encoder.allocate_inference_cache
    get_state=model.get_current_state
    row=dict(attention_prefix_finite=[],attention_prefix_identical=[],decoded_cache_finite=[])
    decode_count=0
    def initialized_storage(*args,**kwargs):
        cache=allocate(*args,**kwargs)
        for entry in cache.values():
            if entry[0] is not None and entry[0].ndim==5:
                entry[0].fill_(fill)
        return cache
    def checked_state(*args,**kwargs):
        nonlocal decode_count
        params=kwargs['inference_params']
        prefill=kwargs.get('return_all',False)
        # Inspect at the next call: the sampler's shared-prefix copy has now run.
        if not prefill and decode_count==0:
            for entry in params.key_value_memory_dict.values():
                if entry[0] is not None and entry[0].ndim==5:
                    prefix=entry[0][:,:len(past)]
                    row['attention_prefix_finite'].append(torch.isfinite(prefix).flatten(1).all(1).tolist())
                    row['attention_prefix_identical'].append([bool(torch.equal(x,prefix[0])) for x in prefix])
                    assert torch.isfinite(prefix).all()
                    assert all(torch.equal(x,prefix[0]) for x in prefix)
        value=get_state(*args,**kwargs)
        if prefill:
            row['prefill_offset']=params.seqlen_offset
            assert params.seqlen_offset==len(past)
        else:
            decode_count+=1
            for entry in params.key_value_memory_dict.values():
                if entry[0] is not None and entry[0].ndim==5:
                    valid=entry[0][:,:len(past)+decode_count]
                    row['decoded_cache_finite'].append(bool(torch.isfinite(valid).all()))
                    assert torch.isfinite(valid).all()
            assert torch.isfinite(value).all()
            if decode_count==2:
                row['second_decode_state']=value.detach().cpu()
                raise ProbeComplete()
        return value
    encoder.allocate_inference_cache=initialized_storage
    model.get_current_state=checked_state
    torch.manual_seed(20261102);np.random.seed(20261102)
    try:
        model.sample(batch_size=4,duration=30,past_seq=past,max_sample_len=1000,return_sequences=True)
        raise AssertionError('Expected two diagnostic decoding steps')
    except ProbeComplete:
        return row
    finally:
        encoder.allocate_inference_cache=allocate
        model.get_current_state=get_state


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    frozen=json.loads((ROOT/'experiments/revision/main/frozen_code_config_manifest.json').read_text())
    changes={row['path']:row for row in manifest['changes']}
    for row in frozen['source_files']:
        expected=changes[row['path']]['after_sha256'] if row['path'] in changes else row['sha256']
        assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==expected,row['path']
    for row in frozen['configs']:
        path=ROOT/row['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],row['path']
    proof=[]
    for region in ['swc','scsn']:
        for seed in range(5):
            run=ROOT/'experiments/revision/main/runs'/f'generation_{region}_main_seed_{seed}'
            model,args=load_and_prepare_model(run/'best_model_1.pth',torch.device('cuda:0'),compile=False)
            cls=Catalog.by_name(args.dataset+'-Standard')
            cat=cls(**build_tpp_catalog_init_kwargs(cls,ROOT/'data'/args.dataset,args.catalog_cfg))
            seq=cat.test[0].double();batch=Batch.from_list([seq]).to(torch.device('cuda:0'))
            expected=json.loads((run/'metrics_test_best_1.json').read_text())
            with torch.no_grad():
                out=model.nll_loss(batch,reduction='per_time',use_b_updater=False)
            row=dict(region=region,seed=seed)
            for name in ['time','mag']:
                difference=abs(out[name].mean().item()-expected['nll_test_'+name])
                assert difference<1e-5,(region,seed,name,difference)
                row[name+'_nll_abs_difference']=difference
            if seed==0:
                past=seq.get_subsequence(seq.t_start,seq.t_nll_start).to(torch.device('cuda:0'))
                past.t_nll_start=past.t_start
                poison=probe(model,past,float('nan'))
                poison.pop('second_decode_state')
                row.update(poison)
                a=probe(model,past,0.)['second_decode_state']
                b=probe(model,past,42.)['second_decode_state']
                difference=float((a-b).abs().max())
                assert difference==0.,(region,difference)
                row['unused_storage_effect_on_second_decode']=difference
                torch.manual_seed(20261102)
                samples=model.sample(batch_size=4,duration=2,past_seq=past,max_sample_len=1000,return_sequences=True)
                for sample in samples:
                    assert sample.arrival_times.dtype==torch.float64
                    assert torch.isfinite(sample.inter_times).all() and (sample.inter_times>=0).all()
                    assert torch.isfinite(sample.mag).all()
                    assert (torch.diff(sample.arrival_times)>0).all()
                row['two_day_forecast_counts']=[len(sample) for sample in samples]
                del past,samples
            proof.append(row)
            (OUT/'gpu_and_nll_checks.json').write_text(json.dumps(proof,indent=2))
            print(row,flush=True)
            del model,cat,seq,batch,out
            torch.cuda.empty_cache()
    manifest['validation_status']='passed'
    manifest['validation']='18 targeted tests; actual four-trajectory prefix, correct token-count offset, persistent two-step caches, unused-memory invariance; all ten saved time/magnitude NLLs reproduced within 1e-5'
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
