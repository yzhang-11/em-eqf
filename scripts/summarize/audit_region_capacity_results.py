"""Check immutable inputs, paired budgets, target counts and saved forecasts."""
from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np
import torch

FROZEN=Path('/root/autodl-fs/em_eqf_training/region_capacity_20261105/frozen_project')
sys.path.insert(0,str(FROZEN/'scripts/tuning'))
import region_capacity_common as c


def read(path):return json.loads(Path(path).read_text())


def main():
    torch.set_num_threads(1);c.bootstrap()
    from run_multidomain_pretraining import build_model
    plan=read(c.CONFIGS/'study_plan.json');neural=read(c.REPORTS/'neural_results.json')
    chosen=neural['selected_capacity'];data=torch.load(c.DATA,map_location='cpu',weights_only=False)
    assert c.sha(c.DATA)==plan['data_sha256']
    checks={};runs={}
    for complete in c.RUNS.glob('*/training_complete.json'):
        record=read(complete);name=record['config']['name'];runs[name]=record
        assert c.sha(complete.parent/'best_model.pth')==record['checkpoint_sha256'],name
        assert record['config']['dataset_sha256']==plan['data_sha256'],name
        for metric in complete.parent.glob('metrics_best_model_context_*.json'):
            result=read(metric)
            for region,windows in result['test_windows'].items():
                spec=data['regions'][region]
                assert sum(v['events'] for v in windows)==spec['test_end']-spec['val_end']
                assert all(math.isfinite(v[k]) for v in windows for k in ['time','mag','total'])
    assert len(runs)==70
    paired=0
    for seed in [0,1,2]:
        global_fit=runs[f'global_all_regions_{chosen}_long_seed_{seed}']
        for region in plan['regions']:
            local=runs[f'local_{region}_{chosen}_long_seed_{seed}']
            assert global_fit['initial_state_sha256']==local['initial_state_sha256']
            assert global_fit['training_draw_hashes'][region]==local['training_draw_hashes'][region]
            assert global_fit['scored_event_exposures'][region]==local['scored_event_exposures'][region]
            assert local['completed_region_updates'][region]==800
            assert global_fit['completed_region_updates'][region]==800
            paired+=1
    checks['paired_region_seed_budgets']=paired
    floor_counts={}
    for region,spec in data['regions'].items():
        ids=np.flatnonzero(np.diff(spec['ns'])/86400e9<1e-10)+1
        floor_counts[region]=dict(train=int((ids<spec['train_end']).sum()),
            validation=int(((ids>=spec['train_end'])&(ids<spec['val_end'])).sum()),
            test=int((ids>=spec['val_end']).sum()))
        assert floor_counts[region]['validation']==floor_counts[region]['test']==0
    checks['intervals_below_neural_likelihood_floor_days_1e_10']=floor_counts
    replays=[]
    center_sensitivity=[]
    for name in [f'global_all_regions_{chosen}_long_seed_2',
                 f'global_all_regions_large_long_seed_2',
                 f'local_italy_{chosen}_long_seed_2']:
        complete=runs[name];cfg=complete['config']
        saved=read(c.RUNS/name/'metrics_best_model_context_1024.json')
        model=build_model('stats_film',cfg['seed'],data['vocabulary'],cfg['backbone'])
        state=torch.load(c.RUNS/name/'best_model.pth',map_location='cuda:0',weights_only=False)
        model.load_state_dict(state['model_state_dict'],strict=True)
        pools=c.evaluation_pools(data,'test',1024)
        pools={r:pools[r] for r in cfg['sources']}
        actual,_=c.evaluate(model,pools)
        differences=[abs(actual['regions'][r][k]-saved['test']['regions'][r][k])
                     for r in cfg['sources'] for k in ['time','mag','total']]
        assert max(differences)<2e-5,(name,max(differences))
        replays.append(dict(name=name,max_absolute_nll_difference=max(differences)))
        if cfg['scope']=='global' and cfg['size']==chosen:
            from run_multidomain_pretraining import make_batch
            sample=make_batch([pools['japan_east_honshu'][0]])
            target_start=int(sample.nll_event_mask[0].nonzero()[0])
            change_at=target_start+64
            with torch.no_grad():
                reference=model.get_context(sample).clone()
                sample.mag[:,change_at:]+=.3
                modified=model.get_context(sample)
            error=float((reference[:,:change_at+1]-modified[:,:change_at+1]).abs().max())
            assert error<1e-5,error
            checks['future_magnitude_perturbation_prefix_max_error']=error
            sample=make_batch([pools['japan_east_honshu'][0]])
            with torch.no_grad():
                time_query=sample.inter_times.clone()
                reference=model.get_context(sample)
                time_before=model.get_inter_time_dist(reference).log_prob(time_query)
                mag_before=model.get_magnitude_dist(reference).log_prob(sample.mag)
                extra=.5*sample.inter_times[:,change_at:].clone()
                sample.inter_times[:,change_at:]+=extra
                sample.arrival_times[:,change_at:]+=extra.cumsum(-1)
                changed=model.get_context(sample)
                time_after=model.get_inter_time_dist(changed).log_prob(time_query)
                mag_after=model.get_magnitude_dist(changed).log_prob(sample.mag)
            prefix=slice(target_start,change_at+1)
            time_error=float((time_before[:,prefix]-time_after[:,prefix]).abs().max())
            mag_error=float((mag_before[:,prefix]-mag_after[:,prefix]).abs().max())
            assert max(time_error,mag_error)<2e-4,(time_error,mag_error)
            checks['future_time_perturbation_prefix_log_density_max_error']=dict(time=time_error,mag=mag_error)
            import src.models.mha.rotary_embedding_time as rotary
            original=rotary.compute_nonzero_center_per_sample
            try:
                rotary.compute_nonzero_center_per_sample=lambda times,*args,**kwargs:torch.zeros(
                    times.shape[0],device=times.device,dtype=times.dtype)
                sample=make_batch([pools['japan_east_honshu'][0]])
                with torch.no_grad():
                    query=sample.inter_times.clone()
                    before=model.get_context(sample)
                    t0=model.get_inter_time_dist(before).log_prob(query)
                    m0=model.get_magnitude_dist(before).log_prob(sample.mag)
                    extra=.5*sample.inter_times[:,change_at:].clone()
                    sample.inter_times[:,change_at:]+=extra
                    sample.arrival_times[:,change_at:]+=extra.cumsum(-1)
                    after=model.get_context(sample)
                    t1=model.get_inter_time_dist(after).log_prob(query)
                    m1=model.get_magnitude_dist(after).log_prob(sample.mag)
                fixed=dict(time=float((t0[:,prefix]-t1[:,prefix]).abs().max()),
                           mag=float((m0[:,prefix]-m1[:,prefix]).abs().max()))
                assert max(fixed.values())<1e-6,fixed
                checks['future_time_perturbation_with_fixed_xpos_center_log_density_error']=fixed
            finally:rotary.compute_nonzero_center_per_sample=original
        import src.models.mha.rotary_embedding_time as rotary
        original=rotary.compute_nonzero_center_per_sample
        try:
            rotary.compute_nonzero_center_per_sample=lambda times,*args,**kwargs:torch.zeros(
                times.shape[0],device=times.device,dtype=times.dtype)
            fixed,_=c.evaluate(model,pools)
            change=max(abs(fixed['regions'][r][k]-actual['regions'][r][k])
                       for r in cfg['sources'] for k in ['time','mag','total'])
            assert change<1e-3,(name,change)
            center_sensitivity.append(dict(name=name,max_region_nll_difference=change,
                macro_total_difference=fixed['macro']['total']-actual['macro']['total']))
        finally:rotary.compute_nonzero_center_per_sample=original
        del model;torch.cuda.empty_cache()
    checks['checkpoint_replays']=replays
    checks['xpos_fixed_center_numerical_sensitivity']=center_sensitivity
    etas=read(c.REPORTS/'etas/test_results.json')['rows']
    for row in etas:
        spec=data['regions'][row['region']]
        assert row['events']==spec['test_end']-spec['val_end']
        assert all(math.isfinite(row[k]) for k in ['time','mag','total'])
    checks['etas_rows_same_test_counts']=len(etas)
    protected=read(c.REPORTS/'data.json')['protected_before']
    protected.update(read(c.STUDY/'reports/compatibility_20261105/source_snapshot.json')['protected'])
    assert all(c.sha(path)==expected for path,expected in protected.items())
    checks['protected_inputs']=protected
    checks['frozen_source_files_verified']=len(read(c.REPORTS/'frozen_source.json'))
    checks['neural_rows']=len(neural['rows'])
    checks['passed']=True
    c.save(c.REPORTS/'final_audit.json',checks)
    print(json.dumps(checks),flush=True)


if __name__=='__main__':main()
