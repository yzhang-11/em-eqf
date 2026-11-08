"""Pre-test, validation-triggered checks against an undertrained local baseline."""
from pathlib import Path
import hashlib
import json
import math
import random
import sys
import time
import torch

FROZEN=Path('/root/autodl-fs/em_eqf_training/region_capacity_20261105/frozen_project')
sys.path.insert(0,str(FROZEN/'scripts/tuning'))
import region_capacity_common as c


def read(path):return json.loads(Path(path).read_text())


def main():
    torch.set_num_threads(1);c.bootstrap()
    from run_multidomain_pretraining import build_model
    plan=read(c.CONFIGS/'study_plan.json');selected=read(c.REPORTS/'capacity_selection.json')['size']
    out=c.REPORTS/'convergence_controls';out.mkdir(exist_ok=True)
    regions=[];evidence={}
    for region in plan['regions']:
        result=read(c.RUNS/f'local_{region}_{selected}_long_seed_0/training_complete.json')
        values={v['step']:v['validation']['macro']['total'] for v in result['history']}
        if result['best_step']==800 and values[600]-values[800]>.002:
            regions.append(region);evidence[region]=dict(best_step=800,improvement_600_to_800=values[600]-values[800])
    control_plan=dict(regions=regions,size=selected,seeds=[0,1,2],max_updates=1600,
        criterion='Seed-0 local long best at final 800 and validation improvement from 600 to 800 >0.002.',
        evidence=evidence,test_used_for_selection=False,dataset_sha256=c.sha(c.DATA),
        purpose='Supplemental higher-budget local control; does not replace the primary matched-budget comparison.')
    c.save(out/'plan.json',control_plan)
    data=torch.load(c.DATA,map_location='cpu',weights_only=False)
    target=c.STORE/'convergence_controls';target.mkdir(exist_ok=True)
    completed={}
    for region in regions:
        for seed in [0,1,2]:
            name=f'local_{region}_{selected}_long_seed_{seed}_steps_1600';folder=target/name;folder.mkdir(exist_ok=True)
            path=folder/'training_complete.json'
            if path.exists():completed[name]=read(path);continue
            cfg=dict(protocol='regional_local_convergence_1600_v1',name=name,method='stats_film',
                size=selected,seed=seed,region=region,scope='local_extended',length='long',
                region_id_map=data['vocabulary'],backbone=plan['backbones'][selected],
                optimizer_steps=1600,learning_rate=3e-4,weight_decay=.01,dataset_sha256=control_plan['dataset_sha256'])
            model=build_model('stats_film',seed,data['vocabulary'],cfg['backbone']);initial=c.state_hash(model)
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
            rng=random.Random(seed*100003+data['vocabulary'][region]*1009);spec=data['regions'][region]
            draws=rng.choices(range(len(spec['block_starts'])),weights=spec['block_targets'],k=1600)
            prefix_hash=hashlib.sha256(json.dumps(draws[:800]).encode()).hexdigest()
            validation={region:c.evaluation_pools({'regions':{region:spec},'vocabulary':data['vocabulary']},'val',1024)[region]}
            history=[];best=math.inf;start=time.time();exposures=0
            for step,block in enumerate(draws,1):
                model.train();batch=c.training_batch(data,region,block,'long');n=batch.nll_event_mask.sum(-1)
                loss=(model.nll_loss(batch,reduction='per_event')['total']*n).sum()/n.sum()
                assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
                lr=3e-4*min(1.,step/160)*max(.05,.5*(1+math.cos(math.pi*step/1600)))
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.step();exposures+=int(n.sum())
                if step%100==0:
                    metrics,_=c.evaluate(model,validation);history.append(dict(step=step,validation=metrics))
                    if metrics['macro']['total']<best:
                        best=metrics['macro']['total'];torch.save(dict(config=cfg,model_state_dict=model.state_dict(),
                            step=step,validation=metrics),folder/'best_model.pth')
                    print(json.dumps(dict(run=name,step=step,validation=metrics['macro']['total'])),flush=True)
            state=torch.load(folder/'best_model.pth',map_location='cpu',weights_only=False)
            result=dict(config=cfg,checkpoint=str(folder/'best_model.pth'),checkpoint_sha256=c.sha(folder/'best_model.pth'),
                best_step=state['step'],validation=state['validation'],initial_state_sha256=initial,
                first_800_draws_sha256=prefix_hash,scored_event_exposures=exposures,
                history=history,seconds=time.time()-start)
            c.save(path,result);completed[name]=result;del model,optimizer;torch.cuda.empty_cache()
    c.save(out/'training_complete.json',dict(plan=control_plan,runs=completed))
    # Choices above are frozen before the main runner makes any test forecasts.
    while not (c.REPORTS/'neural_results.json').exists():time.sleep(20)
    rows=[]
    for name,result in completed.items():
        cfg=result['config'];region=cfg['region']
        primary=read(c.RUNS/f'local_{region}_{selected}_long_seed_{cfg["seed"]}/training_complete.json')
        assert primary['initial_state_sha256']==result['initial_state_sha256']
        assert primary['training_draw_hashes'][region]==result['first_800_draws_sha256']
        model=build_model('stats_film',cfg['seed'],data['vocabulary'],cfg['backbone'])
        state=torch.load(result['checkpoint'],map_location='cuda:0',weights_only=False);model.load_state_dict(state['model_state_dict'])
        pools=c.evaluation_pools({'regions':{region:data['regions'][region]},'vocabulary':data['vocabulary']},'test',1024)
        metrics,_=c.evaluate(model,pools)
        rows.append(dict(region=region,seed=cfg['seed'],best_step=result['best_step'],**metrics['regions'][region]))
        del model;torch.cuda.empty_cache()
    c.save(out/'test_results.json',dict(plan=control_plan,rows=rows,initialization_and_training_prefix_verified=True))
    print('CONVERGENCE_CONTROLS_COMPLETE',flush=True)


if __name__=='__main__':main()
