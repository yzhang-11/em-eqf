"""Reconstruct pilot split provenance, probe trained-model causality, and audit budgets."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts/run')]
import run_multidomain_pretraining as runner
from src.data import Batch, Sequence
from mamba_ssm.utils.generation import InferenceParams
import src.models.mamba.mamba2_rotary as mamba_module

STUDY = ROOT/'experiments/revision/multiregion_pretraining'
REPORTS = STUDY/'reports/multidomain_id'
ORIGINAL_RUNS = STUDY/'runs/multidomain_id'
REGIONS = ['italy', 'japan_kyushu', 'new_zealand']


def read(path):
    return json.loads(path.read_text())


def save(name, payload):
    runner.save_json(REPORTS/name, payload)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tagged_data():
    path = ROOT/'data/MultiRegion/catalogs/pilot_2007_2019.pt'
    data = torch.load(path, map_location='cpu', weights_only=False)
    vocab = read(REPORTS/'region_vocabulary.json')
    for region, pools in data.items():
        for sequences in pools.values():
            for seq in sequences:
                seq.region_id = torch.full((len(seq),), vocab[region], dtype=torch.long)
    return data, vocab, sha(path)


def audit_data(data, data_sha):
    description = read(STUDY/'reports/pilot_data_20261104.json')
    assert description['dataset_sha256'] == data_sha
    per_region = {}
    physical = {}
    for region in REGIONS:
        config = description['regions'][region]
        frame = pd.read_csv(config['processed_path'])
        frame.time = pd.to_datetime(frame.time, utc=True, format='ISO8601')
        assert frame.time.is_monotonic_increasing and not frame.time.duplicated().any()
        assert len(frame) == config['records']
        calibration = frame[frame.time < '2014-01-01'].iloc[:256]
        ns = pd.DatetimeIndex(calibration.time).as_unit('ns').asi8.astype(np.float64)
        gaps = np.diff(ns)/86400e9
        mc = config['mc']
        b = np.log1p(1/np.mean((calibration.magnitude-mc)/.1))/(.1*np.log(10))
        metadata = [mc,float(b),float(gaps.mean()),float(gaps.min()),float(gaps.max()),.1,config['metadata'][6]]
        np.testing.assert_allclose(metadata,config['metadata'],rtol=1e-12,atol=1e-12)
        train_start = calibration.time.iloc[-1]+pd.Timedelta(nanoseconds=1)
        sets = {}; split_report = {}
        for split, start, end in [('train',train_start,pd.Timestamp('2014-01-01',tz='UTC')),
                                   ('val',pd.Timestamp('2014-01-01',tz='UTC'),pd.Timestamp('2016-01-01',tz='UTC')),
                                   ('test',pd.Timestamp('2016-01-01',tz='UTC'),pd.Timestamp('2020-01-01',tz='UTC'))]:
            selected = frame[(frame.time>=start)&(frame.time<end)]
            raw_ns = pd.DatetimeIndex(selected.time).as_unit('ns').asi8
            days = raw_ns.astype(np.float64)/86400e9
            mags = selected.magnitude.to_numpy(np.float64)
            starts = list(range(1,len(days)-192+1,128))
            assert len(starts)==len(data[region][split])
            scored = []; consumed = set()
            counts=[]
            for left, seq in zip(starts,data[region][split]):
                right=left+192
                arrival=days[left:right]-days[left-1]
                expected=np.diff(np.r_[0.,arrival,arrival[-1]])
                np.testing.assert_array_equal(seq.inter_times.numpy(),expected)
                np.testing.assert_array_equal(seq.mag.numpy(),mags[left:right])
                np.testing.assert_allclose(seq.region_params.numpy(),np.tile(metadata,(192,1)),rtol=1e-12,atol=1e-12)
                assert seq.t_nll_start==arrival[63]
                batch=Batch.from_list([seq])
                assert batch.start_idx.item()==64 and batch.end_idx.item()==192
                assert batch.nll_event_mask.sum().item()==128
                counts.append(128)
                consumed.update(raw_ns[left-1:right].tolist())
                scored.extend(raw_ns[left+64:right].tolist())
            assert len(scored)==len(set(scored)),(region,split,'overlapping scored events')
            sets[split]=consumed
            split_report[split]=dict(windows=len(starts),scored_events=len(scored),
                                      first_event=selected.time.iloc[0].isoformat(),last_event=selected.time.iloc[-1].isoformat(),
                                      unique_scored_events=True,all_windows_128_scored=True)
            physical[(region,split)]={(int(t.value),round(float(lat),5),round(float(lon),5))
                                      for t,lat,lon in zip(selected.time,selected.latitude,selected.longitude)}
        assert not (sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test'])
        assert not set(pd.DatetimeIndex(calibration.time).as_unit('ns').asi8.tolist())&set.union(*sets.values())
        per_region[region]=dict(processed_sha256=sha(Path(config['processed_path'])),
                                metadata_recomputed_from_first_256=True,calibration_end=calibration.time.iloc[-1].isoformat(),
                                splits=split_report,train_val_test_event_disjoint=True)
    pair_overlap=[]
    names=list(physical)
    for i,left in enumerate(names):
        for right in names[i+1:]:
            if left[0]!=right[0]:
                overlap=physical[left]&physical[right]
                assert not overlap,(left,right)
                pair_overlap.append(dict(left=list(left),right=list(right),overlap=len(overlap)))
    result=dict(passed=True,dataset_sha256=data_sha,regions=per_region,cross_region_overlap=pair_overlap,
                event_keys='absolute UTC timestamp + latitude/longitude rounded to 5 decimals',
                scope='serialized windows reconstructed from processed catalogs; raw agency catalogs are retrospective')
    save('leakage_data_audit.json',result)
    print(json.dumps(dict(data_audit_passed=True,regions=len(per_region))),flush=True)


def heads(model, context):
    return dict(time=model.hypernet_time(model._hidden(context)),b=model._get_b_pred(context))


def reference_scan64(x,dt,A,B,C,chunk_size,D=None,z=None,dt_bias=None,dt_softplus=False,**kwargs):
    """Independent dense causal SSM formula, audit only; production kernels stay unchanged."""
    dtype=x.dtype
    dt=dt.double()
    if dt_bias is not None:
        dt=dt+dt_bias.double()
    if dt_softplus:
        dt=torch.nn.functional.softplus(dt)
    dt=dt.clamp(*kwargs.get('dt_limit',(0.,float('inf'))))
    repeats=x.shape[2]//B.shape[2]
    bb=B.double().repeat_interleave(repeats,dim=2)
    cc=C.double().repeat_interleave(repeats,dim=2)
    accumulated=dt.cumsum(1).permute(0,2,1)
    length=x.shape[1]
    mask=torch.ones((length,length),device=x.device,dtype=torch.bool).tril()
    span=(accumulated[:,:,:,None]-accumulated[:,:,None,:]).masked_fill(~mask,0)
    decay=torch.exp(A.double()[None,:,None,None]*span)*mask
    kernel=torch.matmul(cc.permute(0,2,1,3),bb.permute(0,2,3,1))*decay
    y=torch.matmul(kernel,(x.double()*dt[:,:,:,None]).permute(0,2,1,3)).permute(0,2,1,3)
    if D is not None:
        skip=D.double()[None,None,:,None] if D.ndim==1 else D.double()[None,None]
        y=y+x.double()*skip
    if z is not None:
        y=y*torch.nn.functional.silu(z.double())
    if kwargs.get('return_final_states'):
        weight=torch.exp(A.double()[None,:,None]*(accumulated[:,:,-1:]-accumulated)).permute(0,2,1)*dt
        state=torch.einsum('blh,blhp,blhn->bhpn',weight,x.double(),bb).to(dtype)
        return y.to(dtype),state
    return y.to(dtype)


def difference(first, second):
    maximum=0.
    for key in first:
        torch.testing.assert_close(first[key],second[key],rtol=1e-5,atol=1e-4)
        maximum=max(maximum,float((first[key]-second[key]).abs().max()))
    return maximum


def prefix_sequence(seq, cut):
    return Sequence(torch.cat([seq.inter_times[:cut],seq.inter_times.new_zeros(1)]),
                    t_nll_start=float(seq.arrival_times[cut-1]),mag=seq.mag[:cut].clone(),
                    region_params=seq.region_params[:cut].clone(),region_id=seq.region_id[:cut].clone())


@torch.no_grad()
def audit_models(data):
    replays=[];causality=[];online=[];numerical=[]
    fast_scan=mamba_module.mamba_chunk_scan_combined
    production_precision=None
    for seed in range(3):
        for kind, targets in [('known',REGIONS)]+[('local',[r]) for r in REGIONS]:
            name=f'known_stats_film_seed_{seed}' if kind=='known' else f'local_{targets[0]}_seed_{seed}'
            path=ORIGINAL_RUNS/name/'best_model.pth'
            reference=read(path.parent/'metrics.json')
            assert reference['config']['initialization'] is None
            model,cfg=runner.load_multidomain_model(path)
            if production_precision is None:
                # Building the first model lazily imports a module that selects
                # 'high' precision, just as in the original training runner.
                production_precision=torch.get_float32_matmul_precision()
            mamba_module.mamba_chunk_scan_combined=fast_scan
            torch.set_float32_matmul_precision(production_precision)
            replay,_=runner.evaluate(model,{r:data[r]['test'] for r in targets})
            err=max(abs(replay['regions'][r][k]-reference['test']['regions'][r][k])
                    for r in targets for k in ['time','mag','total'])
            # Production enables TF32 and fused scans; audit against a material
            # 1e-3 nats bound, while retaining every unrounded discrepancy.
            assert err<1e-3,(name,err)
            mamba_module.mamba_chunk_scan_combined=reference_scan64
            torch.set_float32_matmul_precision('highest')
            independent,_=runner.evaluate(model,{r:data[r]['test'] for r in targets})
            backend_error=max(abs(replay['regions'][r][k]-independent['regions'][r][k])
                              for r in targets for k in ['time','mag','total'])
            assert backend_error<.002,(name,backend_error)
            replays.append(dict(run=name,max_nll_error=err,test=replay,reference64=independent,
                                max_reference_nll_error=backend_error,checkpoint_sha256=sha(path)))
            progress=read(path.parent/'progress.json')['history']
            best=min(progress,key=lambda x:x['validation_nll'])
            assert best['step']==reference['best_step']
            for region in targets:
                samples=data[region]['test']
                for window in [0,len(samples)//2,len(samples)-1]:
                    seq=samples[window]
                    batch=runner.make_batch([seq])
                    original=model.get_context(batch)
                    # Independent likelihood aggregation: score exactly event indices 64..191.
                    event_sum=model.get_inter_time_dist(original).log_prob(batch.inter_times.clamp_min(1e-10))[:,64:192].sum()
                    last=model.get_inter_time_dist(original[:,192]).log_survival(batch.inter_times[:,192].clamp_min(1e-10))
                    boundary=batch.inter_times[:,64]-(batch.arrival_times[:,64]-batch.t_nll_start)
                    previous=model.get_inter_time_dist(original[:,64]).log_survival(boundary)
                    manual_time=-(event_sum+last-previous).squeeze()/128
                    manual_mag=-model.get_magnitude_dist(original).log_prob(batch.mag)[:,64:192].mean()
                    loss=model.nll_loss(batch,reduction='per_event')
                    torch.testing.assert_close(manual_time,loss['time'][0],rtol=1e-7,atol=1e-7)
                    torch.testing.assert_close(manual_mag,loss['mag'][0],rtol=1e-7,atol=1e-7)
                    for cut in [64,128]:
                        changed=copy.deepcopy(seq)
                        changed.mag[cut:]+=.7
                        changed.inter_times[cut:-1]=changed.inter_times[cut:-1]*1.7+.001
                        changed.arrival_times=changed.inter_times.cumsum(-1)[:-1]
                        changed.t_end=float(changed.inter_times.sum())
                        perturbed=model.get_context(runner.make_batch([changed]))
                        before=heads(model,original[:,:cut+1])
                        after=heads(model,perturbed[:,:cut+1])
                        future_error=difference(before,after)
                        prefix=model.get_context(runner.make_batch([prefix_sequence(seq,cut)]))
                        prefix_error=difference(heads(model,original[:,cut:cut+1]),heads(model,prefix[:,-1:]))
                        mamba_module.mamba_chunk_scan_combined=fast_scan
                        torch.set_float32_matmul_precision(production_precision)
                        fast_original=model.get_context(batch)
                        fast_perturbed=model.get_context(runner.make_batch([changed]))
                        f1=heads(model,fast_original[:,:cut+1]);f2=heads(model,fast_perturbed[:,:cut+1])
                        fast_error=max(float((f1[k]-f2[k]).abs().max()) for k in f1)
                        p1=model.get_inter_time_dist(fast_original).log_prob(batch.inter_times)[:,:cut+1]
                        p2=model.get_inter_time_dist(fast_perturbed).log_prob(batch.inter_times)[:,:cut+1]
                        numerical.append(dict(run=name,region=region,window=window,cut=cut,
                                              fused_head_error=fast_error,fused_logprob_error=float((p1-p2).abs().max())))
                        mamba_module.mamba_chunk_scan_combined=reference_scan64
                        torch.set_float32_matmul_precision('highest')
                        causality.append(dict(run=name,region=region,window=window,cut=cut,
                                              future_change_head_error=future_error,prefix_head_error=prefix_error))
                    if seed==0 and kind=='known' and window==0:
                        cut=64
                        cache=InferenceParams(max_seqlen=193,max_batch_size=1)
                        history=runner.make_batch([prefix_sequence(seq,cut)]).get_sample_batch()
                        current=model.get_current_state(history,inference_params=cache)
                        maximum=difference(heads(model,original[:,cut:cut+1]),heads(model,current))
                        for j in range(cut,192):
                            token=Batch(inter_times=batch.inter_times[:,j:j+1],arrival_times=batch.arrival_times[:,j:j+1],
                                        mag=batch.mag[:,j:j+1],input_mask=batch.input_mask[:,j:j+1],
                                        region_params=batch.region_params[:,j:j+1],region_id=batch.region_id[:,j:j+1])
                            current=model.get_current_state(token,inference_params=cache)
                            maximum=max(maximum,difference(heads(model,original[:,j+1:j+2]),heads(model,current)))
                        online.append(dict(region=region,window=window,streamed_targets=128,max_head_error=maximum))
            del model
            torch.cuda.empty_cache()
            print(json.dumps(dict(replayed=name,max_nll_error=err)),flush=True)
    mamba_module.mamba_chunk_scan_combined=fast_scan
    torch.set_float32_matmul_precision(production_precision)
    result=dict(passed=True,replayed_checkpoints=replays,causal_probes=causality,online_probes=online,
                replay_tolerance_nats=1e-3,max_saved_nll_error=max(x['max_nll_error'] for x in replays),
                causal_backend='independent float64 dense causal SSM formula + highest float32 GEMM; weights unchanged',
                production_matmul_precision=production_precision,
                numerical_probes=numerical,
                max_fused_head_error=max(x['fused_head_error'] for x in numerical),
                max_fused_logprob_error=max(x['fused_logprob_error'] for x in numerical),
                max_reference_nll_error=max(x['max_reference_nll_error'] for x in replays),
                max_future_head_error=max(x['future_change_head_error'] for x in causality),
                max_prefix_head_error=max(x['prefix_head_error'] for x in causality),
                warning='inspect prediction outputs; time-embedding vectors may change because reciprocal factors cancel')
    save('leakage_model_audit.json',result)


def budget_control(data,vocabulary,data_sha,steps,regions):
    runner.RUNS=ORIGINAL_RUNS/'budget_audit'
    runner.CONFIGS=STUDY/'configs/multidomain_id/budget_audit'
    fitted=[]
    for seed in range(3):
        for region in regions:
            name=f'local_{region}_steps_{steps}_seed_{seed}'
            fitted.append(runner.fit(data,vocabulary,data_sha,name,'stats_film',[region],seed,steps,16,[region]))
    save(f'budget_control_{steps}.json',fitted)
    return fitted


def summary():
    data=read(REPORTS/'leakage_data_audit.json')
    models=read(REPORTS/'leakage_model_audit.json')
    results=[]
    for region in REGIONS:
        row=dict(region=region)
        for kind in ['local','known']:
            records=[read(ORIGINAL_RUNS/(f'local_{region}_seed_{s}' if kind=='local' else f'known_stats_film_seed_{s}')/'metrics.json')
                     for s in range(3)]
            row[kind]={k:statistics.mean(x['test']['regions'][region][k] for x in records) for k in ['time','mag','total']}
        for path in sorted(REPORTS.glob('budget_control_*.json')):
            steps=path.stem.replace('budget_control_','')
            rows=[x for x in read(path) if x['config']['targets']==[region]]
            if not rows:
                continue
            assert len(rows)==3
            row[f'local_{steps}']={k:statistics.mean(x['test']['regions'][region][k] for x in rows) for k in ['time','mag','total']}
        row['original_gain_nats']=row['local']['total']-row['known']['total']
        row['geometric_density_ratio']=math.exp(row['original_gain_nats'])
        results.append(row)
    save('leakage_audit_summary.json',dict(data_audit_passed=data['passed'],model_audit_passed=models['passed'],results=results))
    lines=['# 大幅 NLL 差距复核（2026-11-04）','',
           '检查对象：原单区模型各 150 步，与三地区联合模型 450 步。三个种子均使用同一测试集。','',
           '**结论：原始数值可以复算，本次事件、统计量和因果性检查未发现足以解释差距的泄露；但单区 150 步基线偏弱，原差距不能代表充分训练后的收益。等 450 步后，日本改善、意大利差距很小、新西兰联合训练反而较差。**','',
           '## 泄露核查','',
           '- 从处理后的绝对 UTC 时间目录逐窗重建保存的数据，核对每个事件间隔、震级、64 个历史事件和 128 个评分事件。训练/验证/测试事件交集为 0，跨地区同时间同位置事件交集为 0。',
           '- 参考 b、时间均值及范围逐个重算，仅使用各区最早 256 个校准事件；这些事件也未进入训练/验证/测试窗口。所有区训练事件早于 2014 年、验证早于 2016 年、测试始于 2016 年。',
           '- 12 个实际训练检查点重新评估；逐个核对验证最优步数。训练代码随机初始化，训练池仅为配置内源地区 train，检查点选择只看 val。',
           f'- 对实际检查点采用独立 float64 下三角状态空间公式和 highest 精度矩阵计算复核（不修改权重）。{len(models["causal_probes"])} 组探针改变当前及未来震级/间隔，或直接删除全部未来；最大未来扰动参数误差 {models["max_future_head_error"]:.3g}，最大前缀参数差异 {models["max_prefix_head_error"]:.3g}。',
           f'- 生产融合内核存在浮点计算误差：未来扰动时最大参数差异 {models["max_fused_head_error"]:.3g}、最大单事件 log-pdf 差异 {models["max_fused_logprob_error"]:.3g}；高精度公式重算所有检查点测试集，平均 NLL 与原内核最大相差 {models["max_reference_nll_error"]:.3g}，不足以解释 0.623 的原差距。',
           '- 联合 seed=0 在三地区各一个窗口，从 64 个历史事件开始逐个读入 128 个目标，与高精度整窗因果预测核对；预填充使用高精度参考，逐步状态更新沿用原计算。独立重算事件对数概率，核对评分下标和聚合。',
           '- 时间编码的中心依赖整窗，但成对缩放会抵消；没有仅因中间向量变化而修改时间编码。',
           '- 此审计未发现上述范围内的数据泄露。目录属于事后整理版本，未模拟当时实时目录的定位/震级修订和检出延迟；这种局限在单区和联合两组共同存在。','',
           '## 补齐优化步数后的比较','',
           '补跑模型与联合模型保持相同骨干、初始化种子、batch、450 步学习率日程及验证检查频次；每区单独训练 450 步。选择检查点仍只用相应验证集。',
           '原 150 步对照匹配的是三个单区模型的总步数与每区期望训练量；450 步对照匹配每个模型的优化更新次数，单区接触本区的样本数因此更多。两种预算回答不同问题。','',
           '| 地区 | 原单区 150 步 | 单区 450 步 | 联合 450 步 |','|---|---:|---:|---:|']
    for row in results:
        lines.append(f"| {row['region']} | {row['local']['total']:.6f} | {row['local_450']['total']:.6f} | {row['known']['total']:.6f} |")
    for row in results:
        if 'local_1500' in row:
            lines+=['',f"日本主问题的额外预算检查：单区 1500 步预算的三种子平均联合 NLL 为 {row['local_1500']['total']:.6f}，历史验证集均选择第 500 步；第 1000/1500 步验证指标变差。这是补充诊断，不等于已经证明收敛。"]
    lines+=['','以上为 3 种子测试联合 NLL 均值，nats/事件。更多步数控制若存在，记录于 leakage_audit_summary.json 和 budget_control_*.json。','',
            '## 正确解释差距','',
            '日本原始 0.743600→0.120793，差值为 0.622808 nats/事件；几何平均联合密度比为 exp(0.622808)≈1.864。不能解释为预测精度提高 6 倍或提高 84%。时间项可以为负，时间单位也会平移其绝对值，联合 NLL 的百分比不具有稳定含义。','',
            '原日本分项：时间 NLL -1.973930→-2.590927，震级 NLL 2.717530→2.711719。差距几乎全部来自时间项。新对照仍是短预算原型，不能据此声称充分收敛后的优势。','',
            '原始结果与检查点未覆盖。补跑配置在 configs/multidomain_id/budget_audit/，检查点和训练轨迹在 runs/multidomain_id/budget_audit/；事件/模型核查明细分别为 leakage_data_audit.json 与 leakage_model_audit.json。','']
    (REPORTS/'leakage_audit.md').write_text('\n'.join(lines),encoding='utf-8')
    banner='''<!-- budget-audit-20261104 -->
**2026-11-04 补充复核：原单区基线仅 150 步，不能据此宣称充分训练后有同等幅度收益。新增 12 次预算对照：单区 450 步，日本/意大利/新西兰 NLL 为 0.403310/0.618640/0.960935；联合 450 步为 0.120793/0.603148/1.024235。日本单区另给 1500 步预算，验证选择第 500 步，NLL 均值 0.266744。事件和统计量重建、12 检查点复算、108 组因果探针未发现相关泄露。完整说明见 leakage_audit.md（与 results.md 同目录）。**
'''
    for path in [STUDY/'README.md',REPORTS/'results.md']:
        contents=path.read_text()
        if '<!-- budget-audit-20261104 -->' not in contents:
            title,rest=contents.split('\n',1)
            path.write_text(title+'\n\n'+banner+'\n'+rest,encoding='utf-8')
    print(json.dumps(dict(results=results,data_passed=data['passed'],model_passed=models['passed'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['audit','budget','summary','all'],default='all')
    parser.add_argument('--steps',type=int,default=450)
    parser.add_argument('--regions',nargs='+',choices=REGIONS,default=REGIONS)
    args=parser.parse_args()
    torch.set_num_threads(1)
    data,vocabulary,data_sha=tagged_data()
    if args.stage in ['audit','all']:
        audit_data(data,data_sha)
        audit_models(data)
    if args.stage in ['budget','all']:
        budget_control(data,vocabulary,data_sha,args.steps,args.regions)
    if args.stage in ['summary','all']:
        summary()


if __name__=='__main__':
    main()
