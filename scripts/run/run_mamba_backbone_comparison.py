"""Controlled TPP backbone comparison on the existing multiregion time split."""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import sys
import tarfile
import time

import numpy as np
from omegaconf import OmegaConf
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts/run")]
from run_multidomain_pretraining import build_model,model_settings,make_batch,evaluate,save_json
from run_long_multiregion import pools,evaluation_pools

STUDY=ROOT/"experiments/revision/mamba_backbones_20261105"
REPORTS=STUDY/"reports"
RUNS=Path("/root/autodl-fs/em_eqf_training/mamba_backbones_20261105/runs")
DATA=ROOT/"data/MultiRegion/catalogs/long_sequence_20261105.pt"
ARMS=["mamba2_rotary_xpos","mamba2_rotary","mamba3_rotary_siso","mamba3_rotary_mimo"]
SEEDS=[0,1,2]
STEPS=1500
EVERY=100


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(state):
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def backbone(arm):
    cfg=model_settings("stats_film")
    cfg.resume_path=None
    cfg.load_specific_parts=[]
    common=dict(d_state=64,headdim=32,expand=2)
    if arm.startswith("mamba2"):
        options=dict(layer="Mamba2Rotary",use_conv=False,chunk_size=256,
            rotary_emb_scale_base=8192 if arm.endswith("_xpos") else None,**common)
    else:
        options=dict(layer="Mamba3Rotary",rope_fraction=.5,rotary_emb_base=10000.,
            chunk_size=16 if arm.endswith("_mimo") else 64,
            is_mimo=arm.endswith("_mimo"),mimo_rank=2,**common)
    cfg.mixer_model_config.ssm_cfg=options
    return OmegaConf.to_container(cfg,resolve=True)


def common_key(name):
    return ".mixer." not in name or name.endswith(".mixer.out_proj.weight")


def common_initialization(seed,vocabulary,config=None):
    reference=build_model("stats_film",seed,vocabulary,backbone(ARMS[0]) if config is None else config)
    shared={name:value.detach().cpu().clone() for name,value in reference.state_dict().items() if common_key(name)}
    del reference
    return shared


def create_model(arm,seed,vocabulary,shared,config=None):
    model=build_model("stats_film",seed,vocabulary,backbone(arm) if config is None else config)
    state=model.state_dict()
    for name,value in shared.items():
        if name not in state or state[name].shape!=value.shape:
            raise ValueError(f"Common initialization mismatch: {arm}:{name}")
        state[name]=value.to(state[name].device)
    model.load_state_dict(state,strict=True)
    torch.manual_seed(seed+1000)
    return model


def setup():
    REPORTS.mkdir(parents=True,exist_ok=True)
    RUNS.mkdir(parents=True,exist_ok=True)
    link=STUDY/"runs"
    if not link.exists():
        link.symlink_to(RUNS,target_is_directory=True)
    data=torch.load(DATA,map_location="cpu",weights_only=False)
    train=pools(data,"long")
    validation=evaluation_pools(data,"val")
    return data,train,validation


def source_files():
    paths=[Path(__file__)]
    for directory in ["src/models","src/data","scripts/run"]:
        paths.extend((ROOT/directory).rglob("*.py"))
    return sorted(set(paths))


def study_plan(data):
    return dict(protocol="mamba_backbone_multiregion_v1",arms=ARMS,seeds=SEEDS,
        initialization="scratch, paired data order and shared non-mixer/SSM-out-projection initialization",
        dataset=str(DATA),dataset_sha256=sha(DATA),
        split="train before 2014; validation 2014-2015; test 2016-2019",
        sources=sorted(data["regions"]),
        targets=sorted(name for name,value in data["regions"].items() if value["core"]),
        model="RegionalMixerTPP stats-FiLM; identical heads and metadata normalization",
        d_model=64,n_layer=2,d_state=64,headdim=32,mimo_rank=2,
        train_context=256,train_scored=1024,batch_size=2,scored_events_per_update=2048,
        validation_context=256,validation_scored=128,validation_stride=128,
        max_updates=STEPS,validation_every=EVERY,learning_rate=3e-4,
        optimizer="AdamW",weight_decay=.01,gradient_clip=1.,
        scheduler="10% warmup then cosine floor 5%",
        checkpoint_selection="minimum core-region validation macro total NLL",
        test_selection=False,test_access="only after every planned fit has completed",
        metrics=["time NLL","magnitude NLL","total NLL"],units="nats/event, lower is better",
        calibration="metadata from earliest 256 training events, never scored; allowed as observed warmup history",
        capacity="matched width/state/head dimensions; parameter counts reported, MIMO has more parameters",
        physical_time="same tau-normalized event timestamps; M3 keeps native phase plus real-time phase",
        xpos="M2 production-style xPos=8192 plus unscaled M2 control; both M3 variants unscaled",
        train_unique_scored_events=sum(value["train_targets"] for value in data["regions"].values()),
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in source_files()},
        backbones={arm:backbone(arm) for arm in ARMS})


def freeze_plan(plan):
    path=REPORTS/"study_plan.json"
    if path.exists():
        if json.loads(path.read_text())!=plan:
            raise ValueError("Frozen comparison plan/source hashes changed; do not mix runs")
        return
    snapshot=RUNS.parent/"dataset_snapshot.pt"
    if not snapshot.exists():
        shutil.copyfile(DATA,snapshot)
    if sha(snapshot)!=plan["dataset_sha256"]:
        raise ValueError("Comparison dataset snapshot checksum mismatch")
    save_json(path,plan)
    with tarfile.open(REPORTS/"code_snapshot.tar.gz","w:gz") as output:
        for path in source_files():
            output.add(path,arcname=str(path.relative_to(ROOT)))
    save_json(REPORTS/"environment.json",dict(python=sys.executable,torch=torch.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability())))


def preflight(data,train,validation):
    shared=common_initialization(0,data["vocabulary"])
    selected=[train[region][0] for region in sorted(train)[:2]]
    batch=make_batch(selected)
    assert int(batch.nll_event_mask.sum())==2048
    rows=[]
    for arm in ARMS:
        print("PREFLIGHT_BEGIN",arm,flush=True)
        model=create_model(arm,0,data["vocabulary"],shared)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        timings=[]
        for step in range(5):
            started=time.perf_counter()
            model.train()
            loss=model.nll_loss(batch,reduction="per_event")["total"].mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert torch.isfinite(norm)
            optimizer.step()
            torch.cuda.synchronize()
            if step>=2:timings.append(time.perf_counter()-started)
        metrics,_=evaluate(model,{region:items[:2] for region,items in validation.items()})
        row=dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),
            mean_step_seconds=float(np.mean(timings)),validation=metrics["macro"],
            common_initialization_sha256=tensor_hash({k:v for k,v in model.state_dict().items() if common_key(k)}))
        rows.append(row)
        print(json.dumps(row),flush=True)
        del model,optimizer
        torch.cuda.empty_cache()
    save_json(REPORTS/"preflight.json",dict(rows=rows,source_sha256=study_plan(data)["source_sha256"],
        estimated_training_minutes=sum(row["mean_step_seconds"] for row in rows)*STEPS*len(SEEDS)/60))
    return rows


def fit(data,train,validation,plan,arm,seed,shared):
    name=f"{arm}_seed_{seed}"
    folder=RUNS/name
    config=dict(protocol=plan["protocol"],arm=arm,method="stats_film",seed=seed,
        region_id_map=data["vocabulary"],backbone=plan["backbones"][arm],dataset_sha256=plan["dataset_sha256"],
        shared_initialization_sha256=tensor_hash(shared),max_updates=STEPS,
        scored_events_per_update=2048,validation_every=EVERY)
    done=folder/"training_complete.json"
    if done.exists():
        saved=json.loads(done.read_text())
        assert saved["config"]==config and sha(folder/"best_model.pth")==saved["checkpoint_sha256"]
        print("REUSE",name,flush=True)
        return saved
    folder.mkdir(parents=True,exist_ok=True)
    OmegaConf.save(OmegaConf.create(config),folder/"config.yaml")
    model=create_model(arm,seed,data["vocabulary"],shared,config["backbone"])
    initial_hash=tensor_hash(model.state_dict())
    actual_shared=tensor_hash({k:v for k,v in model.state_dict().items() if common_key(k)})
    assert actual_shared==tensor_hash(shared)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    rng=random.Random(seed)
    regions=plan["sources"]
    counts=dict.fromkeys(regions,0)
    best,history=math.inf,[]
    train_seconds=0.
    started=time.time()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1,STEPS+1):
        tic=time.perf_counter()
        model.train()
        chosen=[]
        for _ in range(2):
            region=rng.choice(regions)
            chosen.append(rng.choice(train[region]))
            counts[region]+=1024
        batch=make_batch(chosen)
        assert int(batch.nll_event_mask.sum())==2048
        losses=model.nll_loss(batch,reduction="per_event")
        loss=losses["total"].mean()
        if not torch.isfinite(loss):raise ValueError(f"Nonfinite loss {name}:{step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise ValueError(f"Nonfinite gradient {name}:{step}")
        lr=3e-4*min(1.,step/(STEPS//10))*max(.05,.5*(1+math.cos(math.pi*step/STEPS)))
        for group in optimizer.param_groups:group["lr"]=lr
        optimizer.step()
        torch.cuda.synchronize()
        train_seconds+=time.perf_counter()-tic
        if step%EVERY==0:
            metrics,_=evaluate(model,validation)
            row=dict(step=step,train_nll=float(loss),validation=metrics,
                elapsed_seconds=time.time()-started,training_seconds=train_seconds)
            history.append(row)
            if metrics["macro"]["total"]<best:
                best=metrics["macro"]["total"]
                temporary=folder/"best_model.tmp"
                torch.save(dict(model_state_dict=model.state_dict(),config=config,
                    step=step,validation=metrics),temporary)
                temporary.replace(folder/"best_model.pth")
            save_json(folder/"progress.json",dict(config=config,history=history,initial_state_sha256=initial_hash))
            print(json.dumps(dict(run=name,step=step,validation=metrics["macro"],
                seconds=round(time.time()-started,1))),flush=True)
    checkpoint=torch.load(folder/"best_model.pth",map_location="cpu",weights_only=False)
    complete=dict(config=config,checkpoint=str(folder/"best_model.pth"),
        checkpoint_sha256=sha(folder/"best_model.pth"),best_step=checkpoint["step"],
        validation=checkpoint["validation"],initial_state_sha256=initial_hash,
        shared_initialization_sha256=actual_shared,elapsed_seconds=time.time()-started,
        training_seconds=train_seconds,scored_event_exposures=counts,
        scored_events_per_second=STEPS*2048/train_seconds,
        peak_gpu_memory_mib=torch.cuda.max_memory_allocated()/2**20,
        parameters=sum(p.numel() for p in model.parameters()),history=history)
    save_json(done,complete)
    print("TRAINING_COMPLETE",name,flush=True)
    del model,optimizer,checkpoint
    torch.cuda.empty_cache()
    return complete


def finalize(data,plan):
    expected=[f"{arm}_seed_{seed}" for seed in SEEDS for arm in ARMS]
    completed={name:json.loads((RUNS/name/"training_complete.json").read_text()) for name in expected}
    # This is the first construction/access of test windows in the runner.
    tests=evaluation_pools(data,"test")
    rows=[]
    for name in expected:
        complete=completed[name]
        assert sha(complete["checkpoint"])==complete["checkpoint_sha256"]
        cfg=complete["config"]
        model=build_model("stats_film",cfg["seed"],data["vocabulary"],cfg["backbone"])
        state=torch.load(complete["checkpoint"],map_location="cuda:0",weights_only=False)
        model.load_state_dict(state["model_state_dict"],strict=True)
        metrics,windows=evaluate(model,tests)
        save_json(RUNS/name/"metrics.json",dict(**complete,test=metrics))
        save_json(RUNS/name/"test_window_nll.json",windows)
        for region in [*plan["targets"],"macro"]:
            values=metrics["macro"] if region=="macro" else metrics["regions"][region]
            rows.append(dict(arm=cfg["arm"],seed=cfg["seed"],region=region,best_step=complete["best_step"],
                **values,parameters=complete["parameters"],seconds=complete["training_seconds"],
                scored_events_per_second=complete["scored_events_per_second"]))
        del model,state
    with (REPORTS/"all_results.csv").open("w",newline="") as output:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregates=[]
    for arm in ARMS:
        for region in [*plan["targets"],"macro"]:
            group=[row for row in rows if row["arm"]==arm and row["region"]==region]
            aggregates.append(dict(arm=arm,region=region,n=len(group),
                **{key:dict(mean=statistics.mean(row[key] for row in group),
                            sd=statistics.stdev(row[key] for row in group))
                   for key in ["time","mag","total","scored_events_per_second"]},
                parameters=group[0]["parameters"]))
    for seed in SEEDS:
        hashes={completed[f"{arm}_seed_{seed}"]["shared_initialization_sha256"] for arm in ARMS}
        assert len(hashes)==1
        assert completed[f"mamba2_rotary_xpos_seed_{seed}"]["initial_state_sha256"]==completed[f"mamba2_rotary_seed_{seed}"]["initial_state_sha256"]
        assert len({json.dumps(completed[f"{arm}_seed_{seed}"]["scored_event_exposures"],sort_keys=True) for arm in ARMS})==1
    summary=dict(plan=plan,completed_fits=len(completed),metrics=aggregates,
        shared_initialization_and_training_order_verified=True,
        test_windows={region:len(items) for region,items in tests.items()},
        lower_is_better=True)
    save_json(REPORTS/"summary.json",summary)
    lines=["# Controlled Mamba TPP backbone comparison","",
        "All results are held-out nats/event (lower is better), mean +/- sample SD over seeds 0/1/2.",
        "All 12 fits completed before test scoring. Architecture/checkpoints use validation only.",
        "Regional stats-FiLM and heads are identical; shared parameters and per-seed data order are paired.",
        "The four backbones have matched hidden/state/head sizes, with different parameter counts.",
        "M2-xPos matches the current positional-scaling option; M2-no-xPos is the scaling control.",
        "M3 uses native learned phase plus real event time. This fixed-budget result does not establish universal superiority.","",
        "| Backbone | Joint NLL | Time NLL | Magnitude NLL | Parameters |","|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        row=next(item for item in aggregates if item["arm"]==arm and item["region"]=="macro")
        cells=[f"{row[key]['mean']:.5f} +/- {row[key]['sd']:.5f}" for key in ["total","time","mag"]]
        lines.append(f"| {arm} | "+" | ".join(cells)+f" | {row['parameters']} |")
    lines+=["","Training: 256 context + 1024 scored events, batch 2, 1500 AdamW updates.",
        "Evaluation: shared 256 context + 128 scored events, stride 128.",
        "Train before 2014; validation 2014-2015; test 2016-2019.",
        "Training includes all seven prepared regions. Held-out evaluation covers Italy, Kyushu, New Zealand.",
        "Scored events are disjoint from metadata-calibration events; calibration events may serve as past context.",
        "Primary metric is the equal-region macro mean of conditional joint NLL.","",
        f"Raw metrics: {REPORTS/'all_results.csv'}",f"Checkpoints: {RUNS}",""]
    (REPORTS/"results.md").write_text("\n".join(lines))
    print("COMPARISON_COMPLETE",REPORTS/"summary.json",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--all",action="store_true")
    parser.add_argument("--finalize",action="store_true")
    args=parser.parse_args()
    assert args.preflight or args.all or args.finalize
    torch.set_num_threads(1)
    data,train,validation=setup()
    with (REPORTS/"run.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.preflight:
            preflight(data,train,validation)
        if args.all or args.finalize:
            plan=study_plan(data)
            freeze_plan(plan)
            if args.all:
                for seed in SEEDS:
                    shared=common_initialization(seed,data["vocabulary"],plan["backbones"][ARMS[0]])
                    torch.save(shared,REPORTS/f"common_initialization_seed_{seed}.pth")
                    for arm in ARMS:fit(data,train,validation,plan,arm,seed,shared)
            finalize(data,plan)


if __name__=="__main__":main()
