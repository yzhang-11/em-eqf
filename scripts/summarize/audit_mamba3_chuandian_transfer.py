"""Verify saved weights, temporal provenance, identical targets and replays."""
from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import torch

ROOT=Path('/root/autodl-tmp/em_eqf')
STORE=Path('/root/autodl-fs/em_eqf_training/mamba3_chuandian_transfer_20261105')
sys.path.insert(0,str(STORE/'frozen_project/scripts/tuning'))
import mamba3_transfer_common as c


def read(path):return json.loads(Path(path).read_text())


def main(guard=False):
    torch.set_num_threads(1);c.bootstrap()
    tag='chuandian_warmup_guard_20261105' if guard else 'chuandian_multiregion_transfer_20261105'
    down=ROOT/'experiments/revision/controls/reports'/tag
    runs=ROOT/'experiments/revision/controls/runs'/tag
    result=read(down/'test_results.json');regional=read(c.REPORTS/'neural_results.json')
    assert c.sha(c.DATA)==regional['plan']['data_sha256']
    regional_records=list(c.RUNS.glob('*/training_complete.json'))
    assert len(regional_records)==66
    for path in regional_records:
        info=read(path)
        assert info['config']['dataset_sha256']==regional['plan']['data_sha256']
        assert info['config']['backbone']['mixer_model_config']['n_layer']==2
        assert c.sha(path.parent/'best_model.pth')==info['checkpoint_sha256']
    regional_replays=[]
    target_precision=torch.get_float32_matmul_precision()
    target_tf32=torch.backends.cuda.matmul.allow_tf32
    if not guard:
        from run_mamba3_transfer_pretraining import install_builder
        build=install_builder()
        regional_data=torch.load(c.DATA,map_location='cpu',weights_only=False)
        pools=c.evaluation_pools(regional_data,'test',1024)
        for name in ['global_all_regions_mamba3_oracle_residual_long_seed_2',
                     'global_all_regions_mamba2_oracle_residual_long_seed_0']:
            folder=c.RUNS/name;info=read(folder/'training_complete.json');cfg=info['config']
            model=build('stats_film',cfg['seed'],regional_data['vocabulary'],cfg['backbone'])
            state=torch.load(folder/'best_model.pth',map_location='cuda:0',weights_only=False)
            model.load_state_dict(state['model_state_dict'],strict=True)
            actual,_=c.evaluate(model,pools);saved=read(folder/'metrics_context_1024.json')['test']
            difference=max(abs(actual['regions'][r][k]-saved['regions'][r][k])
                for r in actual['regions'] for k in ['time','mag','total'])
            assert difference<2e-4,(name,difference)
            regional_replays.append(dict(name=name,max_absolute_nll_difference=difference))
            del model;torch.cuda.empty_cache()
    # The legacy TPP builder imports an NCDE module that sets TF32 globally.
    # Target runners construct MixerModel directly and use the default FP32
    # matmul policy. Restore their actual policy before replaying task heads.
    torch.set_float32_matmul_precision(target_precision)
    torch.backends.cuda.matmul.allow_tf32=target_tf32
    data=torch.load(STORE/'chuandian_transfer.pt',map_location='cpu',weights_only=False)
    assert c.sha(STORE/'chuandian_transfer.pt')==result['plan']['data_sha256']
    assert c.sha(data['source'])==data['source_sha256']
    source_data=torch.load(STORE/'transfer_sources_before_20110212.pt',map_location='cpu',weights_only=False)
    cutoff=pd.Timestamp('2011-02-12',tz='UTC').value
    assert all(int(spec['physical_ns'][-1])<cutoff for spec in source_data['regions'].values())
    assert 'chuandian' not in source_data['regions']
    source_sha=c.sha(STORE/'transfer_sources_before_20110212.pt')
    source_records=list((c.RUNS/'transfer_pretraining').glob('*/training_complete.json'))
    assert len(source_records)==6
    for path in source_records:
        info=read(path)
        assert info['config']['dataset_sha256']==source_sha
        assert info['config']['protocol']=='past_only_transfer_pretraining_20110212_v1'
        assert c.sha(path.parent/'best_model.pth')==info['checkpoint_sha256']
    dates={key:(pd.Timestamp(value)-pd.Timestamp('1970-01-01')).total_seconds()/86400
           for key,value in data['split_dates'].items()}
    for task in data['tasks'].values():
        origins=task['origins'];stop=origins+task['Tfore'];split=task['splits']
        assert np.all(stop[split['train']]<=dates['train_end'])
        assert np.all(origins[split['validation']]>=dates['train_end'])
        assert np.all(stop[split['validation']]<=dates['validation_end'])
        assert np.all(origins[split['test']]>=dates['validation_end'])
        assert np.all(stop[split['test']]<=dates['test_end'])
    records=list(runs.glob('*/training_complete.json'));assert len(records)==66
    loaded=0;frozen_verified=0;predictions_verified=0
    for path in records:
        info=read(path);name=path.parent.name;task_name=name.split('_')[0];task=data['tasks'][task_name]
        ids=task['splits']['test']
        with np.load(path.parent/'predictions_test.npz') as predictions:
            np.testing.assert_array_equal(predictions['origin'],task['origins'][ids])
            np.testing.assert_array_equal(predictions['target'],task['labels'][ids])
            assert np.isfinite(predictions['prediction']).all()
        predictions_verified+=1
        if '_rf_seed_' in name:continue
        cfg=info['config'];assert cfg['data_sha256']==result['plan']['data_sha256']
        if guard:assert info['epochs_completed']>=25
        assert c.sha(path.parent/'best_model.pth')==info['checkpoint_sha256']
        state=torch.load(path.parent/'best_model.pth',map_location='cpu',weights_only=False)['model_state_dict']
        if cfg['family'].startswith('mamba'):
            assert cfg['backbone']['mixer_model_config']['n_layer']==2
            assert cfg['backbone']['mixer_model_config']['d_model']==64
            if cfg['mode']!='scratch':
                source=Path(cfg['source_checkpoint'])
                assert c.sha(source)==cfg['source_loading']['source_sha256']
                assert cfg['source_loading']['missing_keys']==cfg['source_loading']['unexpected_keys']==[]
                source_weights=torch.load(source,map_location='cpu',weights_only=False)['model_state_dict']
                loaded+=1
                if cfg['mode'] in ['frozen','last_block']:
                    for name,value in state.items():
                        if not name.startswith('encoder.'):continue
                        if cfg['mode']=='last_block' and name.startswith(('encoder.layers.1.','encoder.norm_f.')):continue
                        expected=source_weights['base_model.'+name]
                        assert torch.equal(value,expected),(path,name)
                    frozen_verified+=1
    assert loaded==36 and frozen_verified==24
    spec=importlib.util.spec_from_file_location('frozen_chuandian_runner',STORE/'frozen_downstream/run_chuandian_encoder_transfer.py')
    task_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(task_module)
    replays=[]
    names=['classification_mamba3_oracle_residual_frozen_seed_2',
           'classification_mamba3_oracle_residual_full_seed_2',
           'regression_mamba3_oracle_residual_last_block_seed_2',
           'regression_lstm_features_scratch_seed_2']
    for name in names:
        folder=runs/name;info=read(folder/'training_complete.json');cfg=info['config']
        model=task_module.TransferModel(cfg['family'],cfg['mode'],cfg['backbone'],cfg['seed'])
        state=torch.load(folder/'best_model.pth',map_location='cuda:0',weights_only=False)
        model.load_state_dict(state['model_state_dict'],strict=True)
        task=data['tasks'][cfg['task']];ids=task['splits']['test']
        actual=task_module.predict(model,data,task,ids)
        with np.load(folder/'predictions_test.npz') as saved:expected=saved['prediction'].copy()
        difference=float(np.max(np.abs(actual-expected)))
        assert difference<5e-5,(name,difference)
        replays.append(dict(name=name,max_prediction_difference=difference))
        del model;torch.cuda.empty_cache()
    assert regional['paired_initialization_and_exposures_verified']
    assert result['frozen_weights_and_paired_heads_verified']
    audit=dict(passed=True,warmup_guard=guard,source_pretraining_observations_strictly_before='2011-02-12',
        target_replay_float32_matmul_precision=target_precision,target_replay_allow_tf32=target_tf32,
        transfer_source_fits=6,new_regional_fits=regional['completed_new_fits'],
        regional_checkpoint_hashes_verified=len(regional_records),
        regional_metric_replays=regional_replays,
        target_result_files_with_identical_targets=predictions_verified,
        complete_encoder_loads_verified=loaded,best_checkpoints_with_frozen_weights_verified=frozen_verified,
        target_checkpoint_replays=replays,source_dataset_sha256=source_sha,
        target_dataset_sha256=result['plan']['data_sha256'],frozen_source_files=len(read(c.REPORTS/'frozen_source.json')))
    c.save(down/'final_audit.json',audit)
    c.save(c.REPORTS/('warmup_guard_audit.json' if guard else 'final_audit.json'),audit)
    print(json.dumps(audit),flush=True)


if __name__=='__main__':main('--warmup-guard' in sys.argv)
