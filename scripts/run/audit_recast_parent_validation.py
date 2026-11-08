"""Record zero-update target validation scores of the fixed RECAST parents."""
import argparse
import importlib.util
from pathlib import Path


def main(root):
    location = importlib.util.spec_from_file_location('finetune_driver', root / 'recast_driver.py')
    driver = importlib.util.module_from_spec(location)
    location.loader.exec_module(driver)
    plan = driver.read(root / 'plan.json')
    driver.verify(root, plan)
    torch, yaml, Study, *_ = driver.runtime(root)
    reference = Path(plan['reference'])
    cached = driver.read(root / 'parent_validation.json')['rows'] if (root / 'parent_validation.json').exists() else []
    rows = []
    for target in plan['targets']:
        for seed in [0, 1, 2]:
            original = reference / 'gr' / target / f's{seed}'
            config = yaml.safe_load((original / 'transfer.yaml').read_text())
            study = Study(config)
            data = study.load('development.pt')
            source = original / 'runs' / f'{target}__source__s{seed}__lr0.0003'
            parent = driver.read(source / 'result.json')
            checkpoint = source / 'best_model.pth'
            assert driver.sha(checkpoint) == parent['checkpoint_sha256']
            model = study._build_model(seed, data['vocabulary'])
            region = data['regions'][target]
            bounds = {target: (region['train_end'], region['val_end'])}
            scratch, _ = study.evaluate(model, data, bounds)
            prior = next((r for r in cached if (r['target'], r['seed'], r['parent_sha256']) ==
                          (target, seed, parent['checkpoint_sha256'])), None)
            if prior is None:
                model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
                value, _ = study.evaluate(model, data, bounds)
                initial_validation = value['macro']['time']
            else:
                initial_validation = prior['validation']
            assert driver.sha(checkpoint) == parent['checkpoint_sha256']
            row = dict(target=target, seed=seed, target_updates=0,
                       validation=initial_validation, scratch_validation=scratch['macro']['time'],
                       source_updates=parent['updates'],
                       parent_best_step=parent['best_step'], parent_checkpoint=str(checkpoint),
                       parent_sha256=parent['checkpoint_sha256'])
            rows.append(row)
            print(row, flush=True)
            del model
            torch.cuda.empty_cache()
    driver.write(root / 'parent_validation.json', dict(passed=True, rows=rows,
        scope='zero target parameter updates; observed target history and auxiliary calibration remain available'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-fs/em_eqf_training/unified_tpp_20261106/benchmark_finetune_convergence_h4096_3seed'))
    main(parser.parse_args().root.resolve())
