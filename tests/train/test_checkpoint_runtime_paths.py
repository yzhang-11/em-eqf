from omegaconf import OmegaConf
import torch

from src.train import config_setup


def test_current_paths_override_checkpoint_paths_without_changing_hyperparameters():
    checkpoint = {'hyperparameters': {
        'model': 'original', 'Mc': 3., 'learning_rate': .001,
        'save_dir': '/old/run', 'resume_path': '/old/source.pth',
        'catalog_cfg': {'root_dir': '/old/cache', 'catalog_file': '/old/events.csv', 'mag_completeness': 3.},
    }}
    runtime = OmegaConf.create({
        'model': 'different', 'Mc': 7., 'learning_rate': .5,
        'save_dir': '/new/run', 'resume_path': '/new/source.pth',
        'catalog_cfg': {'root_dir': '/new/cache', 'catalog_file': '/new/events.csv', 'mag_completeness': 7.},
    })
    args = config_setup.load_args_from_checkpoint(runtime, checkpoint)
    assert (args.model, args.Mc, args.learning_rate) == ('original', 3., .001)
    assert args.catalog_cfg.mag_completeness == 3.
    assert args.catalog_cfg.root_dir == '/new/cache'
    assert args.catalog_cfg.catalog_file == '/new/events.csv'
    assert args.save_dir == '/new/run'
    assert args.resume_path == '/new/source.pth'
    assert checkpoint['hyperparameters']['catalog_cfg']['root_dir'] == '/old/cache'


def test_missing_runtime_paths_preserve_checkpoint_values():
    saved = {'model': 'original', 'save_dir': '/old/run', 'catalog_cfg': {'root_dir': '/old/cache'}}
    checkpoint = {'hyperparameters': saved}
    assert OmegaConf.to_container(config_setup.load_args_from_checkpoint(None, checkpoint)) == saved
    runtime = OmegaConf.create({'save_dir': None, 'catalog_cfg': {}})
    assert OmegaConf.to_container(config_setup.load_args_from_checkpoint(runtime, checkpoint)) == saved


def test_model_loading_uses_adjacent_configuration_for_runtime_paths(tmp_path, monkeypatch):
    path = tmp_path / 'best_model_1.pth'
    path.touch()
    OmegaConf.save(OmegaConf.create({'model': 'different', 'save_dir': None,
                                    'catalog_cfg': {'root_dir': '/new/cache'}}), tmp_path / 'config.yaml')
    checkpoint = {'hyperparameters': {'model': 'original', 'save_dir': '/old/run',
                                     'catalog_cfg': {'root_dir': '/old/cache'}}}
    monkeypatch.setattr(config_setup.torch, 'load', lambda *a, **kw: checkpoint)
    monkeypatch.setattr(config_setup.ModelBuilder, 'by_name', lambda name: lambda: lambda args, device: torch.nn.Linear(1, 1))
    monkeypatch.setattr(config_setup, 'load_model_from_checkpoint', lambda model, saved: (model, None, None))
    model, args = config_setup.load_and_prepare_model(path, torch.device('cpu'), compile=False)
    assert args.model == 'original'
    assert args.catalog_cfg.root_dir == '/new/cache'
    assert args.save_dir == str(tmp_path)
    assert not model.training
