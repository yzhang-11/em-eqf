import math

import pytest
import torch
from torch import nn

from src.data import Batch
from src.models.tpp.global_mixer_tpp import GlobalMixerTPP
from src.models.tpp.mixer_tpp import MixerTPP
from test_region_mixer_tpp import TinyBase, args, sequence


def test_input_embedding_is_repeated_for_each_observed_event():
    settings = args()
    base = TinyBase(settings)
    base.encoder = nn.Linear(10, 8)
    base.encoder.d_model = 8
    parent = MixerTPP(base, nn.Linear(8,6), nn.Sequential(nn.Linear(8,1)),
                       dropout=0., predict_b=True, b_range=(.2,2.))
    model = GlobalMixerTPP(parent, settings, num_regions=2)
    sequences = [sequence(), sequence()]
    for i,seq in enumerate(sequences,1):
        seq.region_id = torch.full((len(seq),),i,dtype=torch.long)
    batch=Batch.from_list(sequences)
    features=model.base_model.input_adapter(batch)['features']
    torch.testing.assert_close(features[0,:,:2],features[1,:,:2])
    assert not torch.equal(features[0,0,2:],features[1,0,2:])
    torch.testing.assert_close(features[0,:4,2:],model.region_embedding.weight[1].double().expand(4,-1))
    assert torch.count_nonzero(features[:,-1,2:])==0
    model.nll_loss(batch)['total'].sum().backward()
    assert model.region_embedding.weight.grad[1:].abs().sum()>0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual Mamba kernels')
def test_global_cuda_sampling_and_checkpoint_roundtrip(tmp_path):
    from omegaconf import OmegaConf
    from src.models.builders import ModelBuilder
    settings=OmegaConf.load('experiments/revision/main/configs/pretrain_azdx_main_seed_0.yaml')
    for k,v in args().items():settings[k]=v
    settings.mixer_model_config.input_dim=10
    settings.use_b_updater=False
    settings.dropout=0.
    settings.mixer_model_config.dropout_prob=0.
    settings.mixer_model_config.input_proj_dropout=0.
    device=torch.device('cuda:0')
    def build():
        base=ModelBuilder.by_name('mixer_tpp')()(settings,device)
        return GlobalMixerTPP(base,settings,num_regions=3).eval()
    model=build()
    with torch.no_grad():
        model.hypernet_time.weight.zero_()
        model.hypernet_time.bias[:2*model.num_components].fill_(math.log(math.expm1(1.)))
        model.hypernet_time.bias[2*model.num_components:].zero_()
        model.hypernet_mag[0].weight.zero_();model.hypernet_mag[0].bias.fill_(1.)
    seq=sequence(mc=3.,tau=.01).to(device)
    seq.region_id=torch.full((len(seq),),3,device=device,dtype=torch.long)
    batch=Batch.from_list([seq])
    with torch.no_grad():expected=model.get_context(batch)
    torch.save(model.state_dict(),tmp_path/'model.pth')
    restored=build();restored.load_state_dict(torch.load(tmp_path/'model.pth',weights_only=True))
    with torch.no_grad():torch.testing.assert_close(expected,restored.get_context(batch))
    generated=restored.sample(4,.02,past_seq=seq,max_sample_len=64)
    assert torch.all(generated.region_id==3)
    for i in range(4):
        assert (generated.mag[i,:int(generated.end_idx[i])]>=3.).all()
