"""Reuse the existing NTPP time model with a likelihood-only magnitude density head."""
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
import torch

from src.models.tpp.mixer_tpp import MixerTPP
from src.models.tpp.magnitude_density import MagnitudeDensityHead
from src.train.config_setup import load_and_prepare_model


def mark_history_values(marks, intervals, mc):
    return np.stack([marks-mc, (marks-mc)**2, marks>=mc+.5, marks>=mc+1.,
                     marks>=mc+2., np.log1p(intervals)], axis=-1)


def causal_mark_features(marks, intervals, mc):
    values=mark_history_values(marks,intervals,mc)
    result=[]
    for length in [32,128,512]:
        alpha=1./length
        filtered=lfilter([alpha],[1.,-(1.-alpha)],values,axis=1)
        weight=1.-(1.-alpha)**np.arange(1,marks.shape[1]+1)
        filtered/=weight[None,:,None]
        result.append(np.concatenate([np.zeros_like(filtered[:,:1]),filtered[:,:-1]],axis=1))
    return np.concatenate(result,axis=-1).astype(np.float32)


class MagnitudeTPP(MixerTPP):
    def __init__(self, original, checkpoint, freeze_time=True):
        assert original.b_filter is None
        super().__init__(original.base_model,original.hypernet_time,original.hypernet_mag,
                         dropout=original.dropout.p,predict_b=False,use_b_updater=False,
                         loss_weights={'time_weight':1.,'mag_weight':1.,'b_weight':0.,'b_smooth_weight':0.},
                         loss_reduction=original.reduction,b_range=(original.b_min,original.b_max),
                         bg_model=original.bg_model)
        if freeze_time:
            for parameter in self.parameters():parameter.requires_grad_(False)
        config=checkpoint['config']
        self.feature_mode=config['features']
        assert self.feature_mode in {'static','context','history','context_history'}
        self.density_head=MagnitudeDensityHead(checkpoint['input_dim'],config['family'],config['components'],
                                               checkpoint['Mc'],checkpoint['step'],hidden=config['hidden'])
        self.density_head.load_state_dict(checkpoint['state_dict'])
        self.density_head.to(device=self.device)
        self.register_buffer('feature_mean',torch.as_tensor(checkpoint['normalization_mean'],device=self.device))
        self.register_buffer('feature_std',torch.as_tensor(checkpoint['normalization_std'],device=self.device))
        self._sampling=False
        self._history_sum=None
        self._history_weight=None
        self._offline_features=None

    def get_context(self,batch,inference_params=None):
        context=super().get_context(batch,inference_params)
        if self.feature_mode in {'history','context_history'}:
            history=causal_mark_features(batch.mag.detach().double().cpu().numpy(),
                batch.inter_times.detach().double().cpu().numpy(),float(self.mag_completeness))
            history=torch.as_tensor(history,device=context.device,dtype=context.dtype)
            self._offline_features=history if self.feature_mode=='history' else torch.cat([context,history],-1)
        else:self._offline_features=context
        return context

    def _update_mark_history(self,batch):
        if self.feature_mode not in {'history','context_history'}:return
        marks,intervals=batch.mag.double(),batch.inter_times.double()
        if marks.shape[1]>1:
            values=mark_history_values(marks.cpu().numpy(),intervals.cpu().numpy(),float(self.mag_completeness))
            sums=[];weights=[]
            for length in [32,128,512]:
                alpha=1./length
                sums.append(lfilter([alpha],[1.,-(1.-alpha)],values,axis=1)[:,-1])
                weights.append(1.-(1.-alpha)**marks.shape[1])
            self._history_sum=torch.as_tensor(np.stack(sums,axis=1),device=marks.device)
            self._history_weight=torch.as_tensor(weights,device=marks.device).expand(marks.shape[0],-1).clone()
            return
        mc=float(self.mag_completeness)
        mark=marks[:,0];interval=intervals[:,0]
        values=torch.stack([mark-mc,(mark-mc)**2,(mark>=mc+.5).double(),(mark>=mc+1.).double(),
                            (mark>=mc+2.).double(),torch.log1p(interval)],-1)
        if self._history_sum is None:
            self._history_sum=values.new_zeros((len(mark),3,6))
            self._history_weight=values.new_zeros((len(mark),3))
        if self._history_sum.shape[0]!=len(mark):
            assert self._history_sum.shape[0]==1
            self._history_sum=self._history_sum.expand(len(mark),-1,-1).clone()
            self._history_weight=self._history_weight.expand(len(mark),-1).clone()
        alpha=values.new_tensor([1./32,1./128,1./512])
        self._history_sum=(1-alpha[None,:,None])*self._history_sum+alpha[None,:,None]*values[:,None,:]
        self._history_weight=(1-alpha[None,:])*self._history_weight+alpha[None,:]

    def get_current_state(self,batch,*args,**kwargs):
        context=super().get_current_state(batch,*args,**kwargs)
        if self._sampling:self._update_mark_history(batch)
        return context

    def get_magnitude_dist(self,context,predict_b=None,return_b=False,inter_times=None,filter_params=None):
        if self._sampling:
            if self.feature_mode in {'history','context_history'}:
                if self._history_sum is None:history=context.new_zeros((context.shape[0],1,18))
                else:
                    history=(self._history_sum/self._history_weight.clamp_min(1e-15).unsqueeze(-1)).flatten(1)
                    history=history.to(context.dtype).unsqueeze(1).expand(context.shape[0],context.shape[1],-1)
                features=history if self.feature_mode=='history' else torch.cat([context,history],-1)
            else:features=context
        else:
            features=self._offline_features
            assert features is not None and features.shape[:2]==context.shape[:2]
        if self.feature_mode=='static':features=context.new_zeros(context.shape[:-1]+(1,))
        distribution=self.density_head((features-self.feature_mean)/self.feature_std)
        return (distribution,None) if return_b else distribution

    def sample(self,*args,**kwargs):
        self._sampling=True;self._history_sum=None;self._history_weight=None
        try:return super().sample(*args,**kwargs)
        finally:self._sampling=False

    def compute_b_value(self,*args,**kwargs):
        raise NotImplementedError('This model predicts a magnitude probability distribution, not a b-value.')


def load_magnitude_model(head_path,device,freeze_time=True):
    saved=torch.load(head_path,map_location='cpu',weights_only=False)
    original,args=load_and_prepare_model(saved['base_checkpoint'],device,compile=False)
    model=MagnitudeTPP(original,saved,freeze_time=freeze_time).to(device).eval()
    return model,args
