"""Use a fitted GR-GAS state with the original frozen NTPP time model."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.models.tpp.mixer_tpp import MixerTPP
from src.models.tpp.magnitude_density import MagnitudeGridDistribution
from src.models.tpp.magnitude_dynamics import filter_gas,gr_statistics,parameters
from src.train.config_setup import load_and_prepare_model


class GASMagnitudeTPP(MixerTPP):
    def __init__(self,original,saved):
        assert original.b_filter is None
        super().__init__(original.base_model,original.hypernet_time,original.hypernet_mag,
            dropout=original.dropout.p,predict_b=False,use_b_updater=False,
            loss_weights={'time_weight':1.,'mag_weight':1.,'b_weight':0.,'b_smooth_weight':0.},
            loss_reduction=original.reduction,b_range=(original.b_min,original.b_max),bg_model=original.bg_model)
        for value in self.parameters():value.requires_grad_(False)
        self.gas_theta=np.asarray(saved['theta'],dtype=float)
        self.gas_family=saved['config']['family']
        if self.gas_family not in {'gr_free','gr_contractive'}:raise ValueError(self.gas_family)
        self.density_head=SimpleNamespace(mag_min=saved['Mc'],step=saved['step'])
        self._sampling=False;self._gas_sample_state=None;self._gas_offline_state=None

    def get_context(self,batch,inference_params=None):
        context=super().get_context(batch,inference_params)
        marks=batch.mag.detach().cpu().numpy();mc=self.density_head.mag_min;step=self.density_head.step
        values=[]
        for row in marks:
            valid=row>=mc-1e-6;n=int(valid.sum())
            assert valid[:n].all() and not valid[n:].any()
            excess=np.rint((row[:n].astype(float)-mc)/step)*step
            result=filter_gas(self.gas_theta,self.gas_family,excess,step,10.-mc,return_trace=True)
            values.append(np.r_[result['state'],np.full(len(row)-n,result['final_state'])])
        self._gas_offline_state=torch.as_tensor(np.stack(values),device=context.device,dtype=context.dtype)
        return context

    def _update_gas_history(self,batch):
        marks=batch.mag.detach().cpu().numpy()
        mc=self.density_head.mag_min;step=self.density_head.step
        if marks.shape[1]>1:
            assert marks.shape[0]==1
            excess=np.rint((marks[0].astype(float)-mc)/step)*step
            result=filter_gas(self.gas_theta,self.gas_family,excess,step,10.-mc,return_trace=True)
            self._gas_sample_state=np.array([result['final_state']])
            return
        mean,rho,alpha,_,_=parameters(self.gas_theta,self.gas_family)
        if self._gas_sample_state is None:self._gas_sample_state=np.full(len(marks),mean)
        if len(self._gas_sample_state)!=len(marks):
            assert len(self._gas_sample_state)==1
            self._gas_sample_state=np.repeat(self._gas_sample_state,len(marks))
        for i,mark in enumerate(marks[:,0]):
            excess=round((float(mark)-mc)/step)*step
            score=gr_statistics(self._gas_sample_state[i],excess,step,10.-mc)[1]
            self._gas_sample_state[i]=mean+rho*(self._gas_sample_state[i]-mean)+alpha*score

    def get_current_state(self,batch,*args,**kwargs):
        context=super().get_current_state(batch,*args,**kwargs)
        if self._sampling:self._update_gas_history(batch)
        return context

    def _get_b_pred(self,context,predict_b=None,inter_times=None,filter_params=None):
        if self._sampling:
            values=self._gas_sample_state
            if values is None:values=np.array([self.gas_theta[0]])
            state=torch.as_tensor(values,device=context.device,dtype=context.dtype)[:,None]
            state=state.expand(context.shape[:2])
        else:
            state=self._gas_offline_state
            assert state is not None and state.shape==context.shape[:2]
        return .2+1.8*state.sigmoid()

    def get_magnitude_dist(self,context,predict_b=None,return_b=False,inter_times=None,filter_params=None):
        b=self._get_b_pred(context)
        rate=b*np.log(10.)
        raw=(8.*torch.atanh(rate.log()/8.)).unsqueeze(-1)
        dist=MagnitudeGridDistribution(raw,'gr',1,self.density_head.mag_min,self.density_head.step)
        return (dist,b) if return_b else dist

    def sample(self,*args,**kwargs):
        self._sampling=True;self._gas_sample_state=None
        try:return super().sample(*args,**kwargs)
        finally:self._sampling=False


def load_gas_magnitude_model(path,device,base_seed=0):
    saved=json.loads(Path(path).read_text());config=saved['config']
    root=Path('/root/autodl-tmp/em_eqf')
    base=root/'experiments/revision/main/runs'/f"generation_{config['region']}_{config['variant']}_seed_{base_seed}"/'best_model_1.pth'
    original,args=load_and_prepare_model(base,device,compile=False)
    return GASMagnitudeTPP(original,saved).to(device).eval(),args
