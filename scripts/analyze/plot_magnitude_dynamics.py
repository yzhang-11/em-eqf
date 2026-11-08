"""Notebook-based views of saved, unsmoothed GR-GAS predictions."""
from pathlib import Path
import hashlib
import json

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

from src.utils.b_value_plot import plot_b_value
from src.utils.plot_style import save_pub_figure

WIDTH=183./25.4
COLORS={'gas':'#167D9A','original':'#777777','other':'#B9BEC5','gain':'#D88A38'}


def style():
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':7.5,'axes.titlesize':8.5,'axes.labelsize':8.,'xtick.labelsize':7.,
        'ytick.labelsize':7.,'legend.fontsize':7.2,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.6,'lines.linewidth':.9,'legend.frameon':False,'svg.fonttype':'none',
        'pdf.fonttype':42,'figure.dpi':120,'savefig.dpi':300})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_results(root):
    root=Path(root);base=root/'experiments/revision/magnitude_models'
    figures=root/'notebooks/figs/gr_gas_comparison';metrics=root/'notebooks/metrics/gr_gas_comparison'
    figures.mkdir(parents=True,exist_ok=True);metrics.mkdir(parents=True,exist_ok=True)
    groups={};records=[];inputs=[]
    scores=pd.read_csv(base/'reports/dynamics/test_summary.csv')
    for region,epoch in [('swc','1970-01-01'),('scsn','1981-01-01')]:
        for variant in ['main','sensitivity']:
            event_path=base/'features'/f'{region}_{variant}_events.npz'
            meta_path=event_path.with_suffix('.json');meta=json.loads(meta_path.read_text())
            with np.load(event_path) as f:
                mask=f['split']==3;times=f['times'][mask].copy();magnitude=f['magnitude'][mask].copy()
            gas_path=base/'reports/dynamics/predictions'/f'{region}_{variant}_gr_free.npz'
            with np.load(gas_path) as f:
                np.testing.assert_array_equal(f['times'],times)
                np.testing.assert_array_equal(f['magnitude'],magnitude)
                gas_b=f['value'].copy();gas_logp=f['logp'].copy()
            originals=[];original_logp=[]
            for seed in range(5):
                path=base/'features'/f'{region}_{variant}_seed_{seed}.npz'
                with np.load(path) as f:
                    originals.append(f['old_b'][mask].copy());original_logp.append(f['old_logp'][mask].copy())
                inputs.append(path)
            originals=np.asarray(originals);original_logp=np.asarray(original_logp)
            assert np.isfinite(originals).all() and np.isfinite(gas_b).all()
            assert np.all(np.diff(times)>0)
            candidate=scores[(scores.region==region)&(scores.variant==variant)&(scores.model=='gr_free')].iloc[0]
            assert abs(float(-gas_logp.mean())-candidate.nll_event)<1e-12
            epoch_number=float((pd.Timestamp(epoch)-pd.Timestamp('1970-01-01'))/pd.Timedelta(days=1))
            dates=times+epoch_number
            period=meta['periods']['test']
            item=dict(region=region,variant=variant,mc=float(meta['Mc']),times=times,dates=dates,
                magnitude=magnitude,gas_b=gas_b,original_b=originals,gas_logp=gas_logp,
                original_logp=original_logp,start=period['start']+epoch_number,end=period['end']+epoch_number,
                days=period['days'],events=len(times))
            groups[(region,variant)]=item
            calendar_dates=pd.to_datetime(dates,unit='D',origin='unix')
            frame=pd.DataFrame(dict(relative_days=times,date=calendar_dates,magnitude=magnitude,
                gas_b=gas_b,gas_logp=gas_logp,**{f'original_b_seed_{s}':originals[s] for s in range(5)},
                **{f'original_logp_seed_{s}':original_logp[s] for s in range(5)}))
            frame.to_csv(metrics/f'{region}_{variant}_source_data.csv',index=False)
            records.append(dict(region=region,variant=variant,Mc=meta['Mc'],events=len(times),
                original_fits=5,new_fits=1,zoom_days=180.,zoom_events=int((dates<item['start']+180.).sum()),
                excluded_events=0,source_precision='float64 relative days retained; calendar offset added for display'))
            inputs += [event_path,meta_path,gas_path]
    for name in ['test_summary.csv','b_variation_fit_scores.csv','initialization_validation_scores.csv','scale_sensitivity_validation.csv']:
        path=base/'reports/dynamics'/name;inputs.append(path)
        pd.read_csv(path).to_csv(metrics/name,index=False)
    contract=dict(conclusion='GAS reduces adjacent jumps; long-term b variation and cumulative gains must be inspected separately',
        archetype='quantitative grid',backend='Python matplotlib, existing notebook/helper workflow',
        figure_width_mm=183,formats=['png','pdf','svg'],png_dpi=300,
        data=records,inputs=[dict(path=str(q),sha256=sha(q)) for q in inputs],
        reuse='b(t) structure from b_t_estimation; shared plot_b_value and save_pub_figure; cumulative-likelihood notebook comparison logic',
        smoothing=False,downsampling=False,zoom='first 180 calendar days, fixed independently of model outputs',
        uncertainty='all five original fitted traces shown; fit variation is not a confidence interval',
        inference='existing paired time-block intervals include zero; plots do not establish significance',
        time_axis='b_i before mark i is drawn at target event time; connecting lines are visualization, not a continuous-time model')
    (metrics/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    return dict(root=root,base=base,figures=figures,metrics=metrics,groups=groups,contract=contract)


def export(fig,results,name):
    target=results['figures']/name
    save_pub_figure(fig,target.with_suffix('.png'),dpi=300,file_format='png')
    save_pub_figure(fig,target.with_suffix('.pdf'),file_format='pdf')
    fig.savefig(target.with_suffix('.svg'),bbox_inches='tight')
    return fig


def calendar_axis(axis):
    # Use explicit calendar ticks: date64-to-datetime conversion in this runtime
    # offsets UTC-midnight labels; floating source-day coordinates stay intact.
    left,right=map(float,axis.get_xlim())
    first=pd.to_datetime(left,unit='D',origin='unix')
    last=pd.to_datetime(right,unit='D',origin='unix')
    if right-left>730.:
        interval=max(1,int(np.ceil((last.year-first.year)/4.)))
        dates=pd.DatetimeIndex([pd.Timestamp(year=y,month=1,day=1) for y in range(first.year,last.year+1,interval)])
        labels=dates.strftime('%Y')
    else:
        interval=max(1,int(np.ceil((right-left)/30./4.)))
        dates=pd.date_range(first.to_period('M').start_time,last,freq=f'{interval}MS')
        labels=dates.strftime('%Y-%m')
    positions=((dates-pd.Timestamp('1970-01-01'))/pd.Timedelta(days=1)).to_numpy(dtype=float)
    keep=(positions>=left-1e-8)&(positions<=right+1e-8)
    axis.xaxis.set_major_locator(ticker.FixedLocator(positions[keep]))
    axis.xaxis.set_major_formatter(ticker.FixedFormatter(labels[keep]))
    axis.xaxis.set_minor_locator(ticker.NullLocator())
    axis.set_xlabel('Date');axis.grid(axis='y',alpha=.18,linewidth=.5)


def add_b_trace(axis,dates,values,color,label,linewidth,alpha=1.):
    # Use the shared notebook b-value plotting implementation without smoothing.
    # Calendar offsets are supplied as floating matplotlib days, preserving sub-day positions.
    seq={'arrival_times':torch.tensor(dates,dtype=torch.float64)}
    plot_b_value(seq,torch.as_tensor(values),fig=axis.figure,ax=axis,show=False,
        smooth=False,start_time=None,label=label)
    line=axis.lines[-1];line.set_color(color);line.set_linewidth(linewidth);line.set_alpha(alpha)
    if axis.legend_ is not None:axis.legend_.remove()


def plot_b_timeline(results,variant='main'):
    style();fig,axes=plt.subplots(2,2,figsize=(WIDTH,4.3),sharey=True)
    for row,region in enumerate(['swc','scsn']):
        group=results['groups'][(region,variant)]
        for col in range(2):
            axis=axes[row,col]
            for seed in [1,2,3,4,0]:
                add_b_trace(axis,group['dates'],group['original_b'][seed],
                    COLORS['original'] if seed==0 else COLORS['other'],
                    f'NN-GR seed {seed}',.72 if seed==0 else .4,1. if seed==0 else .65)
            add_b_trace(axis,group['dates'],group['gas_b'],COLORS['gas'],'GR-GAS',.95)
            finish=group['end'] if col==0 else min(group['start']+180.,group['end'])
            axis.set_xlim(group['start'],finish);axis.set_ylim(.15,2.05)
            calendar_axis(axis);axis.set_ylabel('Predicted b')
            scope=f"Full test, n={group['events']:,}" if col==0 else f"First 180 days, n={(group['dates']<finish).sum():,}"
            axis.set_title(f"{region.upper()}  Mc={group['mc']:.1f} | {scope}",loc='left')
            axis.text(-.14,1.045,chr(97+2*row+col),transform=axis.transAxes,fontweight='bold',fontsize=9.)
    handles=[Line2D([],[],color=COLORS['original'],lw=.9,label='NN-GR seed 0'),
        Line2D([],[],color=COLORS['other'],lw=.7,label='Other 4 NN-GR fits'),
        Line2D([],[],color=COLORS['gas'],lw=1.1,label='GR-GAS')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,1.035),ncol=3)
    fig.text(.5,-.005,'Raw pre-event predictions; no post-processing or smoothing.',ha='center',fontsize=7.)
    fig.subplots_adjust(left=.09,right=.99,bottom=.10,top=.91,hspace=.53,wspace=.20)
    return export(fig,results,f'gr_gas_b_time_{variant}')


def plot_b_metrics(results):
    style();frame=pd.read_csv(results['metrics']/'b_variation_fit_scores.csv')
    groups=[('swc','main'),('swc','sensitivity'),('scsn','main'),('scsn','sensitivity')]
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,2.55),sharey=True)
    keys=['mean_abs_delta_b','b_sd'];labels=['Mean adjacent |Δb|','Whole-period SD of b']
    for i,(axis,key,label) in enumerate(zip(axes,keys,labels)):
        for j,(region,variant) in enumerate(groups):
            part=frame[(frame.region==region)&(frame.variant==variant)]
            old=part[part.model=='original_GR'][key].to_numpy()
            new=float(part[part.model=='GR_GAS'][key].iloc[0])
            mean=float(old.mean())
            axis.plot([mean,new],[j,j],color='#BAC0C6',lw=1.)
            axis.scatter(old,np.full(len(old),j),s=12,facecolor='none',edgecolor=COLORS['other'],linewidth=.7)
            axis.scatter([mean],[j],s=30,facecolor='white',edgecolor=COLORS['original'],linewidth=1.,zorder=4)
            axis.scatter([new],[j],s=34,marker='D',color=COLORS['gas'],zorder=5)
        axis.set_xlabel(label);axis.set_xlim(left=0);axis.grid(axis='x',alpha=.18)
        axis.set_title('Short-term jumps' if i==0 else 'Long-term variation',loc='left')
        axis.text(-.14,1.045,chr(97+i),transform=axis.transAxes,fontweight='bold',fontsize=9.)
    axes[0].set_yticks(range(4),[f"{r.upper()}, Mc={results['groups'][(r,v)]['mc']:.1f}" for r,v in groups]);axes[0].invert_yaxis()
    handles=[Line2D([],[],marker='o',ls='',mfc='white',mec=COLORS['original'],label='NN-GR mean (5 fits)'),
             Line2D([],[],marker='D',ls='',color=COLORS['gas'],label='GR-GAS')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.56,1.055),ncol=2)
    fig.text(.55,-.025,'Small open circles: individual original fits; spread is not a confidence interval.',ha='center',fontsize=7.)
    fig.subplots_adjust(left=.18,right=.99,bottom=.23,top=.82,wspace=.25)
    return export(fig,results,'gr_gas_b_variation')


def plot_cumulative_gain(results):
    style();fig,axes=plt.subplots(1,2,figsize=(WIDTH,2.65))
    curves=[]
    for i,region in enumerate(['swc','scsn']):
        group=results['groups'][(region,'main')];axis=axes[i]
        gains=np.cumsum(group['gas_logp'][None,:]-group['original_logp'],axis=1)
        dates=np.r_[group['start'],group['dates']]
        for seed in range(5):
            axis.plot(dates,np.r_[0.,gains[seed]],color=COLORS['other'],lw=.6,alpha=.8)
        axis.plot(dates,np.r_[0.,gains.mean(0)],color=COLORS['gas'],lw=1.1)
        axis.axhline(0,color='#404040',ls='--',lw=.6)
        axis.set_xlim(group['start'],group['end']);calendar_axis(axis)
        axis.set_ylabel('Cumulative log gain (nats)')
        axis.set_title(f"{region.upper()}  Mc={group['mc']:.1f}",loc='left')
        axis.text(-.065,1.08,chr(97+i),transform=axis.transAxes,fontweight='bold',fontsize=9.)
        for seed in range(5):
            curves.append(pd.DataFrame(dict(region=region,seed=seed,relative_days=group['times'],cumulative_gain=gains[seed])))
    pd.concat(curves,ignore_index=True).to_csv(results['metrics']/'cumulative_gain_curves.csv',index=False)
    handles=[Line2D([],[],color=COLORS['other'],lw=.8,label='Against each original fit'),
        Line2D([],[],color=COLORS['gas'],lw=1.1,label='Mean gain')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,1.065),ncol=2)
    fig.text(.52,-.035,'Positive favors GAS. Paired time-block intervals include zero.',ha='center',fontsize=7.)
    fig.subplots_adjust(left=.105,right=.99,bottom=.23,top=.82,wspace=.34)
    return export(fig,results,'gr_gas_cumulative_gain')


def plot_sensitivity(results):
    style();initial=pd.read_csv(results['metrics']/'initialization_validation_scores.csv')
    scales=pd.read_csv(results['metrics']/'scale_sensitivity_validation.csv')
    fig,axes=plt.subplots(2,2,figsize=(WIDTH,4.55),sharex='row',sharey='row')
    for col,region in enumerate(['swc','scsn']):
        part=initial[(initial.region==region)&(initial.variant=='main')&(initial.family=='gr_free')].sort_values('initial_index')
        axis=axes[0,col]
        x=np.arange(1,6)
        y=(part.validation_nll.to_numpy()-part.validation_nll.min())*1e6
        axis.plot(x,y,'o-',color=COLORS['gas'],lw=.9,ms=3.5)
        axis.set_xticks(x,[str(v) for v in x])
        axis.set_xlabel('Optimization start (H and α varied)')
        axis.set_ylabel('Δ validation NLL (×10⁻⁶)')
        mc=results['groups'][(region,'main')]['mc']
        axis.set_title(f"{region.upper()}  Mc={mc:.1f} | 5 starts",loc='left')
        axis.set_ylim(-.05,2.15);axis.grid(axis='y',alpha=.18)
        axis=axes[1,col]
        part=scales[(scales.region==region)&(scales.variant=='main')&(scales.family=='gr_free')]
        for key,color,label,marker in [('half_life',COLORS['gas'],'Memory H','o'),('gain',COLORS['gain'],'Update α','s')]:
            points=part[part.axis==key].sort_values('multiplier')
            assert points.within_model_constraints.all()
            assert (points.multiplier>0).all(), 'Log-axis multipliers must be positive.'
            base=float(points[points.multiplier==1.].validation_nll.iloc[0])
            axis.plot(points.multiplier,points.validation_nll-base,color=color,marker=marker,lw=.9,ms=3.,label=label)
        axis.axhline(0,color='#555555',ls='--',lw=.6)
        axis.set_xscale('log',base=2);axis.set_xticks([.25,.5,1.,2.,4.],['0.25','0.5','1','2','4'])
        axis.set_xlabel('Multiplier of learned coefficient')
        axis.set_ylabel('Δ validation NLL (nats/event)')
        axis.set_title('Post-fit perturbation, no refitting',loc='left');axis.grid(axis='y',alpha=.18)
        for row in range(2):axes[row,col].text(-.065,1.08,chr(97+row*2+col),transform=axes[row,col].transAxes,fontweight='bold',fontsize=9.)
    handles,labels=axes[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.55,-.015),ncol=2)
    fig.subplots_adjust(left=.105,right=.99,bottom=.12,top=.93,hspace=.62,wspace=.30)
    return export(fig,results,'gr_gas_sensitivity')
