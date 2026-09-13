"""Scientific figures for the separately recorded trajectory follow-up."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'production/trajectory_distribution_2026_09_13'
CACHE=ROOT/'production/assets/trajectory_distribution_2026_09_13'
FAMILIES=['spline_pca','cvae','conditional_ae','unconditional_vae','condition_ridge']
LABELS={'spline_pca':'Spline + PCA','cvae':'CVAE','conditional_ae':'CAE','unconditional_vae':'VAE','condition_ridge':'Condition Ridge'}
COLORS={'spline_pca':'#bb7226','cvae':'#27827b','conditional_ae':'#965382','unconditional_vae':'#365a8c','condition_ridge':'#7c7c7c'}


def main():
    people=pd.read_csv(OUT/'participant_scores.csv')
    for geometry in ['raw_rms','axis_balanced']:
        fig,axes=plt.subplots(2,2,figsize=(11,7.1))
        for row,dim in enumerate([3,8]):
            for col,metric in enumerate(['energy','mmd2']):
                ax=axes[row,col]
                part=people[(people.geometry==geometry)&((people.latent_dim==dim)|(people.model_family=='condition_ridge'))]
                for j,family in enumerate(FAMILIES):
                    vals=part[part.model_family==family].sort_values('subject')[metric].to_numpy()
                    jitter=np.random.default_rng(2026).uniform(-.16,.16,len(vals))
                    ax.scatter(j+jitter,vals,s=16,alpha=.58,color=COLORS[family],edgecolors='none')
                    ax.plot([j-.23,j+.23],[vals.mean()]*2,color='black',lw=2)
                refs=people[(people.geometry==geometry)&(people.model_family=='context_resample')][metric]
                ax.axhline(refs.mean(),color='#238642',ls='--',lw=1.2,label='Context-resampling mean')
                ax.axhline(0,color='gray',lw=.6)
                ax.set_xticks(range(len(FAMILIES)),[LABELS[f] for f in FAMILIES],rotation=15,ha='right',fontsize=9)
                ax.set_title(f'n = {dim} | '+('Energy discrepancy' if metric=='energy' else 'Unbiased MMD squared'))
                ax.set_ylabel('Tracker units' if geometry=='raw_rms' and metric=='energy' else 'Dimensionless')
                ax.grid(axis='y',alpha=.2)
        axes[0,0].legend(frameon=False,fontsize=8)
        title='Complete normalized paths: common spatial units' if geometry=='raw_rms' else 'Sensitivity: spatial axes scaled by training variation'
        fig.suptitle(title,fontsize=14)
        fig.text(.5,.012,'Each point: one participant after seed averaging. Black bars: means across 28 participants. Lower discrepancy is better.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.035,1,.96))
        fig.savefig(OUT/f'{geometry}_comparison.png',dpi=220)
        plt.close(fig)
    # Fixed first participant and seed, independent of the new score rankings.
    paths={
        'spline_pca':CACHE/'spline_pca_z8_fold2_subject01.npz',
        'cvae':CACHE/'cvae_z8_seed42_fold2_subject01.npz',
        'unconditional_vae':CACHE/'unconditional_vae_z8_seed42_fold2_subject01.npz',
    }
    samples={family:np.load(path) for family,path in paths.items()}
    curves=[samples['spline_pca']['recorded'],*[samples[f]['generated'] for f in paths]]
    names=['Recorded query','Spline + PCA','CVAE','VAE']
    colors=['black',*[COLORS[f] for f in paths]]
    fig,axes=plt.subplots(2,4,figsize=(12,5.3),sharex=True,sharey='row')
    phase=np.linspace(0,1,100)
    for col,(values,name,color) in enumerate(zip(curves,names,colors)):
        for axis in [0,1]:
            ax=axes[axis,col]
            low,high=np.quantile(values[:,:,axis],[.1,.9],axis=0)
            ax.fill_between(phase,low,high,color=color,alpha=.18)
            ax.plot(phase,values[:,:,axis].mean(0),color=color,lw=1.6)
            ax.grid(alpha=.2)
            if axis==0: ax.set_title(name)
            else: ax.set_xlabel('Normalized window phase')
    axes[0,0].set_ylabel('Lateral x (tracker units)')
    axes[1,0].set_ylabel('Forward y (tracker units)')
    fig.suptitle('subject01 | n = 8 | neural seed 42 | fold 2',fontsize=13)
    fig.text(.5,.01,'Lines: mean coordinate paths. Shading: pointwise 10th-90th percentiles of movements, not confidence intervals.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95))
    fig.savefig(OUT/'mean_and_spread_example.png',dpi=220)
    plt.close(fig)
    print('Rendered three trajectory follow-up figures')


if __name__=='__main__':main()
