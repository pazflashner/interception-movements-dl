"""Measured trajectory examples and feature ECDFs for the visual course revision."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from production.course_evidence import *
from production.course_figures import COLORS

SAVE=OUT/'course_assets'
FEATURE_NAMES={'initiation_time_s':'Initiation time (s)','movement_time_s':'Movement time (s)','peak_speed_tracker_units_s':'Peak speed (units/s)','time_to_peak_speed':'Relative time to peak','path_length':'Path length (units)','straight_line_dist':'Straight-line distance (units)','curvature_index':'Path / straight-line distance','max_lateral_deviation':'Maximum deviation (units)','n_submovements':'Speed-peak count','end_x':'Endpoint x (units)','end_y':'Endpoint y (units)'}

def save(fig,name):
    fig.savefig(SAVE/name,dpi=240,bbox_inches='tight');plt.close(fig)

def main():
    from scripts.run_review_controls import load_trials
    from scripts.run_corrected_study import load_per_trial_checkpoint
    from src.evaluate import reconstruct
    from src.baseline_spline import SplinePCARepresentation
    from threadpoolctl import threadpool_limits
    import torch
    torch.set_num_threads(1);threadpool_limits(1)
    trials=load_trials();source=ROOT/'studies/final_strategy_evaluation/runs'
    provenance=json.loads((OUT/'data_figure_provenance.json').read_text())
    trial=next(t for t in trials if t['metadata']['trial_id']==provenance['trial_id'])
    split=json.loads((source/'cvae/fold2/cvae_z3_seed42/split.json').read_text())
    train=[t for t in trials if t['metadata']['subject'] in split['train_subjects']]
    manifest=[];recons={}
    for n in (3,8):
        for f in FAMILIES:
            if f=='spline_pca':
                run=source/f/f'fold2/spline_pca_z{n}'
                settings=json.loads((run/'result.json').read_text())
                rep=SplinePCARepresentation(n,include_timing=False,standardize_coefficients=settings['standardize_spline_coefficients']).fit(train)
                pred=rep.decode(rep.encode([trial]))[0][0]
            else:
                model,norm=load_per_trial_checkpoint(source/f/f'fold2/{f}_z{n}_seed42/checkpoint.pt','cpu')
                pred=reconstruct(model,[trial],norm,'cpu')[0].reshape(100,2)
            recons[n,f]=pred
            manifest.append(dict(model=f,n=n,subject='subject01',fold=2,seed=None if f=='spline_pca' else 42,trial_id=provenance['trial_id'],mse=float(np.mean((pred-trial['pos_norm'])**2))))
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axs=plt.subplots(2,4,figsize=(7.8,4.8),sharex=True,sharey=True,layout='constrained')
        for row,n in enumerate((3,8)):
            for col,f in enumerate(FAMILIES):
                ax=axs[row,col];p=recons[n,f];real=trial['pos_norm']
                ax.plot(real[:,0],real[:,1],c='black',lw=1.6,label='Recorded')
                ax.plot(p[:,0],p[:,1],c=COLORS[col],ls='--',lw=1.5,label='Reconstruction')
                ax.set_aspect('equal',adjustable='box');ax.set_title(f'{LABELS[f]}, n={n}',fontsize=10)
                ax.set_xlim(-2.8,2.8);ax.set_xticks([-2,2])
                ax.set_xlabel('x');ax.set_ylabel('y (units)' if col==0 else '')
        fig.legend(*axs[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
        save(fig,'reconstruction_examples.png')
        fig,axs=plt.subplots(1,4,figsize=(7.6,3.5),sharex=True,sharey=True,layout='constrained')
        for ax,(n,f) in zip(axs,[(3,'spline_pca'),(3,'unconditional_vae'),(8,'spline_pca'),(8,'unconditional_vae')]):
            p=recons[n,f];real=trial['pos_norm'];col=COLORS[FAMILIES.index(f)]
            ax.plot(real[:,0],real[:,1],c='black',lw=2,label='Recorded')
            ax.plot(p[:,0],p[:,1],c=col,ls='--',lw=1.8,label='Reconstruction')
            ax.set_aspect('equal',adjustable='box');ax.set_xlim(-2.8,2.8);ax.set_xticks([-2,2]);ax.set_xlabel('x')
            ax.set_title(f'{LABELS[f]}\nn = {n}',fontsize=12)
        axs[0].set_ylabel('y (tracker units)')
        fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
        save(fig,'reconstruction_slide.png')
        fig,axs=plt.subplots(1,2,figsize=(7.8,2.6),layout='constrained')
        for axis in (0,1):
            ax=axs[axis];real=trial['pos_norm'];phase=np.linspace(0,1,100)
            ax.plot(phase,real[:,axis],c='black',lw=1.8,label='Recorded')
            for n,ls in [(3,'--'),(8,'-')]:
                for f,col in [('spline_pca',COLORS[0]),('unconditional_vae',COLORS[3])]:ax.plot(phase,recons[n,f][:,axis],c=col,ls=ls,lw=1.2,label=f'{LABELS[f]}, n={n}')
            ax.set_xlabel('Normalized time (phase)');ax.set_ylabel(('x' if axis==0 else 'y')+' (units)')
        fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False,fontsize=8)
        save(fig,'reconstruction_phase.png')
        fig,axs=plt.subplots(2,5,figsize=(8.4,4.8),sharex=True,sharey=True,layout='constrained')
        for row,n in enumerate((3,8)):
            groups=[]
            for f in FAMILIES:
                a=np.load(OUT/'assets/trajectory_distribution_2026_09_13'/f'{f}_z{n}_{"" if f=="spline_pca" else "seed42_"}fold2_subject01.npz')
                groups.append(a['generated'])
            groups=[a['recorded']]+groups
            for ax,paths,label,col in zip(axs[row],groups,['Recorded']+[LABELS[f] for f in FAMILIES],['black']+COLORS):
                for p in paths[:30]:ax.plot(p[:,0],p[:,1],c=col,lw=.8,alpha=.30)
                ax.plot(paths.mean(0)[:,0],paths.mean(0)[:,1],c=col,lw=1.6)
                ax.set_aspect('equal',adjustable='box');ax.set_title(f'{label}, n={n}' if label!='Recorded' else label,fontsize=9)
                ax.set_xticks([-2,2]);ax.set_xlabel('x')
            axs[row,0].set_ylabel(f'n={n}: y (units)')
        save(fig,'generation_both_dimensions.png')
        fig,axs=plt.subplots(1,5,figsize=(10.5,3.5),sharex=True,sharey=True,layout='constrained')
        for ax,paths,label,col in zip(axs,groups,['Recorded']+[LABELS[f] for f in FAMILIES],['black']+COLORS):
            for p in paths[:30]:ax.plot(p[:,0],p[:,1],c=col,lw=1,alpha=.30)
            mean=paths.mean(0);ax.plot(mean[:,0],mean[:,1],c=col,lw=1.8)
            ax.set_aspect('equal',adjustable='box');ax.set_title(label,fontsize=13);ax.set_xticks([-2,2]);ax.set_xlabel('x',fontsize=11)
        axs[0].set_ylabel('y (tracker units)',fontsize=11)
        save(fig,'generation_slide_n8.png')
        fig,axs=plt.subplots(1,2,figsize=(7.5,2.8),layout='constrained')
        axs[0].imshow(plt.imread(OUT/'presentation_assets/task_schematic.png'));axs[0].axis('off');axs[0].set_title('Task display')
        for t in trials:
            if t['metadata']['subject']=='subject01':
                p=t['pos_norm'];axs[1].plot(p[:,0],p[:,1],alpha=.2,c='#087c83',lw=.7)
        axs[1].set_aspect('equal',adjustable='box');axs[1].set_title('subject01: 180 recorded trials');axs[1].set_xlabel('x (tracker units)');axs[1].set_ylabel('y (tracker units)')
        save(fig,'task_and_recordings.png')
        for n in (3,8):
            data={f:np.load(OUT/'assets/feature_redundancy_2026_09_14'/f'{f}_z{n}_{"" if f=="spline_pca" else "seed42_"}fold2_subject01.npz') for f in FAMILIES}
            features=list(data['cvae']['features']);real=data['cvae']['recorded_features']
            for fs,name,shape,size in [(features,f'features_all_n{n}.png',(4,3),(8.0,8.2)),(['initiation_time_s','movement_time_s','peak_speed_tracker_units_s','curvature_index'],f'features_selected_n{n}.png',(2,2),(7.8,3.7))]:
                fig,axs=plt.subplots(*shape,figsize=size,layout='constrained',sharey=True)
                for ax,f in zip(axs.flat,fs):
                    j=features.index(f)
                    for vals,label,col in [(real[:,j],'Recorded','black')]+[(data[k]['generated_features'][:,j],LABELS[k],COLORS[i]) for i,k in enumerate(FAMILIES)]:
                        vals=np.sort(vals);ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',label=label,c=col,lw=1.5 if label=='Recorded' else 1.1)
                    ax.set_xlabel(FEATURE_NAMES[f]);ax.set_ylim(0,1.03);ax.grid(alpha=.14)
                for ax in list(axs.flat)[len(fs):]:ax.axis('off')
                for ax in axs[:,0]:ax.set_ylabel('Cumulative proportion')
                fig.legend(*axs[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=5,frameon=False,fontsize=8)
                save(fig,name)
    (SAVE/'visual_examples_provenance.json').write_text(json.dumps(dict(reconstruction_selection=provenance['selection'],reconstruction_examples=manifest,generation_subject='subject01',generation_selection='First 30 stored draws and query paths; identifier-selected participant; no ranking of generation error',feature_panels='Timing, peak speed and curvature selected for distinct scientific meanings, not goodness of fit',equal_xy_axes=True),indent=2)+'\n')

if __name__=='__main__':main()
