"""Figures from frozen auxiliary results; no optimizer or model training."""
import json,pickle,shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch
from production.remaining_evidence import ROOT,RESULTS,AUDIT,CONTROL,FAMILIES,LABELS,TARGETS,FEATURES

COLORS={'spline_pca':'#bb7226','cvae':'#27827b','conditional_ae':'#965382','unconditional_vae':'#365a8c'}


def save(fig,assets,name):
    fig.savefig(assets/(name+'.png'),dpi=300,bbox_inches='tight');plt.close(fig)


def remaining_figures(e,assets):
    fig,axs=plt.subplots(1,2,figsize=(7,3.1),sharey=True)
    for ax,(name,frame) in zip(axs,[('CVAE n=3',e['fingerprint_people'].query('latent_dim == 3')),('VAE n=8',e['uvae_fp_people'])]):
        wide=frame.pivot(index='subject',columns='arm',values='mean_ks').sort_index()
        for j,arm in enumerate(['population','wrong']):
            d=wide[arm]-wide['own'];ax.scatter(j+np.linspace(-.12,.12,len(d)),d,s=15,alpha=.7,color=['#bb7226','#365a8c'][j])
            ax.plot([j-.18,j+.18],[d.mean()]*2,color='black',lw=2)
        ax.axhline(0,color='grey',lw=.8,ls='--');ax.set_xticks([0,1],['Population','Other person']);ax.set_title(name);ax.grid(axis='y',alpha=.15)
    axs[0].set_ylabel('Control KS minus own KS');fig.tight_layout();save(fig,assets,'fingerprint_controls')
    fig,axs=plt.subplots(2,2,figsize=(7,4.3),sharex=True)
    for col,dim in enumerate([3,8]):
        for row,endpoint in enumerate(['initiation','movement']):
            ax=axs[row,col]
            for family in ['spline_pca','cvae']:
                primary=e['means'].query('model_family == @family and latent_dim == @dim').iloc[0][endpoint+'_time_mae_ms']
                sub=e['timing'].query('model_family == @family and latent_dim == @dim and endpoint == @endpoint').set_index('comparison')
                vals=[primary,sub.loc['common_mlp','mae_ms_mean'],sub.loc['validation_calibrated_original','mae_ms_mean']]
                ax.plot(range(3),vals,'o-',color=COLORS[family],label=LABELS[family],ms=4)
            ax.set_title(f'{endpoint.capitalize()} | n={dim}',fontsize=10);ax.set_xticks(range(3),['Original','Common MLP','Calibrated']);ax.set_ylabel('MAE (ms)');ax.grid(alpha=.15);ax.tick_params(labelsize=8)
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.07,1,1));save(fig,assets,'timing_heads')
    # All 14 outcomes, including negative scores. Color clipping never clips labels.
    probes=e['probes'].query('fingerprint == "mean" and latent_dim in [3,8]')
    columns=[(f,n) for n in [3,8] for f in FAMILIES]
    values=np.array([[probes.set_index(['model_family','latent_dim','target']).loc[(f,n,t),'r2_oof_mean'] for f,n in columns] for t in TARGETS])
    fig,ax=plt.subplots(figsize=(8.1,5.7));im=ax.imshow(values,vmin=-1,vmax=1,cmap='RdBu',aspect='auto')
    ax.set_xticks(range(8),[f'{LABELS[f]}\nn={n}' for f,n in columns],fontsize=8)
    ax.set_yticks(range(14),[FEATURES[t.rsplit('_',1)[0]]+' ('+t.rsplit('_',1)[1]+')' for t in TARGETS],fontsize=8)
    for i in range(14):
        for j in range(8):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if abs(values[i,j])>.65 else 'black')
    ax.axvline(3.5,color='white',lw=2);fig.colorbar(im,ax=ax,label='Pooled out-of-fold R² (color range -1 to 1)',shrink=.85)
    fig.tight_layout();save(fig,assets,'probe_scores')
    # Cohort-weighted count proportions, with seed averaging for generated rows.
    counts=[]
    real=e['component_raw'].query('kind == "recorded"')
    for k in range(1,5):counts.append({'label':'Recorded','k':k,'p':real.assign(hit=real.mj_n_components==k).groupby('subject').hit.mean().mean()})
    for label,frame in [('CVAE n=3',e['component_raw'].query('kind == "generated" and latent_dim == 3')),('CVAE n=8',e['component_raw'].query('kind == "generated" and latent_dim == 8')),('VAE n=8',e['uvae_component_raw'])]:
        for k in range(1,5):counts.append({'label':label,'k':k,'p':frame.assign(hit=frame.mj_n_components==k).groupby(['subject','seed']).hit.mean().groupby('subject').mean().mean()})
    counts=pd.DataFrame(counts);counts.to_csv(ROOT/'production/component_count_plot_values.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,2.9))
    for i,(label,color) in enumerate(zip(['Recorded','CVAE n=3','CVAE n=8','VAE n=8'],['black','#72ada6','#27827b','#365a8c'])):
        vals=counts.query('label == @label').sort_values('k');ax.bar(np.arange(1,5)+(i-1.5)*.19,vals.p,width=.18,label=label,color=color)
    ax.set_xticks([1,2,3,4]);ax.set_xlabel('Selected minimum-jerk components');ax.set_ylabel('Participant-balanced proportion');ax.legend(ncol=2,frameon=False,fontsize=8);ax.grid(axis='y',alpha=.15)
    fig.tight_layout();save(fig,assets,'component_counts')
    shutil.copyfile(RESULTS/'event_audit/pre_go_examples.png',assets/'event_examples.png')
    latent_figure(assets)
    component_figure(e,assets)
    dashboard_diagram(assets)


def latent_figure(assets):
    from scripts.run_review_controls import load_trials
    from scripts.run_corrected_study import load_per_trial_checkpoint,context_query_for_trials
    from src.evaluate import encode_trials
    import config
    trials=load_trials();source=ROOT/'studies/final_strategy_evaluation/runs'
    split=json.loads((source/'cvae/fold0/cvae_z3_seed42/split.json').read_text())
    test=[t for t in trials if t['metadata']['subject'] in split['test_subjects']]
    parts=context_query_for_trials(test,config.CONTEXT_QUERY_SEED)
    fig=plt.figure(figsize=(8,3.8));palette=plt.get_cmap('tab10');manifest=[]
    for panel,family in enumerate(['cvae','unconditional_vae']):
        model,norm=load_per_trial_checkpoint(source/f'{family}/fold0/{family}_z3_seed42/checkpoint.pt','cpu');model.eval()
        mu=encode_trials(model,test,norm,'cpu')[0];ax=fig.add_subplot(1,2,panel+1,projection='3d')
        for j,p in enumerate(parts):
            q=mu[p.query_indices];c=mu[p.context_indices].mean(0)
            ax.scatter(*q.T,s=8,alpha=.38,color=palette(j),depthshade=False)
            ax.scatter(*c,s=75,marker='X',color=palette(j),edgecolors='black',linewidths=.5,label=p.subject)
            manifest.append(dict(family=family,fold=0,seed=42,subject=p.subject,n_context=len(p.context_indices),n_query=len(p.query_indices)))
        ax.set_title(LABELS[family]+' n=3 | fold 0, seed 42',fontsize=9)
        ax.set_xlabel('z1',fontsize=8);ax.set_ylabel('z2',fontsize=8);ax.set_zlabel('z3',fontsize=8);ax.tick_params(labelsize=7);ax.view_init(elev=23,azim=-55)
    handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=7,frameon=False,fontsize=7)
    fig.subplots_adjust(bottom=.14,wspace=.08);save(fig,assets,'latent_three_dimensions')
    (ROOT/'production/latent_figure_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')


def component_figure(e,assets):
    from scripts.run_review_controls import prepare_matched_movement
    from src.submovements import _prepare_velocity,reconstruct_velocity,minimum_jerk_velocity
    old=pickle.load((CONTROL/'component_inputs.pkl').open('rb'))
    new=pickle.load((AUDIT/'uvae8_component_inputs.pkl').open('rb'))
    chosen=[]
    for label,items,frame,kind in [('Recorded',old,e['component_raw'],'recorded'),('CVAE n=8',old,e['component_raw'],'generated'),('VAE n=8',new,e['uvae_component_raw'],'generated')]:
        subset=[(r,w) for r,w,_,_ in items if r['subject']=='subject01' and r['kind']==kind and (kind=='recorded' or (r['latent_dim']==8 and r['seed']==42))]
        r,w=min(subset,key=lambda item:str(item[0]['job_id']))
        fitted=frame.set_index('job_id').loc[r['job_id']];p=np.array(json.loads(fitted.mj_parameters_json))
        movement,hz=prepare_matched_movement(w,r['movement_time_s'],r['initiation_time_s']);t,v=_prepare_velocity(movement,hz,10)
        end=max(t[-1],float(np.max(p[:,0]+p[:,1])));time=np.arange(0,end+.5/hz,1/hz)
        observed=np.zeros((len(time),2));observed[:len(v)]=v
        predicted=reconstruct_velocity(time,p)
        chosen.append((label,r,fitted,p,time,observed,predicted,t[-1]))
    fig,axs=plt.subplots(1,3,figsize=(8,3.1),sharey=True)
    for ax,(label,r,fitted,p,time,observed,predicted,end) in zip(axs,chosen):
        ax.plot(time,np.linalg.norm(observed,axis=1),color='black',lw=1.3,label='Observed input')
        ax.plot(time,np.linalg.norm(predicted,axis=1),color='#365a8c',lw=1.2,label='Fitted sum')
        for i,(onset,duration,x,y) in enumerate(p):
            ax.plot(time,np.linalg.norm(minimum_jerk_velocity(time,onset,duration,[x,y]),axis=1),ls='--',lw=.9,label=f'Component {i+1}')
        ax.axvline(end,color='grey',lw=.7,ls=':');ax.set_title(f'{label} | k={len(p)}\nE={fitted.mj_fit_error:.3f}',fontsize=9);ax.set_xlabel('Time from movement onset (s)',fontsize=8);ax.tick_params(labelsize=7);ax.grid(alpha=.15)
    axs[0].set_ylabel('Speed (tracker units/s)',fontsize=8);axs[-1].legend(fontsize=6,frameon=False)
    fig.tight_layout();save(fig,assets,'component_examples')
    (ROOT/'production/component_figure_provenance.json').write_text(json.dumps([dict(label=x[0],job_id=x[1]['job_id'],selected_count=len(x[3]),normalized_error=float(x[2].mj_fit_error)) for x in chosen],indent=2)+'\n')


def dashboard_diagram(assets):
    fig,ax=plt.subplots(figsize=(8,2.7));ax.set_xlim(0,10);ax.set_ylim(0,3);ax.axis('off')
    boxes=[(.1,1.7,2.8,'Choose model and capacity\nVAE8 default; CVAE / CAE / spline'),(3.5,1.7,2.8,'Choose a fingerprint\nTraining center or enrolled person'),(6.9,1.7,2.8,'Explore latent coordinates\nInspect decoded path and timing'),(.1,.15,2.8,'Task controls where supported\nConditional models; spline timing'),(3.5,.15,2.8,'Sample a distribution\nShared covariance; chosen seed'),(6.9,.15,2.8,'Compare and export\nHeld-out scores and feature plots')]
    for x,y,w,txt in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,.95,boxstyle='round,pad=.04',facecolor='#f3f5f4',edgecolor='#50756e',lw=.8));ax.text(x+w/2,y+.475,txt,ha='center',va='center',fontsize=8)
    for x1,y1,x2,y2 in [(2.95,2.18,3.45,2.18),(6.35,2.18,6.85,2.18),(8.3,1.65,8.3,1.2),(4.9,1.65,4.9,1.2),(2.95,.62,3.45,.62),(6.35,.62,6.85,.62)]:
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='->',mutation_scale=12,color='#50756e'))
    fig.tight_layout();save(fig,assets,'dashboard_workflow')
