"""Build the first manuscript review section from frozen participant results.
Outputs and QA assets stay in production; does not train or select new models.
"""
from pathlib import Path
import sys,json,pickle,math
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'production';ASSETS=OUT/'assets/methods_reconstruction';ASSETS.mkdir(parents=True,exist_ok=True)
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config
from matplotlib.patches import FancyArrowPatch,Rectangle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate
from reports.article_layout import Article
from scripts.run_review_controls import load_trials
from scripts.run_corrected_study import load_per_trial_checkpoint
from src.evaluate import reconstruct
from src.baseline_spline import SplinePCARepresentation
torch.set_num_threads(1);threadpool_limits(1)
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','savefig.facecolor':'white'})
AUDIT=OUT/'audit_2026_09_08';SOURCE=ROOT/'studies/final_strategy_evaluation';CORRECTED=ROOT/'studies/review_corrected_evaluation'
COLORS={'cvae':'#27827b','unconditional_vae':'#365a8c','conditional_ae':'#365a8c','spline_pca':'#bb7226'}
LABELS={'cvae':'CVAE','unconditional_vae':'VAE','conditional_ae':'Conditional AE','spline_pca':'Spline + PCA'}

class ReviewArticle(Article):
    def __init__(self,*args):
        super().__init__(*args)
        self.styles['title'].fontSize=15
        self.styles['title'].leading=18
        self.styles['title'].spaceAfter=6
        self.styles['authors'].spaceAfter=7
        self.styles['caption'].spaceAfter=6
        self.styles['heading'].spaceBefore=8
        self.styles['body'].leading=13
    def finish(self):
        def footer(c,doc):
            c.saveState();c.setFont('Times-Roman',8)
            c.drawString(22*mm,12*mm,'Review draft 3 | Methods, reconstruction and generation | 10 September 2026')
            c.drawRightString(188*mm,12*mm,str(doc.page));c.restoreState()
        SimpleDocTemplate(str(self.output),pagesize=(210*mm,297*mm),leftMargin=22*mm,rightMargin=22*mm,
                          topMargin=17*mm,bottomMargin=20*mm,title=self.title,author='Simaan Libbiss and Paz Flashner').build(self.story,onFirstPage=footer,onLaterPages=footer)

def box(ax,x,y,w,h,text,size=9):
    ax.add_patch(Rectangle((x,y),w,h,facecolor='#f5f5f3',edgecolor='#777777',lw=.7))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.35)

def arrow(ax,x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=10,lw=.8,color='#555555'))

def figures(avg):
    fig,ax=plt.subplots(figsize=(7.1,1.8));ax.set(xlim=(0,10),ylim=(0,2.6));ax.axis('off')
    labels=[('3-D tracker\n240 samples/s',0,2.0),('Frame grid +\n10 Hz filter',2.65,2.1),('Target motion\nto final sample',5.3,2.0),('Resample and\ntranslate: 100 x 2',7.95,2.05)]
    for label,x,w in labels:box(ax,x,.9,w,1.05,label,size=8)
    for x1,x2 in [(2,2.65),(4.75,5.3),(7.3,7.95)]:arrow(ax,x1,1.425,x2,1.425)
    ax.text(5,.3,'Detected onset supplies two duration labels; seconds are withheld from the encoder.',ha='center',fontsize=9)
    fig.savefig(ASSETS/'preprocessing.png',dpi=260,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.1,2.15));ax.set(xlim=(0,10),ylim=(0,3));ax.axis('off')
    ax.text(0,2.77,'Conditional variational autoencoder',fontsize=10,weight='bold')
    box(ax,0,1.75,2.3,.78,'200 coordinates\n+ 5 conditions',8);box(ax,3,1.75,3.05,.78,'Encoder: 256 -> 256\nmean and log variance\nn latent values',8);box(ax,6.8,1.75,3.2,.78,'Latent + conditions\nDecoder: 256 -> 256\n200 coords + 2 log-times',8)
    arrow(ax,2.3,2.14,3,2.14);arrow(ax,6.05,2.14,6.8,2.14)
    ax.text(0,1.35,'Spline + PCA',fontsize=10,weight='bold')
    box(ax,0,.25,2.3,.78,'100 x 2 positions\nfit 18 coefficients',8);box(ax,3,.25,3.05,.78,'Training matrix: N_train x 18\nPCA -> n scores',7.5);box(ax,6.8,.25,3.2,.78,'Inverse PCA: 18 coefficients\nSpline basis: 100 x 2',7.8)
    arrow(ax,2.3,.64,3,.64);arrow(ax,6.05,.64,6.8,.64)
    fig.savefig(ASSETS/'models.png',dpi=260,bbox_inches='tight');plt.close(fig)
    fams=['spline_pca','unconditional_vae','conditional_ae','cvae'];fig,axs=plt.subplots(1,2,figsize=(7.1,2.5),sharey=True)
    for ax,dim in zip(axs,[3,8]):
        for j,fam in enumerate(fams):
            vals=avg[(avg.model_family==fam)&(avg.latent_dim==dim)].sort_values('subject').trajectory_mse.to_numpy()
            ax.scatter(j+np.linspace(-.14,.14,len(vals)),vals,s=12,c=COLORS[fam],alpha=.55,edgecolors='none')
            ax.plot([j-.24,j+.24],[vals.mean()]*2,color='black',lw=1.5)
        ax.set_xticks(range(4),['Spline','VAE','CAE','CVAE']);ax.set_yscale('log');ax.grid(axis='y',alpha=.18);ax.set_title(f'n = {dim}');ax.set_ylim(.0009,7)
    axs[0].set_ylabel('Trajectory MSE (tracker units squared)')
    fig.tight_layout();fig.savefig(ASSETS/'participant_mse.png',dpi=280,bbox_inches='tight');plt.close(fig)

def examples(avg):
    trials=load_trials();lookup={t['metadata']['trial_id']:t for t in trials}
    score=avg[(avg.latent_dim==3)&avg.model_family.isin(LABELS)].groupby('subject').trajectory_mse.mean().sort_values(kind='stable')
    selected=[]
    for quantile in [.5,.9]:
        subject=score.index[math.ceil(quantile*len(score))-1];fold=int(avg[avg.subject==subject].outer_fold.iloc[0]);rows=[]
        for family in LABELS:
            for path in (CORRECTED/f'runs/{family}/fold{fold}').glob('*_z3*'):
                f=pd.read_csv(path/'reconstruction_predictions.csv');f=f[f.subject==subject].copy();f['family']=family;rows.append(f)
        errors=pd.concat(rows).groupby(['family','trial_id']).trajectory_mse_tracker_units2.mean().groupby('trial_id').mean().sort_values(kind='stable')
        tid=errors.index[math.ceil(.5*len(errors))-1];selected.append(dict(quantile=quantile,subject=subject,fold=fold,trial_id=tid,selection_score=float(score.loc[subject]),trial=lookup[tid]))
    fig,axs=plt.subplots(2,2,figsize=(7.1,4.75));manifest=[]
    for row,ex in enumerate(selected):
        split=json.loads((SOURCE/f"runs/cvae/fold{ex['fold']}/cvae_z3_seed42/split.json").read_text());train=[t for t in trials if t['metadata']['subject'] in split['train_subjects']];t=ex['trial'];curves={}
        for dim in [3,8]:
            for family in ['cvae','unconditional_vae' if dim==3 else 'conditional_ae']:
                model,norm=load_per_trial_checkpoint(SOURCE/f"runs/{family}/fold{ex['fold']}/{family}_z{dim}_seed42/checkpoint.pt",'cpu')
                pred=reconstruct(model,[t],norm,'cpu')[0]
                curves[(dim,family)]=np.asarray(pred).reshape(100,2)
            setting=json.loads((SOURCE/f"runs/spline_pca/fold{ex['fold']}/spline_pca_z{dim}/result.json").read_text())
            rep=SplinePCARepresentation(dim,include_timing=False,standardize_coefficients=setting['standardize_spline_coefficients']).fit(train)
            curves[(dim,'spline_pca')]=rep.decode(rep.encode([t]))[0][0]
        allpoints=np.concatenate([t['pos_norm'],*curves.values()]);lo=allpoints.min(0);hi=allpoints.max(0);pad=np.maximum((hi-lo)*.1,.2)
        for col,dim in enumerate([3,8]):
            ax=axs[row,col];ax.plot(*t['pos_norm'].T,color='black',lw=1.6,label='Recorded',zorder=5)
            for fam in ['spline_pca','cvae','unconditional_vae' if dim==3 else 'conditional_ae']:
                pred=curves[(dim,fam)];ax.plot(*pred.T,color=COLORS[fam],lw=1.25,ls='--' if fam=='cvae' else '-',label=LABELS[fam])
                manifest.append(dict(subject=ex['subject'],trial_id=ex['trial_id'],fold=ex['fold'],dim=dim,family=fam,seed=42 if fam!='spline_pca' else None,mse=float(np.mean((pred-t['pos_norm'])**2)),participant_rank_target=ex['quantile'],sp=int(t['metadata']['sp']),side=int(t['metadata']['side'])))
            ax.set_xlim(-2,2);ax.set_ylim(-1,17);ax.set_xticks([-2,-1,0,1,2]);ax.grid(alpha=.15)
            ax.set_title(f"{ex['subject']} | n={dim}",fontsize=9);ax.set_xlabel('x (tracker units)');ax.set_ylabel('y (tracker units)')
            ax.legend(fontsize=6.8,loc='best',frameon=False)
    fig.tight_layout(h_pad=1.6,w_pad=1.2);fig.savefig(ASSETS/'reconstructions.png',dpi=300,bbox_inches='tight');plt.close(fig)
    pd.DataFrame(manifest).to_csv(ASSETS/'example_manifest.csv',index=False)
    (ASSETS/'example_selection.json').write_text(json.dumps([{k:v for k,v in ex.items() if k!='trial'} for ex in selected],indent=2))
    return selected


def data_figures():
    """Describe available recordings without changing inclusion or choosing on model error."""
    from collections import Counter
    from matplotlib.lines import Line2D
    import config
    with (ROOT/'studies/strategy_window_comparison/data/canonical_trials.pkl').open('rb') as handle:
        trials=pickle.load(handle)
    counts=Counter(t['metadata']['subject'] for t in trials)
    rows=[]
    for subject,retained in sorted(counts.items()):
        available=len(list((config.DATA_RAW_DIR/subject).glob('li_2_*.csv')))
        rows.append(dict(subject=subject,available=available,retained=retained,excluded=available-retained))
    cohort=pd.DataFrame(rows)
    assert cohort.available.sum()==4763 and cohort.retained.sum()==4732
    cohort.to_csv(OUT/'cohort_trial_counts.csv',index=False)
    subject=min(counts)
    group=[t for t in trials if t['metadata']['subject']==subject]
    ordered=sorted(group,key=lambda t:(t['move_end_idx']-t['go_signal_idx'],t['metadata']['trial_id']))
    selected=ordered[math.ceil(len(ordered)/2)-1]
    color={1:'#365a8c',2:'#b36b29',3:'#27827b'}
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.65))
    for t in group:
        pos=t['pos_go_to_arrival_norm'][:,:2]
        axs[0].plot(*pos.T,color=color[int(t['metadata']['sp'])],alpha=.35,lw=.65,ls='-' if t['metadata']['side']==1 else '--')
    pos=selected['pos_go_to_arrival_norm'][:,:2]
    axs[1].plot(*pos.T,color='#365a8c',lw=1.4)
    axs[1].scatter(*pos[0],c='black',s=20,zorder=5)
    axs[1].scatter(*pos[-1],c='#b36b29',s=25,marker='s',zorder=5)
    for ax,title in zip(axs,[f'{subject}: all {len(group)} trials',f'{subject}: median-duration trial']):
        ax.set_title(title,fontsize=10);ax.set_xlim(-1.5,2.5);ax.set_ylim(-1,16);ax.set_xticks([-1,0,1,2])
        ax.set_xlabel('x (tracker units)');ax.set_ylabel('y (tracker units)');ax.grid(alpha=.15)
    handles=[Line2D([0],[0],color=color[i],label=f'Category {i}') for i in [1,2,3]]
    handles += [Line2D([0],[0],color='black',ls=ls,label=name) for ls,name in [('-','Target from left'),('--','Target from right')]]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,.15,1,1));fig.savefig(ASSETS/'recorded_examples.png',dpi=280,bbox_inches='tight');plt.close(fig)
    start,end=selected['go_signal_idx'],selected['move_end_idx']
    raw=selected['pos_raw'][start:end+1,1];filtered=selected['pos_filtered'][start:end+1,1]
    time=np.arange(len(filtered))/240
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.6),sharey=True)
    axs[0].plot(time,raw-raw[0],color='#aaaaaa',label='Recorded grid',lw=.8)
    axs[0].plot(time,filtered-filtered[0],color='#365a8c',label='Filtered',lw=1.1)
    axs[0].set_xlabel('Time from target motion (s)');axs[0].set_title(f'{len(filtered)} samples at 240 Hz',fontsize=10);axs[0].legend(frameon=False,fontsize=8)
    axs[1].plot(np.linspace(0,1,len(filtered)),filtered-filtered[0],color='#365a8c',lw=.8)
    axs[1].scatter(np.linspace(0,1,100),selected['pos_go_to_arrival_norm'][:,1],c='#b36b29',s=8,label='100 resampled points')
    axs[1].set_xlabel('Normalized time (phase)');axs[1].set_title('Fixed-length representation',fontsize=10);axs[1].legend(frameon=False,fontsize=8)
    axs[0].set_ylabel('Forward position (tracker units)')
    for ax in axs: ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(ASSETS/'resampling_example.png',dpi=280,bbox_inches='tight');plt.close(fig)
    (OUT/'data_figure_provenance.json').write_text(json.dumps(dict(subject=subject,trials_shown=len(group),trial_id=selected['metadata']['trial_id'],selection='First participant in identifier order; lower median trial duration, ties ordered by trial ID; no model-error selection.',original_window_samples=len(filtered),resampled_points=100,task_image=str(config.STIMULI_DIR/'instructions2.jpg')),indent=2)+'\n')


def main():
    from production.manuscript_sections import write_sections
    pp=pd.read_csv(AUDIT/'independent_participant_metrics.csv')
    avg=pp.groupby(['model_family','latent_dim','outer_fold','subject'],as_index=False).trajectory_mse.mean()
    means=avg.groupby(['model_family','latent_dim']).trajectory_mse.mean()
    from production.report_equations import render_equations
    from production.generation_section import generation_figures
    figures(avg);chosen=examples(avg);data_figures();render_equations(ASSETS);generation_figures(ASSETS)
    dest=OUT/'Interception_Movements_Methods_Reconstruction_Review.pdf'
    a=ReviewArticle(dest,'Compact Representations of Human Interception Movements',
                   'Simaan Libbiss and Paz Flashner<br/>Workshop on Deep Learning, Tel Aviv University<br/>Research supervision: Prof. Jason Friedman | Course advisor: Moni Shahar')
    write_sections(a,ASSETS,means,chosen,config.STIMULI_DIR/'instructions2.jpg')
    a.finish();print(dest,flush=True)


if __name__=='__main__':main()
