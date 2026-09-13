"""Scientific figures from saved evidence; never changes checkpoints or scores."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil
from production.course_evidence import *
from production.report_equations import render_equations

COLORS = ['#65798b','#ae824b','#7563a8','#087c83']

def prepare_figures():
    """Rebuild equations and reuse the delivered scientific figures on a fresh checkout.

    Full regeneration (build_figures) additionally requires the ignored decoded arrays.
    """
    saved=OUT/'course_assets'
    names=['trajectory_energy.png','generated_paths.png','fingerprint_controls.png']
    if not all((saved/n).exists() for n in names):
        build_figures()
    render_equations(ASSETS)
    for name in names:shutil.copy2(saved/name,ASSETS/name)

def build_figures():
    render_equations(ASSETS)
    with plt.rc_context({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False}):
        fig, axs = plt.subplots(1,2,figsize=(8.3,2.65),layout='constrained')
        for ax,n in zip(axs,(3,8)):
            vals=[value(f,n,'energy','raw_rms') for f in FAMILIES]
            ax.bar(range(4),vals,color=COLORS,width=.65)
            ax.set_xticks(range(4),['Spline\n+ PCA','CAE','CVAE','VAE'])
            ax.set_ylim(0,.15);ax.set_title(f'n = {n}')
            ax.set_ylabel('Trajectory energy discrepancy')
            for i,v in enumerate(vals): ax.text(i,v+.003,f'{v:.4f}',ha='center',fontsize=9)
        fig.savefig(ASSETS/'trajectory_energy.png',dpi=220);plt.close(fig)
        cache=OUT/'assets/trajectory_distribution_2026_09_13'
        fig,axs=plt.subplots(1,5,figsize=(8.3,3.1),layout='constrained',sharex=True,sharey=True)
        all_paths=[]
        for n in (8,):
            for f in FAMILIES:
                name=f'{f}_z{n}_'+('' if f=='spline_pca' else 'seed42_')+'fold2_subject01.npz'
                a=np.load(cache/name)
                all_paths.append(a['generated'])
            recorded=a['recorded']
        for ax,paths,title,color in zip(axs,[recorded]+all_paths,['Recorded']+[LABELS[f] for f in FAMILIES],['#30343b']+COLORS):
            for p in paths[:30]: ax.plot(p[:,0],p[:,1],color=color,alpha=.23,lw=.7)
            ax.plot(paths.mean(axis=0)[:,0],paths.mean(axis=0)[:,1],color=color,lw=1.8)
            ax.set_title(title,fontsize=10);ax.set_aspect('equal',adjustable='box');ax.set_xlabel('x')
        axs[0].set_ylabel('y (tracker units)')
        fig.savefig(ASSETS/'generated_paths.png',dpi=240);plt.close(fig)
        # These are source arrays from the frozen generation run, not illustrative simulations.
        fig,ax=plt.subplots(figsize=(6.3,2.2),layout='constrained')
        x=np.arange(3); width=.22
        data=np.array(control_means())
        for k,(lab,col) in enumerate(zip(['Own context','Population center','Other participants'],['#087c83','#8498ac','#b3bbc3'])):
            ax.bar(x+(k-1)*width,data[:,k],width,label=lab,color=col)
        ax.set_xticks(x,['CVAE n=3','CVAE n=8','VAE n=8']);ax.set_ylabel('Mean KS (11 features)');ax.set_ylim(0,.4)
        ax.legend(frameon=False,ncol=3,fontsize=9,loc='upper center',bbox_to_anchor=(.5,1.22))
        fig.savefig(ASSETS/'fingerprint_controls.png',dpi=240);plt.close(fig)
    saved=OUT/'course_assets';saved.mkdir(exist_ok=True)
    for name in ['trajectory_energy.png','generated_paths.png','fingerprint_controls.png']:
        shutil.copy2(ASSETS/name,saved/name)

if __name__=='__main__': build_figures()
