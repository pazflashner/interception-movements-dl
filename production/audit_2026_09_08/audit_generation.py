"""Reproduce neural generative scores and independently check distance formulas.
No new comparative inference; this is part of the integrity gate.
"""
from pathlib import Path
import sys,json,pickle,hashlib
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp
from threadpoolctl import threadpool_limits
import config
from scripts.run_review_controls import load_trials
from scripts.run_corrected_study import load_per_trial_checkpoint,context_query_for_trials,training_latent_noise_covariance
from src.evaluate import encode_trials
from src.features import compute_trial_features,features_from_generated_window,kinematic_features_for_dim
from src.context_query import DistanceReference,distribution_distances
from src.vae_model import encode_trial_condition
torch.set_num_threads(1);threadpool_limits(1)
SOURCE=ROOT/'studies/final_strategy_evaluation';CORRECTED=ROOT/'studies/review_corrected_evaluation'

def independent_distances(e,g,reference,features):
    a=(e[features].to_numpy()-reference.centre)/reference.scale
    b=(g[features].to_numpy()-reference.centre)/reference.scale
    ab=cdist(a,b);aa=cdist(a,a);bb=cdist(b,b)
    energy=2*ab.mean()-aa.mean()-bb.mean()
    mmd=(np.exp(-reference.gamma*aa**2).sum()-len(a))/(len(a)*(len(a)-1))+(np.exp(-reference.gamma*bb**2).sum()-len(b))/(len(b)*(len(b)-1))-2*np.exp(-reference.gamma*ab**2).mean()
    stats={}
    for f in features:
        x,y=np.sort(e[f]),np.sort(g[f]);support=np.sort(np.r_[x,y])
        stats['ks_'+f]=float(np.max(np.abs(np.searchsorted(x,support,side='right')/len(x)-np.searchsorted(y,support,side='right')/len(y))))
    return dict(energy_distance=energy,mmd_rbf=mmd,**stats)

def main():
    trials=load_trials();features=kinematic_features_for_dim(2);rows=[];fold_data={}
    for fold in range(4):
        split=json.loads((CORRECTED/f'runs/cvae/fold{fold}/cvae_z3_seed42/split.json').read_text())
        train=[t for t in trials if t['metadata']['subject'] in split['train_subjects']];test=[t for t in trials if t['metadata']['subject'] in split['test_subjects']]
        ref=DistanceReference.fit(pd.DataFrame([compute_trial_features(t) for t in train]),features)
        empirical={s.subject:pd.DataFrame([compute_trial_features(test[i]) for i in s.query_indices]) for s in context_query_for_trials(test,config.CONTEXT_QUERY_SEED)}
        fold_data[fold]=(train,test,ref,empirical)
    for idx,path in enumerate(sorted((SOURCE/'runs').glob('*/*/*/checkpoint.pt'))):
        rel=path.parent.relative_to(SOURCE/'runs');fold=int(rel.parts[1][4:]);train,test,ref,empirical=fold_data[fold]
        destination=OUT/'generation_audit'/('_'.join(rel.parts)+'.csv');destination.parent.mkdir(exist_ok=True)
        if destination.exists():rows.extend(pd.read_csv(destination).to_dict('records'));continue
        model,norm=load_per_trial_checkpoint(path,'cpu');model.eval()
        trainmu,trainlv,_,subjects=encode_trials(model,train,norm,'cpu');mu=encode_trials(model,test,norm,'cpu')[0]
        residual=trainmu.copy()
        for s in np.unique(subjects):residual[np.array(subjects)==s]-=trainmu[np.array(subjects)==s].mean(0)
        # Same documented covariance, independently expressed as matrix products.
        centred=residual-residual.mean(0)
        cov=centred.astype(float).T@centred.astype(float)/(len(residual)-1)+np.diag(np.exp(trainlv).mean(0))+np.eye(model.latent_dim)*1e-6
        actualcov=training_latent_noise_covariance(model,train,norm,'cpu')
        np.testing.assert_allclose(cov,actualcov,atol=1e-7,rtol=1e-6)
        savedref=json.loads((CORRECTED/'runs'/rel/'distance_reference.json').read_text())
        np.testing.assert_allclose(ref.centre,savedref['centre'],atol=1e-12)
        np.testing.assert_allclose(ref.scale,savedref['scale'],atol=1e-12)
        np.testing.assert_allclose(ref.gamma,savedref['rbf_gamma'],atol=1e-12)
        saved=pd.read_csv(CORRECTED/'runs'/rel/'context_query_fidelity.csv').set_index('subject');runrows=[]
        for split in context_query_for_trials(test,config.CONTEXT_QUERY_SEED):
            mean=mu[split.context_indices].mean(0);rng=np.random.default_rng(config.CONTEXT_QUERY_SEED+sum(map(ord,split.subject)))
            # Use precisely the frozen covariance after independently checking it.
            z=rng.multivariate_normal(mean,actualcov,size=120).astype(np.float32)
            query=[test[i] for i in split.query_indices];chosen=rng.integers(0,len(query),size=120)
            cond=np.stack([encode_trial_condition(query[i]['metadata'],5) for i in chosen])
            with torch.no_grad():
                xx,tt=model.decode(torch.from_numpy(z),torch.from_numpy(cond));x=(xx.numpy()*norm.train_std+norm.train_mean).reshape(120,100,2);timing=norm.denormalise_timing(tt.numpy())
            gen=pd.DataFrame([features_from_generated_window(p,max(float(t[0]),.001),max(float(t[1]),0),config.WINDOW_GO_TO_ARRIVAL) for p,t in zip(x,timing)])
            assert np.isfinite(gen[features]).all().all()
            dist=independent_distances(empirical[split.subject],gen,ref,features)
            active=list(dist)
            np.testing.assert_allclose([dist[k] for k in active],saved.loc[split.subject,active],atol=1e-6,rtol=1e-6)
            runrows.append(dict(run=str(rel),subject=split.subject,n_generated=120,n_query=len(query),max_metric_difference=max(abs(dist[k]-saved.loc[split.subject,k]) for k in active),movement_floor_count=int((timing[:,0]<.001).sum()),initiation_floor_count=int((timing[:,1]<=0).sum()),crop_cap_count=int((timing[:,1]/(np.maximum(timing[:,0],.001)+timing[:,1])>.98).sum()),**dist))
        pd.DataFrame(runrows).to_csv(destination,index=False);rows.extend(runrows)
        if idx%4==0:print('generation verified',idx+1,'/96',flush=True)
    result=pd.DataFrame(rows);result.to_csv(OUT/'generation_audit.csv',index=False)
    (OUT/'generation_summary.json').write_text(json.dumps(dict(runs=result.run.nunique(),participant_runs=len(result),generated_samples=int(result.n_generated.sum()),max_metric_difference=result.max_metric_difference.max(),movement_floor_count=int(result.movement_floor_count.sum()),initiation_floor_count=int(result.initiation_floor_count.sum()),crop_cap_count=int(result.crop_cap_count.sum())),indent=2))
    print((OUT/'generation_summary.json').read_text(),flush=True)

if __name__=='__main__':main()
