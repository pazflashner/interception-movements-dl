"""Descriptive frozen-model perturbations; no training or causal inference.
Shuffle whole valid condition tuples within each test participant, five fixed
permutations. Change both encoder/decoder conditions or decoder only. A separate
nesting check constructs an equivalent conditional network from every UVAE.
"""
from pathlib import Path
import sys,json,copy
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
import config
from scripts.run_review_controls import load_trials
from scripts.run_corrected_study import load_per_trial_checkpoint
from src.vae_model import TrajectoryDataset
torch.set_num_threads(1);threadpool_limits(1)

def main():
    trials=load_trials();rows=[];nested=[]
    source=ROOT/'studies/final_strategy_evaluation/runs'
    for path in sorted(source.glob('*/*/*/checkpoint.pt')):
        rel=path.parent.relative_to(source);family=rel.parts[0];fold=int(rel.parts[1][4:]);model,norm=load_per_trial_checkpoint(path,'cpu');model.eval()
        if model.latent_dim not in [3,8]:continue
        split=json.loads((path.parent/'split.json').read_text());test=[t for t in trials if t['metadata']['subject'] in split['test_subjects']]
        ds=TrajectoryDataset(test);x=torch.from_numpy((ds.trajectories-norm.train_mean)/norm.train_std);c=torch.from_numpy(ds.conditions)
        with torch.no_grad():
            mu=model.encode(x,c)[0];y,tt=model.decode(mu,c)
            original=np.mean((y.numpy()*norm.train_std+norm.train_mean-ds.trajectories)**2,1)
            if not model.use_condition:
                equivalent=copy.deepcopy(model);equivalent.use_condition=True
                equivalent.encoder[0].weight[:,-5:]=0;equivalent.decoder[0].weight[:,-5:]=0
                my=equivalent.encode(x,c)[0];yy,t2=equivalent.decode(my,c)
                assert torch.equal(my,mu) and torch.equal(yy,y) and torch.equal(t2,tt)
                nested.append(dict(run=str(rel),equal_outputs=True));continue
            ids=np.array(ds.subjects)
            for repeat in range(5):
                rng=np.random.default_rng(20260908+repeat);permuted=np.arange(len(test))
                for s in np.unique(ids):
                    idx=np.flatnonzero(ids==s);permuted[idx]=rng.permutation(idx)
                wrong=c[permuted]
                for arm in ['both','decoder_only']:
                    mz=model.encode(x,wrong)[0] if arm=='both' else mu
                    yy=model.decode(mz,wrong)[0]
                    mse=np.mean((yy.numpy()*norm.train_std+norm.train_mean-ds.trajectories)**2,1)
                    for s in np.unique(ids):
                        mask=ids==s
                        rows.append(dict(family=family,dim=model.latent_dim,fold=fold,run=str(rel),subject=s,repeat=repeat,arm=arm,correct_mse=original[mask].mean(),shuffled_mse=mse[mask].mean()))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'condition_shuffle_raw.csv',index=False)
    pp=frame.groupby(['family','dim','subject','arm'])[['correct_mse','shuffled_mse']].mean().reset_index()
    pp['shuffled_minus_correct']=pp.shuffled_mse-pp.correct_mse
    pp.to_csv(OUT/'condition_shuffle_participant.csv',index=False)
    summary=pp.groupby(['family','dim','arm']).agg(correct_mse=('correct_mse','mean'),shuffled_mse=('shuffled_mse','mean'),mean_increase=('shuffled_minus_correct','mean'),correct_better_n=('shuffled_minus_correct',lambda x:int((x>0).sum())))
    summary.to_csv(OUT/'condition_shuffle_summary.csv');pd.DataFrame(nested).to_csv(OUT/'conditional_contains_unconditional.csv',index=False)
    print(summary.to_string());print('UVAE networks exactly embedded as conditional:',len(nested))

if __name__=='__main__':main()
