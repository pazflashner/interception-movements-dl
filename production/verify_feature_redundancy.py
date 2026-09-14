"""Independent rank, correction and decoded-feature checks for the sensitivity."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import rankdata,wilcoxon
from scipy.spatial.distance import cdist
from production.course_evidence import OUT,FAMILIES,check_frozen

def main():
    check_frozen();folder=OUT/'feature_redundancy_2026_09_14'
    tests=pd.read_csv(folder/'paired_comparisons.csv');people=pd.read_csv(folder/'participant_scores.csv')
    for r in tests.itertuples():
        part=people[(people.feature_view==r.feature_view)&(people.latent_dim==r.latent_dim)]
        w=part.pivot(index='subject',columns='model_family',values=r.metric)
        assert len(w)==28 and not w.isna().any().any()
        d=np.round(w[r.model_b]-w[r.model_a],12)
        p=wilcoxon(d,zero_method='pratt',method='auto').pvalue if np.any(d) else 1
        assert abs(p-r.p)<1e-12
    p=tests.p.to_numpy();order=np.argsort(p);m=len(p);assert m==72
    bh=np.minimum(1,np.minimum.accumulate((p[order]*m/np.arange(1,m+1))[::-1])[::-1])
    holm=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(m,0,-1)))
    assert np.allclose(bh,tests.q_bh_72.to_numpy()[order],atol=1e-12)
    assert np.allclose(holm,tests.p_holm_72.to_numpy()[order],atol=1e-12)
    # Rank every training column directly, then correlate the ranks.
    from scripts.run_review_controls import load_trials
    from src.features import compute_trial_features
    trials=load_trials();frame=pd.DataFrame([compute_trial_features(t) for t in trials])
    protocol=json.loads((folder/'protocol.json').read_text());priority=protocol['priority']
    selections=json.loads((folder/'selection.json').read_text());saved=pd.read_csv(folder/'training_correlations.csv')
    def corr(df):
        x=df[priority].to_numpy(float);assert np.isfinite(x).all()
        return np.corrcoef(rankdata(x,axis=0),rowvar=False)
    maximum_rho=0.
    for s in selections:
        split=json.loads((ROOT/f'studies/final_strategy_evaluation/runs/cvae/fold{s["fold"]}/cvae_z3_seed42/split.json').read_text())
        assert s['train_subjects']==split['train_subjects']
        assert not set(s['train_subjects'])&set(split['test_subjects']+split['val_subjects'] if 'val_subjects' in split else split['test_subjects']+split['validation_subjects'])
        tr=frame[frame.subject.isin(s['train_subjects'])];assert len(tr)==s['n_train_trials']
        pooled=corr(tr);within=np.nanmedian(np.stack([corr(g) for _,g in tr.groupby('subject')]),axis=0)
        keep=[]
        for j,f in enumerate(priority):
            if not any(abs(pooled[j,k])>=.9 and abs(within[j,k])>=.9 and pooled[j,k]*within[j,k]>0 for k in keep):keep.append(j)
        assert [priority[j] for j in keep]==s['retained']
        for r in saved[saved.fold==s['fold']].itertuples():
            j,k=priority.index(r.feature_a),priority.index(r.feature_b)
            diff=max(abs(pooled[j,k]-r.pooled_rho),abs(within[j,k]-r.median_within_rho));maximum_rho=max(maximum_rho,diff);assert diff<1e-12
    # Recompute all three discrepancies for the actual illustrated participant,
    # using distance matrices and empirical CDFs independent of the runner.
    raw=pd.read_csv(folder/'per_seed_scores.csv');cases=0;maximum_score=0.
    tr=frame[frame.subject.isin(selections[2]['train_subjects'])]
    for n in (3,8):
        for f in FAMILIES:
            seed=-1 if f=='spline_pca' else 42
            a=np.load(OUT/f'assets/feature_redundancy_2026_09_14/{f}_z{n}_{"" if seed==-1 else "seed42_"}fold2_subject01.npz')
            for view,features in [('original',list(a['features'])),('reduced',selections[2]['retained'])]:
                js=[list(a['features']).index(k) for k in features];x=a['recorded_features'][:,js];y=a['generated_features'][:,js]
                ks=[]
                for j in range(len(js)):
                    v=np.sort(np.concatenate([x[:,j],y[:,j]]))
                    ks.append(np.max(abs(np.searchsorted(np.sort(x[:,j]),v,side='right')/len(x)-np.searchsorted(np.sort(y[:,j]),v,side='right')/len(y))))
                vals=tr[features].to_numpy(float);assert np.isfinite(vals).all()
                center=vals.mean(0);sd=vals.std(0);sd=np.where(sd>1e-6,sd,1)
                if 'n_submovements' in features:sd[features.index('n_submovements')]=max(sd[features.index('n_submovements')],1)
                selected=np.random.default_rng(2026).choice(len(vals),min(512,len(vals)),replace=False)
                ref=(vals[selected]-center)/sd;ds=cdist(ref,ref,'sqeuclidean');gamma=1/np.median(ds[ds>0])
                x=(x-center)/sd;y=(y-center)/sd
                xx,yy,xy=cdist(x,x),cdist(y,y),cdist(x,y)
                energy=2*xy.mean()-xx.mean()-yy.mean()
                kxx,kyy,kxy=[np.exp(-gamma*d*d) for d in (xx,yy,xy)]
                np.fill_diagonal(kxx,0);np.fill_diagonal(kyy,0)
                mmd=kxx.sum()/(len(x)*(len(x)-1))+kyy.sum()/(len(y)*(len(y)-1))-2*kxy.mean()
                r=raw[(raw.subject=='subject01')&(raw.model_family==f)&(raw.latent_dim==n)&(raw.seed==seed)&(raw.feature_view==view)].iloc[0]
                diff=max(abs(np.mean(ks)-r.mean_ks),abs(energy-r.energy_distance),abs(mmd-r.mmd_rbf));maximum_score=max(maximum_score,diff);assert diff<1e-11;cases+=1
    reduced=tests[(tests.feature_view=='reduced')&(tests.model_a=='spline_pca')&tests.model_b.isin(['cvae','unconditional_vae'])]
    assert len(reduced)==12 and (reduced.p_holm_72<.00332).all() and (reduced.mean_b<reduced.mean_a).all()
    vc=tests[(tests.feature_view=='reduced')&(tests.latent_dim==8)&(tests.model_a=='cvae')&(tests.model_b=='unconditional_vae')]
    assert len(vc)==3 and (vc.q_bh_72>.05).all()
    result=dict(paired_tests_recomputed=72,correlation_pairs_recomputed=len(saved),training_splits_checked=4,decoded_feature_cases=cases,max_correlation_difference=maximum_rho,max_score_difference=maximum_score,claims_verified=True)
    (folder/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

if __name__=='__main__':main()
