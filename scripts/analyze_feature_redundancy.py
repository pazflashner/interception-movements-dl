"""Training-only correlation screen and frozen-generation sensitivity.

Protocol is committed alongside the outputs. No model or historical score edits.
"""
from pathlib import Path
import sys,json,itertools,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from scipy.stats import false_discovery_control
from threadpoolctl import threadpool_limits
import scripts.analyze_trajectory_distributions as A
from src.statistical_tests import paired_wilcoxon,holm_adjust

OUT=ROOT/'production/feature_redundancy_2026_09_14'
CACHE=ROOT/'production/assets/feature_redundancy_2026_09_14'

def select_features(frame,priority,threshold=.9):
    pooled=frame[priority].corr(method='spearman')
    within=np.stack([g[priority].corr(method='spearman').to_numpy() for _,g in frame.groupby('subject')])
    median=pd.DataFrame(np.nanmedian(within,axis=0),index=priority,columns=priority)
    retained=[];decisions=[]
    for f in priority:
        matches=[r for r in retained if abs(pooled.loc[f,r])>=threshold and abs(median.loc[f,r])>=threshold and pooled.loc[f,r]*median.loc[f,r]>0]
        reason=matches[0] if matches else None
        if reason is None:retained.append(f)
        decisions.append(dict(feature=f,retained=reason is None,representative=reason,
                              pooled_rho=float(pooled.loc[f,reason]) if reason else None,
                              median_within_rho=float(median.loc[f,reason]) if reason else None))
    return retained,decisions,pooled,median

def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((OUT/'protocol.json').read_text())
    A.torch.set_num_threads(1);threadpool_limits(1)
    weights={str(p.relative_to(ROOT)):A.sha(p) for p in A.SOURCE.glob('*/*/*/checkpoint.pt')}
    trials=A.load_trials();features=A.kinematic_features_for_dim(2)
    assert set(features)==set(protocol['priority'])
    feature_rows=pd.DataFrame([A.compute_trial_features(t) for t in trials])
    selections=[];correlations=[];all_rows=[]
    for fold in range(4):
        sp=json.loads((A.SOURCE/f'cvae/fold{fold}/cvae_z3_seed42/split.json').read_text())
        train,validation,test=[[t for t in trials if t['metadata']['subject'] in sp[k+'_subjects']] for k in ('train','validation','test')]
        training=feature_rows[feature_rows.subject.isin(sp['train_subjects'])]
        retained,decisions,pooled,within=select_features(training,protocol['priority'])
        selections.append(dict(fold=fold,train_subjects=sp['train_subjects'],n_train_trials=len(training),retained=retained,decisions=decisions))
        for f,g in itertools.combinations(features,2):correlations.append(dict(fold=fold,feature_a=f,feature_b=g,pooled_rho=pooled.loc[f,g],median_within_rho=within.loc[f,g]))
        print('TRAINING SCREEN',fold,'retained',len(retained),retained,flush=True)
        views={'original':features,'reduced':retained}
        refs={v:A.DistanceReference.fit(training,fs) for v,fs in views.items()}
        splits=A.context_query_for_trials(test,A.config.CONTEXT_QUERY_SEED)
        truths={s.subject:pd.DataFrame([A.compute_trial_features(test[i]) for i in s.query_indices]) for s in splits}
        specs=[(f,n,seed) for f in protocol['models'] for n in (3,8) for seed in ([-1] if f=='spline_pca' else [42,43,44])]
        for family,dim,seed in specs:
            name=f'{family}_z{dim}'+('' if seed==-1 else f'_seed{seed}')
            dest=CACHE/f'{name}_fold{fold}.csv'
            if dest.exists():all_rows.extend(pd.read_csv(dest).to_dict('records'));continue
            run=A.SOURCE/family/f'fold{fold}'/name
            if family=='spline_pca':
                setting=json.loads((run/'result.json').read_text())
                rep=A.SplinePCARepresentation(dim,include_timing=False,standardize_coefficients=setting['standardize_spline_coefficients']).fit(train)
                head=A.TimingRidge.fit(rep,train,validation)
                train_mu=rep.encode(train);mu=rep.encode(test)
                cov=A._training_noise_covariance(train_mu,np.array([t['metadata']['subject'] for t in train]))
            else:
                model,norm=A.load_per_trial_checkpoint(run/'checkpoint.pt','cpu');model.eval()
                mu=A.encode_trials(model,test,norm,'cpu')[0]
                cov=A.training_latent_noise_covariance(model,train,norm,'cpu')
            saved=pd.read_csv(A.CORRECTED/family/f'fold{fold}'/name/'context_query_fidelity.csv').set_index('subject')
            rows=[]
            for s in splits:
                query=[test[i] for i in s.query_indices]
                rng=np.random.default_rng(A.config.CONTEXT_QUERY_SEED+sum(map(ord,s.subject)))
                z=rng.multivariate_normal(mu[s.context_indices].mean(0),cov,size=120)
                chosen=rng.integers(0,len(query),120)
                cond=np.stack([A.encode_trial_condition(query[i]['metadata'],5) for i in chosen])
                if family=='spline_pca':generated=rep.decode(z)[0];timing=head.predict(z,cond)
                else:
                    with torch.no_grad():xx,tt=model.decode(torch.as_tensor(z,dtype=torch.float32),torch.as_tensor(cond,dtype=torch.float32))
                    generated=(xx.numpy()*norm.train_std+norm.train_mean).reshape(120,100,2);timing=norm.denormalise_timing(tt.numpy())
                gen=pd.DataFrame([A.features_from_generated_window(p,max(float(t[0]),.001),max(float(t[1]),0),A.config.WINDOW_GO_TO_ARRIVAL) for p,t in zip(generated,timing)])
                np.savez_compressed(CACHE/f'{name}_fold{fold}_{s.subject}.npz',generated_features=gen[features].to_numpy(),recorded_features=truths[s.subject][features].to_numpy(),features=np.array(features))
                for view,fs in views.items():
                    distances=A.distribution_distances(truths[s.subject],gen,fs,refs[view])
                    error=0.
                    if view=='original':
                        error=max(abs(distances[k]-saved.loc[s.subject,k]) for k in ['energy_distance','mmd_rbf']+[f'ks_{f}' for f in fs])
                        assert error<1e-10,(name,s.subject,error)
                    rows.append(dict(model_family=family,latent_dim=dim,seed=seed,fold=fold,subject=s.subject,feature_view=view,n_features=len(fs),mean_ks=np.mean([distances[f'ks_{f}'] for f in fs]),energy_distance=distances['energy_distance'],mmd_rbf=distances['mmd_rbf'],reproduction_error=error))
            pd.DataFrame(rows).to_csv(dest,index=False);all_rows.extend(rows)
            print('Scored',name,'fold',fold,flush=True)
    raw=pd.DataFrame(all_rows);assert len(raw)==1120
    raw.to_csv(OUT/'per_seed_scores.csv',index=False)
    keys=['model_family','latent_dim','fold','subject','feature_view']
    people=raw.groupby(keys,as_index=False)[protocol['endpoints']].mean();people.to_csv(OUT/'participant_scores.csv',index=False)
    summary=people.groupby(['model_family','latent_dim','feature_view'],as_index=False)[protocol['endpoints']].mean();summary.to_csv(OUT/'summary.csv',index=False)
    tests=[]
    for view in ('original','reduced'):
        for n in (3,8):
            for metric in protocol['endpoints']:
                w=people[(people.feature_view==view)&(people.latent_dim==n)].pivot(index='subject',columns='model_family',values=metric)
                for f,g in itertools.combinations(protocol['models'],2):
                    d=w[g]-w[f]
                    tests.append(dict(feature_view=view,latent_dim=n,metric=metric,model_a=f,model_b=g,n=28,mean_a=w[f].mean(),mean_b=w[g].mean(),a_better_n=int((d>0).sum()),p=paired_wilcoxon(d).pvalue))
    tests=pd.DataFrame(tests);assert len(tests)==72
    tests['q_bh_72']=false_discovery_control(tests.p.to_numpy(),method='bh');tests['p_holm_72']=holm_adjust(tests.p.to_numpy())
    tests.to_csv(OUT/'paired_comparisons.csv',index=False)
    pd.DataFrame(correlations).to_csv(OUT/'training_correlations.csv',index=False)
    (OUT/'selection.json').write_text(json.dumps(selections,indent=2)+'\n')
    assert all(A.sha(ROOT/p)==h for p,h in weights.items())
    (OUT/'verification.json').write_text(json.dumps(dict(unchanged_checkpoints=len(weights),neural_training=False,model_person_seed_evaluations=560,feature_view_evaluations=len(raw),max_original_score_reproduction_error=float(raw.reproduction_error.max()),paired_tests=72,training_only_feature_selection=True,retained_counts=[len(s['retained']) for s in selections]),indent=2)+'\n')
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
