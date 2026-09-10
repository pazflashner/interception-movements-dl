"""Refit all classical configurations and reproduce their generative metrics."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_generation import *
from src.baseline_spline import SplinePCARepresentation
from src.confirmatory_spline import TimingRidge,_training_noise_covariance
from src.confirmatory_controls import ConditionRidge

def main():
    trials=load_trials();features=kinematic_features_for_dim(2);rows=[]
    for fold in range(4):
        split=json.loads((CORRECTED/f'runs/cvae/fold{fold}/cvae_z3_seed42/split.json').read_text())
        train,val,test=[[t for t in trials if t['metadata']['subject'] in split[k]] for k in ['train_subjects','validation_subjects','test_subjects']]
        ref=DistanceReference.fit(pd.DataFrame([compute_trial_features(t) for t in train]),features)
        ridge=ConditionRidge.fit(train,val)
        for family,dim,seed in [('spline_pca',d,config.CONTEXT_QUERY_SEED) for d in [2,3,4,8]]+[('condition_ridge',0,s) for s in [42,43,44]]:
            name=f'spline_pca_z{dim}' if family=='spline_pca' else f'condition_ridge_seed{seed}'
            run=CORRECTED/f'runs/{family}/fold{fold}/{name}';saved=pd.read_csv(run/'context_query_fidelity.csv').set_index('subject')
            savedref=json.loads((run/'distance_reference.json').read_text())
            refdiff=max(float(np.max(np.abs(ref.centre-np.array(savedref['centre'])))),float(np.max(np.abs(ref.scale-np.array(savedref['scale'])))),abs(ref.gamma-savedref['rbf_gamma']))
            if family=='spline_pca':
                settings=json.loads((run/'result.json').read_text());rep=SplinePCARepresentation(dim,include_timing=False,standardize_coefficients=settings['standardize_spline_coefficients']).fit(train)
                timing=TimingRidge.fit(rep,train,val);codes=rep.encode(test);cov=_training_noise_covariance(rep.encode(train),np.array([t['metadata']['subject'] for t in train]))
            for s in context_query_for_trials(test,config.CONTEXT_QUERY_SEED):
                query=[test[i] for i in s.query_indices];rng=np.random.default_rng(seed+sum(map(ord,s.subject)))
                if family=='spline_pca':
                    z=rng.multivariate_normal(codes[s.context_indices].mean(0),cov,size=120);chosen=rng.integers(0,len(query),120)
                    cond=np.stack([encode_trial_condition(query[i]['metadata']) for i in chosen]);x=rep.decode(z)[0];tt=timing.predict(z,cond)
                else:
                    chosen=rng.integers(0,len(query),120);x,tt=ridge.sample([query[i]['metadata'] for i in chosen],rng)
                gen=pd.DataFrame([features_from_generated_window(p,max(float(t[0]),.001),max(float(t[1]),0),config.WINDOW_GO_TO_ARRIVAL) for p,t in zip(x,tt)])
                assert np.isfinite(gen[features]).all().all()
                emp=pd.DataFrame([compute_trial_features(t) for t in query]);dist=independent_distances(emp,gen,ref,features);cols=list(dist)
                passed=np.allclose([dist[k] for k in cols],saved.loc[s.subject,cols],atol=1e-6,rtol=1e-6)
                if not passed:print('MISMATCH',family,fold,dim,seed,s.subject,'reference diff',refdiff,flush=True)
                rows.append(dict(family=family,fold=fold,dim=dim,seed=seed,subject=s.subject,passed=passed,reference_difference=refdiff,max_metric_difference=max(abs(dist[k]-saved.loc[s.subject,k]) for k in cols),**{'new_'+k:v for k,v in dist.items()},**{'saved_'+k:saved.loc[s.subject,k] for k in cols}))
        print('classical generation fold',fold,'verified',flush=True)
    frame=pd.DataFrame(rows);assert len(frame)==196;frame.to_csv(OUT/'classical_generation_audit.csv',index=False)
    (OUT/'classical_generation_summary.json').write_text(json.dumps(dict(runs=28,participant_runs=len(frame),failed=int((~frame.passed).sum()),generated_samples=len(frame)*120,max_metric_difference=frame.max_metric_difference.max()),indent=2))

if __name__=='__main__':main()
