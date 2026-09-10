"""Export 12 live references and reproduce saved seed-42 generative evaluations.

Neural weights are read-only. Spline/Ridge fits reproduce frozen train/validation
settings and are exported as portable numerical arrays. Original evidence stays intact.
"""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
import config
from scripts.run_review_controls import load_trials
from scripts.run_corrected_study import load_per_trial_checkpoint,context_query_for_trials,training_latent_noise_covariance
from src.evaluate import encode_trials
from src.baseline_spline import SplinePCARepresentation
from src.confirmatory_spline import TimingRidge,_training_noise_covariance
from src.features import compute_trial_features,features_from_generated_window,kinematic_features_for_dim
from src.context_query import DistanceReference,distribution_distances
from src.vae_model import encode_trial_condition
from src.dashboard_models import MODEL_DIMS,DEFAULT_MODEL,DEFAULT_DIM,reference_name,SplineReference

OUT=ROOT/'studies/review_corrected_evaluation/results/dashboard/multimodel'
SOURCE=ROOT/'studies/final_strategy_evaluation/runs'
CORRECTED=ROOT/'studies/review_corrected_evaluation/runs'


def export_spline(rep,timing):
    zero=rep.decode(np.zeros((1,rep.n_components)))[0].reshape(200)
    weights=rep.decode(np.eye(rep.n_components))[0].reshape(rep.n_components,200)-zero
    return dict(latent_dim=rep.n_components,trajectory_intercept=zero.tolist(),trajectory_weights=weights.tolist(),
        timing_input_mean=timing.x_scaler.mean_.tolist(),timing_input_scale=timing.x_scaler.scale_.tolist(),
        timing_weights=timing.model.coef_.tolist(),timing_intercept=timing.model.intercept_.tolist(),
        timing_target_mean=timing.y_mean.tolist(),timing_target_std=timing.y_std.tolist(),
        timing_alpha=float(timing.alpha),standardize_coefficients=rep.standardize_coefficients)


def main():
    torch.set_num_threads(1);threadpool_limits(1)
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'spline').mkdir(exist_ok=True)
    (OUT/'examples').mkdir(exist_ok=True)
    trials=load_trials();features=kinematic_features_for_dim(2)
    stats={};fingerprints=[];generated=[];empirical=[];checks=[];inventory=[]
    example_subject=min(t['metadata']['subject'] for t in trials)
    for fold in range(4):
        split=json.loads((SOURCE/f'cvae/fold{fold}/cvae_z3_seed42/split.json').read_text())
        train=[t for t in trials if t['metadata']['subject'] in split['train_subjects']]
        validation=[t for t in trials if t['metadata']['subject'] in split['validation_subjects']]
        test=[t for t in trials if t['metadata']['subject'] in split['test_subjects']]
        splits=context_query_for_trials(test,config.CONTEXT_QUERY_SEED)
        reference=DistanceReference.fit(pd.DataFrame([compute_trial_features(t) for t in train]),features)
        truth={s.subject:pd.DataFrame([compute_trial_features(test[i]) for i in s.query_indices]) for s in splits}
        for subject,frame in truth.items():
            empirical.append(frame.assign(outer_fold=fold))
        for family,dims in MODEL_DIMS.items():
            for dim in dims:
                name=reference_name(family,dim);run=SOURCE/family/f'fold{fold}'/name
                result=json.loads((run/'result.json').read_text())
                if family=='spline_pca':
                    rep=SplinePCARepresentation(dim,include_timing=False,standardize_coefficients=result['standardize_spline_coefficients']).fit(train)
                    timing_model=TimingRidge.fit(rep,train,validation)
                    train_mu=rep.encode(train);mu=rep.encode(test)
                    covariance=_training_noise_covariance(train_mu,np.asarray([t['metadata']['subject'] for t in train]))
                    payload=export_spline(rep,timing_model);adapter=SplineReference(payload)
                    if fold==0:(OUT/'spline'/f'{name}.json').write_text(json.dumps(payload))
                else:
                    model,norm=load_per_trial_checkpoint(run/'checkpoint.pt','cpu');model.eval()
                    train_mu=encode_trials(model,train,norm,'cpu')[0];mu=encode_trials(model,test,norm,'cpu')[0]
                    covariance=training_latent_noise_covariance(model,train,norm,'cpu')
                if fold==0:
                    # Match the equal-participant population control, not a trial-weighted mean.
                    train_splits=context_query_for_trials(train,config.CONTEXT_QUERY_SEED)
                    population=np.mean([train_mu[s.context_indices].mean(0) for s in train_splits],axis=0)
                    stats[name]=dict(training_center=population.tolist(),training_scale=(train_mu.std(0)+1e-8).tolist(),shared_covariance=covariance.tolist(),outer_fold=0,training_seed=None if family=='spline_pca' else 42)
                    inventory.append(dict(family=family,dim=dim,reference=name,
                        checkpoint_sha256=None if family=='spline_pca' else hashlib.sha256((run/'checkpoint.pt').read_bytes()).hexdigest()))
                saved=pd.read_csv(CORRECTED/family/f'fold{fold}'/name/'context_query_fidelity.csv').set_index('subject')
                for s in splits:
                    center=mu[s.context_indices].mean(0)
                    if fold==0:
                        fingerprints.append(dict(run=name,model_family=family,latent_dim=dim,subject=s.subject,n_context=len(s.context_indices),n_query=len(s.query_indices),**{f'z{i+1}':float(v) for i,v in enumerate(center)}))
                    rng=np.random.default_rng(config.CONTEXT_QUERY_SEED+sum(map(ord,s.subject)))
                    z=rng.multivariate_normal(center,covariance,size=120)
                    query=[test[i] for i in s.query_indices]
                    chosen=rng.integers(0,len(query),size=120)
                    conditions=np.stack([encode_trial_condition(query[i]['metadata'],5) for i in chosen])
                    if family=='spline_pca':
                        x=rep.decode(z)[0];t=timing_model.predict(z,conditions)
                        xx,tt=adapter.predict(z,conditions)
                        np.testing.assert_allclose(xx,x,atol=1e-10);np.testing.assert_allclose(tt,t,atol=1e-10)
                    else:
                        with torch.no_grad():
                            xx,tt=model.decode(torch.as_tensor(z,dtype=torch.float32),torch.as_tensor(conditions,dtype=torch.float32))
                        x=(xx.numpy()*norm.train_std+norm.train_mean).reshape(120,100,2)
                        t=norm.denormalise_timing(tt.numpy())
                    t[:,0]=np.maximum(t[:,0],.001);t[:,1]=np.maximum(t[:,1],0)
                    frame=pd.DataFrame([features_from_generated_window(p,float(ti[0]),float(ti[1]),config.WINDOW_GO_TO_ARRIVAL) for p,ti in zip(x,t)])
                    distances=distribution_distances(truth[s.subject],frame,features,reference)
                    keys=['energy_distance','mmd_rbf',*[f'ks_{f}' for f in features]]
                    error=max(abs(distances[k]-saved.loc[s.subject,k]) for k in keys)
                    assert error<1e-5,(name,fold,s.subject,error)
                    checks.append(dict(family=family,dim=dim,fold=fold,subject=s.subject,max_distance_error=error))
                    generated.append(frame.assign(subject=s.subject,model_family=family,latent_dim=dim,outer_fold=fold,training_seed=42 if family!='spline_pca' else -1,sample_index=np.arange(120)))
                    if s.subject==example_subject:
                        np.savez_compressed(OUT/'examples'/f'{name}.npz',generated=x,timing=t,recorded=np.stack([q['pos_norm'] for q in query]),query_ids=np.asarray([q['metadata']['trial_id'] for q in query]),context_ids=np.asarray([test[i]['metadata']['trial_id'] for i in s.context_indices]),subject=s.subject,fold=fold)
                print(f'Exported {family} n={dim} fold={fold}',flush=True)
    pd.concat(generated,ignore_index=True).to_csv(OUT/'generated_features.csv',index=False)
    pd.concat(empirical,ignore_index=True).to_csv(OUT/'empirical_features.csv',index=False)
    pd.DataFrame(fingerprints).to_csv(OUT/'subject_fingerprints.csv',index=False)
    (OUT/'latent_stats.json').write_text(json.dumps(stats,indent=2))
    pd.DataFrame(checks).to_csv(OUT/'reproduction_checks.csv',index=False)
    means=pd.read_csv(ROOT/'production/audit_2026_09_08/all_model_means.csv')
    for metric in ['mean_ks','energy_distance','mmd_rbf']:
        winner=means.loc[means[metric].idxmin()]
        assert winner.model_family==DEFAULT_MODEL and winner.latent_dim==DEFAULT_DIM
    means.to_csv(OUT/'model_means.csv',index=False)
    manifest=dict(version='multimodel-dashboard-v1',models={k:list(v) for k,v in MODEL_DIMS.items()},default_model=DEFAULT_MODEL,default_dim=DEFAULT_DIM,
        default_reason='Lowest cohort-average mean KS, energy discrepancy and MMD squared in the tested model matrix; numerical ranking, not significant superiority on every contrast.',
        live_fold=0,live_seed=42,validation_seed=42,validation_generated_per_person=120,validation_participants=28,
        live_test_subjects=sorted({r['subject'] for r in fingerprints}),
        inventory=inventory,example_subject=example_subject,source='Frozen matched study; no new neural training',
        max_reproduction_error=max(x['max_distance_error'] for x in checks))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Verified',len(checks),'participant evaluations; max error',manifest['max_reproduction_error'],flush=True)


if __name__=='__main__':main()
