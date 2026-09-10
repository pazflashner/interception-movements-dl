"""Exploratory model comparisons and UVAE8 controls, gated by frozen-run audit."""
from pathlib import Path
import sys,json,pickle,itertools
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon,false_discovery_control
from threadpoolctl import threadpool_limits
import config
from scripts.run_review_controls import load_trials,component_job,run_component_jobs,MJ_FEATURES
from scripts.generate_final_samples import generate_run
from scripts.analyze_review_controls import component_metrics
from scripts.run_corrected_study import load_per_trial_checkpoint,context_query_for_trials,training_latent_noise_covariance
from src.evaluate import encode_trials
from src.features import compute_trial_features,features_from_generated_window,kinematic_features_for_dim
from src.context_query import DistanceReference,distribution_distances
from src.vae_model import encode_trial_condition
torch.set_num_threads(1);threadpool_limits(1)
SOURCE=ROOT/'studies/final_strategy_evaluation';CORRECTED=ROOT/'studies/review_corrected_evaluation';OLD=CORRECTED/'results/review_controls'
METRICS=['trajectory_mse','movement_time_mae_ms','initiation_time_mae_ms','mean_ks','ks_n_submovements','energy_distance','mmd_rbf']
DIST=['mean_ks','energy_distance','mmd_rbf']

def paired(d):
    d=np.round(np.asarray(d,dtype=float),12)
    return float(wilcoxon(d,zero_method='pratt',alternative='two-sided',method='auto').pvalue) if np.any(d) else 1.

def holm(p):
    p=np.asarray(p);order=np.argsort(p);out=np.empty_like(p)
    out[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)))
    return out

def compare():
    pp=pd.read_csv(OUT/'independent_participant_metrics.csv')
    avg=pp.groupby(['model_family','latent_dim','outer_fold','subject'])[METRICS].mean().reset_index()
    avg.groupby(['model_family','latent_dim'])[METRICS].mean().to_csv(OUT/'all_model_means.csv')
    families=['cvae','conditional_ae','unconditional_vae','spline_pca','condition_ridge'];rows=[];foldrows=[]
    for dim in [3,8]:
        for fa,fb in itertools.combinations(families,2):
            a=avg[(avg.model_family==fa)&(avg.latent_dim==(0 if fa=='condition_ridge' else dim))].set_index('subject').sort_index()
            b=avg[(avg.model_family==fb)&(avg.latent_dim==(0 if fb=='condition_ridge' else dim))].set_index('subject').loc[a.index]
            assert len(a)==len(b)==28
            for metric in METRICS:
                d=b[metric]-a[metric]
                rows.append(dict(dim=dim,model_a=fa,model_b=fb,metric=metric,n=28,mean_a=a[metric].mean(),mean_b=b[metric].mean(),mean_b_minus_a=d.mean(),median_b_minus_a=d.median(),a_better_n=int((d>0).sum()),b_better_n=int((d<0).sum()),p=paired(d)))
                for f in range(4):
                    foldrows.append(dict(dim=dim,model_a=fa,model_b=fb,metric=metric,fold=f,mean_b_minus_a=d[a.outer_fold==f].mean()))
    frame=pd.DataFrame(rows);assert len(frame)==140
    frame['q_bh_140']=false_discovery_control(frame.p);frame['p_holm_140']=holm(frame.p)
    frame.to_csv(OUT/'all_pairs_140.csv',index=False);pd.DataFrame(foldrows).to_csv(OUT/'all_pairs_fold_effects.csv',index=False)
    print('140 model comparisons complete',flush=True)

def fingerprint_run(train,test,model,norm,cov,fold,seed):
    destination=OUT/f'uvae8_fingerprint_fold{fold}_seed{seed}.csv'
    if destination.exists():return
    features=kinematic_features_for_dim(2)
    savedref=json.loads((CORRECTED/f'runs/unconditional_vae/fold{fold}/unconditional_vae_z8_seed{seed}/distance_reference.json').read_text())
    ref=DistanceReference(tuple(savedref['features']),np.array(savedref['centre']),np.array(savedref['scale']),savedref['rbf_gamma'])
    testmu=encode_trials(model,test,norm,'cpu')[0];trainmu=encode_trials(model,train,norm,'cpu')[0]
    splits=context_query_for_trials(test,config.CONTEXT_QUERY_SEED)
    centres={s.subject:testmu[s.context_indices].mean(0) for s in splits}
    pop=np.mean([trainmu[s.context_indices].mean(0) for s in context_query_for_trials(train,config.CONTEXT_QUERY_SEED)],0)
    rows=[]
    for s in splits:
        query=[test[i] for i in s.query_indices];emp=pd.DataFrame([compute_trial_features(t) for t in query]);mean=centres[s.subject]
        rng=np.random.default_rng(config.CONTEXT_QUERY_SEED+sum(map(ord,s.subject)));own=rng.multivariate_normal(mean,cov,size=120);noise=own-mean
        chosen=rng.integers(0,len(query),size=120);cond=np.stack([encode_trial_condition(query[i]['metadata'],5) for i in chosen])
        donors=[('own',s.subject,mean),('population','training_mean',pop)]+[('wrong',k,v) for k,v in centres.items() if k!=s.subject]
        for arm,donor,centre in donors:
            z=(own if arm=='own' else noise+centre).astype(np.float32)
            with torch.no_grad():
                xx,tt=model.decode(torch.from_numpy(z),torch.from_numpy(cond));x=(xx.numpy()*norm.train_std+norm.train_mean).reshape(120,100,2);timing=norm.denormalise_timing(tt.numpy())
            gen=pd.DataFrame([features_from_generated_window(p,max(float(t[0]),.001),max(float(t[1]),0),config.WINDOW_GO_TO_ARRIVAL) for p,t in zip(x,timing)])
            assert np.isfinite(gen[features]).all().all()
            dist=distribution_distances(emp,gen,features,ref)
            rows.append(dict(fold=fold,seed=seed,subject=s.subject,arm=arm,donor=donor,mean_ks=np.mean([dist['ks_'+f] for f in features]),energy_distance=dist['energy_distance'],mmd_rbf=dist['mmd_rbf']))
    df=pd.DataFrame(rows);saved=pd.read_csv(CORRECTED/f'runs/unconditional_vae/fold{fold}/unconditional_vae_z8_seed{seed}/context_query_fidelity.csv').set_index('subject')
    saved['mean_ks']=saved[['ks_'+f for f in features]].mean(1);own=df[df.arm=='own'].set_index('subject')
    np.testing.assert_allclose(own[DIST],saved.loc[own.index,DIST],atol=1e-6,rtol=1e-6)
    df.to_csv(destination,index=False)

def summarize_fingerprints():
    frame=pd.concat([pd.read_csv(p) for p in sorted(OUT.glob('uvae8_fingerprint_fold*.csv'))]);assert len(frame)==12*7*8
    # Average six wrong-donor distances, then three training seeds, for each person.
    pp=frame.groupby(['fold','seed','subject','arm'])[DIST].mean().groupby(['fold','subject','arm']).mean().reset_index()
    pp.to_csv(OUT/'uvae8_fingerprint_participant.csv',index=False)
    pp.groupby('arm')[DIST].mean().to_csv(OUT/'uvae8_fingerprint_summary.csv')
    own=pp[pp.arm=='own'].set_index('subject').sort_index();rows=[]
    for arm in ['population','wrong']:
        other=pp[pp.arm==arm].set_index('subject').loc[own.index]
        for metric in DIST:
            d=other[metric]-own[metric]
            rows.append(dict(control=arm,metric=metric,own_mean=own[metric].mean(),control_mean=other[metric].mean(),own_better_n=int((d>0).sum()),n=28,p=paired(d)))
    df=pd.DataFrame(rows);df['q_bh_6']=false_discovery_control(df.p);df.to_csv(OUT/'uvae8_fingerprint_paired.csv',index=False)

def summarize_components():
    frame=pd.read_csv(OUT/'uvae8_matched_components.csv');old=pd.read_csv(OLD/'matched_components.csv');real=old[old.kind=='recorded']
    assert len(frame)==840 and frame.job_id.is_unique and frame.mj_fit_completed.eq(True).all()
    rows=[]
    for (fold,seed,subject),g in frame.groupby(['outer_fold','seed','subject']):
        e=real[real.subject==subject];assert len(g)==10 and len(e)>0
        rows.append(dict(outer_fold=fold,seed=seed,subject=subject,**component_metrics(e,g)))
    metrics=['ks_'+f for f in MJ_FEATURES]+['count_total_variation','count_jsd','count_total_variation_bic','count_jsd_bic']
    pp=pd.DataFrame(rows);pp.to_csv(OUT/'uvae8_component_participant_raw.csv',index=False)
    avg=pp.groupby(['outer_fold','subject'])[metrics].mean().reset_index();avg.to_csv(OUT/'uvae8_component_participant.csv',index=False)
    cv=pd.read_csv(OLD/'matched_component_participant.csv');cv=cv[cv.latent_dim==8].set_index('subject').sort_index();uv=avg.set_index('subject').loc[cv.index];tests=[]
    for metric in metrics:
        d=cv[metric]-uv[metric]
        tests.append(dict(metric=metric,cvae8_mean=cv[metric].mean(),uvae8_mean=uv[metric].mean(),uvae_better_n=int((d>0).sum()),n=28,p=paired(d)))
    result=pd.DataFrame(tests);result['q_bh_9']=false_discovery_control(result.p);result.to_csv(OUT/'uvae8_vs_cvae8_components.csv',index=False)
    ref=pd.read_csv(OLD/'matched_component_sampling.csv').query('latent_dim==8').set_index('metric')
    rr=[]
    for metric in ref.index:
        rr.append(dict(metric=metric,observed_uvae8=avg[metric].mean(),empirical_plugin_mean=ref.loc[metric,'reference_mean'],empirical_plugin_95=ref.loc[metric,'reference_95']))
    pd.DataFrame(rr).to_csv(OUT/'uvae8_components_sampling_reference.csv',index=False)
    sens=pd.read_csv(OUT/'uvae8_component_sensitivity.csv');base=frame.set_index('job_id').loc[sens.job_id]
    summary=dict(fits=len(frame),completed=int(frame.mj_fit_completed.sum()),selected_converged=int(frame.mj_selected_optimizer_converged.sum()),all_candidates_converged=int(frame.mj_all_candidates_converged.sum()),sensitivity_fits=len(sens),sensitivity_count_agreement=int((sens.mj_n_components.to_numpy()==base.mj_n_components.to_numpy()).sum()),sensitivity_selected_converged=int(sens.mj_selected_optimizer_converged.sum()))
    (OUT/'uvae8_component_diagnostics.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)

def main():
    audit=json.loads((OUT/'integrity_summary.json').read_text());gen=json.loads((OUT/'generation_summary.json').read_text())
    assert audit['failed']==0 and audit['neural_checkpoints']==96 and gen['runs']==96
    compare();trials=load_trials();items=[]
    # Verify reuse of recorded component inputs, not just their filenames.
    olditems=pickle.load(open(OLD/'component_inputs.pkl','rb'));realitems={i[0]['trial_id']:i for i in olditems if i[0]['kind']=='recorded'}
    queryids=set()
    for fold in range(4):
        split=json.loads((CORRECTED/f'runs/cvae/fold{fold}/cvae_z3_seed42/split.json').read_text());train=[t for t in trials if t['metadata']['subject'] in split['train_subjects']];test=[t for t in trials if t['metadata']['subject'] in split['test_subjects']]
        for s in context_query_for_trials(test,config.CONTEXT_QUERY_SEED):
            for i in s.query_indices:
                t=test[i];tid=t['metadata']['trial_id'];old=realitems[tid];queryids.add(tid)
                np.testing.assert_array_equal(t['pos_norm'],old[1]);assert old[2:]==(2,400)
                assert old[0]['movement_time_s']==(t['move_end_idx']-t['move_start_idx'])/240 and old[0]['initiation_time_s']==(t['move_start_idx']-t['go_signal_idx'])/240
        for seed in [42,43,44]:
            model,norm=load_per_trial_checkpoint(SOURCE/f'runs/unconditional_vae/fold{fold}/unconditional_vae_z8_seed{seed}/checkpoint.pt','cpu');model.eval();cov=training_latent_noise_covariance(model,train,norm,'cpu')
            fingerprint_run(train,test,model,norm,cov,fold,seed)
            name=f'unconditional_vae_z8_fold{fold}_seed{seed}'
            for row,window in generate_run(model,norm,test,10,'cpu',seed,name,cov):
                row.update(kind='generated',outer_fold=fold,latent_dim=8,seed=seed,fit_seed_id=f"{row['run']}-{row['subject']}-{row['sample_id']}");row['job_id']=row['fit_seed_id'];items.append((row,window,2,400))
            print('candidate prepared',fold,seed,flush=True)
    assert queryids==set(realitems) and len(queryids)==2376
    (OUT/'recorded_component_reuse.json').write_text(json.dumps(dict(verified_inputs=2376,operator='same 100-point go window, timing crop, 2 restarts, 400 nfev',unchanged=True),indent=2))
    summarize_fingerprints();pickle.dump(items,open(OUT/'uvae8_component_inputs.pkl','wb'))
    fitted=run_component_jobs(items,OUT/'uvae8_matched_components.csv',6,'process')
    sensitivity=[(r,w,4,700) for r,w,_,_ in items if r['seed']==42 and r['sample_id']<2]
    run_component_jobs(sensitivity,OUT/'uvae8_component_sensitivity.csv',6,'process')
    # A disclosed diagnostic only: never replace fixed-budget primary rows.
    nonconverged=set(fitted.loc[~fitted.mj_selected_optimizer_converged,'job_id'])
    targeted=[(r,w,4,700) for r,w,_,_ in items if r['job_id'] in nonconverged]
    if targeted:
        run_component_jobs(targeted,OUT/'uvae8_nonconverged_targeted_recheck.csv',1,'thread')
    summarize_components()

if __name__=='__main__':main()
