"""Independently recompute auxiliary tests and pooled probe scores."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(OUT))
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon,false_discovery_control
from compare_and_control import holm,paired
R=ROOT/'studies/review_corrected_evaluation/results';S=ROOT/'studies/final_strategy_evaluation/results';checks=[]

def close(name,a,b,atol=1e-10):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    checks.append(dict(name=name,passed=bool(np.allclose(a,b,atol=atol,rtol=1e-9)),max_difference=float(np.max(np.abs(a-b)))))

def main():
    fp=pd.concat([pd.read_csv(p) for p in sorted((R/'review_controls/fingerprint').glob('*.csv'))])
    cols=['mean_ks','energy_distance','mmd_rbf']
    fp=fp.groupby(['latent_dim','seed','subject','arm'])[cols].mean().groupby(['latent_dim','subject','arm']).mean().reset_index()
    old=pd.read_csv(R/'review_controls/fingerprint_paired.csv');ps=[]
    for _,r in old.iterrows():
        a=fp[(fp.latent_dim==r.latent_dim)&(fp.arm=='own')].set_index('subject').sort_index();b=fp[(fp.latent_dim==r.latent_dim)&(fp.arm==r.control)].set_index('subject').loc[a.index]
        p=paired(b[r.metric]-a[r.metric]);ps.append(p);close('fingerprint '+str(r.latent_dim)+'/'+r.control+'/'+r.metric,[a[r.metric].mean(),b[r.metric].mean(),p],[r.own_mean,r.control_mean,r.wilcoxon_p])
    close('fingerprint BH12',false_discovery_control(ps),old.p_fdr_bh)
    mj=pd.read_csv(R/'review_controls/matched_component_participant_raw.csv');cols=[c for c in mj if c.startswith(('ks_','count_'))]
    mj=mj.groupby(['latent_dim','subject'])[cols].mean().reset_index();old=pd.read_csv(R/'review_controls/matched_component_paired.csv');ps=[]
    a=mj[mj.latent_dim==3].set_index('subject').sort_index();b=mj[mj.latent_dim==8].set_index('subject').loc[a.index]
    for _,r in old.iterrows():
        p=paired(a[r.metric]-b[r.metric]);ps.append(p);close('matched components '+r.metric,[a[r.metric].mean(),b[r.metric].mean(),p],[r.n3_mean,r.n8_mean,r.p])
    close('matched components BH9',false_discovery_control(ps),old.p_fdr_bh)
    raw=pd.read_csv(R/'timing_fairness/predictions.csv');rows=[]
    for endpoint in ['movement','initiation']:
        frame=raw.assign(mae_ms=np.abs(raw[endpoint+'_true_s']-raw[endpoint+'_pred_s'])*1000)
        keys=['comparison','model_family','latent_dim','seed','subject']
        p=frame.groupby(keys,dropna=False).mae_ms.mean().groupby([k for k in keys if k!='seed']).mean().reset_index();p['endpoint']=endpoint;rows.append(p)
    pp=pd.concat(rows);old=pd.read_csv(R/'timing_fairness/paired_comparisons.csv');ps=[]
    for _,r in old.iterrows():
        w=pp[(pp.comparison==r.comparison)&(pp.latent_dim==r.latent_dim)&(pp.endpoint==r.endpoint)].pivot(index='subject',columns='model_family',values='mae_ms')
        p=wilcoxon(w.cvae-w.spline_pca,zero_method='pratt').pvalue;ps.append(p)
        close('timing '+r.comparison+str(r.latent_dim)+r.endpoint,[w.cvae.mean(),w.spline_pca.mean(),p],[r.cvae_mean_ms,r.spline_mean_ms,r.wilcoxon_p])
    close('timing BH8',false_discovery_control(ps),old.p_fdr_bh)
    cond=pd.read_csv(S/'condition_effects/condition_strata_features.csv')
    cols=['standardized_median_abs_error','ks_statistic'];avg=cond.groupby(['model_family','latent_dim','subject','feature'])[cols].mean().reset_index()
    old=pd.read_csv(S/'condition_effects/cvae_vs_unconditional_paired.csv');ps=[]
    for _,r in old.iterrows():
        w=avg[(avg.latent_dim==r.latent_dim)&(avg.feature==r.feature)].pivot(index='subject',columns='model_family',values=r.metric);d=w.unconditional_vae-w.cvae
        p=wilcoxon(d,zero_method='pratt').pvalue if not np.allclose(d,0) else 1.;ps.append(p)
        close('condition '+str(r.latent_dim)+r.feature+r.metric,[p,np.median(w.cvae),np.median(w.unconditional_vae)],[r.wilcoxon_p_uncorrected,r.cvae_median,r.unconditional_vae_median])
    close('condition feature BH44',false_discovery_control(ps),old.wilcoxon_p_fdr_bh)
    tr=pd.read_csv(S/'condition_effects/condition_strata_trajectories.csv');tr=tr.groupby(['model_family','latent_dim','subject']).mean_trajectory_mse.mean().reset_index()
    old=pd.read_csv(S/'condition_effects/condition_trajectory_summary.csv');ps=[]
    for _,r in old.iterrows():
        w=tr[tr.latent_dim==r.latent_dim].pivot(index='subject',columns='model_family',values='mean_trajectory_mse');p=wilcoxon(w.unconditional_vae-w.cvae,zero_method='pratt').pvalue;ps.append(p)
        close('condition trajectory raw p '+str(r.latent_dim),p,r.wilcoxon_p_uncorrected)
    close('existing column actually Bonferroni',np.minimum(1,np.array(ps)*len(ps)),old.wilcoxon_p_holm)
    corrected=old.rename(columns={'wilcoxon_p_holm':'legacy_bonferroni_mislabeled_holm'});corrected['wilcoxon_p_holm_corrected']=holm(ps)
    corrected['same_significance_decision']=(corrected.legacy_bonferroni_mislabeled_holm<.05)==(corrected.wilcoxon_p_holm_corrected<.05)
    corrected.to_csv(OUT/'condition_trajectory_corrected_holm.csv',index=False)
    probes=pd.read_csv(R/'behavioral_probes/out_of_fold_predictions.csv');keys=['model_family','latent_dim','training_seed','fingerprint','target'];recomputed=[]
    for values,g in probes.groupby(keys,dropna=False):
        assert len(g)==28 and g.subject.nunique()==28
        ss=np.sum((g.true-g.predicted)**2);den=np.sum((g.true-g.true.mean())**2)
        r2=1-ss/den if den>0 else float(ss==0)
        recomputed.append(dict(zip(keys,values))|dict(r2_oof_pooled=r2,mae_model=np.mean(np.abs(g.true-g.predicted)),mae_train_mean_baseline=np.mean(np.abs(g.true-g.train_mean_baseline))))
    fresh=pd.DataFrame(recomputed).sort_values(keys);old=pd.read_csv(R/'behavioral_probes/pooled_by_seed.csv').sort_values(keys)
    for metric in ['r2_oof_pooled','mae_model','mae_train_mean_baseline']:close('pooled probe '+metric,fresh[metric],old[metric],atol=1e-8)
    pd.DataFrame(checks).to_csv(OUT/'auxiliary_statistics_checks.csv',index=False)
    summary=dict(checks=len(checks),failed=sum(not c['passed'] for c in checks),recomputed_pooled_probe_scores=len(fresh),mislabel='condition trajectory correction is Bonferroni, not Holm',decisions_changed=int((~corrected.same_significance_decision).sum()))
    (OUT/'auxiliary_statistics_summary.json').write_text(json.dumps(summary,indent=2));print(summary)

if __name__=='__main__':main()
