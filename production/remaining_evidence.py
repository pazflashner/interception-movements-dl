"""Read frozen auxiliary evidence and extend descriptive context comparisons.

No model fitting, evaluation-row replacement, or new significance tests.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'production'
RESULTS=ROOT/'studies/review_corrected_evaluation/results'
AUDIT=OUT/'audit_2026_09_08'
CONTROL=RESULTS/'review_controls'
LABELS={'cvae':'CVAE','unconditional_vae':'VAE','conditional_ae':'CAE','spline_pca':'Spline + PCA'}
FAMILIES=['spline_pca','cvae','conditional_ae','unconditional_vae']
FEATURES={'initiation_time_s':'Initiation time','movement_time_s':'Movement time',
          'curvature_index':'Curvature index','peak_speed_tracker_units_s':'Peak speed',
          'path_length':'Path length','max_lateral_deviation':'Lateral deviation','n_submovements':'Speed-peak count'}
TARGETS=[f'{f}_{s}' for f in FEATURES for s in ['mean','sd']]


def load_evidence():
    sources={
      'means':AUDIT/'all_model_means.csv',
      'fingerprint':CONTROL/'fingerprint_summary.csv',
      'fingerprint_paired':CONTROL/'fingerprint_paired.csv',
      'fingerprint_people':CONTROL/'fingerprint_participant.csv',
      'uvae_fp':AUDIT/'uvae8_fingerprint_summary.csv',
      'uvae_fp_paired':AUDIT/'uvae8_fingerprint_paired.csv',
      'uvae_fp_people':AUDIT/'uvae8_fingerprint_participant.csv',
      'timing':RESULTS/'timing_fairness/summary.csv',
      'timing_paired':RESULTS/'timing_fairness/paired_comparisons.csv',
      'model_comparison':RESULTS/'dashboard/model_comparison.csv',
      'probes':RESULTS/'behavioral_probes/summary.csv',
      'probe_predictions':RESULTS/'behavioral_probes/out_of_fold_predictions.csv',
      'direct_predictions':CONTROL/'direct_context_predictions.csv',
      'condition_shuffle':AUDIT/'condition_shuffle_summary.csv',
      'condition_trajectory':OUT/'condition_trajectory_summary.csv',
      'condition_features':ROOT/'studies/final_strategy_evaluation/results/condition_effects/cvae_vs_unconditional_paired.csv',
      'components':CONTROL/'matched_component_summary.csv',
      'component_raw':CONTROL/'matched_components.csv',
      'component_sampling':CONTROL/'matched_component_sampling.csv',
      'component_sensitivity':CONTROL/'matched_component_sensitivity_summary.csv',
      'component_paired':CONTROL/'matched_component_paired.csv',
      'component_diagnostics':CONTROL/'matched_component_diagnostics.csv',
      'uvae_components':AUDIT/'uvae8_vs_cvae8_components.csv',
      'uvae_component_raw':AUDIT/'uvae8_matched_components.csv',
      'outliers':RESULTS/'dashboard/timing_outlier_audit.csv',
      'historical_component_sensitivity':RESULTS/'dashboard/minimum_jerk_sensitivity.csv',
      'kmeans':ROOT/'studies/strategy_window_comparison/results/go_to_arrival/baselines/kmeans_selection_corrected.csv',
    }
    e={k:pd.read_csv(p) for k,p in sources.items()}
    e['event']=json.loads((RESULTS/'event_audit/summary.json').read_text())
    e['uvae_fit_diagnostics']=json.loads((AUDIT/'uvae8_component_diagnostics.json').read_text())
    # All-family descriptive extension of the existing CVAE direct-context control.
    # Reuse the identical query targets and direct context predictions; do not fit probes.
    pred=e['probe_predictions'].query('fingerprint == "mean" and latent_dim in [3,8]')
    joined=pred.merge(e['direct_predictions'],on=['outer_fold','subject','target'],validate='many_to_one',suffixes=('','_checked'))
    assert len(joined)==len(pred)
    np.testing.assert_allclose(joined.true,joined.true_checked,atol=1e-10)
    records=[]
    for (family,dim,seed,target),g in joined.groupby(['model_family','latent_dim','training_seed','target'],dropna=False):
        assert len(g)==28 and g.subject.nunique()==28
        for arm,col in [('latent_probe','predicted'),('direct_context','context_prediction'),('development_constant','development_mean')]:
            records.append(dict(model_family=family,latent_dim=int(dim),seed=seed,target=target,arm=arm,
                                mae=float(np.abs(g.true-g[col]).mean()),r2=float(r2_score(g.true,g[col]))))
    direct=pd.DataFrame(records).groupby(['model_family','latent_dim','target','arm'],as_index=False)[['mae','r2']].mean()
    direct.to_csv(OUT/'all_models_direct_context_summary.csv',index=False)
    e['all_direct']=direct
    cv=e['fingerprint'].copy();cv['model_family']='cvae'
    uv=e['uvae_fp'].copy();uv['model_family']='unconditional_vae';uv['latent_dim']=8
    e['all_fingerprint']=pd.concat([cv,uv],ignore_index=True)
    for (family,dim),g in e['all_fingerprint'].groupby(['model_family','latent_dim']):
        main=e['means'].query('model_family == @family and latent_dim == @dim').iloc[0]
        np.testing.assert_allclose(g.set_index('arm').loc['own',['mean_ks','energy_distance','mmd_rbf']].to_numpy(float),main[['mean_ks','energy_distance','mmd_rbf']].to_numpy(float),atol=1e-12)
    code=['src/context_query.py','src/submovements.py','src/condition_effects.py',
          'scripts/run_corrected_study.py','scripts/run_review_controls.py','scripts/analyze_review_controls.py',
          'scripts/analyze_timing_fairness.py','scripts/analyze_behavioral_probes.py','scripts/audit_pre_go_motion.py',
          'scripts/analyze_condition_effects.py','production/audit_2026_09_08/compare_and_control.py',
          'src/confirmatory_controls.py','src/confirmatory_dashboard.py','scripts/audit_submovement_assumptions.py']
    extra_sources=[RESULTS/'event_audit/summary.json',AUDIT/'uvae8_component_diagnostics.json']
    provenance={'operation':'Report existing results; all-family direct-context comparison reaggregates saved predictions only.',
      'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*sources.values(),*extra_sources]},
      'code':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in code},
      'query_prediction_rows_checked':len(joined),'direct_summary_rows':len(direct),
      'neural_training':False,'new_significance_tests':False}
    (OUT/'remaining_results_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return e
