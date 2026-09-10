"""Archive provenance and assess the impact of classical roundoff on comparisons."""
from pathlib import Path
import sys,json,hashlib,subprocess
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(OUT))
import pandas as pd,numpy as np
from scipy.stats import false_discovery_control
from compare_and_control import paired

def main():
    raw=pd.read_csv(OUT/'independent_participant_metrics.csv');fresh=pd.read_csv(OUT/'classical_generation_audit.csv')
    for _,r in fresh.iterrows():
        mask=(raw.model_family==r.family)&(raw.latent_dim==r.dim)&(raw.outer_fold==r.fold)&(raw.subject==r.subject)
        if r.family=='condition_ridge':mask &= raw.training_seed==r.seed
        assert mask.sum()==1
        for metric in ['energy_distance','mmd_rbf']:raw.loc[mask,metric]=r['new_'+metric]
    metrics=['trajectory_mse','movement_time_mae_ms','initiation_time_mae_ms','mean_ks','ks_n_submovements','energy_distance','mmd_rbf']
    avg=raw.groupby(['model_family','latent_dim','subject'])[metrics].mean().reset_index()
    old=pd.read_csv(OUT/'all_pairs_140.csv');ps=[]
    for _,r in old.iterrows():
        a=avg[(avg.model_family==r.model_a)&(avg.latent_dim==(0 if r.model_a=='condition_ridge' else r.dim))].set_index('subject').sort_index()
        b=avg[(avg.model_family==r.model_b)&(avg.latent_dim==(0 if r.model_b=='condition_ridge' else r.dim))].set_index('subject').loc[a.index]
        ps.append(paired(b[r.metric]-a[r.metric]))
    q=false_discovery_control(ps)
    impact=dict(n_tests=140,changed_raw_p_values=int((np.abs(np.array(ps)-old.p)>1e-12).sum()),max_q_change=float(np.max(np.abs(q-old.q_bh_140))),changed_bh_decisions=int(((q<.05)!=(old.q_bh_140<.05)).sum()),one_thread_max_classical_metric_difference=float(fresh.max_metric_difference.max()),affected_participant_runs=int((~fresh.passed).sum()),resolved_by='The single flagged Ridge row reproduces exactly with 2,4,8,16 numerical threads; see ridge_precision_diagnostic.csv.')
    (OUT/'ridge_precision_inference_impact.json').write_text(json.dumps(impact,indent=2))
    paths=sorted((ROOT/'src').glob('*.py'))+sorted((ROOT/'scripts').glob('*.py'))+[ROOT/'config.py',ROOT/'reports/concise_article.py',ROOT/'SIMAAN_MONI_REPORT_HANDOFF_2026-09-08.md',ROOT/'studies/strategy_window_comparison/data/canonical_trials.pkl']
    records=[dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in paths]
    (OUT/'source_manifest.json').write_text(json.dumps(records,indent=2))
    checkpoint=pd.read_csv(OUT/'checkpoint_audit.csv')
    conditions={}
    for fam,g in checkpoint.groupby('family'):
        conditions[fam]=dict(checkpoints=len(g),min_encoder_swap_change=float(g.encoder_swap_change.min()),min_decoder_swap_change=float(g.decoder_swap_change.min()))
    (OUT/'condition_wiring_summary.json').write_text(json.dumps(conditions,indent=2))
    result=dict(integrity=json.loads((OUT/'integrity_summary.json').read_text()),neural_generation=json.loads((OUT/'generation_summary.json').read_text()),classical_generation=json.loads((OUT/'classical_generation_summary.json').read_text()),auxiliary_statistics=json.loads((OUT/'auxiliary_statistics_summary.json').read_text()),preprocessing=json.loads((OUT/'preprocessing_loss_summary.json').read_text()),ridge_precision=impact,candidate_components=json.loads((OUT/'uvae8_component_diagnostics.json').read_text()),trained_models_modified=False,existing_reports_modified=False,neural_retraining=False)
    result['interpretation']='Frozen confirmatory model execution reproduced; the one classical single-thread mismatch is explained and independently reproduced with other thread counts. Auxiliary correction mislabel and dormant code defects remain documented; no claim of optimal training or physiological validation.'
    (OUT/'audit_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(impact,indent=2))

if __name__=='__main__':main()
