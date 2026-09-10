"""Check numerical claims and source preservation for the detailed review draft.

Requires local study artifacts. Does not train models or rerun component fitting.
"""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
from pypdf import PdfReader
from scipy.stats import false_discovery_control
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from production.remaining_evidence import load_evidence,RESULTS,AUDIT,CONTROL,FAMILIES
from production.report_equations import FORMULAS
from src.statistical_tests import paired_wilcoxon


def main():
    e=load_evidence();checks=[]
    def check(name,test):
        assert bool(test),name
        checks.append(name)
    for family in FAMILIES:
        for n in [3,8]:
            d=e['all_direct'].query('model_family == @family and latent_dim == @n').pivot(index='target',columns='arm',values='mae')
            check(f'{family} n={n}: all 14 direct-context MAEs lower',len(d)==14 and (d.direct_context<d.latent_probe).all())
    old_direct=pd.read_csv(CONTROL/'direct_context_summary.csv')
    new_direct=e['all_direct'].query('model_family == "cvae"').replace({'arm':{'latent_probe':'cvae_probe'}})
    joined=new_direct.merge(old_direct,on=['latent_dim','target','arm'],validate='one_to_one')
    check('CVAE direct-context extension reproduces original 84 rows',len(joined)==84 and np.allclose(joined.mae_x,joined.mae_y,atol=1e-12) and np.allclose(joined.r2,joined.pooled_r2,atol=1e-12))
    # Independently recompute paired control p-values from the stored participant rows.
    for name,raw,paired,dim_col,pcol,qcol in [
        ('CVAE',e['fingerprint_people'],e['fingerprint_paired'],'latent_dim','wilcoxon_p','p_fdr_bh'),
        ('VAE8',e['uvae_fp_people'],e['uvae_fp_paired'],None,'p','q_bh_6')]:
        pvals=[]
        for _,r in paired.iterrows():
            g=raw[raw.latent_dim==r.latent_dim] if dim_col else raw
            wide=g.pivot(index='subject',columns='arm',values=r.metric).sort_index()
            difference=wide[r.control]-wide['own'];p=paired_wilcoxon(difference).pvalue;pvals.append(p)
            check(f'{name} {r.get("latent_dim",8)} {r.control} {r.metric}: p and direction',np.isclose(p,r[pcol],atol=1e-12) and (difference>0).sum()==r.own_better_n)
        check(name+' BH adjustment',np.allclose(false_discovery_control(pvals),paired[qcol],atol=1e-12))
    check('All 44 condition-feature tests nonsignificant',len(e['condition_features'])==44 and (e['condition_features'].wilcoxon_p_fdr_bh>=.05).all())
    check('Timing family has eight paired tests',len(e['timing_paired'])==8)
    check('Neither common-head movement contrast survives BH',(e['timing_paired'].query('comparison == "common_mlp" and endpoint == "movement"').p_fdr_bh>=.05).all())
    check('Both common-head initiation contrasts survive BH',(e['timing_paired'].query('comparison == "common_mlp" and endpoint == "initiation"').p_fdr_bh<.05).all())
    tail=e['outliers'].query('model_family == "cvae" and latent_dim == 8 and training_seed == 43').iloc[0]
    check('Reported CVAE timing tail',np.isclose(tail.initiation_time_max_abs_error_ms/1000,11.54610866))
    check('Component fit coverage',len(e['component_raw'])==4056 and len(e['uvae_component_raw'])==840)
    check('CVAE selected fits converge',e['component_raw'].mj_selected_optimizer_converged.all())
    check('VAE selected fits converge in 839/840',e['uvae_component_raw'].mj_selected_optimizer_converged.sum()==839)
    check('CVAE component capacity contrasts nonsignificant',len(e['component_paired'])==9 and (e['component_paired'].p_fdr_bh>=.05).all())
    check('VAE-CVAE component contrasts nonsignificant',len(e['uvae_components'])==9 and np.allclose(e['uvae_components'].q_bh_9,.8642920791641455))
    check('All CVAE component discrepancies above empirical 95th percentile',(e['component_sampling'].observed>e['component_sampling'].reference_95).all())
    uv=e['uvae_components'].set_index('metric');ref=e['component_sampling'].query('latent_dim == 8').set_index('metric')
    check('All seven VAE component discrepancies above empirical 95th percentile',(uv.loc[ref.index,'uvae8_mean']>ref.reference_95).all())
    hist=e['historical_component_sensitivity'].set_index('sensitivity')
    check('Historical 5 Hz and 167 ms counts',round(hist.loc['five_hz_filter','same_threshold_count_rate']*56)==29 and round(hist.loc['jason_167ms_constraints','same_threshold_count_rate']*56)==31)
    counts=pd.read_csv(ROOT/'production/component_count_plot_values.csv')
    check('Count plot proportions sum to one',np.allclose(counts.groupby('label').p.sum(),1))
    # Prior sources and model weights remain the source of truth.
    ckpts=pd.read_csv(AUDIT/'checkpoint_audit.csv')
    for _,r in ckpts.iterrows():
        p=ROOT/'studies/final_strategy_evaluation/runs'/r.run/'checkpoint.pt'
        check('Checkpoint '+r.run,hashlib.sha256(p.read_bytes()).hexdigest()==r.checkpoint_sha256)
    documents=json.loads((ROOT/'studies/review_corrected_evaluation/VERIFICATION.json').read_text())['documents']
    for d in documents:check('Historical PDF '+d['path'],hashlib.sha256((ROOT/d['path']).read_bytes()).hexdigest()==d['sha256'])
    pdf=ROOT/'production/Interception_Movements_Methods_Reconstruction_Review.pdf'
    reader=PdfReader(pdf);text='\n'.join(p.extract_text() for p in reader.pages)
    check('All 19 numbered tables present',all(re.search(r'\bTable '+str(i)+r'\.',text) for i in range(1,20)))
    check('All 18 numbered figures present',all(re.search(r'\bFigure '+str(i)+r'\.',text) for i in range(1,19)))
    check('All 19 equation images generated',all((ROOT/f'production/assets/methods_reconstruction/equation_{name}.png').exists() for name in FORMULAS))
    check('All 15 Jason questions retained',text.count('[Jason:')==15)
    check('No textual hat placeholders','-hat' not in text)
    # Record a page map directly from the final PDF instead of manually guessing offsets.
    starts={}
    for i,p in enumerate(reader.pages,1):
        for line in p.extract_text().splitlines():
            if re.match(r'^([1-9]|1[0-2]) [A-Z]',line):starts[line]=i
    report={'date':'2026-09-10','revision':4,'checks_passed':len(checks),'checks':checks,
      'pdf_pages':len(reader.pages),'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),
      'equations':len(FORMULAS),'figures':18,'tables':19,'red_jason_questions':15,'section_pages':starts,
      'unchanged_neural_checkpoints':len(ckpts),'historical_pdfs_unchanged':len(documents),
      'new_neural_training':False,'new_component_optimizer_runs':False,
      'scope':'Detailed laboratory-review methods and results; not the condensed submission.'}
    (ROOT/'production/DETAILED_REVIEW_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
