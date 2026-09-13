"""Check delivered trajectory follow-up against saved decoded arrays and inputs."""
from pathlib import Path
import hashlib
import json
import sys
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'production/trajectory_distribution_2026_09_13'
CACHE=ROOT/'production/assets/trajectory_distribution_2026_09_13'


def main():
    verification=json.loads((OUT/'verification.json').read_text())
    historical=[]
    for name,expected in verification['protected_file_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected:
            continue
        # A collaborator may legitimately update a draft after the analysis.
        # Verify its recorded source revision, never excuse changed checkpoints.
        updates=verification.get('subsequent_collaborator_updates',{})
        assert name in updates and name.endswith('.pdf'), name
        original=subprocess.run(['git','show',verification['analysis_source_commit']+':'+name.replace('\\','/')],cwd=ROOT,capture_output=True,check=True).stdout
        assert hashlib.sha256(original).hexdigest()==expected
        historical.append(name)
    rows=pd.read_csv(OUT/'per_seed_scores.csv')
    assert len(rows)==1512
    assert rows.feature_reproduction_error.max()<1e-12
    sizes=rows.groupby(['model_family','latent_dim','fold','subject','geometry']).size()
    for (family,dim,fold,subject,geometry),count in sizes.items():
        assert count==(1 if family in ['spline_pca','context_mean_only'] else 3)
    manifest=json.loads((OUT/'context_query_manifest.json').read_text())
    assert len(manifest)==28 and len({r['subject'] for r in manifest})==28
    ids=[]
    for r in manifest:
        assert not set(r['context_ids']) & set(r['query_ids'])
        ids+=r['context_ids']+r['query_ids']
    assert len(ids)==4732 and len(set(ids))==4732
    refs={(r['fold'],r['geometry']):r for r in json.loads((OUT/'distance_references.json').read_text())}
    # First lexical participant in each model/fold/seed, both geometries.
    chosen=rows[~rows.model_family.str.startswith('context_')].sort_values('subject').groupby(['model_family','latent_dim','fold','seed','geometry'],as_index=False).first()
    maximum=0.0
    for r in chosen.itertuples():
        if r.model_family=='condition_ridge':name=f'condition_ridge_seed{r.seed}_blas2'
        else:name=f'{r.model_family}_z{r.latent_dim}'+('' if r.seed==-1 else f'_seed{r.seed}')
        arrays=np.load(CACHE/f'{name}_fold{r.fold}_{r.subject}.npz')
        ref=refs[r.fold,r.geometry]
        x=(arrays['recorded']/ref['axis_scale']).reshape(len(arrays['recorded']),200)/np.sqrt(200)
        y=(arrays['generated']/ref['axis_scale']).reshape(len(arrays['generated']),200)/np.sqrt(200)
        # Independent broadcast expression, not the production cdist routine.
        xx=np.linalg.norm(x[:,None]-x[None,:],axis=-1)
        yy=np.linalg.norm(y[:,None]-y[None,:],axis=-1)
        xy=np.linalg.norm(x[:,None]-y[None,:],axis=-1)
        energy=2*xy.mean()-xx.mean()-yy.mean()
        kxx,kyy,kxy=[np.exp(-ref['rbf_gamma']*d*d) for d in [xx,yy,xy]]
        np.fill_diagonal(kxx,0);np.fill_diagonal(kyy,0)
        mmd=kxx.sum()/(len(x)*(len(x)-1))+kyy.sum()/(len(y)*(len(y)-1))-2*kxy.mean()
        difference=max(abs(energy-r.energy),abs(mmd-r.mmd2))
        assert difference<1e-12
        maximum=max(maximum,difference)
    people=pd.read_csv(OUT/'participant_scores.csv')
    paired=pd.read_csv(OUT/'paired_comparisons.csv')
    for r in paired.itertuples():
        part=people[(people.geometry==r.geometry)&((people.latent_dim==r.latent_dim)|(people.model_family=='condition_ridge'))]
        wide=part.pivot(index='subject',columns='model_family',values=r.metric)
        d=np.round((wide[r.model_b]-wide[r.model_a]).to_numpy(),12)
        p=wilcoxon(d,zero_method='pratt',alternative='two-sided',method='auto').pvalue if np.any(d) else 1
        assert abs(p-r.p)<1e-12
    p=paired.p.to_numpy();order=np.argsort(p);n=len(p)
    bh=np.minimum(1,np.minimum.accumulate((p[order]*n/np.arange(1,n+1))[::-1])[::-1])
    holm=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(n,0,-1)))
    np.testing.assert_allclose(paired.q_bh_80.to_numpy()[order],bh,atol=1e-12)
    np.testing.assert_allclose(paired.p_holm_80.to_numpy()[order],holm,atol=1e-12)
    result=dict(independent_decoded_array_cases=len(chosen),maximum_distance_difference=maximum,paired_tests_rechecked=len(paired),rows_verified=len(rows),context_query_trial_ids_verified=len(ids),protected_files_currently_unchanged=len(verification['protected_file_hashes'])-len(historical),subsequently_updated_pdfs_verified_against_source_revision=historical,test_suite='9 tests passed: new trajectory metric tests and existing condition-control tests')
    (OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
