"""Single source of numerical evidence for the course paper and presentation.

No training or new inference occurs here. Tables are read from audited results.
"""
from pathlib import Path
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'production'
ASSETS = OUT / 'assets/course_2026_09_14'
ASSETS.mkdir(parents=True, exist_ok=True)
FAMILIES = ['spline_pca', 'conditional_ae', 'cvae', 'unconditional_vae']
LABELS = dict(zip(FAMILIES, ['Spline + PCA', 'CAE', 'CVAE', 'VAE']))
LABELS['condition_ridge'] = 'Condition Ridge'
MEANS = pd.read_csv(OUT / 'audit_2026_09_08/all_model_means.csv')
OLD_TESTS = pd.read_csv(OUT / 'audit_2026_09_08/all_pairs_140.csv')
PATH_MEANS = pd.read_csv(OUT / 'trajectory_distribution_2026_09_13/summary.csv')
PATH_TESTS = pd.read_csv(OUT / 'trajectory_distribution_2026_09_13/paired_comparisons.csv')
CVAE_CONTROLS = pd.read_csv(ROOT / 'studies/review_corrected_evaluation/results/review_controls/fingerprint_summary.csv')
VAE_CONTROLS = pd.read_csv(OUT / 'audit_2026_09_08/uvae8_fingerprint_summary.csv')
FROZEN = OUT / 'Interception_Movements_Methods_Reconstruction_Review.pdf'
FROZEN_SHA = 'a1461d66e8034810342be4b89d9ed8e759e79db31c9f2a65d385f51c54b3ba2a'

def check_frozen():
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == FROZEN_SHA, 'Frozen laboratory report changed'

def control_means():
    return [[float((CVAE_CONTROLS[(CVAE_CONTROLS.latent_dim==n)&(CVAE_CONTROLS.arm==arm)] if f=='cvae' else VAE_CONTROLS[VAE_CONTROLS.arm==arm]).iloc[0].mean_ks) for arm in ('own','population','wrong')] for f,n in [('cvae',3),('cvae',8),('unconditional_vae',8)]]

def value(family, dim, metric, geometry=None):
    table = MEANS if geometry is None else PATH_MEANS
    rows = table[(table.model_family == family) & (table.latent_dim == dim)]
    if geometry is not None:
        rows = rows[rows.geometry == geometry]
    assert len(rows) == 1
    return float(rows.iloc[0][metric])

def comparison(a, b, dim, metric, geometry=None):
    table = OLD_TESTS if geometry is None else PATH_TESTS
    dc = 'dim' if geometry is None else 'latent_dim'
    rows = table[(table[dc] == dim) & (table.metric == metric) &
                 (((table.model_a == a) & (table.model_b == b)) |
                  ((table.model_a == b) & (table.model_b == a)))]
    if geometry is not None:
        rows = rows[rows.geometry == geometry]
    assert len(rows) == 1, (a,b,dim,metric,geometry)
    return rows.iloc[0]

def result_rows(metrics, geometry=None, digits=4):
    return [[LABELS[f], str(n)] + [f'{value(f,n,m,geometry):.{digits}f}' for m in metrics]
            for n in (3,8) for f in FAMILIES]

def selected_tests():
    rows = []
    for n in (3,8):
        for f in FAMILIES[:-1]:
            for metric in ('trajectory_mse','mean_ks','energy_distance','mmd_rbf'):
                r = comparison(f,'unconditional_vae',n,metric)
                rows.append([str(n),LABELS[f],metric,float(r.q_bh_140),float(r.p_holm_140)])
    return rows
