"""Check supported references, condition semantics and frozen-score reproduction."""
import json
import numpy as np
import pandas as pd
import pytest
from src.confirmatory_dashboard import ROOT,MULTI_ASSETS,decode,load_selected_model
from src.dashboard_models import MODEL_DIMS,reference_name


@pytest.mark.parametrize('family,dim',[(f,n) for f,dims in MODEL_DIMS.items() for n in dims])
def test_live_family_shape_and_condition_semantics(family,dim):
    if not (MULTI_ASSETS/'manifest.json').exists():
        pytest.skip('Restore the multi-model dashboard bundle for these reference checks')
    model,norm=load_selected_model(family,dim)
    z=np.zeros((2,dim));z[1,0]=.1
    a,t=decode(model,norm,z,1,1,.54)
    b,u=decode(model,norm,z,3,2,.72)
    assert a.shape==(2,100,2) and t.shape==(2,2)
    assert np.isfinite(a).all() and np.isfinite(t).all() and (t>=0).all()
    assert not np.allclose(a[0],a[1])
    if family=='unconditional_vae':
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(t,u)
    elif family=='spline_pca':
        np.testing.assert_array_equal(a,b);assert not np.allclose(t,u)
    else:
        assert not np.allclose(a,b)


def test_registry_rejects_untrained_capacity():
    with pytest.raises(ValueError): reference_name('cvae',16)
    with pytest.raises(ValueError): reference_name('conditional_ae',2)


def test_manifest_default_and_evidence_coverage():
    if not (MULTI_ASSETS/'manifest.json').exists():
        pytest.skip('Restore the multi-model dashboard bundle')
    m=json.loads((MULTI_ASSETS/'manifest.json').read_text())
    assert (m['default_model'],m['default_dim'])==('unconditional_vae',8)
    scores=pd.read_csv(MULTI_ASSETS/'model_means.csv')
    for metric in ['mean_ks','energy_distance','mmd_rbf']:
        row=scores.loc[scores[metric].idxmin()]
        assert (row.model_family,row.latent_dim)==('unconditional_vae',8)
    checks=pd.read_csv(MULTI_ASSETS/'reproduction_checks.csv')
    assert len(checks)==336 and checks.max_distance_error.max()<1e-5
    generated=pd.read_csv(MULTI_ASSETS/'generated_features.csv')
    assert generated.groupby(['model_family','latent_dim','subject']).size().eq(120).all()
    assert generated.groupby(['model_family','latent_dim']).subject.nunique().eq(28).all()
