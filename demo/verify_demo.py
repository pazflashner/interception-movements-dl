"""Exercise the demo UI and compare its predictions with the shared decoder."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from streamlit.testing.v1 import AppTest
from src.confirmatory_dashboard import MULTI_ASSETS, read_json, load_selected_model, decode
from src.dashboard_models import MODEL_DIMS, reference_name


def rerun(app):
    # Streamlit's test adapter expects a list even for a single-select button group.
    for group in app.button_group:
        if not isinstance(group.value, list):
            group.set_value([group.value])
    return app.run()


def verify():
    app = AppTest.from_file(str(ROOT / 'demo/app.py'), default_timeout=40).run()
    assert not app.exception, app.exception
    stats = read_json(str(MULTI_ASSETS / 'latent_stats.json'))
    checked = []
    for family, dims in MODEL_DIMS.items():
        app.selectbox[0].select(family)
        rerun(app)
        for dim in dims:
            app.button_group[0].set_value([dim])
            rerun(app)
            assert not app.exception, app.exception
            model, norm = load_selected_model(family, dim)
            base = np.asarray(stats[reference_name(family, dim)]['training_center'])
            paths, timing = decode(model, norm, base, 2, 1, app.slider[-1].value)
            assert np.isfinite(paths).all() and np.isfinite(timing).all()
            movement, initiation = timing[0]
            expected = [f'{initiation*1000:.0f} ms', f'{movement*1000:.0f} ms',
                        f'{(movement+initiation)*1000:.0f} ms']
            assert [m.value for m in app.metric] == expected, (family, dim, expected)
            assert len(app.tabs) == 0
            assert len(app.slider) == dim + 1
            checked.append((family,dim))
    # Changing a latent coordinate must change the decoded path; UI remains valid.
    app.selectbox[0].select('unconditional_vae')
    rerun(app)
    app.button_group[0].set_value([8])
    rerun(app)
    app.slider[0].set_value(1.5)
    rerun(app)
    assert not app.exception
    model,norm = load_selected_model('unconditional_vae',8)
    s=stats[reference_name('unconditional_vae',8)]
    a=np.asarray(s['training_center']); b=a.copy(); b[0]+=1.5*s['training_scale'][0]
    x,_=decode(model,norm,a,2,1,.64); y,t=decode(model,norm,b,2,1,.64)
    assert not np.allclose(x,y)
    assert app.metric[0].value == f'{t[0,1]*1000:.0f} ms'
    # The fingerprint source is a context center, never a query trajectory.
    if len(app.selectbox[1].options)>1:
        previous_latent = next(c.value for c in app.caption if c.value.startswith('Actual latent z'))
        app.selectbox[1].select(app.selectbox[1].options[1])
        rerun(app)
        assert not app.exception
        current_latent = next(c.value for c in app.caption if c.value.startswith('Actual latent z'))
        assert previous_latent != current_latent
        assert all(s.value == 0 for s in app.slider[:-1])
        # Confirm that the selected preset really reaches the decoder unchanged at zero offsets.
        from src.confirmatory_dashboard import read_csv
        fingerprints=read_csv(str(MULTI_ASSETS/'subject_fingerprints.csv'))
        row=fingerprints[(fingerprints.run==reference_name('unconditional_vae',8)) &
                         (fingerprints.subject==app.selectbox[1].value)].iloc[0]
        center=row[[f'z{i+1}' for i in range(8)]].to_numpy(float)
        _,predicted=decode(model,norm,center,2,1,.64)
        assert app.metric[0].value == f'{predicted[0,1]*1000:.0f} ms'
    print(f'Passed: {len(checked)} model/dimension combinations, timing order, latent response, context selector, no extra tabs.')


if __name__ == '__main__':
    verify()
