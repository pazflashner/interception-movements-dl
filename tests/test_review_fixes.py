"""Regression tests for the September 8 integrity-review fixes."""
import numpy as np
import pytest
import torch
from src.statistical_tests import holm_adjust
from src.preprocessing import lowpass_filter
from src.vae_model import ConvCVAE, ConditionalVAE
from scripts.run_corrected_study import load_per_trial_checkpoint


def test_holm_step_down_and_original_order():
    # The second ordered p is multiplied by 2, not by the family size 3.
    np.testing.assert_allclose(holm_adjust([.04, .01, .03]), [.06, .03, .06])
    np.testing.assert_allclose(holm_adjust([.551873967051506, .06623983383178711]),
                               [.551873967051506, .13247966766357422])
    for invalid in [[], [np.nan], [-.1], [1.1]]:
        with pytest.raises(ValueError):
            holm_adjust(invalid)


def test_filter_at_pad_length_and_first_supported_length():
    short = np.arange(30, dtype=float).reshape(15, 2)
    np.testing.assert_array_equal(lowpass_filter(short), short)
    filtered = lowpass_filter(np.arange(32, dtype=float).reshape(16, 2))
    assert filtered.shape == (16, 2) and np.isfinite(filtered).all()


@pytest.mark.parametrize('variational,use_condition', [(True, True), (True, False), (False, True)])
def test_cnn_ablations_train_and_reload(tmp_path, variational, use_condition):
    torch.manual_seed(19)
    model = ConvCVAE(input_dim=200, channels=2, condition_dim=5, hidden_dim=16,
                    latent_dim=3, encoder_uses_timing=False,
                    variational=variational, use_condition=use_condition)
    x,c = torch.randn(4, 200), torch.randn(4, 5)
    mu,lv = model.encode(x,c)
    a,b = model.decode(mu,c)
    assert a.shape == (4,200) and b.shape == (4,2)
    z1,z2 = model.reparameterize(mu,lv),model.reparameterize(mu,lv)
    assert (not torch.equal(z1,z2)) if variational else torch.equal(z1,mu) and torch.equal(z2,mu)
    if not use_condition:
        assert torch.equal(mu,model.encode(x,c+4)[0])
        assert torch.equal(a,model.decode(mu,c+4)[0])
    (a.square().mean()+b.square().mean()).backward()
    assert model.enc_conv[0].weight.grad.isfinite().all()
    checkpoint = dict(model_state=model.state_dict(), input_dim=200, condition_dim=5,
                      trajectory_channels=2, latent_dim=3, timing_dim=2, encoder_uses_timing=False,
                      train_mean=[0.]*200, train_std=[1.]*200, timing_mean=[0.,0.], timing_std=[1.,1.],
                      config={'model':dict(architecture='cnn',hidden_dim=16,variational=variational,use_condition=use_condition)})
    path = tmp_path/'cnn.pt'; torch.save(checkpoint,path)
    loaded,_ = load_per_trial_checkpoint(path,'cpu')
    assert loaded.variational == variational and loaded.use_condition == use_condition
    torch.testing.assert_close(loaded.encode(x,c)[0],mu)


def test_dashboard_loader_respects_saved_unconditional_flags(tmp_path, monkeypatch):
    import src.confirmatory_dashboard as dashboard
    model=ConditionalVAE(input_dim=200,condition_dim=5,hidden_dim=16,latent_dim=3,
                         encoder_uses_timing=False,variational=False,use_condition=False)
    checkpoint=dict(model_state=model.state_dict(),input_dim=200,condition_dim=5,
                    latent_dim=3,timing_dim=2,encoder_uses_timing=False,
                    train_mean=[0.]*200,train_std=[1.]*200,timing_mean=[0.,0.],timing_std=[1.,1.],
                    config={'model':{'hidden_dim':16,'variational':False,'use_condition':False}})
    path=tmp_path/'model.pt';torch.save(checkpoint,path)
    monkeypatch.setattr(dashboard,'checkpoint_path',lambda _:path)
    dashboard.load_model.clear()
    try:
        loaded,_=dashboard.load_model(3)
        assert not loaded.variational and not loaded.use_condition
    finally:
        dashboard.load_model.clear()
