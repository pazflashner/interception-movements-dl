"""Portable live-model support; no model fitting or query enrollment in the UI."""
from pathlib import Path
import json
import numpy as np
import torch
from src.vae_model import ConditionalVAE, NormStats, encode_trial_condition, inverse_timing

MODEL_DIMS = {'unconditional_vae': (3, 8), 'cvae': (2, 3, 4, 8),
              'conditional_ae': (3, 8), 'spline_pca': (2, 3, 4, 8)}
DEFAULT_MODEL = 'unconditional_vae'
DEFAULT_DIM = 8


def reference_name(family, dim):
    if family not in MODEL_DIMS or dim not in MODEL_DIMS[family]:
        raise ValueError(f'No matched-study reference for {family}, n={dim}')
    return f'{family}_z{dim}' + ('' if family == 'spline_pca' else '_seed42')


class SplineReference:
    """Exact affine export of fixed PCA/spline decoding and the fitted timing Ridge."""
    def __init__(self, payload):
        self.payload = payload
        self.latent_dim = int(payload['latent_dim'])
        self.condition_dim = 5
        self.use_condition = True  # timing only; spatial decoding has no conditions

    def predict(self, latent, conditions):
        p = self.payload
        z = np.atleast_2d(np.asarray(latent, dtype=float))
        trajectory = (z @ np.asarray(p['trajectory_weights']) + p['trajectory_intercept']).reshape(-1,100,2)
        u = (np.hstack([z, conditions]) - p['timing_input_mean']) / p['timing_input_scale']
        y = u @ np.asarray(p['timing_weights']).T + p['timing_intercept']
        timing = inverse_timing(y * p['timing_target_std'] + p['timing_target_mean'], 'log')
        return trajectory, timing


def load_reference(root: Path, assets: Path, family: str, dim: int):
    name = reference_name(family, dim)
    if family == 'spline_pca':
        return SplineReference(json.loads((assets/'spline'/f'{name}.json').read_text())), None
    path = root/'studies/final_strategy_evaluation/runs'/family/'fold0'/name/'checkpoint.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    cfg = checkpoint['config']['model']
    if cfg.get('architecture', 'mlp') != 'mlp':
        raise ValueError('This reference exporter supports the audited MLP models only')
    model = ConditionalVAE(input_dim=checkpoint['input_dim'], condition_dim=checkpoint['condition_dim'],
        latent_dim=dim, hidden_dim=cfg['hidden_dim'], timing_dim=checkpoint['timing_dim'],
        encoder_uses_timing=checkpoint['encoder_uses_timing'],
        variational=checkpoint.get('variational', cfg.get('variational', True)),
        use_condition=checkpoint.get('use_condition', cfg.get('use_condition', True)))
    model.load_state_dict(checkpoint['model_state']); model.eval()
    return model, NormStats.from_checkpoint(checkpoint)


def condition_vector(sp, side, speed, count):
    c = encode_trial_condition({'sp':sp,'side':side,'target_speed_screen_s':speed}, 5)
    return np.repeat(c[None,:],count,axis=0)
