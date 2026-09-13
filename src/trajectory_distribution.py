"""Distribution distances on complete phase-aligned paths, without trial pairing.

These endpoints do not assess physical duration. The default Euclidean geometry
is RMS coordinate error over the complete window; axes retain their common units.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist, pdist


def checked_paths(paths):
    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim != 3 or paths.shape[1:] != (100, 2) or len(paths) < 2:
        raise ValueError("Expected at least two trajectories of shape 100 x 2")
    if not np.isfinite(paths).all():
        raise ValueError("Nonfinite path coordinates must not be silently dropped")
    return paths


@dataclass(frozen=True)
class TrajectoryReference:
    geometry: str
    axis_scale: np.ndarray
    gamma: float

    @classmethod
    def fit(cls, training_paths, geometry="raw_rms", seed=20260913):
        paths = checked_paths(training_paths)
        if geometry == "raw_rms":
            scale = np.ones(2)
        elif geometry == "axis_balanced":
            # One SD per spatial axis, pooling deviations from the training mean
            # path. No pointwise division near the zero-variance anchored start.
            residuals = paths - paths.mean(axis=0, keepdims=True)
            scale = np.sqrt(np.mean(residuals ** 2, axis=(0, 1)))
            scale = np.where(scale > 1e-6, scale, 1.0)
        else:
            raise ValueError("Unknown path geometry")
        vectors = (paths / scale).reshape(len(paths), -1) / np.sqrt(200)
        selected = np.random.default_rng(seed).choice(len(paths), min(512, len(paths)), replace=False)
        distances = pdist(vectors[selected], metric="sqeuclidean")
        positive = distances[distances > 0]
        gamma = 1.0 / np.median(positive) if len(positive) else 1.0
        return cls(geometry, scale, float(gamma))

    def vectors(self, paths):
        paths = checked_paths(paths)
        return (paths / self.axis_scale).reshape(len(paths), -1) / np.sqrt(200)

    def to_dict(self):
        return {"geometry": self.geometry, "axis_scale": self.axis_scale.tolist(), "rbf_gamma": self.gamma}


def trajectory_distances(recorded, generated, reference):
    """Energy V statistic, unbiased MMD squared, and descriptive diagnostics.

Energy uses Euclidean (not squared Euclidean) distances. Kernel diagonals are
excluded only for unbiased MMD; its estimate can legitimately be negative.
"""
    x, y = reference.vectors(recorded), reference.vectors(generated)
    xx, yy, xy = cdist(x, x), cdist(y, y), cdist(x, y)
    n, m = len(x), len(y)
    kxx, kyy, kxy = [np.exp(-reference.gamma * d ** 2) for d in (xx, yy, xy)]
    ux = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
    uy = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
    xvar = np.sum(np.var(x, axis=0, ddof=1))
    yvar = np.sum(np.var(y, axis=0, ddof=1))
    return {
        "energy": float(2 * xy.mean() - xx.mean() - yy.mean()),
        "mmd2": float(ux + uy - 2 * kxy.mean()),
        "mmd2_biased_diagnostic": float(kxx.mean() + kyy.mean() - 2 * kxy.mean()),
        "mean_path_rmse": float(np.linalg.norm(x.mean(0) - y.mean(0))),
        "all_pairs_mse_diagnostic": float(np.mean(xy ** 2)),
        "recorded_dispersion": float(xvar),
        "generated_dispersion": float(yvar),
        "dispersion_ratio": float(yvar / xvar) if xvar > 0 else float("nan"),
    }
