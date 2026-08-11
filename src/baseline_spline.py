"""
Phase 2 – Polynomial spline baseline.

Two different things are measured here, and conflating them makes the CVAE look
far worse than it is:

``evaluate_spline_baseline`` (per-trial fit)
    Fits a spline to each trajectory independently and measures how well it
    reproduces *that same* trajectory. With 5 interior knots and degree 3 that
    is 9 coefficients per modeled coordinate, chosen with the
    answer in hand. It is an **interpolation ceiling** — how much of a
    trajectory survives smoothing — not a competing representation. A 3-dim
    latent cannot and should not beat it.

``evaluate_spline_pca_baseline`` (population fit, matched capacity)
    Fits the spline coefficients on the training subjects, reduces them to
    ``n_components`` with PCA fitted on train only, then projects held-out
    trials through that fixed low-dimensional basis. Same bottleneck as the
    CVAE, same requirement to generalise to unseen subjects. This is the
    comparison that answers "is the learned representation better than the best
    linear one of the same size?"
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import splrep, splev
from sklearn.decomposition import PCA

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def fit_spline(
    pos: np.ndarray,
    n_knots: int = config.SPLINE_N_KNOTS,
    degree: int = config.SPLINE_DEGREE,
) -> np.ndarray:
    """
    Fit a piecewise polynomial spline to a (T, D) trajectory and return
    the reconstructed trajectory of the same shape.
    """
    T, D = pos.shape
    t = np.linspace(0, 1, T)
    # Interior knots evenly spaced
    knots = np.linspace(0, 1, n_knots + 2)[1:-1]
    reconstructed = np.zeros_like(pos)
    for d in range(D):
        tck = splrep(t, pos[:, d], t=knots, k=degree)
        reconstructed[:, d] = splev(t, tck)
    return reconstructed


def evaluate_spline_baseline(trials: list[dict], n_knots: int = config.SPLINE_N_KNOTS) -> dict:
    """
    Evaluate spline reconstruction MSE across all trials.

    Returns dict with per-trial MSEs and overall statistics.
    """
    mses = []
    for trial in trials:
        pos = trial["pos_norm"]
        recon = fit_spline(pos, n_knots=n_knots)
        mse = float(np.mean((pos - recon) ** 2))
        mses.append(mse)

    mses = np.array(mses)
    dimensions = trials[0]["pos_norm"].shape[1]
    n_params = (n_knots + config.SPLINE_DEGREE + 1) * dimensions
    result = {
        "mean_mse": float(np.mean(mses)),
        "std_mse": float(np.std(mses)),
        "median_mse": float(np.median(mses)),
        "per_trial_mse": mses,
        "n_params_per_trial": n_params,
    }
    print(
        f"Spline baseline, per-trial fit ({n_knots} knots, "
        f"{n_params} params/trial fitted to the trial itself): "
        f"MSE = {result['mean_mse']:.6f} +/- {result['std_mse']:.6f}"
    )
    return result


# ── Population baseline at matched dimensionality ────────────────────────────
def _spline_coefficients(pos: np.ndarray, n_knots: int, degree: int) -> np.ndarray:
    """
    Flattened spline coefficients for one (T, D) trajectory.

    ``splrep`` returns a coefficient array as long as the knot vector, with
    ``degree + 1`` trailing zeros of padding. Only the first ``n_knots + degree
    + 1`` entries are real coefficients, so the padding is stripped here — the
    per-dimension blocks must be exactly ``n_coef`` wide or the flattened vector
    cannot be sliced back apart.
    """
    T, D = pos.shape
    n_coef = n_knots + degree + 1
    t = np.linspace(0, 1, T)
    knots = np.linspace(0, 1, n_knots + 2)[1:-1]
    coefs = []
    for d in range(D):
        tck = splrep(t, pos[:, d], t=knots, k=degree)
        coefs.append(np.asarray(tck[1])[:n_coef])
    return np.concatenate(coefs)


def _spline_basis(T: int, n_knots: int, degree: int, n_coef: int) -> np.ndarray:
    """
    Design matrix mapping one dimension's coefficients back to T samples.

    Built once by evaluating each basis function, so reconstruction from
    PCA-compressed coefficients is a plain matrix product. ``splev`` needs a
    coefficient vector as long as the knot vector, so the indicator is placed in
    a full-length zero array and truncated columns are never passed to it.
    """
    t = np.linspace(0, 1, T)
    knots = np.linspace(0, 1, n_knots + 2)[1:-1]
    # splrep on a zero signal gives the knot vector; the basis follows from it.
    full_knots, full_c, _ = splrep(t, np.zeros(T), t=knots, k=degree)
    basis = np.zeros((T, n_coef))
    for j in range(n_coef):
        c = np.zeros(len(full_c))
        c[j] = 1.0
        basis[:, j] = splev(t, (full_knots, c, degree))
    return basis


class SplinePCARepresentation:
    """
    The spline analogue of the CVAE, built to be evaluated by the same battery.

    Encodes each trial as ``n_components`` numbers over the joint space of
    [spline coefficients, standardised timing], fitted on training subjects
    only. That mirrors the CVAE exactly: one low-dimensional code per trial
    carrying both shape and timing, decoded by a map shared across subjects.

    ``encode`` plays the role of the encoder's μ, ``decode`` of the decoder, and
    ``sample_subject`` of drawing from a subject's aggregated posterior — so
    reconstruction, fingerprints, probing and generative fidelity can all be run
    on this representation without special-casing anything.

    The difference from the CVAE is what is being tested: this map is linear and
    has ~10^2 parameters against ~10^5, and its latent has no prior pulling it
    anywhere. If the CVAE's extra machinery buys something, it has to show up
    against this.
    """

    def __init__(
        self,
        n_components: int = config.DEFAULT_LATENT_DIM,
        n_knots: int = config.SPLINE_N_KNOTS,
        degree: int = config.SPLINE_DEGREE,
    ):
        self.n_components = n_components
        self.n_knots = n_knots
        self.degree = degree
        self.n_coef = n_knots + degree + 1

    # ── Feature construction ─────────────────────────────────────────────
    def _raw(self, trials: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """(spline coefficients, timing in seconds) for a list of trials."""
        from src.vae_model import encode_timing

        C = np.stack([
            _spline_coefficients(t["pos_norm"], self.n_knots, self.degree) for t in trials
        ])
        timing = np.stack([encode_timing(t) for t in trials])
        return C, timing

    def fit(self, train_trials: list[dict]) -> "SplinePCARepresentation":
        C, timing = self._raw(train_trials)
        self.coef_mean_, self.coef_std_ = C.mean(0), C.std(0) + 1e-8
        self.timing_mean_, self.timing_std_ = timing.mean(0), timing.std(0) + 1e-8
        X = np.hstack([
            (C - self.coef_mean_) / self.coef_std_,
            (timing - self.timing_mean_) / self.timing_std_,
        ])
        self.pca_ = PCA(n_components=self.n_components).fit(X)
        self.T_ = train_trials[0]["pos_norm"].shape[0]
        self.dimensions_ = train_trials[0]["pos_norm"].shape[1]
        self.basis_ = _spline_basis(self.T_, self.n_knots, self.degree, self.n_coef)
        return self

    # ── Encode / decode ──────────────────────────────────────────────────
    def encode(self, trials: list[dict]) -> np.ndarray:
        """(N, n_components) codes — the analogue of the encoder's μ."""
        C, timing = self._raw(trials)
        X = np.hstack([
            (C - self.coef_mean_) / self.coef_std_,
            (timing - self.timing_mean_) / self.timing_std_,
        ])
        return self.pca_.transform(X)

    def decode(self, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Codes -> (trajectories (N, T, D), timing (N, 2) in seconds)."""
        X = self.pca_.inverse_transform(np.atleast_2d(codes))
        coefficient_width = self.dimensions_ * self.n_coef
        C = X[:, :coefficient_width] * self.coef_std_ + self.coef_mean_
        timing = X[:, coefficient_width:] * self.timing_std_ + self.timing_mean_

        trajs = np.stack([
            np.stack(
                [self.basis_ @ C[i, d * self.n_coef : (d + 1) * self.n_coef] for d in range(self.dimensions_)],
                axis=1,
            )
            for i in range(len(C))
        ])
        return trajs, timing

    def sample_subject(
        self, codes: np.ndarray, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Draw codes from a Gaussian fitted to one subject's codes.

        The counterpart of sampling the CVAE's aggregated posterior. A full
        covariance is used so the linear model is not handicapped by an
        independence assumption the CVAE's aggregated posterior does not make.
        """
        mean = codes.mean(0)
        if len(codes) <= codes.shape[1]:
            return rng.normal(mean, codes.std(0) + 1e-9, size=(n_samples, codes.shape[1]))
        cov = np.cov(codes, rowvar=False) + 1e-9 * np.eye(codes.shape[1])
        return rng.multivariate_normal(mean, cov, size=n_samples)


def evaluate_spline_pca_baseline(
    train_trials: list[dict],
    test_trials: list[dict],
    n_components: int = config.DEFAULT_LATENT_DIM,
    n_knots: int = config.SPLINE_N_KNOTS,
    degree: int = config.SPLINE_DEGREE,
) -> dict:
    """
    Spline coefficients compressed to *n_components* by PCA fitted on train.

    The capacity-matched counterpart to the CVAE: the same number of numbers per
    trial, and a basis that must generalise to subjects it never saw. Reported
    in the same mm^2 units as ``compute_reconstruction_mse``.
    """
    T = train_trials[0]["pos_norm"].shape[0]
    n_coef = n_knots + degree + 1

    train_C = np.stack([_spline_coefficients(t["pos_norm"], n_knots, degree) for t in train_trials])
    test_C = np.stack([_spline_coefficients(t["pos_norm"], n_knots, degree) for t in test_trials])

    pca = PCA(n_components=n_components).fit(train_C)
    test_recon_C = pca.inverse_transform(pca.transform(test_C))

    basis = _spline_basis(T, n_knots, degree, n_coef)
    mses = []
    dimensions = train_trials[0]["pos_norm"].shape[1]
    for i, trial in enumerate(test_trials):
        recon = np.stack(
            [basis @ test_recon_C[i, d * n_coef : (d + 1) * n_coef] for d in range(dimensions)], axis=1
        )
        mses.append(float(np.mean((trial["pos_norm"] - recon) ** 2)))

    mses = np.array(mses)
    result = {
        "mean_mse": float(np.mean(mses)),
        "std_mse": float(np.std(mses)),
        "median_mse": float(np.median(mses)),
        "per_trial_mse": mses,
        "n_components": n_components,
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
    }
    print(
        f"Spline+PCA baseline ({n_components} components, fitted on train subjects, "
        f"{result['explained_variance']:.1%} coef variance): "
        f"MSE = {result['mean_mse']:.6f} +/- {result['std_mse']:.6f}"
    )
    return result
