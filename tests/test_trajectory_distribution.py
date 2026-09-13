import numpy as np
import pytest

from src.trajectory_distribution import TrajectoryReference, trajectory_distances


def test_pairwise_mse_rewards_collapse_but_distribution_metrics_do_not():
    x = np.zeros((20, 100, 2))
    x[:10, :, 0] = -1
    x[10:, :, 0] = 1
    mean_only = np.zeros_like(x)
    ref = TrajectoryReference.fit(x)
    correct = trajectory_distances(x, x, ref)
    collapsed = trajectory_distances(x, mean_only, ref)
    assert collapsed['all_pairs_mse_diagnostic'] < correct['all_pairs_mse_diagnostic']
    assert correct['energy'] == pytest.approx(0, abs=1e-12)
    assert correct['mmd2_biased_diagnostic'] == pytest.approx(0, abs=1e-12)
    assert collapsed['energy'] > correct['energy']
    assert collapsed['mmd2'] > correct['mmd2']
    assert correct['dispersion_ratio'] == pytest.approx(1)
    assert collapsed['dispersion_ratio'] == 0


def test_matches_independent_pair_loop_and_preserves_sample_order_invariance():
    rng = np.random.default_rng(13)
    x, y = rng.normal(size=(4, 100, 2)), rng.normal(size=(5, 100, 2))
    ref = TrajectoryReference.fit(x)
    got = trajectory_distances(x, y, ref)
    d = lambda a, b: np.sqrt(np.mean((a-b)**2))
    avg = lambda a, b: np.mean([d(i, j) for i in a for j in b])
    assert got['energy'] == pytest.approx(2*avg(x,y)-avg(x,x)-avg(y,y))
    k = lambda a,b: np.exp(-ref.gamma*d(a,b)**2)
    off = lambda a: np.mean([k(a[i],a[j]) for i in range(len(a)) for j in range(len(a)) if i != j])
    expected = off(x)+off(y)-2*np.mean([k(a,b) for a in x for b in y])
    assert got['mmd2'] == pytest.approx(expected)
    reordered = trajectory_distances(x[::-1], y[[2,1,4,0,3]], ref)
    for key in got:
        assert got[key] == pytest.approx(reordered[key], abs=1e-12)


def test_time_order_is_retained_and_global_translation_cancels():
    rng = np.random.default_rng(3)
    x = rng.normal(scale=.01, size=(12,100,2))
    x[:,:,1] += np.linspace(0,10,100)
    ref = TrajectoryReference.fit(x)
    reversed_time = trajectory_distances(x, x[:,::-1], ref)
    assert reversed_time['energy'] > 1
    translated = trajectory_distances(x+20, x[:,::-1]+20, ref)
    assert translated['energy'] == pytest.approx(reversed_time['energy'])


def test_axis_reference_uses_training_residual_variation():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(8,100,2))*[2,5]
    ref = TrajectoryReference.fit(x, 'axis_balanced')
    expected = np.sqrt(np.mean((x-x.mean(0))**2, axis=(0,1)))
    np.testing.assert_allclose(ref.axis_scale, expected)
    before = ref.to_dict()
    trajectory_distances(x, x*100, ref)
    assert ref.to_dict() == before


@pytest.mark.parametrize('bad', [np.zeros((1,100,2)), np.zeros((4,2,100)), np.full((4,100,2), np.nan)])
def test_invalid_paths_fail_explicitly(bad):
    with pytest.raises(ValueError):
        TrajectoryReference.fit(bad)
