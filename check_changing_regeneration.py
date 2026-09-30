"""Brief deterministic checks specific to Experiment 7; no experiment replay."""

import numpy as np

from changing_regeneration import Memory, RHO, weighted_estimate
from foraging import regeneration_probability, step
from regeneration_regimes import changed_probability, changed_step


def check():
    stocks = np.repeat(np.arange(5)[:, None], 2, axis=1)
    np.testing.assert_allclose(changed_probability(stocks),
        [[.3, .6], [.2325, .465], [.165, .33], [.0975, .195], [0, 0]])
    np.testing.assert_allclose(regeneration_probability(stocks)[[0, 3, 4]],
                               [[.03, .06], [.2325, .465], [0, 0]])
    # Harvest at capacity precedes regeneration; rest at capacity cannot grow.
    initial = np.array([[4, 4], [4, 4], [0, 0]])
    actions = np.array([1, 2, 2])
    draws = np.array([[.2, .2], [0, 0], [.15, .3]])
    successor, reward = changed_step(initial, actions, draws)
    np.testing.assert_array_equal(successor, [[4, 3], [4, 4], [1, 1]])
    np.testing.assert_array_equal(reward, [1.5, 0, 0])
    np.testing.assert_array_equal(step(initial, actions, draws)[0], [[4, 4], [4, 4], [0, 0]])
    weights = np.array([[.25, 0., .75], [.1, 2., 0.]])
    successors = weights[..., None]*np.array([.3, .7])
    p, r, unknown = weighted_estimate(successors, weights, 1.5*weights)
    np.testing.assert_allclose(p.sum(axis=-1), 1.)
    np.testing.assert_allclose(p[~unknown], np.tile([.3, .7], (4, 1)))
    np.testing.assert_allclose(r[~unknown], 1.5)
    np.testing.assert_array_equal(p[unknown], [[1, 0], [0, 1]])
    np.testing.assert_array_equal(r[unknown], [0, 0])
    np.testing.assert_allclose(RHO**5, .5)
    # Independent model copies; actual counts stay separate from discounted weights.
    history = dict(successors=successors[None], visits=weights[None], reward_sums=1.5*weights[None])
    a, b = Memory(history, 2), Memory(history, 2)
    a.before_rollout()
    a.observe(np.array([0]), np.array([0]), np.array([1.5]), np.array([1]))
    np.testing.assert_array_equal(b.visits, history['visits'])
    np.testing.assert_allclose(a.visits[0, 0, 0], RHO*.25+1)
    np.testing.assert_allclose(a.actual_visits[0, 0, 0], 1.25)
    assert not np.shares_memory(a.visits, b.visits)
    print('Passed: reversed rule/event order, unchanged original rule, fractional/zero-row normalization, block decay and independent memories.')


if __name__ == '__main__':
    check()
