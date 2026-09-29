"""Brief checks of copied state, action pairing, accounting, and RNG restore."""

import copy

import numpy as np

from model_guided_collection import initialize, planned_actions
from q_learning import greedy_probabilities


def main():
    history = dict(q=np.zeros((2, 25, 3)), successors=np.zeros((2, 25, 3, 25), dtype=int),
                   visits=np.zeros((2, 25, 3), dtype=int), reward_sums=np.zeros((2, 25, 3)))
    learners, models = initialize(history)
    for c in range(3):
        assert not np.shares_memory(learners[c].q, history['q'])
        for d in range(c):
            assert not np.shares_memory(learners[c].q, learners[d].q)
            assert not np.shares_memory(models[c].successors, models[d].successors)
    states, actions, rewards, next_states = [np.array(x) for x in ([2, 2], [1, 0], [1., 0.], [3, 7])]
    learners[0].q[:, 24] = 100  # Reset stock must not supply a bootstrap target.
    learners[0].q[0, 3] = [2, 5, 1]
    learners[0].observe(states, actions, rewards, next_states)
    models[0].observe(states, actions, rewards, next_states)
    np.testing.assert_allclose(learners[0].q[0, 2, 1], .1*(1+.99*5))
    assert models[0].successors[0, 2, 1, 3] == 1
    assert models[0].successors.sum() == models[0].visits.sum() == 2
    assert models[1].visits.sum() == models[2].visits.sum() == 0
    assert not history['q'].any() and not history['visits'].any()
    assert not learners[1].q.any() and not learners[2].q.any()
    draws = np.array([[.4, .01], [.8, .9]])
    np.testing.assert_array_equal(learners[0].act(states, draws),
        planned_actions(greedy_probabilities(learners[0].q), states, draws, .1))
    rng = np.random.default_rng(123)
    rng.random((7, 2))
    saved = copy.deepcopy(rng.bit_generator.state)
    expected = rng.random((9, 2))
    restored = np.random.default_rng()
    restored.bit_generator.state = saved
    np.testing.assert_array_equal(restored.random((9, 2)), expected)
    print('Independent conditions, paired action convention, actual-successor counts/bootstrap, RNG restore: OK')


if __name__ == '__main__':
    main()
