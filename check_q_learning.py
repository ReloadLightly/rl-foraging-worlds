"""Brief checks of the new update, random ties, and continuing-task cutoff."""

import numpy as np

from q_learning import QLearner, greedy_probabilities


def main():
    learner = QLearner(2, gamma=.99)
    learner.q[0, 3] = [2, 5, 1]
    learner.q[0, 24] = 100  # Reset state must NOT supply the bootstrap target.
    learner.q[1, 7] = [4, 2, 0]
    learner.observe(np.array([2, 7]), np.array([1, 0]),
                    np.array([1.5, 0]), np.array([3, 7]))
    np.testing.assert_allclose(learner.q[0, 2, 1], .1 * (1.5 + .99 * 5))
    np.testing.assert_allclose(learner.q[1, 7, 0], 4 + .1 * (.99 * 4 - 4))
    assert learner.visits.sum() == 2
    # Gamma zero removes the bootstrap, regardless of successor estimates.
    immediate = QLearner(1, gamma=0)
    immediate.q[0, 3] = 100
    immediate.observe(np.array([2]), np.array([1]), np.array([1.5]), np.array([3]))
    np.testing.assert_allclose(immediate.q[0, 2, 1], .15)
    np.testing.assert_allclose(greedy_probabilities(np.zeros((2, 3))), 1/3)
    # Each of the three tie intervals selects its corresponding action.
    ties = QLearner(3, gamma=.99, epsilon=0)
    np.testing.assert_array_equal(ties.act(np.zeros(3, dtype=int),
                                         np.array([[.1, .5], [.5, .5], [.9, .5]])), [0, 1, 2])
    before = ties.q.copy()
    greedy_probabilities(ties.q)  # Frozen-policy extraction cannot update Q.
    np.testing.assert_array_equal(ties.q, before)
    print('Q-learning target, actual-successor bootstrap, gamma zero, ties, frozen extraction: OK')


if __name__ == '__main__':
    main()
