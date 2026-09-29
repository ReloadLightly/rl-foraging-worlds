"""Brief checks of event order, transitions, numerical ties, and Bellman values."""

import numpy as np

from foraging import (exact_model, finite_window_value, greedy_policy,
                     policy_evaluation, policy_model, state_index, step, value_iteration)


def main():
    states, p, r = exact_model()
    assert p.shape == (25, 3, 25)
    np.testing.assert_allclose(p.sum(axis=2), 1, atol=1e-15)
    assert np.all(p >= 0)
    np.testing.assert_array_equal(state_index(states), np.arange(25))
    # Harvest first: A falls from 1 to 0, so .05 exceeds its NEW .03 chance.
    next_stocks, rewards = step(np.array([[1, 4], [4, 4], [0, 0]]),
                               np.array([0, 1, 0]), np.array([[.05, .0], [.0, .4], [.01, .99]]))
    np.testing.assert_array_equal(next_stocks, [[0, 4], [4, 4], [1, 0]])
    np.testing.assert_array_equal(rewards, [1, 1.5, 0])
    # Both depleted patches can regenerate independently, even after an empty harvest.
    expected = np.zeros(25)
    expected[[0, 1, 5, 6]] = [.97*.94, .97*.06, .03*.94, .03*.06]
    for a in range(3):
        np.testing.assert_allclose(p[0, a], expected)
    myopic = greedy_policy(r)
    np.testing.assert_allclose(myopic[0], np.ones(3)/3)
    np.testing.assert_allclose(policy_evaluation(myopic, p, r, 0), r.max(axis=1))
    # Always resting regenerates stocks but yields zero return forever.
    rest = np.tile([0., 0., 1.], (25, 1))
    np.testing.assert_array_equal(policy_evaluation(rest, p, r, .99), np.zeros(25))
    for gamma in (0, .5, .9, .99):
        policy, value, q, _, residual = value_iteration(p, r, gamma)
        p_pi, r_pi = policy_model(policy, p, r)
        np.testing.assert_allclose(value, r_pi + gamma*p_pi@value, atol=1e-11)
        assert residual < 1e-9
        assert np.all(value >= policy_evaluation(myopic, p, r, gamma) - 1e-10)
        if gamma == 0:
            np.testing.assert_array_equal(policy, myopic)
        np.testing.assert_allclose(finite_window_value(policy, p, r, gamma, 1), r_pi)
    print("Transition normalization, harvest-before-regrowth, independent recovery, ties, and Bellman equations: OK")


if __name__ == "__main__":
    main()
