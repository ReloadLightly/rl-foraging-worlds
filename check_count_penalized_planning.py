"""Brief synthetic checks for the fixed penalty and A/B reward accounting."""

import numpy as np

from learned_world_model import evaluate_policy, plan
from run_count_penalized_planning import planning_rewards


def main():
    r = np.array([[1.5, 1., 0.]])
    original = r.copy()
    adjusted, penalty = planning_rewards(r, np.array([[1, 100, 0]]))
    np.testing.assert_array_equal(r, original)
    np.testing.assert_allclose(penalty, [[1., .1, 1.]])
    np.testing.assert_allclose(planning_rewards(np.zeros(4), np.array([0, 1, 4, 100]))[1], [1., 1., .5, .1])
    np.testing.assert_allclose(adjusted, [[.5, .9, -1.]])
    p = np.ones((1, 3, 1))
    pi, score, _, _ = plan(p, adjusted)
    np.testing.assert_allclose(pi, [[0, 1, 0]])
    prediction = evaluate_policy(pi, p, r)
    np.testing.assert_allclose(score, [90.])
    np.testing.assert_allclose(prediction, [100.])
    np.testing.assert_allclose(prediction-score, evaluate_policy(pi, p, penalty))
    # No data: unknown empirical reward zero, but internal score is negative.
    unknown_reward, _ = planning_rewards(np.zeros((1, 3)), np.zeros((1, 3), dtype=int))
    unknown_pi, unknown_score, _, _ = plan(p, unknown_reward)
    np.testing.assert_allclose(unknown_pi, 1/3)
    np.testing.assert_allclose(unknown_score, [-100.])
    np.testing.assert_allclose(evaluate_policy(unknown_pi, p, np.zeros((1, 3))), [0.])
    print('Fixed count penalty, unchanged empirical rewards, negative scores, A/B accounting: OK')


if __name__ == '__main__':
    main()
