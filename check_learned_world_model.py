"""Brief synthetic checks; no additional foraging experiment or tuning."""

import numpy as np

from foraging import value_iteration
from learned_world_model import EmpiricalModel, estimate, evaluate_policy, plan


def main():
    counts = EmpiricalModel(2, states=2, actions=3)
    # Seed 0 observes two distinct successors; seed 1 observes different rewards.
    counts.observe([0, 0], [0, 0], [1., 3.], [1, 0])
    counts.observe([0, 0], [0, 0], [0., 3.], [0, 0])
    p, r, unknown = estimate(counts.successors, counts.visits, counts.reward_sums)
    np.testing.assert_allclose(p.sum(axis=-1), 1)
    np.testing.assert_array_equal(counts.successors.sum(axis=-1), counts.visits)
    np.testing.assert_allclose(p[0, 0, 0], [.5, .5])
    np.testing.assert_allclose(r[:, 0, 0], [.5, 3.])
    assert unknown[:, 1].all() and counts.visits.sum() == 4
    np.testing.assert_allclose(p[:, 1, :, 1], 1)
    np.testing.assert_allclose(r[:, 1], 0)
    policies, values, _, _ = plan(p, r)
    # Known analytic values: seed 0 leaves the rewarding state with chance .5;
    # seed 1 receives 3 forever. Unknown state 1 has three uniformly tied actions.
    np.testing.assert_allclose(values[:, 0], [.5/(1-.99*.5), 3/(1-.99)])
    np.testing.assert_allclose(policies[:, 1], 1/3)
    np.testing.assert_allclose(values, evaluate_policy(policies, p, r))
    for seed in range(2):
        reference, expected, _, _, _ = value_iteration(p[seed], r[seed], .99)
        np.testing.assert_allclose(policies[seed], reference)
        np.testing.assert_allclose(values[seed], expected)
    print('Counts, independent seeds, unknown defaults, analytic values, planning/ties: OK')


if __name__ == '__main__':
    main()
