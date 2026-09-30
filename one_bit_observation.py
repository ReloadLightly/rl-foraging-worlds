"""Experiment 9 selection: one fixed hand-designed observation, archived data only."""

from time import perf_counter

import numpy as np

from learned_world_model import estimate, plan
from state_representation import STATES

CONFIG = dict(seeds=100, observations_per_seed=250000, gamma=.99,
              collectors=['Q-controlled', 'Model-controlled'],
              representations=['Full (A,B)', 'Total A+B', 'Total + bit A>B'],
              mechanism_states=[[4, 3], [3, 4], [3, 3], [2, 4]], illustration_seed=0,
              coverage_thresholds=[1, 10, 100], oracle_fraction=.9)
PAIRS = np.column_stack((STATES.sum(axis=1), (STATES[:, 0] > STATES[:, 1]).astype(int)))
GROUPS, OBSERVATION = np.unique(PAIRS, axis=0, return_inverse=True)
assert len(GROUPS) == 16


def aggregate(successors, visits, reward_sums):
    """Sum both transition axes through the mapping, keeping each action separate."""
    counts = np.zeros(visits.shape[:-2]+(16, 3, 16), dtype=successors.dtype)
    totals = np.zeros(visits.shape[:-2]+(16, 3), dtype=visits.dtype)
    rewards = np.zeros_like(totals, dtype=reward_sums.dtype)
    for s, z in enumerate(OBSERVATION):
        totals[..., z, :] += visits[..., s, :]
        rewards[..., z, :] += reward_sums[..., s, :]
        for successor, next_z in enumerate(OBSERVATION):
            counts[..., z, :, next_z] += successors[..., s, :, successor]
    np.testing.assert_array_equal(counts.sum(axis=-1), totals)
    np.testing.assert_array_equal(totals.sum(axis=-2), visits.sum(axis=-2))
    np.testing.assert_array_equal(rewards.sum(axis=-2), reward_sums.sum(axis=-2))
    # Direct check at each prespecified origin, including the still-merged pair.
    for z in np.unique(OBSERVATION[[23, 19, 18, 14]]):
        for next_z in range(16):
            direct = successors[..., OBSERVATION == z, :, :][..., OBSERVATION == next_z].sum(axis=(-3, -1))
            np.testing.assert_array_equal(counts[..., z, :, next_z], direct)
    return dict(successors=counts, visits=totals, reward_sums=rewards)


def lift(policy):
    lifted = policy[..., OBSERVATION, :]
    for s, z in enumerate(OBSERVATION):
        np.testing.assert_array_equal(lifted[..., s, :], policy[..., z, :])
    np.testing.assert_allclose(lifted.sum(axis=-1), 1., atol=1e-15)
    return lifted


def select(history):
    assert int(history['total_steps']) == CONFIG['observations_per_seed']
    full = {key:history[key][:2] for key in ('successors', 'visits', 'reward_sums')}
    assert full['visits'].shape == (2, 100, 25, 3)
    assert np.all(full['visits'].sum(axis=(-2, -1)) == 250000)
    np.testing.assert_array_equal(full['successors'].sum(axis=-1), full['visits'])
    started = perf_counter()
    stats = aggregate(**full)
    times = dict(aggregation_seconds=perf_counter()-started)
    started = perf_counter()
    p, r, unknown = estimate(**stats)
    policy, prediction, rounds_, residual = plan(p, r, CONFIG['gamma'])
    times['new_model_fitting_and_planning_seconds'] = perf_counter()-started
    action_values = r + CONFIG['gamma']*np.einsum('...san,...n->...sa', p, prediction)
    selected = dict(**stats, transition=p, rewards=r, unknown_rows=unknown, policy=policy,
        prediction=prediction, action_values=action_values, iterations=rounds_, residuals=residual,
        lifted_policy=lift(policy), lifted_prediction=prediction[..., OBSERVATION],
        states=STATES, groups=GROUPS, observation_mapping=OBSERVATION, training_seeds=np.arange(100))
    print('Selected only the new 16-group policies: two collectors, 100 seeds; no true-model access.', flush=True)
    return selected, times
