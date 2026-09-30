"""Fit full-state and total-stock models from saved observations only."""

from time import perf_counter

import numpy as np

from learned_world_model import estimate, plan

CONFIG = dict(seeds=100, observations_per_seed=250000, gamma=.99,
              collectors=['Q-controlled', 'Model-controlled'],
              representations=['Full (A,B)', 'Total A+B'],
              mechanism_states=[[4, 3], [3, 4]], illustration_seed=0,
              coverage_thresholds=[1, 10, 100], oracle_fraction=.9)
STATES = np.array([(a, b) for a in range(5) for b in range(5)])
OBSERVATION = STATES.sum(axis=-1)


def aggregate(successors, visits, reward_sums):
    """Exact summation through z=A+B on both sides; never merge action identities."""
    counts = np.zeros(visits.shape[:-2]+(9, 3, 9), dtype=successors.dtype)
    totals = np.zeros(visits.shape[:-2]+(9, 3), dtype=visits.dtype)
    rewards = np.zeros_like(totals, dtype=reward_sums.dtype)
    for s, z in enumerate(OBSERVATION):
        totals[..., z, :] += visits[..., s, :]
        rewards[..., z, :] += reward_sums[..., s, :]
        for successor, next_z in enumerate(OBSERVATION):
            counts[..., z, :, next_z] += successors[..., s, :, successor]
    np.testing.assert_array_equal(counts.sum(axis=-1), totals)
    # Preserve each action's totals separately, not just the number of transitions.
    np.testing.assert_array_equal(totals.sum(axis=-2), visits.sum(axis=-2))
    np.testing.assert_array_equal(rewards.sum(axis=-2), reward_sums.sum(axis=-2))
    # Direct indexing check for the prespecified merged observation, all successors.
    members = np.flatnonzero(OBSERVATION == 7)
    for next_z in range(9):
        direct = successors[..., members, :, :][..., OBSERVATION == next_z].sum(axis=(-3, -1))
        np.testing.assert_array_equal(counts[..., 7, :, next_z], direct)
    return dict(successors=counts, visits=totals, reward_sums=rewards)


def lift(policy):
    """A shared action distribution for every full state with the same total."""
    lifted = policy[..., OBSERVATION, :]
    for z in range(9):
        for s in np.flatnonzero(OBSERVATION == z):
            np.testing.assert_array_equal(lifted[..., s, :], policy[..., z, :])
    np.testing.assert_allclose(lifted.sum(axis=-1), 1., atol=1e-15)
    return lifted


def select(archive):
    assert int(archive['total_steps']) == CONFIG['observations_per_seed']
    full = {key:archive[key][:2].copy() for key in ('successors', 'visits', 'reward_sums')}
    assert full['visits'].shape == (2, 100, 25, 3)
    assert np.all(full['visits'].sum(axis=(-2, -1)) == CONFIG['observations_per_seed'])
    np.testing.assert_array_equal(full['successors'].sum(axis=-1), full['visits'])
    started = perf_counter()
    compressed = aggregate(**full)
    times = dict(aggregation_seconds=perf_counter()-started)
    data = dict(training_seeds=np.arange(100), states=STATES, observation_mapping=OBSERVATION)
    for label, stats in [('full', full), ('compressed', compressed)]:
        started = perf_counter()
        p, r, unknown = estimate(**stats)
        policy, prediction, rounds_, residual = plan(p, r, CONFIG['gamma'])
        times[label+'_fitting_and_planning_seconds'] = perf_counter()-started
        for key, value in dict(**stats, transition=p, rewards=r, unknown_rows=unknown,
                               policy=policy, prediction=prediction, iterations=rounds_, residuals=residual).items():
            data[label+'_'+key] = value
        print(f'Selected {label}: both datasets, all 100 seeds; no true-model access.', flush=True)
    # Full-state refits must recover the corresponding existing unpenalized policies.
    np.testing.assert_array_equal(data['full_policy'], archive['policies'][:2, 0])
    np.testing.assert_allclose(data['full_prediction'], archive['predicted_values'][:2, 0], atol=1e-10)
    data['policies'] = np.stack((data['full_policy'], lift(data['compressed_policy'])), axis=1)
    data['predictions'] = np.stack((data['full_prediction'], data['compressed_prediction'][..., OBSERVATION]), axis=1)
    return data, times
