"""Experiment 6 collection: frozen behavior, evolving empirical model.

Only observations enter the model. Extracted checkpoint policies never control
collection; true-model evaluation lives in the separate runner.
"""

import json
from time import perf_counter

import numpy as np

from foraging import WORLD, state_index, step
from learned_world_model import EmpiricalModel, estimate, plan
from model_guided_collection import planned_actions
from run_q_learning import METRICS, load_npz

CONFIG = dict(seeds=100, history_transitions=200000, additional_transitions=50000,
              total_transitions=250000, rollout_steps=1000, checkpoint_every=5000,
              gamma=.99, epsilon=.1, frozen_trajectories_per_seed=20,
              frozen_steps=1000, illustration_seed=0, illustration_state=[3, 4],
              illustration_action=1, low_count_thresholds=[1, 10, 100])
COLLECTORS = ['Q-controlled (saved A)', 'Adaptive model (saved B)', 'Fixed initial model policy']


def initialize(history, initial):
    """Copy the 200k history, never the Experiment 5 endpoint."""
    model = EmpiricalModel(CONFIG['seeds'])
    for name in ('successors', 'visits', 'reward_sums'):
        np.testing.assert_array_equal(history[name], initial[name][1])
        getattr(model, name)[:] = history[name]
        assert not np.shares_memory(getattr(model, name), history[name])
    policy = initial['policies'][1, 0].copy()
    np.testing.assert_array_equal(policy, initial['collector_greedy'][1])
    assert int(initial['additional_steps']) == 0
    assert np.all(model.visits.sum(axis=(-2, -1)) == CONFIG['history_transitions'])
    policy.setflags(write=False)
    return model, policy


def save_checkpoint(output, baseline, additional, model, fixed, curves, times, rngs):
    started = perf_counter()
    p, r, unknown = estimate(model.successors, model.visits, model.reward_sums)
    policy, prediction, iterations, residual = plan(p, r, CONFIG['gamma'])
    times['extraction_seconds'] += perf_counter()-started
    saved_baseline = load_npz(baseline/f'checkpoint_{additional:06d}.npz')
    states = json.dumps([rng.bit_generator.state for group in rngs for rng in group])
    # Same generator states AND same draw shape/order establish paired streams.
    assert states == str(saved_baseline['rng_states'])
    np.testing.assert_array_equal(fixed, load_npz(baseline/'checkpoint_000000.npz')['policies'][1, 0])
    assert not fixed.flags.writeable
    assert np.all(model.visits.sum(axis=(-2, -1)) == CONFIG['history_transitions']+additional)
    np.testing.assert_array_equal(model.successors.sum(axis=-1), model.visits)
    if additional == 0:
        np.testing.assert_array_equal(policy, fixed)
    completed = additional//CONFIG['rollout_steps']
    data = dict(additional_steps=additional, total_steps=CONFIG['history_transitions']+additional,
        successors=model.successors, visits=model.visits, reward_sums=model.reward_sums,
        policies=policy, predicted_values=prediction, unknown_rows=unknown,
        planning_iterations=iterations, planning_residuals=residual,
        collector_greedy=fixed, collector_behavior=(1-CONFIG['epsilon'])*fixed+CONFIG['epsilon']/3,
        rollout_metrics=curves[:completed], collection_metric_names=np.array(METRICS),
        rng_states=np.array(states), **times)
    path = output/f'checkpoint_{additional:06d}.npz'
    with path.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream, **data)
    path.with_suffix('.tmp').replace(path)
    block = curves[max(0, completed-5):completed].mean(axis=(0, 1)) if completed else None
    observation_count = int(model.visits[0, 19, 1])
    belief = model.successors[0, 19, 1, 19]/observation_count
    behavior = ('collection not started' if block is None else
                f'last 5 rollouts: reward {block[0]:.4f}, stocks A/B {block[1]:.3f}/{block[2]:.3f}, depleted {block[3]:.1%}')
    print(f'+{additional:>6,} / total {200000+additional:,} per seed | {behavior} | '
          f'seed 0 (3,4), B: N={observation_count:,}, self-loop={belief:.4f}', flush=True)


def collect(output, baseline, history, initial, seeds):
    model, fixed = initialize(history, initial)
    n, length = CONFIG['seeds'], CONFIG['rollout_steps']
    rngs = [[np.random.default_rng(s) for s in seeds[role]] for role in ('environment', 'actions')]
    curves = np.zeros((CONFIG['additional_transitions']//length, n, len(METRICS)))
    times = dict(collection_seconds=0., extraction_seconds=0., control_planning_seconds=0.)
    checkpoints = sorted(output.glob('checkpoint_*.npz'))
    additional = 0
    if checkpoints:
        saved = load_npz(checkpoints[-1])
        additional = int(saved['additional_steps'])
        np.testing.assert_array_equal(saved['collector_greedy'], fixed)
        for name in ('successors', 'visits', 'reward_sums'):
            getattr(model, name)[:] = saved[name]
        curves[:additional//length] = saved['rollout_metrics']
        times = {key: float(saved[key]) for key in times}
        for rng, state in zip([r for group in rngs for r in group], json.loads(str(saved['rng_states']))):
            rng.bit_generator.state = state
        print(f'Resuming saved +{additional:,} transitions per seed.', flush=True)
    else:
        save_checkpoint(output, baseline, 0, model, fixed, curves, times, rngs)
    for rollout in range(additional//length, CONFIG['additional_transitions']//length):
        started = perf_counter()
        # Exact Experiment 5 seed IDs, role order, shape and stacking order.
        env = np.stack([r.random((length, 2)) for r in rngs[0]], axis=1)
        action_draws = np.stack([r.random((length, 2)) for r in rngs[1]], axis=1)
        stocks = np.tile(WORLD.initial, (n, 1))
        for t in range(length):
            states = state_index(stocks)
            actions = planned_actions(fixed, states, action_draws[t], CONFIG['epsilon'])
            successor, reward = step(stocks, actions, env[t])
            model.observe(states, actions, reward, state_index(successor))
            curves[rollout] += np.column_stack((reward, stocks,
                (stocks == 0).any(axis=1), actions == 2, (actions < 2) & (reward == 0)))
            stocks = successor
        # The actual final successor was counted. Reset is not an observation.
        curves[rollout] /= length
        times['collection_seconds'] += perf_counter()-started
        additional = (rollout+1)*length
        if additional % CONFIG['checkpoint_every'] == 0:
            save_checkpoint(output, baseline, additional, model, fixed, curves, times, rngs)
    np.testing.assert_allclose((model.reward_sums-history['reward_sums']).sum(axis=(-2, -1)),
                               length*curves[:, :, 0].sum(axis=0), atol=1e-8, rtol=0)
    return times
