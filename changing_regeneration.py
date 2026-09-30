"""Three independent memories interact with two worlds; no oracle access here."""

import json
from time import perf_counter

import numpy as np

from foraging import WORLD, state_index, step
from learned_world_model import plan
from model_guided_collection import planned_actions
from q_learning import greedy_probabilities
from regeneration_regimes import changed_step
from run_q_learning import MASTER, METRICS, load_npz

CONFIG = dict(seeds=100, history_transitions=250000, additional_transitions=50000,
              total_transitions=300000, rollout_steps=1000, checkpoint_every=5000,
              snapshot_every=1000, gamma=.99, epsilon=.1, evidence_half_life=5000,
              illustration_seed=0, illustration_state=[3, 3])
RHO = 2**(-CONFIG['rollout_steps']/CONFIG['evidence_half_life'])
CONDITIONS = ['Stable', 'Changed']
METHODS = ['Frozen', 'Cumulative', 'Forgetting']
FIELDS = ('successors', 'visits', 'reward_sums')
SNAPSHOT_KEYS = ('policies', 'predictions', 'planning_residuals', 'planning_iterations',
                 'seed0_successor_probabilities', 'seed0_action_values', 'seed0_actual_visits',
                 'seed0_visit_weights', 'total_evidence', 'unknown_rows', 'fractional_rows')


def collection_seeds():
    return {role: [int(np.random.SeedSequence([MASTER, 7, s, code]).generate_state(
        1, dtype=np.uint64)[0]) for s in range(CONFIG['seeds'])]
        for role, code in [('environment', 0), ('actions', 100)]}


def weighted_estimate(successors, visits, reward_sums):
    """Normalize every positive weight by itself, even below one."""
    unknown = visits == 0
    denominator = np.where(unknown, 1., visits)
    p = successors/denominator[..., None]
    p += unknown[..., None]*np.eye(visits.shape[-2])[:, None, :]
    return p, reward_sums/denominator, unknown


class Memory:
    """The method knows its memory rule, not its environmental condition."""

    def __init__(self, history, method):
        self.method = method
        self.rows = np.arange(len(history['visits']))
        for name in FIELDS:
            dtype = float if method == 2 or name == 'reward_sums' else np.int32
            setattr(self, name, np.array(history[name], dtype=dtype, copy=True))
            # Actual observations are diagnostic accounting, never used in C's fit.
            setattr(self, 'actual_'+name, np.array(history[name], copy=True))
            if method == 0:
                getattr(self, name).setflags(write=False)

    def before_rollout(self):
        if self.method == 2:
            for name in FIELDS:
                getattr(self, name)[:] *= RHO

    def observe(self, states, actions, reward, successor):
        idx = self.rows, states, actions
        for prefix in ('actual_', '') if self.method else ('actual_',):
            getattr(self, prefix+'successors')[idx+(successor,)] += 1
            getattr(self, prefix+'visits')[idx] += 1
            getattr(self, prefix+'reward_sums')[idx] += reward


def initialize(history):
    return [[Memory(history, m) for m in range(3)] for _ in range(2)]


def snapshot(memories, policies, inherited_prediction, first=False):
    weights = {name: np.array([[getattr(m, name) for m in group] for group in memories]) for name in FIELDS}
    p, r, unknown = weighted_estimate(**weights)
    predictions = np.broadcast_to(inherited_prediction, (2, 3)+inherited_prediction.shape).copy()
    residuals, iterations = np.zeros((2, 3, CONFIG['seeds'])), np.zeros((2, 3, CONFIG['seeds']), int)
    if not first:
        selected, predicted, rounds_, residual = plan(p[:, 1:], r[:, 1:], CONFIG['gamma'])
        policies[:, 1:] = selected
        predictions[:, 1:] = predicted
        residuals[:, 1:], iterations[:, 1:] = residual, rounds_
    else:
        q = r+CONFIG['gamma']*np.einsum('...san,...n->...sa', p, predictions)
        np.testing.assert_array_equal(greedy_probabilities(q), policies)
    s = 18  # (3,3), seed 0, fixed before results.
    q0 = r[:, :, 0, s]+CONFIG['gamma']*np.einsum('...an,...n->...a', p[:, :, 0, s], predictions[:, :, 0])
    return dict(policies=policies.copy(), predictions=predictions,
        planning_residuals=residuals, planning_iterations=iterations,
        seed0_successor_probabilities=p[:, :, 0, s], seed0_action_values=q0,
        seed0_actual_visits=np.array([[m.actual_visits[0, s] for m in group] for group in memories]),
        seed0_visit_weights=weights['visits'][:, :, 0, s], total_evidence=weights['visits'].sum(axis=(-2, -1)),
        unknown_rows=unknown.sum(axis=(-2, -1)), fractional_rows=((weights['visits'] > 0) & (weights['visits'] < 1)).sum(axis=(-2, -1)))


def save_checkpoint(output, additional, memories, policies, snapshots, curves, rngs, times, initial):
    block = additional//CONFIG['rollout_steps']
    stats = {prefix+name: np.array([[getattr(m, prefix+name) for m in group] for group in memories])
             for prefix in ('', 'actual_') for name in FIELDS}
    np.testing.assert_array_equal(stats['actual_successors'].sum(axis=-1), stats['actual_visits'])
    assert np.all(stats['actual_visits'].sum(axis=(-2, -1)) == CONFIG['history_transitions']+additional)
    np.testing.assert_allclose(stats['successors'].sum(axis=-1), stats['visits'], rtol=1e-12, atol=1e-9)
    expected = CONFIG['history_transitions']*RHO**block+CONFIG['rollout_steps']*(1-RHO**block)/(1-RHO)
    np.testing.assert_allclose(stats['visits'][:, 2].sum(axis=(-2, -1)), expected, rtol=1e-12)
    for name in FIELDS:
        np.testing.assert_array_equal(stats[name][:, 1], stats['actual_'+name][:, 1])
        for g in range(2):
            np.testing.assert_array_equal(stats[name][g, 0], initial[name])
            assert not getattr(memories[g][0], name).flags.writeable
    np.testing.assert_array_equal(policies[:, 0], np.broadcast_to(snapshots[0]['policies'][0, 0], policies[:, 0].shape))
    np.testing.assert_allclose((stats['actual_reward_sums']-initial['reward_sums']).sum(axis=(-2, -1)),
                               CONFIG['rollout_steps']*curves[:, :, :block, :, 0].sum(axis=2), rtol=0, atol=1e-8)
    data = dict(additional_steps=additional, total_steps=CONFIG['history_transitions']+additional,
        **stats, **{key: np.stack([s[key] for s in snapshots]) for key in SNAPSHOT_KEYS},
        rollout_metrics=curves[:, :, :block], collection_metric_names=np.array(METRICS), **times,
        rng_states=np.array(json.dumps([r.bit_generator.state for group in rngs for r in group])))
    path = output/f'checkpoint_{additional:06d}.npz'
    with path.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream, **data)
    path.with_suffix('.tmp').replace(path)
    print(f'+{additional:,} / total {250000+additional:,} observations per method/condition/seed', flush=True)
    if block:
        recent = curves[:, :, max(0, block-5):block].mean(axis=(2, 3))
        for g, condition in enumerate(CONDITIONS):
            parts = [f'{METHODS[m]}: r={v[0]:.3f}, A/B={v[1]:.2f}/{v[2]:.2f}, empty={v[3]:.1%}' for m,v in enumerate(recent[g])]
            print('  '+condition+' | '+'; '.join(parts), flush=True)
        belief = snapshots[-1]['seed0_successor_probabilities'][1, 2, 2, 18]
        count = stats['actual_visits'][1, 2, 0, 18, 2]
        weight = stats['visits'][1, 2, 0, 18, 2]
        print(f'  Seed 0 changed/forgetting, (3,3)/rest: actual N={count:,}, weight={weight:.2f}, P(self)={belief:.4f}', flush=True)


def collect(output, history, inherited_policy, inherited_prediction, seeds):
    memories = initialize(history)
    policies = np.broadcast_to(inherited_policy, (2, 3)+inherited_policy.shape).copy()
    n, length = CONFIG['seeds'], CONFIG['rollout_steps']
    rngs = [[np.random.default_rng(s) for s in seeds[role]] for role in ('environment', 'actions')]
    curves = np.zeros((2, 3, 50, n, len(METRICS)))
    times = dict(collection_seconds=0., planning_seconds=0.)
    snapshots, additional = [], 0
    checkpoints = sorted(output.glob('checkpoint_*.npz'))
    if checkpoints:
        saved = load_npz(checkpoints[-1])
        additional = int(saved['additional_steps'])
        for g, group in enumerate(memories):
            for m, memory in enumerate(group):
                for prefix in ('', 'actual_'):
                    for name in FIELDS:
                        if prefix == '' and m == 0:
                            np.testing.assert_array_equal(saved[name][g, m], getattr(memory, name))
                        else:
                            getattr(memory, prefix+name)[:] = saved[prefix+name][g, m]
        curves[:, :, :additional//length] = saved['rollout_metrics']
        snapshots = [{key: saved[key][i] for key in SNAPSHOT_KEYS} for i in range(additional//length+1)]
        policies[:] = snapshots[-1]['policies']
        times = {key: float(saved[key]) for key in times}
        for rng, state in zip([r for group in rngs for r in group], json.loads(str(saved['rng_states']))):
            rng.bit_generator.state = state
        print(f'Resuming saved +{additional:,} observations per seed.', flush=True)
    else:
        snapshots.append(snapshot(memories, policies, inherited_prediction, first=True))
        save_checkpoint(output, 0, memories, policies, snapshots, curves, rngs, times, history)
    environments = (step, changed_step)  # Labels/rules stay with environment, outside Memory.
    for rollout in range(additional//length, CONFIG['additional_transitions']//length):
        started = perf_counter()
        for group in memories:
            for memory in group:
                memory.before_rollout()  # Identical schedule in BOTH conditions.
        env = np.stack([r.random((length, 2)) for r in rngs[0]], axis=1)
        action_draws = np.stack([r.random((length, 2)) for r in rngs[1]], axis=1)
        stocks = np.tile(WORLD.initial, (2, 3, n, 1))
        for t in range(length):
            for g, environment in enumerate(environments):
                for m, memory in enumerate(memories[g]):
                    states = state_index(stocks[g, m])
                    actions = planned_actions(policies[g, m], states, action_draws[t], CONFIG['epsilon'])
                    successor, reward = environment(stocks[g, m], actions, env[t])
                    memory.observe(states, actions, reward, state_index(successor))
                    curves[g, m, rollout] += np.column_stack((reward, stocks[g, m],
                        (stocks[g, m] == 0).any(axis=1), actions == 2, (actions < 2) & (reward == 0)))
                    stocks[g, m] = successor
        # Final actual successor counted; reset adds no observation or terminal flag.
        curves[:, :, rollout] /= length
        if rollout == 0:
            for g in range(2):
                for m in (1, 2):
                    np.testing.assert_array_equal(curves[g, m, 0], curves[g, 0, 0])
        times['collection_seconds'] += perf_counter()-started
        started = perf_counter()
        snapshots.append(snapshot(memories, policies, inherited_prediction))
        times['planning_seconds'] += perf_counter()-started
        additional = (rollout+1)*length
        if additional % CONFIG['checkpoint_every'] == 0:
            save_checkpoint(output, additional, memories, policies, snapshots, curves, rngs, times, history)
    return times
