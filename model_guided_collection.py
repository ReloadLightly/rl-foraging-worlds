"""Online collection from copied experience. No true-model evaluation here."""

import json
from time import perf_counter

import numpy as np

from foraging import WORLD, state_index, step
from learned_world_model import EmpiricalModel, estimate, evaluate_policy, plan
from q_learning import QLearner, greedy_probabilities
from run_count_penalized_planning import planning_rewards
from run_q_learning import MASTER, METRICS, load_npz


CONFIG = dict(seeds=100, history_transitions=200000, additional_transitions=50000,
              total_transitions=250000, rollout_steps=1000, checkpoint_every=5000,
              gamma=.99, alpha=.1, epsilon=.1, penalty_coefficient=1.,
              frozen_trajectories_per_seed=20, frozen_steps=1000,
              illustration_seed=0, illustration_state=[3, 4], illustration_action=1,
              low_count_thresholds=[1, 10, 100])
COLLECTORS = ['A: Q-controlled', 'B: model-controlled', 'C: penalty-controlled']
PLANNERS = ['Unpenalized', 'Count-penalized']


def continuation_seeds():
    return {name: [int(np.random.SeedSequence([MASTER, 5, s, role]).generate_state(
                1, dtype=np.uint64)[0]) for s in range(CONFIG['seeds'])]
            for name, role in [('environment', 0), ('actions', 100)]}


def initialize(history):
    """Every condition owns its Q copy and model statistics; only A updates Q."""
    n = len(history['q'])
    learners, models = [], []
    for _ in COLLECTORS:
        learner = QLearner(n, CONFIG['gamma'], CONFIG['alpha'], CONFIG['epsilon'])
        learner.q[:] = history['q']
        learner.visits[:] = history['visits']
        model = EmpiricalModel(n)
        for name in ('successors', 'visits', 'reward_sums'):
            getattr(model, name)[:] = history[name]
        learners.append(learner)
        models.append(model)
    return learners, models


def planned_actions(policy, states, uniforms, epsilon):
    """Exactly QLearner.act's draw/tie conventions, using a planned policy."""
    cdf = policy[np.arange(len(states)), states].cumsum(axis=-1)
    cdf[:, -1] = 1.
    greedy = (uniforms[:, :1] >= cdf).sum(axis=-1)
    return np.where(uniforms[:, 1] < epsilon, (3*uniforms[:, 0]).astype(int), greedy)


def control_policies(learners, models):
    """B/C plan solely from their own accumulated observations."""
    policies = [greedy_probabilities(learners[0].q)]
    for c in (1, 2):
        model = models[c]
        p, r, _ = estimate(model.successors, model.visits, model.reward_sums)
        if c == 2:
            r, _ = planning_rewards(r, model.visits)
        policies.append(plan(p, r, CONFIG['gamma'])[0])
    return np.array(policies)


def extract_both(models):
    """Cross three independent datasets with two fixed planning rules."""
    counts, visits, reward_sums = [np.stack([getattr(m, name) for m in models])
                                 for name in ('successors', 'visits', 'reward_sums')]
    p, r, _ = estimate(counts, visits, reward_sums)
    penalized_r, penalty = planning_rewards(r, visits)
    plain, predicted_plain, rounds_plain, residual_plain = plan(p, r, CONFIG['gamma'])
    cautious, score, rounds_penalty, residual_penalty = plan(p, penalized_r, CONFIG['gamma'])
    predicted_cautious = evaluate_policy(cautious, p, r, CONFIG['gamma'])
    np.testing.assert_allclose(predicted_cautious-score,
                               evaluate_policy(cautious, p, penalty, CONFIG['gamma']), atol=1e-10)
    return dict(policies=np.stack((plain, cautious), axis=1),
                predicted_values=np.stack((predicted_plain, predicted_cautious), axis=1),
                internal_scores=np.stack((predicted_plain, score), axis=1),
                planning_iterations=np.stack((rounds_plain, rounds_penalty), axis=1),
                planning_residuals=np.stack((residual_plain, residual_penalty), axis=1))


def save_checkpoint(output, additional, learners, models, policies, curves, times, rngs):
    started = perf_counter()
    fitted = extract_both(models)
    times['extraction_seconds'] += perf_counter()-started
    # The two behavior planners and the same-rule extraction must agree.
    np.testing.assert_array_equal(policies[1], fitted['policies'][1, 0])
    np.testing.assert_array_equal(policies[2], fitted['policies'][2, 1])
    counts = {name: np.stack([getattr(m, name) for m in models])
              for name in ('successors', 'visits', 'reward_sums')}
    assert np.all(counts['visits'].sum(axis=(-2, -1)) == CONFIG['history_transitions']+additional)
    np.testing.assert_array_equal(counts['successors'].sum(axis=-1), counts['visits'])
    np.testing.assert_array_equal(learners[0].visits, models[0].visits)
    data = dict(additional_steps=additional, total_steps=CONFIG['history_transitions']+additional,
        q=np.stack([agent.q for agent in learners]), **counts, **fitted,
        collector_greedy=policies,
        collector_behavior=(1-CONFIG['epsilon'])*policies+CONFIG['epsilon']/3,
        rollout_metrics=curves[:, :additional//CONFIG['rollout_steps']],
        collection_metric_names=np.array(METRICS), **times,
        rng_states=np.array(json.dumps([r.bit_generator.state for group in rngs for r in group])))
    path = output/f'checkpoint_{additional:06d}.npz'
    with path.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream, **data)
    path.with_suffix('.tmp').replace(path)
    print(f'Saved +{additional:>6,} / total {int(data["total_steps"]):,} per condition/seed | '
          f'collection {times["collection_seconds"]:.2f}s, control planning '
          f'{times["control_planning_seconds"]:.2f}s', flush=True)


def collect(output, history, seeds):
    n, length = CONFIG['seeds'], CONFIG['rollout_steps']
    learners, models = initialize(history)
    rngs = [[np.random.default_rng(s) for s in seeds[name]] for name in ('environment', 'actions')]
    curves = np.zeros((3, CONFIG['additional_transitions']//length, n, len(METRICS)))
    times = dict(collection_seconds=0., control_planning_seconds=0., extraction_seconds=0.)
    checkpoints = sorted(output.glob('checkpoint_*.npz'))
    additional = 0
    if checkpoints:
        saved = load_npz(checkpoints[-1])
        additional = int(saved['additional_steps'])
        for c in range(3):
            learners[c].q[:] = saved['q'][c]
            learners[c].visits[:] = saved['visits'][c]
            for name in ('successors', 'visits', 'reward_sums'):
                getattr(models[c], name)[:] = saved[name][c]
        curves[:, :additional//length] = saved['rollout_metrics']
        policies = saved['collector_greedy']
        times = {key: float(saved[key]) for key in times}
        for rng, state in zip([r for group in rngs for r in group], json.loads(str(saved['rng_states']))):
            rng.bit_generator.state = state
        print(f'Resuming +{additional:,} saved transitions/seed; history not replayed.', flush=True)
    else:
        started = perf_counter()
        policies = control_policies(learners, models)
        times['control_planning_seconds'] += perf_counter()-started
        save_checkpoint(output, 0, learners, models, policies, curves, times, rngs)
    for rollout in range(additional//length, CONFIG['additional_transitions']//length):
        started = perf_counter()
        # Both roles use fresh per-seed streams, shared by all three conditions.
        env = np.stack([r.random((length, 2)) for r in rngs[0]], axis=1)
        action_draws = np.stack([r.random((length, 2)) for r in rngs[1]], axis=1)
        stocks = np.tile(WORLD.initial, (3, n, 1))
        for t in range(length):
            for c in range(3):
                states = state_index(stocks[c])
                actions = (learners[0].act(states, action_draws[t]) if c == 0 else
                           planned_actions(policies[c], states, action_draws[t], CONFIG['epsilon']))
                successor, reward = step(stocks[c], actions, env[t])
                next_states = state_index(successor)
                if c == 0:
                    learners[0].observe(states, actions, reward, next_states)
                models[c].observe(states, actions, reward, next_states)
                curves[c, rollout] += np.column_stack((reward, stocks[c],
                    (stocks[c] == 0).any(axis=1), actions == 2, (actions < 2) & (reward == 0)))
                stocks[c] = successor
        # Actual final successor is already counted and bootstrapped. The next
        # rollout's reset contributes no transition, reward, or terminal flag.
        curves[:, rollout] /= length
        times['collection_seconds'] += perf_counter()-started
        started = perf_counter()
        policies = control_policies(learners, models)
        times['control_planning_seconds'] += perf_counter()-started
        additional = (rollout+1)*length
        if additional % CONFIG['checkpoint_every'] == 0:
            save_checkpoint(output, additional, learners, models, policies, curves, times, rngs)
    return times
