"""Experiment 3: replay Q's experience once, then plan with empirical dynamics."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from foraging import TIE_ATOL, WORLD, policy_evaluation, state_index, step
from learned_world_model import EmpiricalModel, estimate, evaluate_policy, plan
from q_learning import QLearner, greedy_probabilities
from run_experiment import CURVES, OUTCOMES
from run_q_learning import MASTER, describe, fraction_interval, load_npz, write_json


CONFIG = dict(training_seeds=100, transitions=200_000, rollout_steps=1000,
              checkpoint_every=10_000, gamma_train=.99, gamma_plan=.99,
              gamma_eval=.99, alpha=.1, epsilon=.1, q_initial=0.,
              evaluation_epsilon=0., evaluation_trajectories_per_seed=20,
              evaluation_steps=1000, bin_width=20, illustration_seed=0,
              illustration_trajectory=0, illustration_steps=120)
METHODS = ['Q-learning', 'Learned-model planning']
METRICS = OUTCOMES + ['harvest_A_per_step', 'harvest_B_per_step', 'reward_per_step']
ROOT = Path(__file__).resolve().parent
BASELINE = ROOT/'results/q_learning'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(output, seeds):
    """Only Q controls collection. Planning and true-model evaluation run later."""
    n, length = CONFIG['training_seeds'], CONFIG['rollout_steps']
    learner = QLearner(n, CONFIG['gamma_train'], CONFIG['alpha'], CONFIG['epsilon'])
    counts = EmpiricalModel(n)
    env_rngs = [np.random.default_rng(s) for s in seeds['regeneration']]
    act_rngs = [np.random.default_rng(s) for s in seeds['actions']]
    snapshots = []
    elapsed = 0.

    def snapshot(completed):
        assert np.all(counts.visits.sum(axis=(1, 2)) == completed)
        np.testing.assert_array_equal(counts.successors.sum(axis=-1), counts.visits)
        np.testing.assert_array_equal(counts.visits, learner.visits)
        snapshots.append(dict(steps=completed, q=learner.q.copy(),
                              visits=counts.visits.copy(), successors=counts.successors.copy(),
                              reward_sums=counts.reward_sums.copy(), seconds=elapsed))
        print(f'Collected {completed:>7,}/seed | {elapsed:.2f} s', flush=True)

    snapshot(0)
    for rollout in range(CONFIG['transitions']//length):
        started = perf_counter()
        # Original environment/action streams, draw shape and order, and update.
        environment_draws = np.stack([r.random((length, 2)) for r in env_rngs], axis=1)
        action_draws = np.stack([r.random((length, 2)) for r in act_rngs], axis=1)
        stocks = np.tile(WORLD.initial, (n, 1))
        for t in range(length):
            states = state_index(stocks)
            actions = learner.act(states, action_draws[t])
            successor, reward = step(stocks, actions, environment_draws[t])
            next_states = state_index(successor)
            learner.observe(states, actions, reward, next_states)
            counts.observe(states, actions, reward, next_states)
            stocks = successor
        # The real final successor was counted and bootstrapped. Restarting the
        # next rollout creates no observation and does not enter the model.
        elapsed += perf_counter()-started
        completed = (rollout+1)*length
        if completed % CONFIG['checkpoint_every'] == 0:
            snapshot(completed)
    data = {key: np.stack([c[key] for c in snapshots]) for key in snapshots[0]}
    data['rng_states'] = np.array(json.dumps([r.bit_generator.state for r in env_rngs+act_rngs]))
    np.savez_compressed(output/'collection.npz', **data)
    return data


def compare_replay(collection):
    """Archive comparisons are diagnostics; no baseline table enters learning."""
    q_error, visit_error = [], []
    for i, completed in enumerate(collection['steps']):
        prior = load_npz(BASELINE/f'checkpoint_{completed:06d}.npz')
        q_error.append(float(np.max(np.abs(collection['q'][i]-prior['q'][0]))))
        visit_error.append(int(np.max(np.abs(collection['visits'][i]-prior['visits'][0]))))
    original_rng = json.loads(str(prior['rng_states']))[:2*CONFIG['training_seeds']]
    return dict(checkpoint_q_max_abs_difference=q_error,
                checkpoint_visit_max_abs_difference=visit_error,
                all_checkpoint_q_tables_bitwise_equal=all(e == 0 for e in q_error),
                all_checkpoint_visit_counts_equal=all(e == 0 for e in visit_error),
                final_rng_states_equal=json.loads(str(collection['rng_states'])) == original_rng,
                final_q_max_abs_difference=q_error[-1], final_visit_max_abs_difference=visit_error[-1])


def plan_and_evaluate(collection, true_model):
    policies, actual, predicted, iterations, residuals, runtimes, eval_times = [], [], [], [], [], [], []
    for i, completed in enumerate(collection['steps']):
        started = perf_counter()
        p, r, unknown = estimate(collection['successors'][i], collection['visits'][i],
                                 collection['reward_sums'][i])
        np.testing.assert_allclose(p.sum(axis=-1), 1, atol=1e-14)
        model_policy, prediction, count, residual = plan(p, r, CONFIG['gamma_plan'])
        runtimes.append(perf_counter()-started)
        policies.append(np.stack((greedy_probabilities(collection['q'][i]), model_policy)))
        predicted.append(prediction)
        iterations.append(count)
        residuals.append(residual)
        # Separate evaluation: neither these arrays nor the returned values are
        # passed back to the empirical estimator, planner, or collector.
        started = perf_counter()
        actual.append(evaluate_policy(policies[-1], true_model['transition'],
                                      true_model['immediate_rewards'], CONFIG['gamma_eval']))
        eval_times.append(perf_counter()-started)
        print(f'Planned {completed:>7,}/seed | true V: Q {actual[-1][0, :, 24].mean():.4f}, '
              f'model {actual[-1][1, :, 24].mean():.4f} | predicted {prediction[:, 24].mean():.4f}', flush=True)
    # One agreement check against the established single-policy evaluator.
    for method in range(2):
        expected = policy_evaluation(policies[-1][method, 0], true_model['transition'],
                                     true_model['immediate_rewards'], CONFIG['gamma_eval'])
        np.testing.assert_allclose(actual[-1][method, 0], expected, atol=1e-10)
    return dict(checkpoint_steps=collection['steps'], checkpoint_policies=np.array(policies),
                checkpoint_values=np.array(actual), checkpoint_predicted_values=np.array(predicted),
                planning_iterations=np.array(iterations), optimality_residuals=np.array(residuals),
                planning_seconds=np.array(runtimes), exact_evaluation_seconds=np.array(eval_times),
                final_transition=p, final_rewards=r, final_unknown=unknown,
                oracle_policy=true_model['policies'][-1, 2], oracle_values=true_model['values'][-1, 2],
                states=true_model['states'], method_names=np.array(METHODS))


def evaluation_seeds():
    shape = (CONFIG['training_seeds'], CONFIG['evaluation_trajectories_per_seed'])
    return {name: np.array([[np.random.SeedSequence([MASTER, 3, s, t, role]).generate_state(
                1, dtype=np.uint64)[0] for t in range(shape[1])] for s in range(shape[0])])
            for name, role in [('regeneration', 0), ('actions', 100)]}


def simulate(policies):
    """Frozen policies; pair regeneration and action uniforms within each path."""
    n, paths, steps, width = (CONFIG[k] for k in ('training_seeds',
        'evaluation_trajectories_per_seed', 'evaluation_steps', 'bin_width'))
    seeds = evaluation_seeds()
    # Independent RNG for every (training seed, trajectory, role). Both frozen
    # policies receive the same draws, evaluated against their own stocks/CDFs.
    env = np.stack([np.random.default_rng(int(s)).random((steps, 2))
                    for s in seeds['regeneration'].flat], axis=1)
    action = np.stack([np.random.default_rng(int(s)).random(steps)
                       for s in seeds['actions'].flat], axis=1)
    seed_rows = np.repeat(np.arange(n), paths)
    stocks = np.tile(WORLD.initial, (2, n*paths, 1))
    totals = np.zeros((2, n*paths, len(METRICS)))
    cumulative = np.zeros((2, n*paths))
    curve_totals = np.zeros((2, n*paths, len(CURVES)))
    seed_curves = np.zeros((2, n, steps//width, len(CURVES)))
    trace_stocks = np.zeros((2, steps+1, 2), dtype=np.int8)
    trace_actions = np.zeros((2, steps), dtype=np.int8)
    trace_rewards = np.zeros((2, steps))
    trace_stocks[:, 0] = stocks[:, 0]
    for t in range(steps):
        for m in range(2):
            cdf = policies[m, seed_rows, state_index(stocks[m])].cumsum(axis=-1)
            cdf[:, -1] = 1.
            actions = (action[t, :, None] >= cdf).sum(axis=-1)
            successor, rewards = step(stocks[m], actions, env[t])
            depleted = stocks[m] == 0
            rest = actions == 2
            unsuccessful = (actions < 2) & (rewards == 0)
            success = rewards > 0
            cumulative[m] += rewards
            totals[m] += np.column_stack((CONFIG['gamma_eval']**t*rewards, rewards,
                stocks[m], depleted, depleted.any(axis=1), depleted.all(axis=1),
                rest, unsuccessful, success, success & (actions == 0),
                success & (actions == 1), rewards))
            curve_totals[m] += np.column_stack((rewards, cumulative[m], stocks[m],
                depleted.any(axis=1), rest, unsuccessful))
            if (t+1) % width == 0:
                binned = curve_totals[m]/width
                binned[:, 1] = cumulative[m]
                seed_curves[m, :, t//width] = binned.reshape(n, paths, -1).mean(axis=1)
                curve_totals[m] = 0
            trace_stocks[m, t+1] = successor[0]
            trace_actions[m, t], trace_rewards[m, t] = actions[0], rewards[0]
            stocks[m] = successor
    totals[:, :, 2:] /= steps
    totals = totals.reshape(2, n, paths, -1)
    np.testing.assert_allclose(totals[..., 8:11].sum(axis=-1), 1)
    np.testing.assert_allclose(totals[..., -1], totals[..., -3]+1.5*totals[..., -2])
    return dict(trajectory_metrics=totals, seed_metrics=totals.mean(axis=2),
                metric_names=np.array(METRICS), seed_curves=seed_curves,
                curve_names=np.array(CURVES), bin_centers=np.arange(0, steps, width)+(width-1)/2,
                bin_ends=np.arange(width, steps+1, width), trace_stocks=trace_stocks,
                trace_actions=trace_actions, trace_rewards=trace_rewards,
                trace_regeneration_uniforms=env[:, 0], trace_action_uniforms=action[:, 0],
                evaluation_regeneration_seeds=seeds['regeneration'], evaluation_action_seeds=seeds['actions'],
                method_names=np.array(METHODS))


def summarize(data, simulation, replay):
    values = data['checkpoint_values'][-1, :, :, 24]
    prediction = data['checkpoint_predicted_values'][-1, :, 24]
    error = prediction-values[1]
    oracle = float(data['oracle_values'][24])
    return dict(oracle_value=oracle, threshold_90_percent=.9*oracle, replay=replay,
        uncertainty='Mean +/- 1.96 SEM across 100 training seeds; fractions: Wilson 95% intervals; pointwise, unadjusted',
        primary_endpoint='Exact V_pi(4,4), gamma=.99, fixed 200000-transition checkpoint',
        methods={name: dict(final_value=describe(values[m]),
                           fraction_of_oracle=describe(values[m]/oracle),
                           reaching_90_percent=fraction_interval(values[m] >= .9*oracle))
                 for m, name in enumerate(METHODS)},
        paired_model_minus_q=describe(values[1]-values[0]),
        fraction_model_better=fraction_interval(values[1] > values[0]+TIE_ATOL),
        prediction=dict(predicted_value=describe(prediction), predicted_minus_actual=describe(error),
                        absolute_error=describe(np.abs(error)), rmse=float(np.sqrt(np.mean(error**2))),
                        overprediction_fraction=fraction_interval(error > 0)),
        unvisited_rows=describe(data['final_unknown'].sum(axis=(1, 2))),
        max_planning_residual=float(data['optimality_residuals'].max()),
        seed_0=dict(q_value=float(values[0, 0]), model_value=float(values[1, 0]),
                    model_prediction=float(prediction[0])),
        frozen_behavior={name: {metric: describe(simulation['seed_metrics'][m, :, k])
                               for k, metric in enumerate(METRICS)} for m, name in enumerate(METHODS)},
        paired_frozen_model_minus_q={metric: describe(simulation['seed_metrics'][1, :, k]-
                                                   simulation['seed_metrics'][0, :, k])
                                    for k, metric in enumerate(METRICS)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/learned_world_model'))
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    from learned_world_model_figures import plot_all
    if args.plot_only or (args.output/'results.npz').exists():
        plot_all(load_npz(args.output/'results.npz'), load_npz(args.output/'frozen_behavior.npz'), args.output)
        print('Figures regenerated from saved results; no collection or planning rerun.')
        return
    started = perf_counter()
    baseline_manifest = json.loads((BASELINE/'manifest.json').read_text())
    seeds = dict(regeneration=baseline_manifest['seeds']['regeneration'],
                 actions=baseline_manifest['seeds']['learners']['0.99'])
    sources = ['foraging.py', 'q_learning.py', 'run_q_learning.py', 'run_experiment.py',
               'learned_world_model.py', 'run_learned_world_model.py', 'check_learned_world_model.py',
               'learned_world_model_figures.py', 'figures.py', 'q_learning_figures.py']
    manifest_path = args.output/'manifest.json'
    hashes = {name: sha(ROOT/name) for name in sources}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest['config'] == CONFIG and manifest['source_sha256'] == hashes
    else:
        manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), config=CONFIG,
            experiment_id=3, master_seed=MASTER, collection_seeds=seeds,
            source_sha256=hashes, baseline_manifest_sha256=sha(BASELINE/'manifest.json'),
            baseline_results_sha256=sha(BASELINE/'results.npz'),
            true_model_sha256=sha(ROOT/'results/first_world/model.npz'),
            python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
            platform=platform.platform(), openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS', 'unset'),
            unknown_rows='Zero-reward self-loop; explicit unknown mask; assumption, not an observation',
            information_boundary='Planner receives only one seed model at a time, batched independently; all true-model use is separate evaluation',
            collection='Original gamma=.99 environment/action seeds, draw shape, Q update order; resets not counted',
            planner='Policy iteration from uniform policies; gamma=.99; uniform ties within absolute 1e-10, rtol=0; stop at stable policy',
            evaluation='Frozen final policies; 20 independent 1000-step trajectories per training seed; paired regeneration and action uniforms',
            evaluation_seed_recipe='SeedSequence([20260929, 3, training_seed, trajectory, role]); roles 0 regeneration, 100 actions',
            measurement='Stocks/depletion before acting; harvest and rest fractions over all decisions; aggregate 20 trajectories within seed before uncertainty',
            illustration='Training seed 0, trajectory 0, first 120 decisions; fixed before run',
            runtime_definitions=dict(collection='Random draws, environment steps, Q updates, count/reward accumulation; excludes snapshots and I/O',
                planning='Empirical estimation, normalization check, policy iteration and own-model value; all 21 checkpoints',
                exact_evaluation='True-model policy evaluation only', frozen_simulation='Draw generation, trajectories and within-seed aggregation'),
            reproduction='OPENBLAS_NUM_THREADS=1 python run_learned_world_model.py --output results/learned_world_model_repeat',
            plotting='python run_learned_world_model.py --plot-only')
        write_json(manifest_path, manifest)
    collection_path = args.output/'collection.npz'
    collection = load_npz(collection_path) if collection_path.exists() else collect(args.output, seeds)
    replay = compare_replay(collection)
    print('Replay comparison:', json.dumps(replay), flush=True)
    # True model is loaded only after the observation archive has been saved.
    true_model = load_npz(ROOT/'results/first_world/model.npz')
    data = plan_and_evaluate(collection, true_model)
    sim_started = perf_counter()
    simulation = simulate(data['checkpoint_policies'][-1])
    manifest['frozen_simulation_seconds'] = perf_counter()-sim_started
    np.savez_compressed(args.output/'frozen_behavior.npz', **simulation)
    data['final_q'] = collection['q'][-1]
    data['final_visits'] = collection['visits'][-1]
    np.savez_compressed(args.output/'results.npz', **data)
    summary = summarize(data, simulation, replay)
    write_json(args.output/'summary.json', summary)
    manifest.update(collection_seconds=float(collection['seconds'][-1]),
                    planning_seconds=float(data['planning_seconds'].sum()),
                    final_checkpoint_planning_seconds=float(data['planning_seconds'][-1]),
                    exact_evaluation_seconds=float(data['exact_evaluation_seconds'].sum()),
                    total_collection_transitions=CONFIG['training_seeds']*CONFIG['transitions'],
                    total_frozen_evaluation_transitions=2*CONFIG['training_seeds']*
                        CONFIG['evaluation_trajectories_per_seed']*CONFIG['evaluation_steps'])
    plot_all(data, simulation, args.output)
    manifest['completion_invocation_seconds'] = perf_counter()-started
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    print('Primary paired model-minus-Q:', summary['paired_model_minus_q'], flush=True)
    print('Prediction minus actual:', summary['prediction']['predicted_minus_actual'], flush=True)
    print('Runtime:', {k: v for k, v in manifest.items() if k.endswith('_seconds')}, flush=True)


if __name__ == '__main__':
    main()
