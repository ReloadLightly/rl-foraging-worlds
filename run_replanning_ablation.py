"""Experiment 6: one fixed-policy collector; reuse all Experiment 5 baselines."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from foraging import ACTIONS
from learned_world_model import evaluate_policy
from replanning_ablation import CONFIG, COLLECTORS, collect
from run_count_penalized_planning import changes, simulate_new
from run_q_learning import describe, fraction_interval, load_npz, write_json

ROOT = Path(__file__).resolve().parent
BASE = ROOT/'results/model_guided_collection'
HISTORY = ROOT/'results/learned_world_model/collection.npz'
BEHAVIOR_KEYS = ['trajectory_metrics', 'seed_metrics', 'seed_curves', 'trace_stocks', 'trace_actions', 'trace_rewards']


def comparison(delta):
    return dict(value_difference=describe(delta), **changes(delta))


def evaluate(checkpoints, old, true_model):
    """Only called after every new policy is selected, with no feedback to collection."""
    policies = np.stack([c['policies'] for c in checkpoints])
    values = evaluate_policy(policies, true_model['transition'], true_model['immediate_rewards'], CONFIG['gamma'])
    data = dict(additional_steps=old['additional_steps'], total_steps=old['total_steps'],
        collector_names=np.array(COLLECTORS), collection_metric_names=old['collection_metric_names'],
        oracle_values=old['oracle_values'], oracle_policy=old['oracle_policy'], states=old['states'],
        action_names=old['action_names'])
    for key, new in [('checkpoint_policies', policies), ('checkpoint_values', values),
                     ('checkpoint_predictions', np.stack([c['predicted_values'] for c in checkpoints]))]:
        data[key] = np.concatenate((old[key][:, :2, 0], new[:, None]), axis=1)
    for key, saved_key in [('checkpoint_visits', 'visits'), ('collector_greedy', 'collector_greedy'),
                           ('collector_behavior', 'collector_behavior')]:
        data[key] = np.concatenate((old[key][:, :2], np.stack([c[saved_key] for c in checkpoints])[:, None]), axis=1)
    data['rollout_metrics'] = np.concatenate((old['rollout_metrics'][:2], checkpoints[-1]['rollout_metrics'][None]))
    for c in range(3):
        np.testing.assert_array_equal(data['checkpoint_policies'][0, c], policies[0])
        np.testing.assert_allclose(data['checkpoint_values'][0, c], values[0], atol=1e-10)
    # First rollout must coincide exactly: same initial B policy, states and draws.
    np.testing.assert_array_equal(data['rollout_metrics'][1, 0], data['rollout_metrics'][2, 0])
    assert np.all(values <= data['oracle_values']+1e-9)
    return data


def frozen_evaluation(policy):
    old = load_npz(BASE/'frozen_behavior.npz')
    baseline = {key: old[key][:2, 0] for key in BEHAVIOR_KEYS}
    for key in ['metric_names', 'curve_names', 'bin_centers', 'bin_ends', 'evaluation_regeneration_seeds',
                'evaluation_action_seeds', 'trace_regeneration_uniforms', 'trace_action_uniforms']:
        baseline[key] = old[key]
    result = simulate_new(policy, baseline)  # Simulates only the new entry, copies A/B outcomes.
    result['method_names'] = np.array(COLLECTORS)
    for key in BEHAVIOR_KEYS:
        np.testing.assert_array_equal(result[key][:2], old[key][:2, 0])
    return result


def seed_zero(checkpoints, data):
    prior = json.loads((BASE/'seed_0.json').read_text())
    counts = np.column_stack((np.array(prior['visit_counts'])[:, :2], [c['visits'][0, 19, 1] for c in checkpoints]))
    loops = np.column_stack((np.array(prior['observed_self_loop_counts'])[:, :2],
                            [c['successors'][0, 19, 1, 19] for c in checkpoints]))
    return dict(seed=0, state=[3, 4], action='harvest B', additional_steps=data['additional_steps'].tolist(),
        collector_names=COLLECTORS, visit_counts=counts.tolist(), observed_self_loop_counts=loops.tolist(),
        empirical_self_loop=(loops/counts).tolist(), true_self_loop_probability=prior['true_self_loop_probability'],
        collector_action_probabilities=data['collector_behavior'][:, :, 0, 19].tolist(),
        final_extracted_action_probabilities=data['checkpoint_policies'][-1, :, 0, 19].tolist(),
        predicted_values_from_state=data['checkpoint_predictions'][:, :, 0, 19].tolist(),
        actual_values_from_state=data['checkpoint_values'][:, :, 0, 19].tolist(),
        final_initial_actual_values=data['checkpoint_values'][-1, :, 0, 24].tolist())


def summarize(data, frozen, checkpoints):
    value = data['checkpoint_values'][-1, :, :, 24]
    prediction = data['checkpoint_predictions'][-1, :, :, 24]
    error = prediction-value
    oracle = float(data['oracle_values'][24])
    summary = dict(primary_endpoint='Saved adaptive B minus new fixed collector; SAME unpenalized extraction; true V_pi(4,4), gamma=.99; total 250000',
        uncertainty='Mean +/- 1.96 SEM over 100 paired training seeds; Wilson fraction intervals; descriptive pointwise checkpoint intervals, unadjusted',
        oracle_value=oracle, threshold_90_percent=.9*oracle,
        primary_adaptive_minus_fixed=comparison(value[1]-value[2]),
        contextual_fixed_minus_Q=comparison(value[2]-value[0]),
        contextual_adaptive_minus_Q=comparison(value[1]-value[0]),
        paired_prediction_error_adaptive_minus_fixed=describe(error[1]-error[2]),
        paired_absolute_error_adaptive_minus_fixed=describe(abs(error[1])-abs(error[2])),
        policies={}, coverage={}, collection={}, frozen_behavior={},
        descriptive_checkpoint_adaptive_minus_fixed={str(int(t)): comparison(v[1]-v[2])
            for t, v in zip(data['additional_steps'], data['checkpoint_values'][..., 24])})
    for c, name in enumerate(COLLECTORS):
        summary['policies'][name] = dict(final_value=describe(value[c]),
            percentile_10=float(np.quantile(value[c], .1)), reaching_90_percent=fraction_interval(value[c] >= .9*oracle),
            original_reward_prediction=describe(prediction[c]), predicted_minus_actual=describe(error[c]),
            absolute_prediction_error=describe(abs(error[c])), prediction_rmse=float(np.sqrt(np.mean(error[c]**2))),
            change_from_historical_start=describe(value[c]-data['checkpoint_values'][0, c, :, 24]))
        visits = data['checkpoint_visits'][-1, c]
        initial = data['checkpoint_visits'][0, c]
        scarce = initial < 10
        sparse = (visits < 10).mean(axis=0)
        order = np.argsort(-sparse.ravel(), kind='stable')[:10]
        summary['coverage'][name] = dict(
            initial_rows_below={str(t): describe((initial < t).sum(axis=(-2, -1))) for t in CONFIG['low_count_thresholds']},
            final_rows_below={str(t): describe((visits < t).sum(axis=(-2, -1))) for t in CONFIG['low_count_thresholds']},
            initially_scarce_rows_reaching_10=describe((scarce & (visits >= 10)).sum(axis=(-2, -1))),
            initially_scarce_rows_with_any_new_observation=describe((scarce & (visits > initial)).sum(axis=(-2, -1))),
            new_visits_to_initially_below_10=describe(((visits-initial)*scarce).sum(axis=(-2, -1))),
            most_frequently_below_10=[dict(state=data['states'][i//3].tolist(), action=ACTIONS[i%3],
                fraction_of_seeds=float(sparse.ravel()[i]), mean_count=float(visits[:, i//3, i%3].mean())) for i in order])
        summary['collection'][name] = {
            window: {str(metric): describe(data['rollout_metrics'][c, selection, :, k].mean(axis=0))
                     for k, metric in enumerate(data['collection_metric_names'])}
            for window, selection in [('all_50_rollouts', slice(None)), ('last_5_rollouts', slice(-5, None))]}
        summary['frozen_behavior'][name] = {str(metric): describe(frozen['seed_metrics'][c, :, k])
                                           for k, metric in enumerate(frozen['metric_names'])}
    summary['paired_frozen_adaptive_minus_fixed'] = {
        str(metric): describe(frozen['seed_metrics'][1, :, k]-frozen['seed_metrics'][2, :, k])
        for k, metric in enumerate(frozen['metric_names'])}
    summary['paired_collection_adaptive_minus_fixed'] = {
        str(metric): describe((data['rollout_metrics'][1, :, :, k]-data['rollout_metrics'][2, :, :, k]).mean(axis=0))
        for k, metric in enumerate(data['collection_metric_names'])}
    initial = data['checkpoint_visits'][0, 0] < 10
    covered = (initial[None] & (data['checkpoint_visits'][-1] >= 10)).sum(axis=(-2, -1))
    summary['paired_initially_scarce_rows_reaching_10_adaptive_minus_fixed'] = describe(covered[1]-covered[2])
    summary['maximum_checkpoint_planning_residual'] = float(max(c['planning_residuals'].max() for c in checkpoints))
    summary['checks'] = ['200k histories equal saved Experiment 5 checkpoint zero; independent copies',
        'Fixed policy read-only and identical to initial B policy at every checkpoint',
        'Exact Experiment 5 RNG states at every checkpoint; identical first-rollout summaries to B',
        'Successor counts sum to visits; totals 200000+additional per seed; rollout rewards equal reward-sum increments',
        'True model loaded after collection and policy selection; baseline frozen outcomes copied exactly']
    return summary


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/replanning_ablation'))
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    from replanning_figures import plot_all
    if args.plot_only or (args.output/'summary.json').exists():
        plot_all(load_npz(args.output/'results.npz'), load_npz(args.output/'frozen_behavior.npz'),
                 json.loads((args.output/'seed_0.json').read_text()), args.output)
        print('Plot-only: no collection, planning or evaluation repeated.')
        return
    started = perf_counter()
    prior = json.loads((BASE/'manifest.json').read_text())
    sources = ['replanning_ablation.py', 'run_replanning_ablation.py', 'replanning_figures.py',
        'model_guided_collection.py', 'learned_world_model.py', 'run_count_penalized_planning.py',
        'run_learned_world_model.py', 'run_q_learning.py', 'q_learning.py', 'foraging.py', 'figures.py', 'q_learning_figures.py']
    inputs = [HISTORY, BASE/'manifest.json', BASE/'results.npz', BASE/'frozen_behavior.npz', BASE/'seed_0.json']
    inputs += sorted(BASE.glob('checkpoint_*.npz'))
    manifest = dict(experiment_id=6, config=CONFIG, collectors=COLLECTORS,
        started_utc=datetime.now(timezone.utc).isoformat(), collection_seeds=prior['collection_seeds'],
        collection_streams='Loaded Experiment 5 IDs; environment then action roles; each generator draws (1000,2) per rollout, stacked on axis 1. No new experiment-ID stream.',
        input_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs},
        source_sha256={name: sha(ROOT/name) for name in sources},
        python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
        platform=platform.platform(), openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS', 'unset'),
        primary='Adaptive B minus fixed initial B collector, SAME unpenalized extraction, total 250000 transitions',
        behavior='Read-only copy of Experiment 5 checkpoint_000000 policies[1,0]; epsilon .1; no control replanning or Q updates',
        fitting='Per-seed empirical successor frequencies and reward means; unvisited rows marked unknown, assumed zero-reward self-loops',
        information_boundary='All collection and empirical policy selection finish before loading true-model arrays. Extracted policies never feed collection.',
        checkpoints='0 then every 5000 additional; sufficient statistics, policies, unknown mask, metrics, RNG states and cumulative runtimes; atomic saves',
        resume='Restore last checkpoint and RNG states; any interrupted tail repeated deterministically, not counted twice. Completed output defaults to plotting only.',
        evaluation='Only NEW final extracted policies simulated: 20 paired 1000-step paths/seed; saved Experiment 5 evaluation seed IDs; baselines copied',
        measurement='Pre-action stocks; depleted means either stock zero. Average trajectories/rollouts within seed before uncertainty.',
        collection_transitions=5000000, frozen_evaluation_transitions=2000000,
        reproduction='OPENBLAS_NUM_THREADS=1 python run_replanning_ablation.py --output results/replanning_ablation_repeat',
        plotting='python run_replanning_ablation.py --plot-only')
    path = args.output/'manifest.json'
    if path.exists():
        saved = json.loads(path.read_text())
        assert all(saved[k] == manifest[k] for k in ('config', 'source_sha256', 'input_sha256', 'collection_seeds'))
        manifest = saved
    else:
        write_json(path, manifest)
    archive = load_npz(HISTORY)
    assert int(archive['steps'][-1]) == CONFIG['history_transitions']
    history = {key: archive[key][-1] for key in ('successors', 'visits', 'reward_sums')}
    initial = load_npz(BASE/'checkpoint_000000.npz')
    manifest.update(collect(args.output, BASE, history, initial, manifest['collection_seeds']))
    checkpoints = [load_npz(p) for p in sorted(args.output.glob('checkpoint_*.npz'))]
    np.testing.assert_array_equal([c['additional_steps'] for c in checkpoints], np.arange(0, 50001, 5000))
    # The new collector and all extracted policies are complete before evaluation.
    eval_started = perf_counter()
    true_path = ROOT/'results/first_world/model.npz'
    true_model = load_npz(true_path)
    data = evaluate(checkpoints, load_npz(BASE/'results.npz'), true_model)
    diagnostic = seed_zero(checkpoints, data)
    manifest['true_model_sha256'] = sha(true_path)
    manifest['exact_evaluation_seconds'] = perf_counter()-eval_started
    np.savez_compressed(args.output/'results.npz', **data)
    write_json(args.output/'seed_0.json', diagnostic)
    sim_started = perf_counter()
    sim_path = args.output/'frozen_behavior.npz'
    frozen = load_npz(sim_path) if sim_path.exists() else frozen_evaluation(data['checkpoint_policies'][-1, 2])
    manifest['frozen_evaluation_invocation_seconds'] = perf_counter()-sim_started
    np.savez_compressed(sim_path, **frozen)
    summary = summarize(data, frozen, checkpoints)
    write_json(args.output/'summary.json', summary)
    plot_all(data, frozen, diagnostic, args.output)
    manifest['completion_invocation_seconds'] = perf_counter()-started
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(path, manifest)
    print('PRIMARY:', json.dumps(summary['primary_adaptive_minus_fixed']), flush=True)
    for name, result in summary['policies'].items():
        print(name, 'value', result['final_value']['mean'], 'p10', result['percentile_10'],
              '>=90% oracle', result['reaching_90_percent']['count'],
              'prediction error', result['predicted_minus_actual']['mean'], flush=True)
    print('Runtime:', {k: v for k, v in manifest.items() if k.endswith('_seconds')}, flush=True)


if __name__ == '__main__':
    main()
