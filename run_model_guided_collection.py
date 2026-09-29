"""Experiment 5: collect once with three controllers, then evaluate six policies."""

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
from model_guided_collection import CONFIG, COLLECTORS, PLANNERS, collect, continuation_seeds
from run_count_penalized_planning import changes, simulate_new
from run_q_learning import describe, fraction_interval, load_npz, write_json


ROOT = Path(__file__).resolve().parent
BASE = ROOT/'results/learned_world_model'


def evaluate_checkpoints(checkpoints, true_model):
    """Read only after collection is complete; never feed values back to control."""
    policies = np.stack([c['policies'] for c in checkpoints])
    values = evaluate_policy(policies, true_model['transition'], true_model['immediate_rewards'], CONFIG['gamma'])
    data = dict(additional_steps=np.array([c['additional_steps'] for c in checkpoints]),
        total_steps=np.array([c['total_steps'] for c in checkpoints]), checkpoint_policies=policies,
        checkpoint_values=values, checkpoint_predictions=np.stack([c['predicted_values'] for c in checkpoints]),
        checkpoint_internal_scores=np.stack([c['internal_scores'] for c in checkpoints]),
        checkpoint_visits=np.stack([c['visits'] for c in checkpoints]),
        collector_greedy=np.stack([c['collector_greedy'] for c in checkpoints]),
        collector_behavior=np.stack([c['collector_behavior'] for c in checkpoints]),
        rollout_metrics=checkpoints[-1]['rollout_metrics'],
        collection_metric_names=checkpoints[-1]['collection_metric_names'],
        collector_names=np.array(COLLECTORS), planner_names=np.array(PLANNERS),
        oracle_values=true_model['values'][-1, 2], oracle_policy=true_model['policies'][-1, 2],
        states=true_model['states'], action_names=np.array(ACTIONS))
    assert np.all(values <= data['oracle_values']+1e-9)
    # The common starting histories must reproduce the saved starting references.
    original = load_npz(BASE/'results.npz')
    cautious = load_npz(ROOT/'results/count_penalized_planning/results.npz')
    for c in range(3):
        np.testing.assert_array_equal(policies[0, c, 0], original['checkpoint_policies'][-1, 1])
        np.testing.assert_array_equal(policies[0, c, 1], cautious['checkpoint_policies'][-1, 2])
        np.testing.assert_allclose(values[0, c, 0], original['checkpoint_values'][-1, 1], atol=1e-10)
        np.testing.assert_allclose(values[0, c, 1], cautious['checkpoint_values'][-1, 2], atol=1e-10)
    return data


def frozen_evaluation(policies):
    """Reuse Exp. 4's one-policy evaluator six times; it simulates no baselines."""
    baseline = load_npz(BASE/'frozen_behavior.npz')
    keys = ['trajectory_metrics', 'seed_metrics', 'seed_curves', 'trace_stocks', 'trace_actions', 'trace_rewards']
    outputs = {key: [] for key in keys}
    for c in range(3):
        rows = {key: [] for key in keys}
        for planner in range(2):
            result = simulate_new(policies[c, planner], baseline)
            for key in keys:
                rows[key].append(result[key][-1])  # The two baseline entries are only copied, never simulated.
            print(f'Frozen evaluation: {COLLECTORS[c]} / {PLANNERS[planner]}', flush=True)
        for key in keys:
            outputs[key].append(np.array(rows[key]))
    data = {key: np.array(value) for key, value in outputs.items()}
    for key in ['metric_names', 'curve_names', 'bin_centers', 'bin_ends', 'evaluation_regeneration_seeds',
                'evaluation_action_seeds', 'trace_regeneration_uniforms', 'trace_action_uniforms']:
        data[key] = baseline[key]
    data.update(collector_names=np.array(COLLECTORS), planner_names=np.array(PLANNERS))
    return data


def seed_zero(checkpoints, data, true_model):
    s, a, seed = 19, 1, 0  # (3,4), harvest B, preselected seed 0.
    counts = np.array([c['visits'][:, seed, s, a] for c in checkpoints])
    loops = np.array([c['successors'][:, seed, s, a, s] for c in checkpoints])
    return dict(seed=seed, state=[3, 4], action='harvest B',
        additional_steps=data['additional_steps'].tolist(), visit_counts=counts.tolist(),
        observed_self_loop_counts=loops.tolist(), empirical_self_loop=(loops/counts).tolist(),
        true_self_loop_probability=float(true_model['transition'][s, a, s]),
        collector_action_probabilities=data['collector_behavior'][:, :, seed, s].tolist(),
        collector_greedy_probabilities=data['collector_greedy'][:, :, seed, s].tolist(),
        final_policy_probabilities=data['checkpoint_policies'][-1, :, :, seed, s].tolist(),
        predicted_values_from_state=data['checkpoint_predictions'][:, :, :, seed, s].tolist(),
        actual_values_from_state=data['checkpoint_values'][:, :, :, seed, s].tolist(),
        final_initial_actual_values=data['checkpoint_values'][-1, :, :, seed, 24].tolist())


def comparison(delta):
    return dict(value_difference=describe(delta), **changes(delta))


def summarize(data, frozen, checkpoints):
    value = data['checkpoint_values'][-1, ..., 24]
    prediction = data['checkpoint_predictions'][-1, ..., 24]
    error = prediction-value
    oracle = float(data['oracle_values'][24])
    names = [f'{c} / {p}' for c in COLLECTORS for p in PLANNERS]
    summary = dict(primary_endpoint='B-controlled minus A-controlled, SAME unpenalized planner, true V_pi(4,4), gamma=.99, total 250000',
        uncertainty='Mean +/- 1.96 SEM over 100 training seeds; paired seed differences; Wilson fraction intervals; pointwise, unadjusted',
        oracle_value=oracle, threshold_90_percent=.9*oracle,
        primary_B_minus_A_unpenalized=comparison(value[1, 0]-value[0, 0]),
        policies={}, secondary_comparisons={}, coverage={}, collection={}, frozen_behavior={})
    for c in range(3):
        for p in range(2):
            name = names[2*c+p]
            e = error[c, p]
            summary['policies'][name] = dict(final_value=describe(value[c, p]),
                percentile_10=float(np.quantile(value[c, p], .1)),
                reaching_90_percent=fraction_interval(value[c, p] >= .9*oracle),
                original_reward_prediction=describe(prediction[c, p]), predicted_minus_actual=describe(e),
                absolute_prediction_error=describe(abs(e)), prediction_rmse=float(np.sqrt(np.mean(e**2))),
                internal_score=describe(data['checkpoint_internal_scores'][-1, c, p, :, 24]),
                change_from_historical_start=describe(value[c, p]-data['checkpoint_values'][0, c, p, :, 24]))
            summary['frozen_behavior'][name] = {str(metric): describe(frozen['seed_metrics'][c, p, :, k])
                                               for k, metric in enumerate(frozen['metric_names'])}
        visits = data['checkpoint_visits'][-1, c]
        initial = data['checkpoint_visits'][0, c]
        low_fraction = (visits < 10).mean(axis=0)
        # Fixed <10 threshold; report the ten most frequently sparse rows, tie by state/action order.
        order = np.argsort(-low_fraction.ravel(), kind='stable')[:10]
        summary['coverage'][COLLECTORS[c]] = dict(
            final_rows_below={str(t): describe((visits < t).sum(axis=(-2, -1))) for t in CONFIG['low_count_thresholds']},
            initial_rows_below={str(t): describe((initial < t).sum(axis=(-2, -1))) for t in CONFIG['low_count_thresholds']},
            final_minimum_visits=describe(visits.min(axis=(-2, -1))),
            new_visits_to_initially_below_10=describe(((visits-initial)*(initial < 10)).sum(axis=(-2, -1))),
            most_frequently_below_10=[dict(state=data['states'][i//3].tolist(), action=ACTIONS[i%3],
                fraction_of_seeds=float(low_fraction.ravel()[i]),
                mean_count=float(visits[:, i//3, i%3].mean())) for i in order])
        summary['collection'][COLLECTORS[c]] = {
            window: {str(metric): describe(data['rollout_metrics'][c, selection, :, k].mean(axis=0))
                     for k, metric in enumerate(data['collection_metric_names'])}
            for window, selection in [('all_50_rollouts', slice(None)), ('last_5_rollouts', slice(-5, None))]}
    for p in range(2):
        for c, d in [(1, 0), (2, 0), (2, 1)]:
            if (p, c, d) == (0, 1, 0):
                continue
            summary['secondary_comparisons'][f'{COLLECTORS[c]} minus {COLLECTORS[d]} / {PLANNERS[p]}'] = comparison(value[c, p]-value[d, p])
    for c in range(3):
        summary['secondary_comparisons'][f'{COLLECTORS[c]} / penalized minus unpenalized'] = comparison(value[c, 1]-value[c, 0])
    summary['paired_primary_prediction_error_change'] = describe(error[1, 0]-error[0, 0])
    summary['paired_primary_absolute_error_change'] = describe(abs(error[1, 0])-abs(error[0, 0]))
    summary['paired_primary_frozen_B_minus_A_unpenalized'] = {
        str(metric): describe(frozen['seed_metrics'][1, 0, :, k]-frozen['seed_metrics'][0, 0, :, k])
        for k, metric in enumerate(frozen['metric_names'])}
    summary['maximum_checkpoint_planning_residual'] = float(max(c['planning_residuals'].max() for c in checkpoints))
    return summary


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/model_guided_collection'))
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    from model_guided_figures import plot_all
    if args.plot_only or (args.output/'summary.json').exists():
        plot_all(load_npz(args.output/'results.npz'), load_npz(args.output/'frozen_behavior.npz'),
                 json.loads((args.output/'seed_0.json').read_text()), args.output)
        print('Plot-only: no collection, planning, or evaluation repeated.')
        return
    started = perf_counter()
    sources = ['model_guided_collection.py', 'run_model_guided_collection.py', 'check_model_guided_collection.py',
               'model_guided_figures.py', 'learned_world_model.py', 'run_count_penalized_planning.py',
               'run_learned_world_model.py', 'run_q_learning.py', 'run_experiment.py', 'q_learning.py',
               'foraging.py', 'figures.py', 'q_learning_figures.py']
    manifest = dict(experiment_id=5, config=CONFIG, collectors=COLLECTORS, planners=PLANNERS,
        started_utc=datetime.now(timezone.utc).isoformat(), collection_seeds=continuation_seeds(),
        collection_seed_recipe='SeedSequence([20260929,5,seed,role]), roles 0 environment and 100 actions; shared across conditions',
        input_history_sha256=sha(BASE/'collection.npz'), source_sha256={name: sha(ROOT/name) for name in sources},
        python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
        platform=platform.platform(), openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS', 'unset'),
        primary='B versus A collection, same unpenalized extraction, fixed total 250000 transitions',
        information_boundary='Collection and all checkpoint policy selection finish before loading true-model arrays; counts independent by seed and condition',
        initial_Q_copies='Copied for all conditions; only A updates Q; B/C historical Q copies are unused',
        planning='B/C before first rollout and after every 1000 new transitions, gamma .99, uniform ties at absolute 1e-10; C fixed coefficient 1, no clipping',
        checkpoints='0 and every 5000 additional transitions, atomic sufficient statistics plus Q, policies, metrics, RNG states; reset boundaries only',
        resume='Restores last checkpoint. Any uncheckpointed tail after interruption is deterministically repeated from its saved RNG states, never counted twice.',
        evaluation='Six final policies; 20 paired 1000-step trajectories per seed; no learning/exploration; average within training seed first',
        evaluation_seed_recipe='Reuse established independent evaluation streams: SeedSequence([20260929,3,training_seed,trajectory,role]); roles 0 regeneration,100 actions',
        measurement='Pre-action stocks, depleted if zero; trajectory/rollout summaries aggregated within training seed before uncertainty',
        prediction='Original empirical reward in prediction; penalized internal score saved separately; true world unchanged',
        collection_transitions=15000000, frozen_evaluation_transitions=12000000,
        reproduction='OPENBLAS_NUM_THREADS=1 python run_model_guided_collection.py --output results/model_guided_collection_repeat',
        plotting='python run_model_guided_collection.py --plot-only')
    path = args.output/'manifest.json'
    if path.exists():
        prior = json.loads(path.read_text())
        assert all(prior[k] == manifest[k] for k in ('config', 'source_sha256', 'input_history_sha256', 'collection_seeds'))
        manifest = prior
    else:
        write_json(path, manifest)
    archive = load_npz(BASE/'collection.npz')
    history = {key: archive[key][-1] for key in ('q', 'successors', 'visits', 'reward_sums')}
    assert int(archive['steps'][-1]) == CONFIG['history_transitions']
    np.testing.assert_array_equal(history['successors'].sum(axis=-1), history['visits'])
    manifest.update(collect(args.output, history, manifest['collection_seeds']))
    checkpoints = [load_npz(p) for p in sorted(args.output.glob('checkpoint_*.npz'))]
    np.testing.assert_array_equal([c['additional_steps'] for c in checkpoints], np.arange(0, 50001, 5000))
    # All six policies at every checkpoint have now been selected without true data.
    eval_started = perf_counter()
    true_path = ROOT/'results/first_world/model.npz'
    true_model = load_npz(true_path)
    data = evaluate_checkpoints(checkpoints, true_model)
    diagnostic = seed_zero(checkpoints, data, true_model)
    manifest['true_model_sha256'] = sha(true_path)
    manifest['exact_evaluation_seconds'] = perf_counter()-eval_started
    np.savez_compressed(args.output/'results.npz', **data)
    write_json(args.output/'seed_0.json', diagnostic)
    sim_path = args.output/'frozen_behavior.npz'
    sim_started = perf_counter()
    frozen = load_npz(sim_path) if sim_path.exists() else frozen_evaluation(data['checkpoint_policies'][-1])
    manifest['frozen_evaluation_invocation_seconds'] = perf_counter()-sim_started
    np.savez_compressed(sim_path, **frozen)
    summary = summarize(data, frozen, checkpoints)
    write_json(args.output/'summary.json', summary)
    plot_all(data, frozen, diagnostic, args.output)
    manifest['completion_invocation_seconds'] = perf_counter()-started
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(path, manifest)
    print('PRIMARY:', json.dumps(summary['primary_B_minus_A_unpenalized']), flush=True)
    for name, result in summary['policies'].items():
        print(name, 'true value', result['final_value']['mean'], 'p10', result['percentile_10'],
              '>=90% oracle', result['reaching_90_percent']['count'],
              'prediction error', result['predicted_minus_actual']['mean'], flush=True)
    print('Runtime:', {k: v for k, v in manifest.items() if k.endswith('_seconds')}, flush=True)


if __name__ == '__main__':
    main()
