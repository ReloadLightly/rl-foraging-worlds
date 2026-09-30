"""Experiment 7: collect once, then evaluate policy tracking in two regimes."""

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from changing_regeneration import CONFIG, CONDITIONS, METHODS, RHO, SNAPSHOT_KEYS, collect, collection_seeds, weighted_estimate
from foraging import ACTIONS, WORLD, exact_model, harvest, state_index
from learned_world_model import evaluate_policy, plan
from regeneration_regimes import changed_probability
from run_count_penalized_planning import changes
from run_q_learning import describe, fraction_interval, load_npz, write_json

ROOT = Path(__file__).resolve().parent
HISTORY = ROOT/'results/model_guided_collection/checkpoint_050000.npz'


def evaluation_models():
    """True models and separate oracles, called only AFTER all policy selection."""
    states, original, reward = exact_model()
    changed = np.zeros_like(original)
    for s, stock in enumerate(states):
        for a in range(3):
            post, observed_reward = harvest(stock[None], np.array([a]))
            assert observed_reward[0] == reward[s, a]
            probability = changed_probability(post[0])
            for grows in product((0, 1), repeat=2):
                mass = np.prod(np.where(grows, probability, 1-probability))
                if mass > 0:
                    changed[s, a, state_index(post[0]+grows)] += mass
    transition = np.stack((original, changed))
    rewards = np.stack((reward, reward))
    np.testing.assert_allclose(transition.sum(axis=-1), 1., rtol=0, atol=1e-15)
    oracle, value, rounds_, residual = plan(transition, rewards, CONFIG['gamma'])
    saved = load_npz(ROOT/'results/first_world/model.npz')
    np.testing.assert_array_equal(original, saved['transition'])
    np.testing.assert_allclose(value[0], saved['values'][-1, 2], atol=1e-9)
    return dict(states=states, transition=transition, rewards=rewards, oracle_policies=oracle,
                oracle_values=value, oracle_planning_iterations=rounds_, oracle_residuals=residual)


def inherited_predictions(policies, history):
    """Evaluate selected policies in the ORIGINAL inherited model, also separately."""
    p, r, _ = weighted_estimate(**history)
    return evaluate_policy(policies, p, r, CONFIG['gamma'])


def evaluate(saved, truth, history):
    policy = saved['policies']
    # Snapshot x condition x method x seed x state; true models broadcast only over the matching condition.
    value = evaluate_policy(policy, truth['transition'][None, :, None, None],
                            truth['rewards'][None, :, None, None], CONFIG['gamma'])
    times = np.arange(0, CONFIG['additional_transitions']+1, CONFIG['snapshot_every'])
    assert len(policy) == len(times)
    gap = truth['oracle_values'][None, :, None, None, 24]-value[..., 24]
    assert np.all(gap >= -1e-9)
    # Fixed 0,1000,...,50000 grid, with half-weight at each endpoint.
    tracking = np.trapz(gap, x=times, axis=0)/CONFIG['additional_transitions']
    data = {key: saved[key] for key in SNAPSHOT_KEYS}
    data.update(snapshot_steps=times, total_steps=CONFIG['history_transitions']+times,
        actual_values=value, oracle_gaps=gap, time_averaged_oracle_gap=tracking,
        inherited_model_predictions=inherited_predictions(policy, history),
        rollout_metrics=saved['rollout_metrics'], metric_names=saved['collection_metric_names'],
        final_actual_visits=saved['actual_visits'], final_visit_weights=saved['visits'],
        condition_names=np.array(CONDITIONS), method_names=np.array(METHODS), action_names=np.array(ACTIONS),
        **truth)
    return data


def paired_difference(delta, lower_is_better=False):
    return dict(difference=describe(delta), lower_is_better=lower_is_better,
                **changes(-delta if lower_is_better else delta))


def summarize(data):
    tracking = data['time_averaged_oracle_gap']
    value = data['actual_values'][..., 24]
    predicted = data['predictions'][..., 24]
    error = predicted-value
    steps = data['snapshot_steps']
    duration = CONFIG['additional_transitions']
    time_absolute_error = np.trapz(abs(error), x=steps, axis=0)/duration
    time_error = np.trapz(error, x=steps, axis=0)/duration
    # Policy churn counts changed mixed action distributions, not sampled actions.
    churn = np.any(np.diff(data['policies'], axis=0) != 0, axis=-1).sum(axis=(0, -1))
    temporal_value_change = abs(np.diff(value, axis=0)).mean(axis=0)
    summary = dict(primary='Changed: forgetting minus cumulative time-averaged oracle-value gap; LOWER is better',
        primary_comparison=paired_difference(tracking[1, 2]-tracking[1, 1], True),
        stable_comparison=paired_difference(tracking[0, 2]-tracking[0, 1], True),
        integration='Trapezoidal integration on prespecified policy snapshots 0,1000,...,50000, divided by 50000; not realized online regret',
        uncertainty='Mean +/-1.96 SEM across 100 paired training seeds; Wilson fraction intervals; pointwise curve intervals, unadjusted',
        conditions={})
    for g, condition in enumerate(CONDITIONS):
        oracle = float(data['oracle_values'][g, 24])
        result = dict(oracle_value=oracle, threshold_90_percent=.9*oracle, methods={},
            paired_forgetting_minus_cumulative=dict(
                time_averaged_gap=paired_difference(tracking[g, 2]-tracking[g, 1], True),
                final_true_value=paired_difference(value[-1, g, 2]-value[-1, g, 1]),
                final_absolute_prediction_error=describe(abs(error[-1, g, 2])-abs(error[-1, g, 1])),
                time_averaged_absolute_prediction_error=describe(time_absolute_error[g, 2]-time_absolute_error[g, 1]),
                temporal_mean_absolute_value_change=describe(temporal_value_change[g, 2]-temporal_value_change[g, 1]),
                policy_state_changes=describe(churn[g, 2]-churn[g, 1]),
                interaction={str(metric): describe((data['rollout_metrics'][g, 2, :, :, k]-
                    data['rollout_metrics'][g, 1, :, :, k]).mean(axis=0)) for k,metric in enumerate(data['metric_names'])}),
            paired_final_values_vs_frozen={METHODS[m]: paired_difference(value[-1, g, m]-value[-1, g, 0]) for m in (1, 2)})
        for m, method in enumerate(METHODS):
            result['methods'][method] = dict(time_averaged_oracle_gap=describe(tracking[g, m]),
                initial_true_value=describe(value[0, g, m]), final_true_value=describe(value[-1, g, m]),
                final_reaching_90_percent=fraction_interval(value[-1, g, m] >= .9*oracle),
                ever_below_90_percent_after_start=fraction_interval((value[1:, g, m] < .9*oracle).any(axis=0)),
                final_predicted_return=describe(predicted[-1, g, m]),
                final_inherited_model_prediction=describe(data['inherited_model_predictions'][-1, g, m, :, 24]),
                final_inherited_model_prediction_error=describe(data['inherited_model_predictions'][-1, g, m, :, 24]-value[-1, g, m]),
                final_inherited_model_absolute_error=describe(abs(data['inherited_model_predictions'][-1, g, m, :, 24]-value[-1, g, m])),
                final_prediction_error=describe(error[-1, g, m]), final_absolute_prediction_error=describe(abs(error[-1, g, m])),
                time_averaged_prediction_error=describe(time_error[g, m]),
                time_averaged_absolute_prediction_error=describe(time_absolute_error[g, m]),
                policy_state_changes=describe(churn[g, m]), temporal_mean_absolute_value_change=describe(temporal_value_change[g, m]),
                interaction={str(metric): describe(data['rollout_metrics'][g, m, :, :, k].mean(axis=0))
                             for k,metric in enumerate(data['metric_names'])},
                final_total_evidence=describe(data['total_evidence'][-1, g, m]),
                final_fractional_positive_rows=describe(data['fractional_rows'][-1, g, m]),
                final_unknown_rows=describe(data['unknown_rows'][-1, g, m]))
        summary['conditions'][condition] = result
    summary['maximum_empirical_planning_residual'] = float(data['planning_residuals'].max())
    summary['original_history_mass_final_forgetting'] = CONFIG['history_transitions']*RHO**50
    summary['original_history_fraction_final_forgetting'] = float(summary['original_history_mass_final_forgetting']/data['total_evidence'][-1, 0, 2, 0])
    return summary


def seed_zero(data):
    return dict(seed=0, state=[3, 3], action_names=list(ACTIONS), successor_states=data['states'].tolist(),
        condition_names=CONDITIONS, method_names=METHODS, snapshot_steps=data['snapshot_steps'].tolist(),
        estimated_successor_probabilities=data['seed0_successor_probabilities'].tolist(),
        predicted_action_values=data['seed0_action_values'].tolist(),
        selected_probabilities=data['policies'][:, :, :, 0, 18].tolist(),
        behavior_probabilities=((1-CONFIG['epsilon'])*data['policies'][:, :, :, 0, 18]+CONFIG['epsilon']/3).tolist(),
        actual_visits=data['seed0_actual_visits'].tolist(), evidence_weights=data['seed0_visit_weights'].tolist(),
        true_successor_probabilities=data['transition'][:, 18].tolist(),
        actual_policy_values_from_state=data['actual_values'][:, :, :, 0, 18].tolist(),
        predicted_policy_values_from_state=data['predictions'][:, :, :, 0, 18].tolist())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/changing_regeneration'))
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    from changing_regeneration_figures import plot_all
    if args.plot_only or (args.output/'summary.json').exists():
        plot_all(load_npz(args.output/'results.npz'), args.output)
        print('Plot-only: no interaction, policy selection, or true evaluation repeated.')
        return
    started = perf_counter()
    sources = ['regeneration_regimes.py', 'changing_regeneration.py', 'run_changing_regeneration.py',
               'changing_regeneration_figures.py', 'check_changing_regeneration.py',
               'foraging.py', 'learned_world_model.py', 'model_guided_collection.py', 'q_learning.py',
               'run_q_learning.py', 'run_count_penalized_planning.py', 'figures.py', 'q_learning_figures.py']
    manifest = dict(experiment_id=7, config=CONFIG, rho=RHO, conditions=CONDITIONS, methods=METHODS,
        started_utc=datetime.now(timezone.utc).isoformat(), collection_seeds=collection_seeds(),
        seed_recipe='SeedSequence([20260929,7,seed,role]), roles 0 environment,100 actions; paired across all six method/condition combinations',
        draw_order='Each rollout: environment group then action group; each seed generator draws (1000,2), stacked on axis 1. All methods/conditions apply shared draws to own states.',
        input_history_sha256=sha(HISTORY), source_sha256={name:sha(ROOT/name) for name in sources},
        python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
        platform=platform.platform(), openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS','unset'),
        initialization='Experiment 5 checkpoint_050000, collector B (index1), unpenalized policy (index0), 250000 observations; no historical retraining',
        environment='Stable original rule; changed g*(1-.9*m/4) below capacity, zero at capacity, from first new transition. No regime signal to agents.',
        forgetting='Before every rollout in both worlds multiply successor/visit weights and reward sums by rho; add unit-weight observations. Blockwise, without reconstructed historical ages.',
        fitting='Every positive row normalized by ACTUAL weight, even below1; only zero rows assume zero-reward self-loop. No penalty.',
        actual_counts='Separate cumulative successor counts, visits and reward sums for accounting, including frozen agent; never used to fit forgetting model',
        primary='Changed forgetting minus cumulative time-averaged oracle-value gap; trapezoid on fixed 0:1000:50000 snapshot grid divided by50000; lower better, not online regret',
        information_boundary='Memory has no condition label. All interaction and empirical policy selection finish before enumerating/evaluating true models or oracles.',
        collection_transitions=30000000, historical_plus_new_per_seed=300000,
        measurement='Pre-action stocks and either stock zero; interaction summaries averaged within training seed before across-seed uncertainty',
        resume='Every5000: atomic stats, RNG states, rollout metrics and all preceding1000-step snapshots. Restore last checkpoint; interrupted unsaved tail deterministically repeated without double counting.',
        reproduction='OPENBLAS_NUM_THREADS=1 python run_changing_regeneration.py --output results/changing_regeneration_repeat',
        plotting='python run_changing_regeneration.py --plot-only')
    path = args.output/'manifest.json'
    if path.exists():
        saved = json.loads(path.read_text())
        assert all(saved[k] == manifest[k] for k in ('config','source_sha256','input_history_sha256','collection_seeds'))
        manifest = saved
    else:
        write_json(path, manifest)
    archive = load_npz(HISTORY)
    assert int(archive['total_steps']) == CONFIG['history_transitions']
    history = {key:archive[key][1] for key in ('successors','visits','reward_sums')}
    assert np.all(history['visits'].sum(axis=(-2,-1)) == CONFIG['history_transitions'])
    np.testing.assert_array_equal(history['successors'].sum(axis=-1), history['visits'])
    manifest.update(collect(args.output, history, archive['policies'][1, 0], archive['predicted_values'][1, 0], manifest['collection_seeds']))
    final = load_npz(args.output/'checkpoint_050000.npz')
    eval_started = perf_counter()
    truth = evaluation_models()
    data = evaluate(final, truth, history)
    manifest['true_evaluation_and_oracles_seconds'] = perf_counter()-eval_started
    np.savez_compressed(args.output/'results.npz', **data)
    write_json(args.output/'seed_0.json', seed_zero(data))
    summary = summarize(data)
    plot_all(data, args.output)
    manifest['completion_invocation_seconds'] = perf_counter()-started
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(path, manifest)
    write_json(args.output/'summary.json', summary)
    print('PRIMARY:', json.dumps(summary['primary_comparison']), flush=True)
    print('STABLE:', json.dumps(summary['stable_comparison']), flush=True)
    for condition,r in summary['conditions'].items():
        print(condition, 'oracle',r['oracle_value'], flush=True)
        for method,v in r['methods'].items():
            print(' ',method,'average gap',v['time_averaged_oracle_gap']['mean'],'final value',v['final_true_value']['mean'],
                  '>=90%',v['final_reaching_90_percent']['count'],'MAE',v['final_absolute_prediction_error']['mean'],flush=True)
    print('Runtime:',{k:v for k,v in manifest.items() if k.endswith('_seconds')},flush=True)


if __name__ == '__main__':
    main()
