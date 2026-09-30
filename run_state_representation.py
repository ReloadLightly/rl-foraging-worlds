"""Experiment 8: same archived experience, two representations; exact evaluation."""

import argparse
import csv
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
from run_count_penalized_planning import changes
from run_q_learning import describe, fraction_interval, load_npz, write_json
from state_representation import CONFIG, STATES, OBSERVATION, select

ROOT = Path(__file__).resolve().parent
HISTORY = ROOT/'results/model_guided_collection/checkpoint_050000.npz'


def paired(delta):
    return dict(difference=describe(delta), **changes(delta))


def evaluate(selected, truth):
    """Called only after all four sets of policies are selected and saved."""
    np.testing.assert_array_equal(STATES, truth['states'])
    actual = evaluate_policy(selected['policies'], truth['transition'], truth['immediate_rewards'], CONFIG['gamma'])
    oracle = truth['values'][-1, 2]
    assert np.all(actual <= oracle+1e-9)
    result = {key:selected[key] for key in ['training_seeds', 'states', 'observation_mapping', 'policies', 'predictions',
              'full_visits', 'compressed_visits', 'full_unknown_rows', 'compressed_unknown_rows']}
    result.update(actual_values=actual, oracle_values=oracle, oracle_policy=truth['policies'][-1, 2],
                  collector_names=np.array(CONFIG['collectors']), representation_names=np.array(CONFIG['representations']),
                  action_names=np.array(ACTIONS))
    value = actual[..., 24]
    result.update(primary_full_minus_compressed=value[1, 0]-value[1, 1],
        full_experience_benefit=value[1, 0]-value[0, 0], compressed_experience_benefit=value[1, 1]-value[0, 1],
        experience_benefit_difference=(value[1, 0]-value[0, 0])-(value[1, 1]-value[0, 1]))
    return result


def mechanism(selected, truth):
    """True diagnostic distributions and collector weights, never planner inputs."""
    indices = np.array([23, 19])  # (4,3), (3,4), chosen before fitting.
    probabilities = np.stack([truth['transition'][indices][..., OBSERVATION == z].sum(axis=-1) for z in range(9)], axis=-1)
    reward = truth['immediate_rewards'][indices]
    # Rewards are deterministic here: joint P(R=r_a, Z'=z' | S=s,A=a)
    # is exactly the next-observation vector, at the displayed reward r_a.
    visits = selected['full_visits'][:, :, indices, :]
    totals = selected['compressed_visits'][:, :, 7, :]
    np.testing.assert_array_equal(visits.sum(axis=-2), totals)
    weights = visits/np.maximum(totals[:, :, None, :], 1)
    observed = totals > 0
    np.testing.assert_allclose(weights.sum(axis=-2)[observed], 1.)
    return dict(states=STATES[indices], state_indices=indices, total_observation=np.array(7),
        true_rewards=reward, true_next_observation_probabilities=probabilities,
        true_optimal_actions=truth['policies'][-1, 2, indices], visits=visits, observed_rows=observed,
        conditional_state_weights=weights, compressed_visit_totals=totals,
        fitted_next_observation_probabilities=selected['compressed_transition'][:, :, 7],
        fitted_rewards=selected['compressed_rewards'][:, :, 7],
        selected_full_actions=selected['full_policy'][:, :, indices],
        selected_compressed_actions=selected['compressed_policy'][:, :, 7])


def summarize(data, diagnostic):
    values, predicted = data['actual_values'][..., 24], data['predictions'][..., 24]
    error = predicted-values
    oracle = float(data['oracle_values'][24])
    summary = dict(primary='Model-controlled histories: FULL minus COMPRESSED true V_pi(4,4), gamma=.99, fixed 250000 observations',
        primary_comparison=paired(data['primary_full_minus_compressed']),
        full_experience_benefit=paired(data['full_experience_benefit']),
        compressed_experience_benefit=paired(data['compressed_experience_benefit']),
        experience_benefit_difference=paired(data['experience_benefit_difference']),
        difference_direction='(model minus Q within FULL) minus (model minus Q within COMPRESSED)',
        uncertainty='Mean +/-1.96 SEM across 100 training seeds, paired within seed; Wilson fraction intervals; no seed exclusions',
        oracle_value=oracle, threshold_90_percent=CONFIG['oracle_fraction']*oracle, policies={}, coverage={}, merged_state_weights={})
    for c, collector in enumerate(CONFIG['collectors']):
        summary['merged_state_weights'][collector] = {}
        for a, action in enumerate(ACTIONS):
            summary['merged_state_weights'][collector][action] = dict(
                observed_seeds=int(diagnostic['observed_rows'][c, :, a].sum()),
                visits_4_3=describe(diagnostic['visits'][c, :, 0, a]), visits_3_4=describe(diagnostic['visits'][c, :, 1, a]),
                weight_4_3_among_observed=describe(diagnostic['conditional_state_weights'][c, diagnostic['observed_rows'][c, :, a], 0, a]),
                weight_3_4_among_observed=describe(diagnostic['conditional_state_weights'][c, diagnostic['observed_rows'][c, :, a], 1, a]))
        for r, (representation, prefix) in enumerate(zip(CONFIG['representations'], ['full', 'compressed'])):
            name = collector+' / '+representation
            summary['policies'][name] = dict(true_value=describe(values[c, r]), predicted_value=describe(predicted[c, r]),
                predicted_minus_actual=describe(error[c, r]), absolute_prediction_error=describe(abs(error[c, r])),
                fraction_90_percent=fraction_interval(values[c, r] >= CONFIG['oracle_fraction']*oracle),
                percentile_10=float(np.quantile(values[c, r], .1)),
                seed_indices_below_90_percent=np.flatnonzero(values[c, r] < CONFIG['oracle_fraction']*oracle).tolist())
            counts = data[prefix+'_visits'][c]
            summary['coverage'][name] = dict(row_count=int(counts.shape[-2]*counts.shape[-1]),
                rows_below={str(t):describe((counts < t).sum(axis=(-2, -1))) for t in CONFIG['coverage_thresholds']},
                fraction_of_rows_below={str(t):describe((counts < t).mean(axis=(-2, -1))) for t in CONFIG['coverage_thresholds']})
    summary['paired_absolute_error_full_minus_compressed_model_data'] = describe(abs(error[1, 0])-abs(error[1, 1]))
    summary['checks'] = ['Exact action-specific count/reward aggregation; successor counts sum to visits',
        'Direct aggregation check for total seven; zero new transitions; 250000 observations per seed/dataset',
        'Lifted policies identical within observation groups and row-stochastic',
        'Full-state refits recover saved Experiment 5 policies and predictions',
        'All policies selected and saved before true model is loaded; no oracle-based selection']
    return summary


def seed_csv(data, output):
    """One paired training-seed row, including all four fixed policies."""
    with (output/'seed_results.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        names = ['Q_full', 'Q_compressed', 'model_full', 'model_compressed']
        writer.writerow(['seed']+[n+'_actual' for n in names]+[n+'_predicted' for n in names]+
            ['model_full_minus_compressed', 'full_model_minus_Q', 'compressed_model_minus_Q', 'experience_benefit_difference'])
        for s in range(CONFIG['seeds']):
            writer.writerow([s]+data['actual_values'][:, :, s, 24].ravel().tolist()+data['predictions'][:, :, s, 24].ravel().tolist()+
                [float(data[key][s]) for key in ['primary_full_minus_compressed','full_experience_benefit','compressed_experience_benefit','experience_benefit_difference']])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/state_representation'))
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    from state_representation_figures import plot_all
    if args.plot_only or (args.output/'summary.json').exists():
        plot_all(load_npz(args.output/'results.npz'), load_npz(args.output/'mechanism.npz'), args.output)
        print('Plot-only: no fitting, planning, evaluation or collection.')
        return
    started = perf_counter()
    sources = ['state_representation.py', 'run_state_representation.py', 'state_representation_figures.py',
               'learned_world_model.py', 'q_learning.py', 'foraging.py', 'run_q_learning.py',
               'run_count_penalized_planning.py', 'figures.py']
    manifest = dict(experiment_id=8, config=CONFIG, started_utc=datetime.now(timezone.utc).isoformat(),
        input_history_sha256=sha(HISTORY), input_path=str(HISTORY.relative_to(ROOT)),
        input_manifest_sha256=sha(ROOT/'results/model_guided_collection/manifest.json'),
        source_sha256={name:sha(ROOT/name) for name in sources}, training_seed_indices=list(range(100)),
        new_transitions=0, new_Q_updates=0, selection='Existing estimate and unpenalized policy iteration, gamma .99, uniform ties within absolute1e-10',
        observation='z=A+B, integers0..8. Sum successor counts on BOTH state axes, visits/reward sums on origin axis; actions preserved.',
        unknown_rows='Existing zero-reward self-loop assumption only for unvisited rows; masks saved.',
        lifting='pi(a|A,B)=pi_compressed(a|A+B), no full-state information added to compressed action selection',
        information_boundary='Fit all full/compressed policies from archived counts and save selected.npz before loading true model; true diagnostics/evaluation only afterwards.',
        limitation='Compressed model is a collector-dependent fitted surrogate, not assumed Markov or the best possible partial-observation policy; collectors had FULL observations.',
        python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
        platform=platform.platform(), openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS','unset'),
        reproduction='OPENBLAS_NUM_THREADS=1 python run_state_representation.py --output results/state_representation_repeat',
        plotting='python run_state_representation.py --plot-only')
    path = args.output/'manifest.json'
    if path.exists():
        old = json.loads(path.read_text())
        assert all(old[k] == manifest[k] for k in ('config','input_history_sha256','source_sha256'))
        manifest = old
    else:
        write_json(path, manifest)
    selected_path = args.output/'selected.npz'
    if selected_path.exists():
        selected = load_npz(selected_path)
        print('Using saved selected policies; no fitting repeated.',flush=True)
    else:
        selected, times = select(load_npz(HISTORY))
        manifest.update(times)
        np.savez_compressed(selected_path, **selected)
        write_json(path, manifest)
    # Information boundary: policy selection is now finished.
    eval_started = perf_counter()
    true_path = ROOT/'results/first_world/model.npz'
    truth = load_npz(true_path)
    data = evaluate(selected, truth)
    diagnostic = mechanism(selected, truth)
    manifest['evaluation_and_diagnostics_seconds'] = perf_counter()-eval_started
    manifest['true_model_sha256'] = sha(true_path)
    np.savez_compressed(args.output/'results.npz', **data)
    np.savez_compressed(args.output/'mechanism.npz', **diagnostic)
    summary = summarize(data, diagnostic)
    seed_csv(data, args.output)
    plot_all(data, diagnostic, args.output)
    manifest['completion_invocation_seconds'] = perf_counter()-started
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(path, manifest)
    write_json(args.output/'summary.json', summary)
    print('PRIMARY:',json.dumps(summary['primary_comparison']),flush=True)
    for key in ['full_experience_benefit','compressed_experience_benefit','experience_benefit_difference']:
        print(key,json.dumps(summary[key]),flush=True)
    for name,value in summary['policies'].items():
        print(name,'true value',value['true_value']['mean'],'prediction',value['predicted_value']['mean'],
              '>=90% oracle',value['fraction_90_percent']['count'],flush=True)
    print('Runtime:',{k:v for k,v in manifest.items() if k.endswith('_seconds')},flush=True)


if __name__ == '__main__':
    main()
