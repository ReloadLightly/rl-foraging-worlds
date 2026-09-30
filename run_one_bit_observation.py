"""One fixed follow-up on Experiment 8: fit one extra observation bit once."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from foraging import ACTIONS
from learned_world_model import evaluate_policy
from one_bit_observation import CONFIG, GROUPS, OBSERVATION, STATES, select
from run_q_learning import describe, fraction_interval, load_npz, write_json
from run_state_representation import paired, sha

ROOT = Path(__file__).resolve().parent
HISTORY = ROOT/'results/model_guided_collection/checkpoint_050000.npz'
BASELINE = ROOT/'results/state_representation/results.npz'


def evaluate(selected, baseline, truth):
    """True evaluation only after the new policies have been selected and saved."""
    np.testing.assert_array_equal(STATES, truth['states'])
    np.testing.assert_array_equal(selected['training_seeds'], baseline['training_seeds'])
    actual = evaluate_policy(selected['lifted_policy'], truth['transition'], truth['immediate_rewards'], CONFIG['gamma'])
    # Reuse all baseline policies, predictions and actual values: no baseline refits/evaluation.
    data = {key:baseline[key] for key in ['oracle_values', 'oracle_policy', 'full_visits', 'compressed_visits']}
    data.update(states=STATES, groups=GROUPS, observation_mapping=OBSERVATION,
        training_seeds=selected['training_seeds'], bit_visits=selected['visits'], bit_unknown_rows=selected['unknown_rows'],
        collector_names=np.array(CONFIG['collectors']), representation_names=np.array(CONFIG['representations']), action_names=np.array(ACTIONS),
        policies=np.concatenate((baseline['policies'], selected['lifted_policy'][:, None]), axis=1),
        predictions=np.concatenate((baseline['predictions'], selected['lifted_prediction'][:, None]), axis=1),
        actual_values=np.concatenate((baseline['actual_values'], actual[:, None]), axis=1))
    return data


def mechanism(selected, data, truth):
    indices = np.array([23, 19, 18, 14])  # (4,3), (3,4), (3,3), (2,4), fixed before fitting.
    z = OBSERVATION[indices]
    probabilities = np.stack([truth['transition'][indices][..., OBSERVATION == k].sum(axis=-1) for k in range(16)], axis=-1)
    actual_q = truth['immediate_rewards'] + CONFIG['gamma']*np.einsum('san,...n->...sa', truth['transition'], data['actual_values'][:, 2])
    visits = data['full_visits'][:, :, indices]
    row_visits = selected['visits'][:, :, z]
    return dict(states=STATES[indices], state_indices=indices, observation_indices=z, groups=GROUPS,
        true_next_observation_probabilities=probabilities, true_rewards=truth['immediate_rewards'][indices],
        true_optimal_actions=truth['policies'][-1, 2, indices], visits=visits,
        conditional_state_weights=visits/np.maximum(row_visits, 1), observed_rows=row_visits>0,
        fitted_next_observation_probabilities=selected['transition'][:, :, z], fitted_rewards=selected['rewards'][:, :, z],
        fitted_action_values=selected['action_values'][:, :, z], actual_action_values=actual_q[:, :, indices],
        predicted_values=selected['prediction'][:, :, z], actual_values=data['actual_values'][:, 2][:, :, indices],
        selected_actions=selected['policy'][:, :, z])


def summarize(data, diagnostic):
    actual, predicted = data['actual_values'][..., 24], data['predictions'][..., 24]
    errors = predicted-actual
    result = dict(primary='Model-controlled history: bit minus total-only true V_pi(4,4), gamma=.99, fixed 250000 observations',
        primary_comparison=paired(actual[1, 2]-actual[1, 1]),
        uncertainty='Mean +/-1.96 SEM across 100 training seeds; contrasts paired within seed; Wilson fraction intervals. No exclusions.',
        qualification='Follow-up on reused histories, not an independent replication. Fixed hand-designed representation, not learned or searched.',
        oracle_value=float(data['oracle_values'][24]), policies={}, comparisons={}, coverage={}, mechanism={})
    for c, collector in enumerate(CONFIG['collectors']):
        result['comparisons'][collector] = dict(bit_minus_total=paired(actual[c, 2]-actual[c, 1]),
            full_minus_bit=paired(actual[c, 0]-actual[c, 2]),
            absolute_error_bit_minus_total=error_comparison(abs(errors[c, 2])-abs(errors[c, 1])),
            absolute_error_bit_minus_full=error_comparison(abs(errors[c, 2])-abs(errors[c, 0])),
            fraction_of_mean_representation_gap_recovered=float((actual[c, 2]-actual[c, 1]).mean()/(actual[c, 0]-actual[c, 1]).mean()))
        for r, (representation, prefix) in enumerate(zip(CONFIG['representations'], ['full', 'compressed', 'bit'])):
            name = collector+' / '+representation
            result['policies'][name] = dict(true_value=describe(actual[c, r]), predicted_value=describe(predicted[c, r]),
                predicted_minus_actual=describe(errors[c, r]), absolute_prediction_error=describe(abs(errors[c, r])),
                percentile_10=float(np.quantile(actual[c, r], .1)),
                fraction_90_percent=fraction_interval(actual[c, r] >= CONFIG['oracle_fraction']*data['oracle_values'][24]),
                seed_indices_below_90_percent=np.flatnonzero(actual[c, r] < CONFIG['oracle_fraction']*data['oracle_values'][24]).tolist())
            counts = data[prefix+'_visits'][c]
            result['coverage'][name] = dict(row_count=int(counts.shape[-2]*3),
                rows_below={str(t):describe((counts<t).sum(axis=(-2,-1))) for t in CONFIG['coverage_thresholds']})
        result['mechanism'][collector] = {}
        for i, state in enumerate(diagnostic['states']):
            result['mechanism'][collector][str(tuple(state))] = dict(
                mean_policy_probabilities=diagnostic['selected_actions'][c, :, i].mean(axis=0).tolist(),
                predicted_value=describe(diagnostic['predicted_values'][c, :, i]),
                actual_value=describe(diagnostic['actual_values'][c, :, i]),
                mean_fitted_action_values=diagnostic['fitted_action_values'][c, :, i].mean(axis=0).tolist())
    result['checks'] = ['Only 16 reachable pairs; preserve action-specific successor, visit and reward totals',
        'Direct aggregation check at all prespecified observation groups; successor counts sum to visits',
        'Lifted policies agree within each observation group and have unit row sums',
        'Only new representation fitted; all new policies saved before loading true model and baseline results']
    return result


def error_comparison(delta):
    return dict(difference=describe(delta), reduced=int(np.sum(delta < -1e-10)),
                increased=int(np.sum(delta > 1e-10)), tied=int(np.sum(abs(delta) <= 1e-10)))


def seed_csv(data, output):
    with (output/'seed_results.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        names = [c+'_'+r for c in ['Q', 'model'] for r in ['full', 'total', 'bit']]
        writer.writerow(['seed']+[n+'_actual' for n in names]+[n+'_predicted' for n in names]+
                        ['Q_bit_minus_total', 'model_bit_minus_total', 'Q_full_minus_bit', 'model_full_minus_bit'])
        for seed in range(100):
            v = data['actual_values'][:, :, seed, 24]
            writer.writerow([seed]+v.ravel().tolist()+data['predictions'][:, :, seed, 24].ravel().tolist()+
                            (v[:, 2]-v[:, 1]).tolist()+(v[:, 0]-v[:, 2]).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/one_bit_observation'))
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    from one_bit_observation_figures import plot_all
    if args.plot_only or (args.output/'summary.json').exists():
        plot_all(load_npz(args.output/'results.npz'), load_npz(args.output/'mechanism.npz'), args.output)
        print('Plot-only: no fitting, evaluation or collection.')
        return
    started = perf_counter()
    source_names = ['one_bit_observation.py', 'run_one_bit_observation.py', 'one_bit_observation_figures.py',
                    'state_representation.py', 'run_state_representation.py', 'learned_world_model.py',
                    'foraging.py', 'q_learning.py', 'run_q_learning.py', 'run_count_penalized_planning.py', 'figures.py']
    input_names = ['results/model_guided_collection/checkpoint_050000.npz',
                   'results/state_representation/results.npz', 'results/state_representation/manifest.json']
    manifest = dict(experiment_id=9, config=CONFIG, groups=GROUPS.tolist(), observation_mapping=OBSERVATION.tolist(),
        training_seed_indices=list(range(100)), started_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={n:sha(ROOT/n) for n in source_names}, input_sha256={n:sha(ROOT/n) for n in input_names},
        selection='Only the new 16-group representation; existing unpenalized estimator/policy iteration, gamma .99, uniform ties atol1e-10.',
        unknown_rows='Zero-reward self-loop for unvisited rows is an assumption; masks saved.',
        information_boundary='Select and save new policies before loading baseline results or true dynamics; true model only evaluates selected policies and mechanism.',
        limitation='Fixed hand-designed follow-up on reused full-observation histories, not an independent replication or best partial-observation policy.',
        new_transitions=0, new_Q_updates=0, new_random_streams=0, baseline_refits=0,
        python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
        platform=platform.platform(), openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS','unset'),
        reproduction='OPENBLAS_NUM_THREADS=1 python run_one_bit_observation.py --output results/one_bit_observation_repeat',
        plotting='python run_one_bit_observation.py --plot-only')
    # Verify the same historical input used by Experiment 8 without refitting any baseline.
    old_manifest = json.loads((ROOT/input_names[-1]).read_text())
    assert old_manifest['input_history_sha256'] == manifest['input_sha256'][input_names[0]]
    path = args.output/'manifest.json'
    if path.exists():
        old = json.loads(path.read_text())
        assert all(old[k] == manifest[k] for k in ['config', 'source_sha256', 'input_sha256'])
        manifest = old
    else:
        write_json(path, manifest)
    selected_path = args.output/'selected.npz'
    if selected_path.exists():
        selected = load_npz(selected_path)
    else:
        selected, times = select(load_npz(HISTORY))
        manifest.update(times)
        np.savez_compressed(selected_path, **selected)
        write_json(path, manifest)
    # Information boundary: all new policies are now fixed on disk.
    eval_started = perf_counter()
    baseline = load_npz(BASELINE)
    truth = load_npz(ROOT/'results/first_world/model.npz')
    data = evaluate(selected, baseline, truth)
    diagnostic = mechanism(selected, data, truth)
    manifest['evaluation_and_diagnostics_seconds'] = perf_counter()-eval_started
    manifest['true_model_sha256'] = sha(ROOT/'results/first_world/model.npz')
    np.savez_compressed(args.output/'results.npz', **data)
    np.savez_compressed(args.output/'mechanism.npz', **diagnostic)
    summary = summarize(data, diagnostic)
    seed_csv(data, args.output)
    plot_all(data, diagnostic, args.output)
    manifest['completion_invocation_seconds'] = perf_counter()-started
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(path, manifest)
    write_json(args.output/'summary.json', summary)
    print('PRIMARY:', json.dumps(summary['primary_comparison']), flush=True)
    for collector, comparison in summary['comparisons'].items():
        print(collector, json.dumps(comparison), flush=True)
    for name, outcome in summary['policies'].items():
        print(name, 'actual', outcome['true_value']['mean'], 'prediction', outcome['predicted_value']['mean'],
              'MAE', outcome['absolute_prediction_error']['mean'], '>=90% oracle', outcome['fraction_90_percent']['count'], flush=True)
    print('Runtime:', {k:v for k,v in manifest.items() if k.endswith('_seconds')}, flush=True)


if __name__ == '__main__':
    main()
