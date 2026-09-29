"""Experiment 4: fixed count caution on Experiment 3 data, with no collection."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from foraging import ACTIONS, TIE_ATOL, WORLD, state_index, step
from learned_world_model import estimate, evaluate_policy, plan
from run_learned_world_model import CURVES, METRICS, evaluation_seeds
from run_q_learning import describe, fraction_interval, load_npz, write_json


ROOT = Path(__file__).resolve().parent
BASE = ROOT/'results/learned_world_model'
METHODS = ['Frozen Q', 'Unpenalized model', 'Count-penalized model']
CONFIG = dict(training_seeds=100, final_checkpoint=200000, gamma=.99,
              penalty_coefficient=1., trajectories_per_seed=20, evaluation_steps=1000,
              evaluation_epsilon=0., bin_width=20, illustration_seed=0,
              illustration_state=[3, 4], illustration_trajectory=0, illustration_steps=120)


def planning_rewards(empirical_rewards, visits):
    """A fixed heuristic in reward units; do not clip negatives or modify inputs."""
    penalty = CONFIG['penalty_coefficient']/np.sqrt(np.maximum(visits, 1))
    return empirical_rewards-penalty, penalty


def select_policies(collection):
    """All policy selection finishes before any true-model access."""
    policies, scores, predictions, rounds, residuals, times = [], [], [], [], [], []
    for i, completed in enumerate(collection['steps']):
        started = perf_counter()
        p, r, unknown = estimate(collection['successors'][i], collection['visits'][i],
                                 collection['reward_sums'][i])
        adjusted, penalty = planning_rewards(r, collection['visits'][i])
        pi, score, count, residual = plan(p, adjusted, CONFIG['gamma'])
        policies.append(pi)
        scores.append(score)  # A: internal value under penalized rewards.
        predictions.append(evaluate_policy(pi, p, r, CONFIG['gamma']))  # B: original reward.
        rounds.append(count)
        residuals.append(residual)
        times.append(perf_counter()-started)
        print(f'Selected {completed:>7,}/seed | A score {score[:, 24].mean():.4f} | '
              f'B predicted return {predictions[-1][:, 24].mean():.4f}', flush=True)
    # Linearity of policy evaluation separates accumulated penalty from reward.
    discounted_penalty = evaluate_policy(policies[-1], p, penalty, CONFIG['gamma'])
    np.testing.assert_allclose(predictions[-1]-scores[-1], discounted_penalty, atol=1e-10)
    return dict(checkpoint_steps=collection['steps'], policies=np.array(policies),
                internal_scores=np.array(scores), original_reward_predictions=np.array(predictions),
                planning_iterations=np.array(rounds), optimality_residuals=np.array(residuals),
                planning_seconds=np.array(times), final_transition=p, final_empirical_rewards=r,
                final_planning_rewards=adjusted, final_penalty=penalty, final_unknown=unknown,
                final_visits=collection['visits'][-1], discounted_penalty=discounted_penalty)


def evaluate_selected(selected, baseline, true_model):
    """C and baseline comparisons are evaluated only after policies are fixed."""
    actual = evaluate_policy(selected['policies'], true_model['transition'],
                             true_model['immediate_rewards'], CONFIG['gamma'])
    # Reuse saved baseline policies/values. Only the new policies are evaluated.
    data = dict(checkpoint_steps=selected['checkpoint_steps'],
        checkpoint_policies=np.concatenate((baseline['checkpoint_policies'], selected['policies'][:, None]), axis=1),
        checkpoint_values=np.concatenate((baseline['checkpoint_values'], actual[:, None]), axis=1),
        checkpoint_predictions=np.stack((baseline['checkpoint_predicted_values'],
                                         selected['original_reward_predictions']), axis=1),
        checkpoint_internal_scores=selected['internal_scores'],
        method_names=np.array(METHODS), prediction_method_names=np.array(METHODS[1:]),
        oracle_values=baseline['oracle_values'], oracle_policy=baseline['oracle_policy'],
        states=baseline['states'])
    # Same estimated model as Experiment 3; the penalty changes only the objective.
    np.testing.assert_array_equal(selected['final_transition'], baseline['final_transition'])
    np.testing.assert_array_equal(selected['final_empirical_rewards'], baseline['final_rewards'])
    np.testing.assert_array_equal(selected['final_unknown'], baseline['final_unknown'])
    np.testing.assert_allclose(data['checkpoint_policies'].sum(axis=-1), 1)
    assert np.all(actual <= baseline['oracle_values']+1e-9)
    return data


def simulate_new(policy, baseline):
    """Adapt Exp. 3's time loop to ONE new policy; reuse all baseline outcomes."""
    n, paths, steps, width = (CONFIG[k] for k in ('training_seeds', 'trajectories_per_seed',
                                                'evaluation_steps', 'bin_width'))
    seeds = evaluation_seeds()  # The experiment-3 recipe is intentionally reused.
    for role in ('regeneration', 'actions'):
        key = 'evaluation_regeneration_seeds' if role == 'regeneration' else 'evaluation_action_seeds'
        np.testing.assert_array_equal(seeds[role], baseline[key])
    env = np.stack([np.random.default_rng(int(s)).random((steps, 2))
                    for s in baseline['evaluation_regeneration_seeds'].flat], axis=1)
    action = np.stack([np.random.default_rng(int(s)).random(steps)
                       for s in baseline['evaluation_action_seeds'].flat], axis=1)
    np.testing.assert_array_equal(env[:, 0], baseline['trace_regeneration_uniforms'])
    np.testing.assert_array_equal(action[:, 0], baseline['trace_action_uniforms'])
    rows = np.repeat(np.arange(n), paths)
    stocks = np.tile(WORLD.initial, (n*paths, 1))
    totals = np.zeros((n*paths, len(METRICS)))
    cumulative = np.zeros(n*paths)
    curve_totals = np.zeros((n*paths, len(CURVES)))
    curves = np.zeros((n, steps//width, len(CURVES)))
    trace_stocks = np.zeros((steps+1, 2), dtype=np.int8)
    trace_actions = np.zeros(steps, dtype=np.int8)
    trace_rewards = np.zeros(steps)
    trace_stocks[0] = stocks[0]
    for t in range(steps):
        cdf = policy[rows, state_index(stocks)].cumsum(axis=-1)
        cdf[:, -1] = 1.
        actions = (action[t, :, None] >= cdf).sum(axis=-1)
        successor, reward = step(stocks, actions, env[t])
        depleted, rest = stocks == 0, actions == 2
        failed, success = (actions < 2) & (reward == 0), reward > 0
        cumulative += reward
        totals += np.column_stack((CONFIG['gamma']**t*reward, reward, stocks, depleted,
            depleted.any(axis=1), depleted.all(axis=1), rest, failed, success,
            success & (actions == 0), success & (actions == 1), reward))
        curve_totals += np.column_stack((reward, cumulative, stocks,
                                        depleted.any(axis=1), rest, failed))
        if (t+1) % width == 0:
            binned = curve_totals/width
            binned[:, 1] = cumulative
            curves[:, t//width] = binned.reshape(n, paths, -1).mean(axis=1)
            curve_totals[:] = 0
        trace_stocks[t+1], trace_actions[t], trace_rewards[t] = successor[0], actions[0], reward[0]
        stocks = successor
    totals[:, 2:] /= steps
    totals = totals.reshape(n, paths, -1)
    np.testing.assert_allclose(totals[..., 8:11].sum(axis=-1), 1)
    np.testing.assert_allclose(totals[..., -1], totals[..., -3]+1.5*totals[..., -2])
    new = dict(trajectory_metrics=totals, seed_metrics=totals.mean(axis=1), seed_curves=curves,
               trace_stocks=trace_stocks, trace_actions=trace_actions, trace_rewards=trace_rewards)
    combined = dict(baseline)
    for key, value in new.items():
        combined[key] = np.concatenate((baseline[key], value[None]), axis=0)
    combined['method_names'] = np.array(METHODS)
    return combined


def changes(difference):
    return dict(improved=int(np.sum(difference > TIE_ATOL)),
                worsened=int(np.sum(difference < -TIE_ATOL)),
                tied=int(np.sum(np.abs(difference) <= TIE_ATOL)))


def summarize(data, behavior, selected):
    value = data['checkpoint_values'][-1, :, :, 24]
    prediction = data['checkpoint_predictions'][-1, :, :, 24]
    errors = prediction-value[1:]
    oracle = float(data['oracle_values'][24])
    return dict(primary_endpoint='Exact true-environment V_pi(4,4), gamma=.99, fixed 200000 transitions',
        uncertainty='Mean +/- 1.96 SEM across 100 training seeds; fractions: Wilson 95%; pointwise, unadjusted',
        oracle_value=oracle, threshold_90_percent=.9*oracle,
        methods={name: dict(final_value=describe(value[m]),
                           percentile_10=float(np.quantile(value[m], .1)),
                           reaching_90_percent=fraction_interval(value[m] >= .9*oracle))
                 for m, name in enumerate(METHODS)},
        paired_new_minus_baseline={name: dict(value_difference=describe(value[2]-value[m]),
                                             **changes(value[2]-value[m]))
                                  for m, name in enumerate(METHODS[:2])},
        prediction={name: dict(predicted_return=describe(prediction[m]),
                              predicted_minus_actual=describe(errors[m]),
                              absolute_error=describe(np.abs(errors[m])),
                              rmse=float(np.sqrt(np.mean(errors[m]**2))),
                              overpredicted=fraction_interval(errors[m] > TIE_ATOL))
                    for m, name in enumerate(METHODS[1:])},
        paired_prediction_error_change=describe(errors[1]-errors[0]),
        paired_absolute_error_change=describe(np.abs(errors[1])-np.abs(errors[0])),
        internal_score_A=describe(selected['internal_scores'][-1, :, 24]),
        expected_discounted_penalty_B_minus_A=describe(selected['discounted_penalty'][:, 24]),
        frozen_behavior={name: {metric: describe(behavior['seed_metrics'][m, :, k])
                               for k, metric in enumerate(METRICS)} for m, name in enumerate(METHODS)},
        paired_frozen_new_minus_baseline={name: {metric: describe(behavior['seed_metrics'][2, :, k]-
                                                    behavior['seed_metrics'][m, :, k])
                                               for k, metric in enumerate(METRICS)}
                                        for m, name in enumerate(METHODS[:2])},
        maximum_planning_residual=float(selected['optimality_residuals'].max()))


def seed_zero(data, selected, true_model):
    """Preselected illustration and diagnostics, not a policy-selection input."""
    p, r = selected['final_transition'][0], selected['final_empirical_rewards'][0]
    pi = data['checkpoint_policies'][-1, :, 0]
    predicted = evaluate_policy(pi, p, r, CONFIG['gamma'])
    scores = selected['internal_scores'][-1, 0]
    unpen_q = r + CONFIG['gamma']*(p @ predicted[1])
    penal_q = selected['final_planning_rewards'][0] + CONFIG['gamma']*(p @ scores)
    state = int(state_index(CONFIG['illustration_state']))
    return dict(seed=0, state=[3, 4], state_index=state, action_names=list(ACTIONS),
        visit_counts=selected['final_visits'][0, state].tolist(),
        penalties=selected['final_penalty'][0, state].tolist(),
        empirical_rewards=r[state].tolist(), planning_rewards=selected['final_planning_rewards'][0, state].tolist(),
        empirical_self_loop_probabilities=p[state, :, state].tolist(),
        true_self_loop_probabilities=true_model['transition'][state, :, state].tolist(),
        unpenalized_action_values=unpen_q[state].tolist(), penalized_action_scores=penal_q[state].tolist(),
        policies={name: dict(action_probabilities=pi[m, state].tolist(),
                            state_predicted_B=float(predicted[m, state]),
                            state_actual_C=float(data['checkpoint_values'][-1, m, 0, state]),
                            initial_predicted_B=float(predicted[m, 24]),
                            initial_actual_C=float(data['checkpoint_values'][-1, m, 0, 24]))
                  for m, name in enumerate(METHODS)},
        penalized_state_score_A=float(scores[state]), penalized_initial_score_A=float(scores[24]))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/count_penalized_planning'))
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    from count_penalized_figures import plot_all
    if args.plot_only or (args.output/'summary.json').exists():
        plot_all(load_npz(args.output/'results.npz'), load_npz(args.output/'frozen_behavior.npz'),
                 json.loads((args.output/'seed_0.json').read_text()), args.output)
        print('Regenerated figures only; no collection, policy selection or evaluation.')
        return
    started = perf_counter()
    sources = ['run_count_penalized_planning.py', 'check_count_penalized_planning.py',
               'count_penalized_figures.py', 'learned_world_model.py', 'run_learned_world_model.py',
               'foraging.py', 'q_learning.py', 'run_q_learning.py', 'run_experiment.py',
               'figures.py', 'q_learning_figures.py']
    inputs = ['collection.npz', 'results.npz', 'frozen_behavior.npz', 'manifest.json']
    manifest = dict(config=CONFIG, experiment_id=4, started_utc=datetime.now(timezone.utc).isoformat(),
        input_sha256={name: sha(BASE/name) for name in inputs},
        source_sha256={name: sha(ROOT/name) for name in sources},
        python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
        platform=platform.platform(), openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS', 'unset'),
        intervention='planning_reward = empirical_reward - 1.0/sqrt(max(N(s,a),1)); no clipping; heuristic, not a confidence bound',
        information_boundary='Select all policies from independent per-seed counts before loading true model; no collection, Q updates, or baseline planning/simulation',
        unknown_rows='Same empirical zero-reward self-loop as Experiment 3; planning reward becomes -1',
        planner='Existing policy iteration, gamma .99, uniform initialization/ties within 1e-10 absolute tolerance, rtol=0',
        evaluation_seed_recipe='Reuse Experiment 3: SeedSequence([20260929,3,training_seed,trajectory,role]), roles 0 regeneration,100 actions',
        pairing='Saved seed IDs; identical draw generation; seed-0 trace draws checked exactly; independent of collection',
        measurement='Stocks/depletion before acting; aggregate 20 trajectories within each training seed before uncertainty',
        predictions='A=penalized score; B=original empirical-reward return; C=true return; calibration uses B-C',
        total_new_collection_transitions=0, total_new_frozen_evaluation_transitions=2000000,
        reproduction='OPENBLAS_NUM_THREADS=1 python run_count_penalized_planning.py --output results/count_penalized_planning_repeat',
        plotting='python run_count_penalized_planning.py --plot-only')
    manifest_path = args.output/'manifest.json'
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        assert all(prior[k] == manifest[k] for k in ('config', 'input_sha256', 'source_sha256'))
        manifest = prior
    else:
        write_json(manifest_path, manifest)
    collection = load_npz(BASE/'collection.npz')
    np.testing.assert_array_equal(collection['steps'], np.arange(0, 200001, 10000))
    assert collection['visits'].shape == (21, 100, 25, 3)
    selected_path = args.output/'planning.npz'
    selected = load_npz(selected_path) if selected_path.exists() else select_policies(collection)
    np.savez_compressed(selected_path, **selected)
    # The entire new policy set is now fixed and saved. True access starts here.
    eval_started = perf_counter()
    true_path = ROOT/'results/first_world/model.npz'
    true_model = load_npz(true_path)
    baseline = load_npz(BASE/'results.npz')
    data = evaluate_selected(selected, baseline, true_model)
    diagnostic = seed_zero(data, selected, true_model)
    manifest['exact_evaluation_and_diagnostics_seconds'] = perf_counter()-eval_started
    manifest['true_model_sha256'] = sha(true_path)
    np.savez_compressed(args.output/'results.npz', **data)
    write_json(args.output/'seed_0.json', diagnostic)
    sim_started = perf_counter()
    behavior = simulate_new(selected['policies'][-1], load_npz(BASE/'frozen_behavior.npz'))
    manifest['frozen_evaluation_seconds'] = perf_counter()-sim_started
    np.savez_compressed(args.output/'frozen_behavior.npz', **behavior)
    summary = summarize(data, behavior, selected)
    write_json(args.output/'summary.json', summary)
    plot_all(data, behavior, diagnostic, args.output)
    manifest['planning_and_original_reward_prediction_seconds'] = float(selected['planning_seconds'].sum())
    manifest['completion_invocation_seconds'] = perf_counter()-started
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    print('Final decisions:', json.dumps(summary['methods']), flush=True)
    print('Paired differences:', json.dumps(summary['paired_new_minus_baseline']), flush=True)
    print('Prediction:', json.dumps(summary['prediction']), flush=True)
    print('Runtime:', {k: v for k, v in manifest.items() if k.endswith('_seconds')}, flush=True)


if __name__ == '__main__':
    main()
