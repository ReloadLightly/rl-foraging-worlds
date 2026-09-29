"""Learn from 100 seeds of experience; evaluate frozen policies with the model.

The world is unchanged. Model arrays never enter QLearner or action selection.
Checkpoints are fixed in advance, and the final checkpoint is the outcome.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from foraging import WORLD, state_index, step
from q_learning import QLearner, greedy_probabilities


CONFIG = dict(training_seeds=100, transitions=200_000, rollout_steps=1000,
              checkpoint_every=10_000, gamma_train=[.99, 0.0], gamma_eval=.99,
              alpha=.1, epsilon=.1, q_initial=0.0, evaluation_epsilon=0.0)
METRICS = ['reward_per_step', 'stock_A', 'stock_B', 'any_depleted_fraction',
           'rest_fraction', 'unsuccessful_harvest_fraction']
MASTER = 20260929


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def seed_table():
    # A seed's stream is stable if methods are reordered or other seeds added.
    def seed(s, role):
        return int(np.random.SeedSequence([MASTER, 2, s, role]).generate_state(
            1, dtype=np.uint64)[0])
    return dict(regeneration=[seed(s, 0) for s in range(100)],
                learners={str(g): [seed(s, role) for s in range(100)]
                          for g, role in [(.99, 199), (0.0, 100)]})


def describe(x):
    x = np.asarray(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    sem = sd / np.sqrt(len(x))
    return dict(mean=mean, seed_sd=sd, sem=sem, ci95=[mean-1.96*sem, mean+1.96*sem],
                minimum=float(x.min()), maximum=float(x.max()),
                quantiles_10_50_90=np.quantile(x, [.1, .5, .9]).tolist())


def fraction_interval(success):
    # Wilson interval for a fraction of independent training seeds.
    n, z = len(success), 1.96
    p = float(np.mean(success))
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return dict(count=int(np.sum(success)), total=n, fraction=p,
                wilson_ci95=[center-half, center+half])


def evaluate(q, model, epsilon=0.0):
    """Exact values of FROZEN policies. No simulation or further Q updates.

    The leading axes are method and training seed. Batched linear solves are
    the same Bellman policy evaluation used in the first experiment.
    """
    policy = (1-epsilon)*greedy_probabilities(q) + epsilon/3
    p_pi = np.einsum('...sa,san->...sn', policy, model['transition'])
    r_pi = (policy * model['immediate_rewards']).sum(axis=-1)
    value = np.linalg.solve(np.eye(25)-CONFIG['gamma_eval']*p_pi, r_pi[..., None])[..., 0]
    return policy, value


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def save_checkpoint(output, completed, learners, curves, training_seconds, rngs, model):
    q = np.stack([agent.q for agent in learners])
    visits = np.stack([agent.visits for agent in learners])
    _, values = evaluate(q, model)
    _, behavior_values = evaluate(q, model, epsilon=CONFIG['epsilon'])
    states = json.dumps([rng.bit_generator.state for group in rngs for rng in group])
    path = output / f'checkpoint_{completed:06d}.npz'
    # Atomic replacement prevents a partial archive being mistaken for a checkpoint.
    with path.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream, completed=completed, q=q, visits=visits,
                            values=values, behavior_values=behavior_values,
                            rollout_metrics=curves[:, :completed//CONFIG['rollout_steps']],
                            training_seconds=training_seconds, rng_states=np.array(states))
    path.with_suffix('.tmp').replace(path)
    print(f'{completed:>7,} transitions/seed | frozen mean V: '
          f'{values[0, :, 24].mean():.4f}, {values[1, :, 24].mean():.4f} | '
          f'training {training_seconds:.2f} s', flush=True)


def train(output, seeds, model):
    n, length = CONFIG['training_seeds'], CONFIG['rollout_steps']
    learners = [QLearner(n, g, CONFIG['alpha'], CONFIG['epsilon']) for g in CONFIG['gamma_train']]
    rngs = [[np.random.default_rng(s) for s in seeds['regeneration']]] + [
        [np.random.default_rng(s) for s in seeds['learners'][str(g)]] for g in CONFIG['gamma_train']]
    curves = np.zeros((2, CONFIG['transitions']//length, n, len(METRICS)))
    checkpoints = sorted(output.glob('checkpoint_*.npz'))
    completed, training_seconds = 0, 0.0
    if checkpoints:
        previous = load_npz(checkpoints[-1])
        completed, training_seconds = int(previous['completed']), float(previous['training_seconds'])
        for m, learner in enumerate(learners):
            learner.q[:] = previous['q'][m]
            learner.visits[:] = previous['visits'][m]
        curves[:, :completed//length] = previous['rollout_metrics']
        for rng, state in zip([r for group in rngs for r in group], json.loads(str(previous['rng_states']))):
            rng.bit_generator.state = state
        print(f'Resuming saved checkpoint at {completed:,} transitions per seed', flush=True)
    else:
        save_checkpoint(output, 0, learners, curves, 0.0, rngs, model)
    for rollout in range(completed//length, CONFIG['transitions']//length):
        started = perf_counter()
        # Independent seed streams, common regeneration opportunities between
        # methods. Draws are generated a rollout at a time, not a huge data cube.
        environment_draws = np.stack([r.random((length, 2)) for r in rngs[0]], axis=1)
        action_draws = [np.stack([r.random((length, 2)) for r in group], axis=1)
                        for group in rngs[1:]]
        stocks = np.tile(WORLD.initial, (2, n, 1))
        for t in range(length):
            for m, learner in enumerate(learners):
                states = state_index(stocks[m])
                actions = learner.act(states, action_draws[m][t])
                successor, reward = step(stocks[m], actions, environment_draws[t])
                learner.observe(states, actions, reward, state_index(successor))
                curves[m, rollout] += np.column_stack((reward, stocks[m],
                    (stocks[m] == 0).any(axis=1), actions == 2, (actions < 2) & (reward == 0)))
                stocks[m] = successor
        # The final successor has ALREADY been used in the update. Reset only
        # the collection state next rollout, retaining Q and visit counts.
        curves[:, rollout] /= length
        training_seconds += perf_counter()-started
        completed = (rollout+1)*length
        if completed % CONFIG['checkpoint_every'] == 0:
            assert all(np.all(agent.visits.sum(axis=(1, 2)) == completed) for agent in learners)
            save_checkpoint(output, completed, learners, curves, training_seconds, rngs, model)
    return training_seconds


def summarize(output, model):
    checkpoints = [load_npz(p) for p in sorted(output.glob('checkpoint_*.npz'))]
    final = checkpoints[-1]
    assert int(final['completed']) == CONFIG['transitions']
    policy, values = evaluate(final['q'], model)
    oracle = float(model['values'][-1, 2, 24])
    summary = dict(oracle_value=oracle, myopic_value=float(model['values'][-1, 1, 24]),
                   random_value=float(model['values'][-1, 0, 24]), threshold_90_percent=.9*oracle,
                   frozen_policy='Final Q tables; epsilon=0, no learning; uniform numerical greedy ties',
                   uncertainty='Mean +/- 1.96 SEM over 100 training seeds; fractions use Wilson intervals',
                   methods={}, paired_difference_gamma_099_minus_0=describe(values[0, :, 24]-values[1, :, 24]))
    for m, g in enumerate(CONFIG['gamma_train']):
        counts = final['visits'][m]
        summary['methods'][str(g)] = dict(
            final_value=describe(values[m, :, 24]),
            fraction_of_oracle=describe(values[m, :, 24]/oracle),
            reaching_90_percent=fraction_interval(values[m, :, 24] >= .9*oracle),
            frozen_epsilon_01_value=describe(final['behavior_values'][m, :, 24]),
            unvisited_state_action_pairs=describe((counts == 0).sum(axis=(1, 2))),
            minimum_state_action_visits=describe(counts.min(axis=(1, 2))),
            final_10_rollouts_training={name: describe(final['rollout_metrics'][m, -10:, :, k].mean(axis=0))
                                        for k, name in enumerate(METRICS)})
    data = dict(checkpoint_steps=np.array([c['completed'] for c in checkpoints]),
                checkpoint_values=np.stack([c['values'] for c in checkpoints]),
                checkpoint_behavior_values=np.stack([c['behavior_values'] for c in checkpoints]),
                final_q=final['q'], final_visits=final['visits'], final_policies=policy,
                final_values=values, rollout_metrics=final['rollout_metrics'],
                metric_names=np.array(METRICS), gamma_train=np.array(CONFIG['gamma_train']),
                oracle_policy=model['policies'][-1, 2], oracle_values=model['values'][-1, 2],
                reference_initial_values=model['values'][-1, :, 24], states=model['states'])
    np.savez_compressed(output/'results.npz', **data)
    write_json(output/'summary.json', summary)
    return data, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/q_learning'))
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(Path('.mplconfig').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    from q_learning_figures import plot_all
    if args.plot_only or (args.output/'results.npz').exists():
        plot_all(load_npz(args.output/'results.npz'), args.output)
        print('Regenerated figures from completed results; no training rerun.')
        return
    wall_started = perf_counter()
    model_path = Path(__file__).parent/'results/first_world/model.npz'
    model = load_npz(model_path)
    seeds = seed_table()
    source_names = ['foraging.py', 'q_learning.py', 'run_q_learning.py', 'check_q_learning.py']
    manifest_path = args.output/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest['config'] == CONFIG and manifest['seeds'] == seeds
        assert manifest['model_sha256'] == hashlib.sha256(model_path.read_bytes()).hexdigest()
        assert all(manifest['source_sha256'][n] == hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                   for n in source_names), 'Training code changed since checkpoint; do not silently mix runs.'
    else:
        manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), config=CONFIG, world=asdict(WORLD),
                        master_seed=MASTER, experiment_id=2, seeds=seeds,
                        python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
                        platform=platform.platform(), model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                        source_sha256={n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                       for n in source_names},
                        checkpoints='0 and every 10000 transitions, chosen before training',
                        pairing='Same per-seed regeneration uniforms; separate stable per-method action streams',
                        evaluation='Exact frozen greedy policy at gamma=.99; final checkpoint only for conclusions',
                        training_runtime='Rollout collection and updates; excludes evaluation, checkpoint I/O, and plotting')
        write_json(manifest_path, manifest)
    manifest['training_seconds'] = train(args.output, seeds, model)
    data, summary = summarize(args.output, model)
    plot_all(data, args.output)
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['completion_invocation_seconds'] = perf_counter()-wall_started
    manifest['total_training_transitions'] = 2*CONFIG['training_seeds']*CONFIG['transitions']
    write_json(manifest_path, manifest)
    for method, result in summary['methods'].items():
        print(f'gamma_train={method}: {result["final_value"]}; >=90% oracle: '
              f'{result["reaching_90_percent"]}', flush=True)
    print(f'Training: {manifest["training_seconds"]:.2f} s; completion invocation: '
          f'{manifest["completion_invocation_seconds"]:.2f} s', flush=True)


if __name__ == '__main__':
    main()
