"""Plan exactly, then compare three fixed policies on paired regeneration draws."""

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

from foraging import (ACTIONS, TIE_ATOL, WORLD, exact_model, finite_window_value,
                     greedy_policy, policy_evaluation, sample_actions, state_index,
                     step, value_iteration)


GAMMAS = [0.0, 0.5, 0.9, 0.99]
POLICIES = ["uniform random", "myopic", "planned gamma=0.99"]
CONFIG = dict(trajectories=2000, steps=1000, simulation_gamma=0.99, bin_width=20,
              trace_trajectory=0, trace_plot_window=[0, 120])
OUTCOMES = ["discounted_return", "cumulative_reward", "mean_stock_A", "mean_stock_B",
            "depleted_A_fraction", "depleted_B_fraction", "any_depleted_fraction",
            "both_depleted_fraction", "rest_fraction", "unsuccessful_harvest_fraction",
            "successful_harvest_fraction"]
CURVES = ["reward", "cumulative_reward", "stock_A", "stock_B",
          "any_depleted_fraction", "rest_fraction", "unsuccessful_harvest_fraction"]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def mean_sem(x, axis=0):
    return x.mean(axis=axis), x.std(axis=axis, ddof=1) / np.sqrt(x.shape[axis])


def describe(x):
    mean, sem = mean_sem(x)
    return dict(mean=float(mean), sem=float(sem), ci95=[float(mean-1.96*sem), float(mean+1.96*sem)],
                trajectory_sd=float(x.std(ddof=1)), quantiles_10_50_90=np.quantile(x, [.1, .5, .9]).tolist())


def make_seeds(master=20260929):
    def seed(role):
        return int(np.random.SeedSequence([master, 1, role]).generate_state(1, dtype=np.uint64)[0])
    return dict(regeneration=seed(0), policies={name: seed(role) for name, role in zip(POLICIES, [100, 101, 102])})


def plan():
    states, p, r = exact_model()
    random, myopic = np.full((25, 3), 1/3), greedy_policy(r)
    policy_rows, value_rows, q_rows, iterations, residuals = [], [], [], [], []
    summary = {}
    for gamma in GAMMAS:
        optimal, value, q, count, residual = value_iteration(p, r, gamma)
        policies = np.stack((random, myopic, optimal))
        values = np.stack((policy_evaluation(random, p, r, gamma),
                           policy_evaluation(myopic, p, r, gamma), value))
        advantage = value - values[1]
        rest_margin = q[:, 2] - q[:, :2].max(axis=1)
        strict_rest = rest_margin > TIE_ATOL
        changed = np.any(np.abs(optimal-myopic) > TIE_ATOL, axis=1)
        policy_rows.append(policies)
        value_rows.append(values)
        q_rows.append(q)
        iterations.append(count)
        residuals.append(residual)
        summary[str(gamma)] = dict(
            initial_values={name: float(v[state_index(WORLD.initial)])
                            for name, v in zip(["uniform random", "myopic", "optimal at this gamma"], values)},
            advantage_by_state=[dict(state=s.tolist(), advantage=float(d)) for s, d in zip(states, advantage)],
            strict_rest_states=states[strict_rest].tolist(),
            strict_rest_margins=rest_margin[strict_rest].tolist(),
            changed_from_myopic_states=states[changed].tolist(),
            optimal_action_sets=[[ACTIONS[a] for a in np.flatnonzero(row)] for row in optimal],
            iterations=count, optimality_residual=residual)
    policies = np.array(policy_rows)
    changes = {f"{GAMMAS[i-1]}_to_{GAMMAS[i]}": states[np.any(
        np.abs(policies[i, 2] - policies[i-1, 2]) > TIE_ATOL, axis=1)].tolist() for i in range(1, 4)}
    arrays = dict(states=states, transition=p, immediate_rewards=r, gammas=np.array(GAMMAS),
                  policy_names=np.array(["uniform random", "myopic", "optimal at this gamma"]),
                  action_names=np.array(ACTIONS), policies=policies, values=np.array(value_rows),
                  optimal_action_values=np.array(q_rows), iterations=np.array(iterations),
                  optimality_residuals=np.array(residuals))
    return arrays, dict(by_gamma=summary, changed_between_discounts=changes)


def simulate(policies, seeds, config=CONFIG):
    n, steps, width = (config[k] for k in ("trajectories", "steps", "bin_width"))
    gamma = config["simulation_gamma"]
    stocks = np.tile(WORLD.initial, (3, n, 1))
    cumulative = np.zeros((3, n))
    totals = np.zeros((3, n, len(OUTCOMES)))
    rng = np.random.default_rng(seeds["regeneration"])
    action_rngs = [np.random.default_rng(seeds["policies"][name]) for name in POLICIES]
    bins = (steps + width - 1) // width
    bin_mean, bin_sem = np.zeros((3, bins, len(CURVES))), np.zeros((3, bins, len(CURVES)))
    bin_totals = np.zeros((3, n, len(CURVES)))
    trace_stocks = np.zeros((3, steps+1, 2), dtype=np.int8)
    trace_actions = np.zeros((3, steps), dtype=np.int8)
    trace_rewards = np.zeros((3, steps))
    trace_uniforms = np.zeros((steps, 2))
    trace_stocks[:, 0] = stocks[:, 0]
    random_hash = hashlib.sha256()
    for t in range(steps):
        uniforms = rng.random((n, 2))  # Shared exogenous opportunities, not shared stocks.
        trace_uniforms[t] = uniforms[0]
        random_hash.update(uniforms.astype("<f8", copy=False).tobytes())
        for m, policy in enumerate(policies):
            actions = sample_actions(policy, stocks[m], action_rngs[m])
            next_stocks, rewards = step(stocks[m], actions, uniforms)
            depleted = stocks[m] == 0  # Measured before action, at observable decision states.
            rest = actions == 2
            unsuccessful = (actions < 2) & (rewards == 0)
            success = rewards > 0
            cumulative[m] += rewards
            totals[m] += np.column_stack((gamma**t * rewards, rewards, stocks[m], depleted,
                                         depleted.any(axis=1), depleted.all(axis=1), rest, unsuccessful, success))
            values = np.column_stack((rewards, cumulative[m], stocks[m], depleted.any(axis=1), rest, unsuccessful))
            bin_totals[m] += values
            if (t+1) % width == 0 or t == steps-1:
                binned = bin_totals[m] / (t % width + 1)
                binned[:, 1] = cumulative[m]  # Cumulative reward uses the bin endpoint.
                bin_mean[m, t//width], bin_sem[m, t//width] = mean_sem(binned)
                bin_totals[m] = 0
            trace_stocks[m, t+1] = next_stocks[0]
            trace_actions[m, t], trace_rewards[m, t] = actions[0], rewards[0]
            stocks[m] = next_stocks
        if (t+1) % 250 == 0:
            print(f"  Simulated {t+1:,}/{steps:,} steps for all three policies", flush=True)
    totals[:, :, 2:] /= steps
    assert np.isfinite(totals).all()
    np.testing.assert_allclose(totals[:, :, 8:11].sum(axis=2), 1)
    starts = np.arange(0, steps, width)
    ends = np.minimum(starts+width, steps)-1
    return dict(policy_names=np.array(POLICIES), outcome_names=np.array(OUTCOMES),
                task_outcomes=totals, final_stocks=stocks, curve_names=np.array(CURVES),
                bin_mean=bin_mean, bin_sem=bin_sem, bin_centers=(starts+ends)/2, bin_ends=ends,
                trace_stocks=trace_stocks, trace_actions=trace_actions, trace_rewards=trace_rewards,
                trace_regeneration_uniforms=trace_uniforms, regeneration_sha256=np.array(random_hash.hexdigest()))


def simulation_summary(data, model):
    values = data["task_outcomes"]
    gamma, steps = CONFIG["simulation_gamma"], CONFIG["steps"]
    policies = model["policies"][-1]
    results = dict(policies={}, paired_differences={}, discount_tail_bound=1.5*gamma**steps/(1-gamma))
    for m, name in enumerate(POLICIES):
        finite = finite_window_value(policies[m], model["transition"], model["immediate_rewards"], gamma, steps)
        result = {key: describe(values[m, :, k]) for k, key in enumerate(OUTCOMES)}
        result["mean_reward"] = describe(values[m, :, 1]/steps)
        # Also show the conditional failure rate, keeping its denominator explicit.
        result["unsuccessful_given_harvest"] = describe(values[m, :, 9]/(1-values[m, :, 8]))
        initial = state_index(WORLD.initial)
        result["exact_finite_window_return"] = float(finite[initial])
        result["exact_infinite_horizon_return"] = float(model["values"][-1, m, initial])
        result["exact_expected_omitted_tail"] = result["exact_infinite_horizon_return"] - float(finite[initial])
        result["simulation_error_in_SEMs"] = (result["discounted_return"]["mean"] - float(finite[initial])) / result["discounted_return"]["sem"]
        results["policies"][name] = result
    for a, b in [(1, 0), (2, 0), (2, 1)]:
        results["paired_differences"][f"{POLICIES[a]} minus {POLICIES[b]}"] = {
            key: describe(values[a, :, k]-values[b, :, k]) for k, key in enumerate(OUTCOMES)}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/first_world"))
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    from figures import plot_all
    if args.plot_only:
        with np.load(args.output/"model.npz", allow_pickle=False) as archive:
            model = dict(archive)
        with np.load(args.output/"simulation.npz", allow_pickle=False) as archive:
            simulation = dict(archive)
        plot_all(model, simulation, args.output)
        return
    started = perf_counter()
    seeds = make_seeds()
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), world=asdict(WORLD),
                    gammas=GAMMAS, config=CONFIG, master_seed=20260929, seeds=seeds,
                    python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
                    platform=platform.platform(), tie_absolute_tolerance=TIE_ATOL, tie_relative_tolerance=0,
                    tie_rule="Uniform mixture over all numerical maximizers; includes empty-harvest/rest equivalences",
                    event_order="Observe stocks, choose action, harvest and reward, regenerate both patches independently",
                    horizon="Continuing MDP; 1000-step observation window is not an environmental terminal state",
                    uncertainty="Trajectory mean +/- 1.96 SEM; paired trajectory differences; pointwise, unadjusted",
                    depletion="Pre-action stock == 0; rest/failure fractions use all decisions as denominator",
                    trace_selection="Trajectory 0, first 120 steps, selected before inspecting outcomes",
                    source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in ("foraging.py", "run_experiment.py", "figures.py", "checks.py")})
    planning_started = perf_counter()
    model, planning = plan()
    manifest["planning_seconds"] = perf_counter()-planning_started
    np.savez_compressed(args.output/"model.npz", **model)
    write_json(args.output/"planning_summary.json", planning)
    for gamma, row in planning["by_gamma"].items():
        print(f"gamma={gamma}: {row['initial_values']}; strict rest states {len(row['strict_rest_states'])}", flush=True)
    sim_started = perf_counter()
    simulation = simulate(model["policies"][-1], seeds)
    manifest["simulation_seconds"] = perf_counter()-sim_started
    np.savez_compressed(args.output/"simulation.npz", **simulation)
    summary = simulation_summary(simulation, model)
    write_json(args.output/"simulation_summary.json", summary)
    plot_all(model, simulation, args.output)
    manifest["total_seconds"] = perf_counter()-started
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output/"manifest.json", manifest)
    for name, metrics in summary["policies"].items():
        print(name, "discounted return", metrics["discounted_return"]["mean"],
              "total harvested reward", metrics["cumulative_reward"]["mean"], flush=True)
    print(f"Saved {args.output}; total {manifest['total_seconds']:.2f} s", flush=True)


if __name__ == "__main__":
    main()
