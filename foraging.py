"""A finite, continuing food-patch MDP and supplied-model planning.

Event order is always: observe stocks -> act/receive reward -> regenerate
BOTH patches independently. A simulation cutoff is not a terminal state.
This is our original stylized environment, inspired by Chapter 3.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class World:
    capacity: int = 4
    growth: tuple = (0.3, 0.6)
    harvest_reward: tuple = (1.0, 1.5)
    initial: tuple = (4, 4)


WORLD = World()
ACTIONS = ("harvest A", "harvest B", "rest")
TIE_ATOL = 1e-10  # Absolute tolerance; no relative tolerance for action ties.


def state_index(stocks, world=WORLD):
    return np.asarray(stocks)[..., 0] * (world.capacity + 1) + np.asarray(stocks)[..., 1]


def harvest(stocks, actions, world=WORLD):
    """Vectorized trajectories; return post-harvest stocks and immediate rewards."""
    post = np.array(stocks, dtype=np.int64, copy=True)
    rewards = np.zeros(len(post))
    for patch, payoff in enumerate(world.harvest_reward):
        success = (actions == patch) & (post[:, patch] > 0)
        post[success, patch] -= 1
        rewards[success] = payoff
    return post, rewards


def regeneration_probability(post, world=WORLD):
    post = np.asarray(post)
    chance = np.asarray(world.growth) * (0.1 + 0.9 * post / world.capacity)
    return np.where(post < world.capacity, chance, 0.0)


def step(stocks, actions, regeneration_uniforms, world=WORLD):
    """One continuing step. Uniforms are exogenous, one per trajectory/patch.

    The just-harvested patch can regenerate too. No done/terminal flag exists.
    """
    post, rewards = harvest(stocks, actions, world)
    regrowth = regeneration_uniforms < regeneration_probability(post, world)
    return post + regrowth, rewards


def exact_model(world=WORLD):
    """Enumerate P[state, action, next_state] and r[state, action]."""
    states = np.array(list(product(range(world.capacity + 1), repeat=2)))
    transition = np.zeros((len(states), len(ACTIONS), len(states)))
    rewards = np.zeros((len(states), len(ACTIONS)))
    for s, stocks in enumerate(states):
        for a in range(len(ACTIONS)):
            post, reward = harvest(stocks[None, :], np.array([a]), world)
            probabilities = regeneration_probability(post[0], world)
            rewards[s, a] = reward[0]
            for grows in product((0, 1), repeat=2):
                probability = np.prod(np.where(grows, probabilities, 1 - probabilities))
                if probability > 0:  # Impossible growth at capacity is excluded.
                    successor = post[0] + grows
                    transition[s, a, state_index(successor, world)] += probability
    return states, transition, rewards


def greedy_policy(action_values, tolerance=TIE_ATOL):
    """Uniformly mix all actions within an absolute tolerance of the maximum."""
    ties = action_values.max(axis=1, keepdims=True) - action_values <= tolerance
    return ties / ties.sum(axis=1, keepdims=True)


def policy_model(policy, transition, rewards):
    p_pi = np.einsum("sa,san->sn", policy, transition)
    r_pi = (policy * rewards).sum(axis=1)
    return p_pi, r_pi


def policy_evaluation(policy, transition, rewards, gamma):
    p_pi, r_pi = policy_model(policy, transition, rewards)
    # Bellman expectation equation: (I - gamma P_pi) V_pi = r_pi.
    return np.linalg.solve(np.eye(len(r_pi)) - gamma * p_pi, r_pi)


def action_values(values, transition, rewards, gamma):
    """Immediate reward plus discounted value of the resulting state."""
    return rewards + gamma * (transition @ values)


def value_iteration(transition, rewards, gamma, tolerance=1e-10):
    """Chapter 4 preview: explore Chapter 3's Bellman optimality equation.

    The stop rule bounds the value error by tolerance via contraction. We
    then evaluate the extracted mixed policy with the exact linear solve.
    """
    values = np.zeros(len(rewards))
    for iteration in range(1, 100000):
        updated = action_values(values, transition, rewards, gamma).max(axis=1)
        change = np.max(np.abs(updated - values))
        values = updated
        if change <= tolerance * (1 - gamma):
            break
    else:
        raise RuntimeError("Value iteration did not converge")
    policy = greedy_policy(action_values(values, transition, rewards, gamma))
    values = policy_evaluation(policy, transition, rewards, gamma)
    q = action_values(values, transition, rewards, gamma)
    residual = float(np.max(np.abs(values - q.max(axis=1))))
    return policy, values, q, iteration, residual


def finite_window_value(policy, transition, rewards, gamma, steps):
    """Expected truncated return of a FIXED policy, not a terminal-state model."""
    p_pi, r_pi = policy_model(policy, transition, rewards)
    value = np.zeros(len(rewards))
    for _ in range(steps):
        value = r_pi + gamma * p_pi @ value
    return value


def sample_actions(policy, stocks, rng, world=WORLD):
    cdf = policy[state_index(stocks, world)].cumsum(axis=1)
    cdf[:, -1] = 1.0  # Repair cumulative floating-point roundoff only.
    return (rng.random(len(stocks))[:, None] >= cdf).sum(axis=1)
