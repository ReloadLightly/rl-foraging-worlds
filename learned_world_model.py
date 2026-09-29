"""Empirical dynamics and planning. Inputs are observations, never world rules."""

import numpy as np

from q_learning import greedy_probabilities


class EmpiricalModel:
    """Separate sufficient statistics for each seed; no sharing across seeds."""

    def __init__(self, seeds, states=25, actions=3):
        self.successors = np.zeros((seeds, states, actions, states), dtype=np.int32)
        self.visits = np.zeros((seeds, states, actions), dtype=np.int32)
        self.reward_sums = np.zeros((seeds, states, actions))
        self.rows = np.arange(seeds)

    def observe(self, states, actions, rewards, next_states):
        # Each row is a different seed, so advanced-index increments are unique.
        self.successors[self.rows, states, actions, next_states] += 1
        self.visits[self.rows, states, actions] += 1
        self.reward_sums[self.rows, states, actions] += rewards


def estimate(successors, visits, reward_sums):
    """Frequency estimates; unknown rows assume zero reward and a self-loop."""
    unknown = visits == 0
    denominator = np.maximum(visits, 1)
    transition = successors / denominator[..., None]
    transition += unknown[..., None] * np.eye(visits.shape[-2])[:, None, :]
    rewards = reward_sums / denominator
    return transition, rewards, unknown


def evaluate_policy(policy, transition, rewards, gamma=.99):
    """Batched version of foraging.policy_evaluation; model passed explicitly."""
    p_pi = np.einsum('...sa,...san->...sn', policy, transition)
    r_pi = (policy * rewards).sum(axis=-1)
    return np.linalg.solve(np.eye(policy.shape[-2])-gamma*p_pi, r_pi[..., None])[..., 0]


def plan(transition, rewards, gamma=.99):
    """Chapter 4 preview: policy iteration, with the established uniform ties.

    Every model starts from a uniform policy. No warm start, oracle, Q table,
    true dynamics, or data from another seed enter the planning calculation.
    """
    policy = np.full_like(rewards, 1/rewards.shape[-1])
    iterations = np.zeros(rewards.shape[:-2], dtype=int)
    active = np.ones_like(iterations, dtype=bool)
    for _ in range(1000):
        values = evaluate_policy(policy, transition, rewards, gamma)
        q = rewards + gamma*np.einsum('...san,...n->...sa', transition, values)
        improved = greedy_probabilities(q)
        iterations[active] += 1
        stable = np.all(improved == policy, axis=(-2, -1))
        active &= ~stable
        if stable.all():
            residual = np.max(np.abs(values-q.max(axis=-1)), axis=-1)
            assert np.all(residual <= 2e-10), 'Policy iteration residual too large'
            return policy, values, iterations, residual
        policy = improved
    raise RuntimeError('Policy iteration did not stabilize')
