"""Tabular Q-learning from observed transitions, with no supplied world model."""

import numpy as np

from foraging import TIE_ATOL


def greedy_probabilities(q):
    """Uniform random greedy ties; accepts (..., actions) arrays."""
    ties = q.max(axis=-1, keepdims=True) - q <= TIE_ATOL
    return ties / ties.sum(axis=-1, keepdims=True)


class QLearner:
    """Independent tables vectorized over training seeds, not shared experience."""

    def __init__(self, seeds, gamma, alpha=0.1, epsilon=0.1):
        self.q = np.zeros((seeds, 25, 3))
        self.visits = np.zeros_like(self.q, dtype=np.int64)
        self.rows = np.arange(seeds)
        self.gamma, self.alpha, self.epsilon = gamma, alpha, epsilon

    def act(self, states, uniforms):
        # One draw selects an action; the independent second draw selects
        # exploration. Exploration includes every action, including greedy ones.
        cdf = greedy_probabilities(self.q[self.rows, states]).cumsum(axis=-1)
        cdf[:, -1] = 1.0
        greedy = (uniforms[:, :1] >= cdf).sum(axis=-1)
        random_actions = (3 * uniforms[:, 0]).astype(int)
        return np.where(uniforms[:, 1] < self.epsilon, random_actions, greedy)

    def observe(self, states, actions, rewards, next_states):
        # The max is taken BEFORE modifying Q. Even the final transition of a
        # rollout bootstraps from its actual successor: there is no terminal flag.
        target = rewards + self.gamma * self.q[self.rows, next_states].max(axis=-1)
        old = self.q[self.rows, states, actions]
        self.q[self.rows, states, actions] += self.alpha * (target - old)
        self.visits[self.rows, states, actions] += 1
