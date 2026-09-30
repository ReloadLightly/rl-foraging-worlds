"""Experiment 7's opt-in regime reversal; original foraging defaults stay intact."""

import numpy as np

from foraging import WORLD, harvest


def changed_probability(post):
    post = np.asarray(post)
    chance = np.asarray(WORLD.growth)*(1.0-.9*post/WORLD.capacity)
    return np.where(post < WORLD.capacity, chance, 0.)


def changed_step(stocks, actions, uniforms):
    """Harvest first, then regenerate both patches under the reversed rule."""
    post, reward = harvest(stocks, actions)
    return post+(uniforms < changed_probability(post)), reward
