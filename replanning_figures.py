"""Experiment 6 figures, made entirely from saved results."""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from figures import ACTION_COLORS, ACTION_LETTERS, save, stock_axes, style
from q_learning_figures import band

COLORS = ['#B46548', '#207F75', '#7960A6']
LABELS = ['Q collector · saved A', 'Adaptive model · saved B', 'Fixed initial model policy']


def plot_values(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    steps = data['additional_steps']/1000
    values = data['checkpoint_values'][..., 24]
    for c in range(3):
        band(axes[0], steps, values[:, c], COLORS[c], LABELS[c])
    axes[0].axhline(data['oracle_values'][24], color='#3C4852', ls='--', label='Saved oracle')
    axes[0].set(title='Same unpenalized planner on each dataset', ylabel='Exact true Vπ(4,4), γ = 0.99')
    axes[0].legend(frameon=False, fontsize=8)
    delta = values[:, 1]-values[:, 2]
    band(axes[1], steps, delta, COLORS[1], 'Adaptive minus fixed')
    axes[1].axhline(0, color='#3C4852', ls=':')
    axes[1].scatter([50], [delta[-1].mean()], s=65, facecolors='white', edgecolors=COLORS[1], zorder=4)
    axes[1].set(title='Paired effect of updating collection behavior', ylabel='Adaptive minus fixed · policy value')
    for ax in axes:
        ax.set(xlabel='Additional transitions per seed (thousands)', xlim=(0, 50))
        ax.grid(alpha=.15)
    fig.suptitle('Does replanning improve the experience an agent collects?', fontsize=16)
    fig.text(.5, .01, 'Same 200,000-transition starting histories and paired collection streams. Fixed endpoint: +50,000 (250,000 total).\n'
             'Bands: pointwise 95% intervals across 100 seed pairs; earlier checkpoints are descriptive.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .94))
    save(fig, output, 'policy_values.png')


def plot_outcomes(data, output):
    actual = data['checkpoint_values'][-1, :, :, 24]
    predicted = data['checkpoint_predictions'][-1, :, :, 24]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    a, b = actual[2], actual[1]
    low, high = min(a.min(), b.min())-1, max(a.max(), b.max())+1
    axes[0].scatter(a, b, color=COLORS[1], alpha=.6, s=26)
    axes[0].plot([low, high], [low, high], color='#3C4852', ls=':')
    axes[0].set(title='Final paired policies', xlabel='Fixed collector · true value',
                ylabel='Adaptive collector · true value', xlim=(low, high), ylim=(low, high), aspect='equal')
    delta = b-a
    mean, half = delta.mean(), 1.96*delta.std(ddof=1)/10
    axes[1].hist(delta, bins=np.linspace(min(0, delta.min())-.1, max(0, delta.max())+.1, 22),
                 color=COLORS[1], alpha=.8, edgecolor='white')
    axes[1].axvline(0, color='#3C4852', ls=':')
    axes[1].axvline(mean, color='#213340')
    axes[1].axvspan(mean-half, mean+half, color='#213340', alpha=.15)
    axes[1].set(title=f'Paired gain {mean:.3f} [{mean-half:.3f}, {mean+half:.3f}]',
                xlabel='Adaptive minus fixed · true value', ylabel='Training seed pairs')
    low, high = min(actual.min(), predicted.min())-2, max(actual.max(), predicted.max())+2
    for c in range(3):
        axes[2].scatter(actual[c], predicted[c], s=22, color=COLORS[c], alpha=.55, label=LABELS[c])
    axes[2].plot([low, high], [low, high], color='#3C4852', ls=':')
    axes[2].set(title='Final model prediction versus reality', xlabel='Actual true-environment value',
                ylabel='Original empirical-reward prediction', xlim=(low, high), ylim=(low, high), aspect='equal')
    axes[2].legend(frameon=False, fontsize=7, loc='upper left')
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle('All final outcomes retained; the primary comparison is adaptive minus fixed', fontsize=15)
    fig.text(.5, .01, 'Each point or histogram observation is one training seed. A 95% interval containing zero does not establish equivalence.\n'
             'Prediction uses each final policy’s own empirical transitions and original rewards; no penalized scores enter this experiment.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .94))
    save(fig, output, 'paired_outcomes_and_predictions.png')


def plot_coverage(data, output):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
    initial = data['checkpoint_visits'][0, 0]
    scarce = initial < 10
    steps = data['additional_steps']/1000
    for c in range(3):
        counts = data['checkpoint_visits'][:, c]
        remaining = (scarce[None] & (counts < 10)).sum(axis=(-2, -1))
        observations = ((counts-initial)*scarce).sum(axis=(-2, -1))
        error = data['checkpoint_predictions'][:, c, :, 24]-data['checkpoint_values'][:, c, :, 24]
        band(axes[0], steps, remaining, COLORS[c], LABELS[c])
        band(axes[1], steps, observations, COLORS[c], LABELS[c])
        band(axes[2], steps, np.abs(error), COLORS[c], LABELS[c])
    axes[0].set(title='Initially scarce rows still below 10', ylabel='State-action rows per seed')
    axes[1].set(title='New observations in initially scarce rows', ylabel='Transitions per seed')
    axes[2].set(title='Original-model prediction error', ylabel='Mean |predicted − actual value|')
    axes[0].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set(xlabel='Additional transitions (thousands)', xlim=(0, 50))
        ax.grid(alpha=.15)
    fig.suptitle('The behavior policy determines which model errors get new evidence', fontsize=16)
    fig.text(.5, .01, 'Scarce means N(s,a) < 10 in the shared 200,000-transition history, fixed before collection.\n'
             'The model is updated in both conditions; only adaptive B changes its collection policy. Bands: 95% across seed means.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .94))
    save(fig, output, 'coverage_and_error.png')


def plot_resources(data, frozen, output):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    titles = ['Reward per decision', 'Stock A before acting', 'Stock B before acting', 'Either patch depleted (%)']
    for k, ax in enumerate(axes[0]):
        scale = 100 if k == 3 else 1
        for c in range(3):
            band(ax, np.arange(1, 51), scale*data['rollout_metrics'][c, :, :, k], COLORS[c], LABELS[c])
        ax.set(title='Collection · '+titles[k], xlabel='New 1,000-step rollout', xlim=(1, 50))
    for k, metric in enumerate([0, 2, 3, 4]):
        ax = axes[1, k]
        scale = 100 if k == 3 else 1
        for c in range(3):
            band(ax, frozen['bin_centers'], scale*frozen['seed_curves'][c, :, :, metric].T, COLORS[c], LABELS[c])
        ax.set(title='Final extracted · '+titles[k], xlabel='Frozen evaluation decision', xlim=(0, 1000))
    for ax in axes.flat:
        ax.grid(alpha=.15)
    for row in range(2):
        for col in (1, 2):
            axes[row, col].set_ylim(0, 4.1)
        axes[row, 3].set_ylim(bottom=-.2)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .95), ncol=3, frameon=False)
    fig.suptitle('Fixed collection behavior can still produce an improved final policy', fontsize=16, y=.99)
    fig.text(.5, .01, 'Top: collection with ε=0.1 and model updates; fixed collection policy never changes.\n'
             'Bottom: newly extracted final policies, no learning/exploration; 20 paired paths averaged within each seed. Bands: 95% across seeds.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .91))
    save(fig, output, 'collection_and_frozen_resources.png')


def plot_seed_zero(data, diagnostic, output):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    steps = np.array(diagnostic['additional_steps'])/1000
    for c in range(3):
        axes[0, 0].plot(steps, np.array(diagnostic['visit_counts'])[:, c], marker='o', ms=3, color=COLORS[c], label=LABELS[c])
        axes[0, 1].plot(steps, np.array(diagnostic['empirical_self_loop'])[:, c], marker='o', ms=3, color=COLORS[c])
        axes[0, 2].plot(steps, np.array(diagnostic['collector_action_probabilities'])[:, c, 1], marker='o', ms=3, color=COLORS[c])
    axes[0, 0].set(title='Harvest-B observations at (3,4)', ylabel='Cumulative visits (log scale)', yscale='log')
    axes[0, 1].axhline(diagnostic['true_self_loop_probability'], color='#3C4852', ls='--', label='True probability · diagnostic only')
    axes[0, 1].set(title='Estimated return to the same state', ylabel='Empirical self-loop probability', ylim=(-.03, 1.03))
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 2].set(title='Behavior probability of harvest B', ylabel='Includes ε=0.1 exploration', ylim=(-.03, 1.03))
    for ax in axes[0]:
        ax.set(xlabel='Additional transitions (thousands)', xlim=(0, 50))
        ax.grid(alpha=.15)
    maps = [data['collector_greedy'][0, 2, 0], data['checkpoint_policies'][-1, 2, 0], data['checkpoint_policies'][-1, 1, 0]]
    titles = ['Fixed behavior · initial greedy policy', 'Fixed dataset · final extracted policy', 'Adaptive dataset · final extracted policy']
    for ax, policy, title in zip(axes[1], maps, titles):
        for (x, y), row in zip(data['states'], policy):
            choices = np.flatnonzero(row)
            for j, a in enumerate(choices):
                ax.add_patch(Rectangle((x-.5+j/len(choices), y-.5), 1/len(choices), 1,
                                      facecolor=ACTION_COLORS[a], alpha=.3, edgecolor='white'))
            ax.text(x, y, '/'.join(ACTION_LETTERS[a] for a in choices), ha='center', va='center', fontsize=9)
        ax.add_patch(Rectangle((2.5, 3.5), 1, 1, fill=False, edgecolor='#7F2424', lw=2))
        stock_axes(ax)
        ax.set_title(title, fontsize=10)
    fig.suptitle('Preselected seed 0: a corrected belief does not require changing this action', fontsize=15, y=.99)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .955), ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .01, 'Seed 0, state (3,4), harvest B were selected before the experiment. Curves show saved checkpoint observations.\n'
             'A/B/R = harvest A/harvest B/rest. Left map controls fixed collection (plus ε); middle and right maps are evaluated without ε.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .91))
    save(fig, output, 'seed_0_beliefs_and_policies.png')


def plot_all(data, frozen, diagnostic, output):
    style()
    plot_values(data, output)
    plot_outcomes(data, output)
    plot_coverage(data, output)
    plot_resources(data, frozen, output)
    plot_seed_zero(data, diagnostic, output)
