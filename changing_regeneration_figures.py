"""Saved-data figures for hidden transition change and the cost of forgetting."""

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
import numpy as np

from figures import ACTION_COLORS, save, style
from q_learning_figures import band

COLORS = ['#7A838B', '#207F75', '#A565A0']
METHODS = ['Frozen', 'Cumulative', 'Forgetting']
CONDITIONS = ['Stable world', 'Changed world']


def adaptation(data, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = data['snapshot_steps']/1000
    for g, condition in enumerate(CONDITIONS):
        for m in range(3):
            band(axes[0, g], x, data['actual_values'][:, g, m, :, 24], COLORS[m], METHODS[m])
        axes[0, g].axhline(data['oracle_values'][g, 24], color='#213340', ls='--', label='This regime’s oracle')
        axes[0, g].set(title=condition, xlabel='New transitions (thousands)', ylabel='Exact true Vπ(4,4), γ=0.99', xlim=(0, 50))
        delta = data['time_averaged_oracle_gap'][g, 2]-data['time_averaged_oracle_gap'][g, 1]
        mean, half = delta.mean(), 1.96*delta.std(ddof=1)/10
        axes[1, g].hist(delta, bins=18, color=COLORS[2], alpha=.8, edgecolor='white')
        axes[1, g].axvline(0, color='#213340', ls=':')
        axes[1, g].axvline(mean, color='#213340')
        axes[1, g].axvspan(mean-half, mean+half, color='#213340', alpha=.12)
        axes[1, g].set(title=f'{"Stable cost" if g == 0 else "Primary"}: {mean:+.3f} [{mean-half:+.3f}, {mean+half:+.3f}]',
            xlabel='Forgetting − cumulative · time-averaged oracle gap', ylabel='Training seed pairs')
    for ax in axes.flat:
        ax.grid(alpha=.15)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle('When a successful world model becomes outdated', fontsize=17)
    fig.text(.5, .01, 'Top: 95% intervals across 100 seeds. Bottom: paired tracking differences; negative favors forgetting.\n'
        'Tracking integrates policy-value gaps over the fixed 0:1,000:50,000 grid. It is not realized online regret.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .95))
    save(fig, output, 'adaptation.png')


def predictions(data, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    x = data['snapshot_steps']/1000
    for g, condition in enumerate(CONDITIONS):
        actual = data['actual_values'][:, g, ..., 24]
        predicted = data['predictions'][:, g, ..., 24]
        for m in range(3):
            band(axes[0, g], x, abs(predicted[:, m]-actual[:, m]), COLORS[m], METHODS[m])
            axes[1, g].scatter(actual[-1, m], predicted[-1, m], s=24, alpha=.5, color=COLORS[m], label=METHODS[m])
        axes[0, g].set(title=condition+' · policy prediction error', xlabel='New transitions (thousands)',
                       ylabel='Mean |predicted − actual Vπ(4,4)|', xlim=(0, 50))
        lo = min(actual[-1].min(), predicted[-1].min())-1
        hi = max(actual[-1].max(), predicted[-1].max())+1
        axes[1, g].plot([lo, hi], [lo, hi], color='#213340', ls=':')
        axes[1, g].set(title=condition+' · final predictions', xlabel='Actual value in this regime',
                      ylabel='Value predicted by own empirical model', xlim=(lo, hi), ylim=(lo, hi), aspect='equal')
    axes[0, 0].legend(frameon=False)
    for ax in axes.flat:
        ax.grid(alpha=.15)
    fig.suptitle('Observed-reward predictions can outlive the dynamics that supported them', fontsize=16)
    fig.text(.5, .01, 'Predictions use each method’s empirical transitions and original observed rewards, with no count penalty.\n'
             'Frozen predictions stay inherited; cumulative and forgetting predictions update. True models are used only for evaluation.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95))
    save(fig, output, 'prediction_errors.png')


def resources(data, output):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    titles = ['Reward per decision', 'Stock A', 'Stock B', 'Either stock depleted (%)']
    for g, condition in enumerate(CONDITIONS):
        for k, ax in enumerate(axes[g]):
            scale = 100 if k == 3 else 1
            for m in range(3):
                band(ax, np.arange(1, 51), scale*data['rollout_metrics'][g, m, :, :, k], COLORS[m], METHODS[m])
            ax.set(title=condition+' · '+titles[k], xlabel='New 1,000-step rollout', xlim=(1, 50))
            if k in (1, 2):
                ax.set_ylim(0, 4.1)
            if k == 3:
                ax.set_ylim(0, 100)
            ax.grid(alpha=.15)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .95), ncol=3, frameon=False)
    fig.suptitle('Actual interaction: adaptation must be earned through changed behavior', fontsize=16, y=.99)
    fig.text(.5, .01, 'All methods act with ε=0.1; learning depends on the memory rule. Stocks/depletion measured before each action.\n'
        'Each method receives 50,000 new transitions per seed in each world. Bands: 95% across seed-level rollout means.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .91))
    save(fig, output, 'interaction_resources.png')


def beliefs(data, output):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    x = data['snapshot_steps']/1000
    action_colors = np.array([to_rgb(c) for c in ACTION_COLORS])
    for g, condition in enumerate(CONDITIONS):
        for m in range(3):
            axes[g, 0].plot(x, data['seed0_successor_probabilities'][:, g, m, 2, 18], color=COLORS[m], label=METHODS[m], lw=2)
            for a in (0, 1):
                advantage = data['seed0_action_values'][:, g, m, a]-data['seed0_action_values'][:, g, m, 2]
                axes[g, a+1].plot(x, advantage, color=COLORS[m], lw=2)
        axes[g, 0].axhline(data['transition'][g, 18, 2, 18], ls='--', color='#213340', label='True probability')
        axes[g, 0].set(title=condition+' · rest self-loop belief', ylabel='P̂((3,3) | (3,3), rest)', ylim=(0, 1))
        for a in (0, 1):
            axes[g, a+1].axhline(0, color='#213340', ls=':')
            axes[g, a+1].set(title=f'Predicted Q(harvest {"AB"[a]}) − Q(rest)', ylabel='Action-value difference')
        # Each RGB cell is the selected mixed policy, preserving ties instead of argmax.
        rgb = data['policies'][:, g, :, 0, 18].transpose(1, 0, 2)@action_colors
        axes[g, 3].imshow(rgb, origin='upper', aspect='auto', extent=(-.5, 50.5, 2.5, -.5), interpolation='nearest')
        axes[g, 3].set(title=condition+' · chosen greedy actions', yticks=range(3), yticklabels=METHODS)
        for ax in axes[g]:
            ax.set(xlabel='New transitions (thousands)', xlim=(0, 50))
            ax.grid(alpha=.12)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.38, .95), ncol=4, frameon=False, fontsize=9)
    action_legend = [Patch(color=c, label=n) for c,n in zip(ACTION_COLORS, ['Harvest A', 'Harvest B', 'Rest'])]
    fig.legend(handles=action_legend, loc='upper center', bbox_to_anchor=(.84, .95), ncol=3, frameon=False, fontsize=9)
    fig.suptitle('Preselected seed 0, state (3,3): beliefs, imagined action values, and decisions', fontsize=16, y=.99)
    fig.text(.5, .01, 'Full successor distributions and absolute action values are saved for every snapshot. Here the rest self-loop summarizes one belief.\n'
        'Positive action-value difference favors that harvest over rest. Action cells show greedy policies; collection adds ε=0.1. Mixed colors denote ties.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .91))
    save(fig, output, 'seed_0_beliefs_and_actions.png')


def plot_all(data, output):
    style()
    adaptation(data, output)
    predictions(data, output)
    resources(data, output)
    beliefs(data, output)
