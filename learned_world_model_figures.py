"""Experiment 3 figures using saved arrays only."""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

from figures import ACTION_COLORS, ACTION_LETTERS, save, stock_axes, style
from q_learning_figures import band


COLORS = ['#C56949', '#207F75']
LABELS = ['Frozen Q policy', 'Learned-model policy']


def plot_values(data, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    values = data['checkpoint_values'][..., 24]
    oracle = data['oracle_values'][24]
    for m in range(2):
        band(axes[0], data['checkpoint_steps']/1000, values[:, m], COLORS[m], LABELS[m])
    axes[0].axhline(oracle, color='#3C4852', ls='--', label='True-model oracle')
    axes[0].set(title='Value from the same collection stream', xlabel='Transitions per seed (thousands)',
                ylabel='Exact Vπ(4,4), γ = 0.99', xlim=(0, 200), ylim=(20, 100))
    axes[0].legend(frameon=False, fontsize=8, loc='lower right')
    final = values[-1]
    lower = 5*np.floor(final.min()/5)-5
    upper = 5*np.ceil(max(oracle, final.max())/5)
    axes[1].scatter(final[0], final[1], color=COLORS[1], s=22, alpha=.65)
    axes[1].plot([lower, upper], [lower, upper], color='#3C4852', ls=':', label='Equal value')
    axes[1].axhline(.9*oracle, color='#848D70', ls='--', label='90% of oracle')
    axes[1].axvline(.9*oracle, color='#848D70', ls='--', lw=.8)
    axes[1].set(title='Final paired outcomes · 100 seeds', xlabel='Frozen Q policy value',
                ylabel='Learned-model policy value', xlim=(lower, upper), ylim=(lower, upper), aspect='equal')
    axes[1].legend(frameon=False, fontsize=8, loc='lower right')
    delta = final[1]-final[0]
    edges = np.arange(5*np.floor(delta.min()/5), 5*np.ceil(delta.max()/5)+2.5, 2.5)
    axes[2].hist(delta, bins=edges, color=COLORS[1], edgecolor='white', alpha=.8)
    mean, half = delta.mean(), 1.96*delta.std(ddof=1)/np.sqrt(len(delta))
    axes[2].axvline(0, color='#3C4852', ls=':')
    axes[2].axvline(mean, color='#213340', lw=1.2)
    axes[2].axvspan(mean-half, mean+half, color='#213340', alpha=.15)
    axes[2].set(title=f'Mean paired gain {mean:.2f} [{mean-half:.2f}, {mean+half:.2f}]',
                xlabel='Learned-model value − Q value', ylabel='Training seed pairs')
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle('Same observations: does estimating dynamics and planning extract more value?', fontsize=16)
    fig.text(.5, .012, 'Final outcome fixed at 200,000 transitions. True dynamics are used only for evaluation and the oracle reference.\n'
             'Curve bands and shaded mean-gain interval: 95% across 100 training seeds (±1.96 SEM). Equal experience is not equal computation.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .93))
    save(fig, output, 'policy_values.png')


def plot_prediction(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    actual = data['checkpoint_values'][:, 1, :, 24]
    predicted = data['checkpoint_predicted_values'][:, :, 24]
    low = min(actual[-1].min(), predicted[-1].min())-2
    high = max(actual[-1].max(), predicted[-1].max())+2
    axes[0].scatter(actual[-1], predicted[-1], c=COLORS[1], s=25, alpha=.65)
    axes[0].plot([low, high], [low, high], color='#3C4852', ls='--', label='Accurate prediction')
    axes[0].set(xlim=(low, high), ylim=(low, high), aspect='equal',
                xlabel='Actual Vπ(4,4) in the true environment',
                ylabel='Predicted Vπ(4,4) in its own learned model', title='Final learned policies · one dot per seed')
    axes[0].legend(frameon=False, fontsize=9)
    # The zero-data assumed model predicts zero; include this initial diagnostic.
    band(axes[1], data['checkpoint_steps']/1000, predicted-actual, COLORS[1], 'Prediction − actual value')
    axes[1].axhline(0, color='#3C4852', ls='--')
    axes[1].set(xlabel='Transitions per seed (thousands)', ylabel='Predicted value − actual value',
                title='Are imagined returns too optimistic?', xlim=(0, 200))
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle('A useful policy can still have an inaccurate imagined future', fontsize=16)
    fig.text(.5, .01, 'Both values evaluate the same frozen learned-model policy at γ = 0.99. Above the diagonal means overprediction.\n'
             'Unknown rows assume zero-reward self-loops. Band: pointwise 95% interval across seeds; the initial no-data model is included.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .085, 1, .93))
    save(fig, output, 'predicted_vs_actual.png')


def plot_resources(simulation, output):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    titles = ['Stock A before acting', 'Stock B before acting', 'Either patch depleted',
              'Harvested reward per decision', 'Cumulative harvested reward', 'Rest decisions']
    for ax, k, title in zip(axes.flat, [2, 3, 4, 0, 1, 5], titles):
        scale = 100 if k in (4, 5) else 1
        x = simulation['bin_ends'] if k == 1 else simulation['bin_centers']
        for m in range(2):
            band(ax, x, scale*simulation['seed_curves'][m, :, :, k].T, COLORS[m], LABELS[m])
        ax.set(title=title, xlabel='Evaluation decision', xlim=(0, 1000))
        if k in (2, 3):
            ax.set_ylim(0, 4.1)
        if k in (4, 5):
            ax.set(ylabel='Percent', ylim=(0, 100))
        ax.grid(alpha=.15)
    fig.suptitle('Frozen behavior after training: which stocks survive?', fontsize=16, y=.99)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, .955), ncol=2, frameon=False)
    fig.text(.5, .01, 'No learning or exploration. 20 paired, independent 1,000-step trajectories per training seed, all from (4,4).\n'
             '20-decision bins. Average trajectories within each seed first; bands show 95% intervals across the 100 seed means.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .91))
    save(fig, output, 'frozen_resources.png')


def plot_illustration(data, simulation, output):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    policies = [*data['checkpoint_policies'][-1, :, 0], data['oracle_policy']]
    titles = ['Q policy · seed 0', 'Learned-model policy · seed 0', 'True-model oracle reference']
    for ax, policy, title in zip(axes[0], policies, titles):
        for (x, y), row in zip(data['states'], policy):
            choices = np.flatnonzero(row)
            for j, action in enumerate(choices):
                ax.add_patch(Rectangle((x-.5+j/len(choices), y-.5), 1/len(choices), 1,
                                      facecolor=ACTION_COLORS[action], alpha=.3, edgecolor='white'))
            ax.text(x, y, '/'.join(ACTION_LETTERS[a] for a in choices), ha='center', va='center')
        stock_axes(ax)
        ax.set_title(title)
    stop = 120
    for m, ax in enumerate(axes[1, :2]):
        for patch in range(2):
            ax.step(np.arange(stop+1), simulation['trace_stocks'][m, :stop+1, patch],
                    where='post', color=ACTION_COLORS[patch], lw=1.6, label=f'Stock {ACTION_LETTERS[patch]}')
        ax.set(title=titles[m]+' · trajectory 0', xlabel='Evaluation decision',
               ylabel='Stock before acting', xlim=(0, stop), ylim=(-.15, 4.15), yticks=range(5))
        ax.legend(frameon=False, fontsize=8, loc='lower right')
        ax.grid(alpha=.15)
    for m in range(2):
        axes[1, 2].plot(np.arange(stop+1), np.r_[0, simulation['trace_rewards'][m, :stop].cumsum()],
                        color=COLORS[m], label=LABELS[m])
    axes[1, 2].set(title='Trajectory 0 · harvested reward', xlabel='Evaluation decisions completed',
                   ylabel='Cumulative reward', xlim=(0, stop))
    axes[1, 2].legend(frameon=False, fontsize=8)
    axes[1, 2].grid(alpha=.15)
    fig.suptitle('Seed 0, chosen before running: a policy and its consequences', fontsize=16, y=.99)
    fig.legend([Patch(facecolor=c, alpha=.5) for c in ACTION_COLORS],
               ['A · harvest A', 'B · harvest B', 'R · rest'], loc='upper center',
               bbox_to_anchor=(.5, .96), ncol=3, frameon=False)
    fig.text(.5, .012, 'Top: final policies; split cells mix tied actions uniformly. Bottom: paired seed-0, trajectory-0 draws, first 120 decisions.\n'
             'The illustration was fixed in advance, not selected for its outcome. All population conclusions use 100 training seeds.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .925))
    save(fig, output, 'seed_0_policies_and_trajectory.png')


def plot_all(data, simulation, output):
    style()
    plot_values(data, output)
    plot_prediction(data, output)
    plot_resources(simulation, output)
    plot_illustration(data, simulation, output)
