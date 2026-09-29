"""Four Experiment 4 figures; all inputs are completed numerical archives."""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

from figures import ACTION_COLORS, ACTION_LETTERS, save, stock_axes, style
from q_learning_figures import band


COLORS = ['#C56949', '#207F75', '#7960A6']
LABELS = ['Frozen Q', 'Unpenalized model', 'Count-penalized model']


def plot_values(data, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    values = data['checkpoint_values'][..., 24]
    oracle = data['oracle_values'][24]
    for m in range(3):
        band(axes[0], data['checkpoint_steps']/1000, values[:, m], COLORS[m], LABELS[m])
    axes[0].axhline(oracle, color='#3C4852', ls='--', label='Saved oracle')
    axes[0].set(title='Same archived experience', xlabel='Transitions in archive per seed (thousands)',
                ylabel='True Vπ(4,4), γ = 0.99', xlim=(0, 200))
    axes[0].legend(frameon=False, fontsize=8, loc='lower right')
    final = values[-1]
    lower, upper = 5*np.floor(final.min()/5)-5, 5*np.ceil(oracle/5)
    for m, ax in enumerate(axes[1:]):
        delta = final[2]-final[m]
        mean, half = delta.mean(), 1.96*delta.std(ddof=1)/np.sqrt(len(delta))
        ax.scatter(final[m], final[2], color=COLORS[2], s=23, alpha=.65)
        ax.plot([lower, upper], [lower, upper], color='#3C4852', ls=':', label='Equal value')
        ax.axhline(.9*oracle, color='#848D70', ls='--', label='90% of oracle')
        ax.axvline(.9*oracle, color='#848D70', ls='--', lw=.8)
        ax.set(title=f'Paired gain {mean:.2f} [{mean-half:.2f}, {mean+half:.2f}]',
               xlabel=LABELS[m]+' · final true value', ylabel='Penalized policy · final true value',
               xlim=(lower, upper), ylim=(lower, upper), aspect='equal')
        ax.legend(frameon=False, fontsize=8, loc='lower right')
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle('Does caution about scarce evidence improve decisions?', fontsize=16)
    fig.text(.5, .012, 'Fixed penalty 1/√max(N,1); no tuning or new collection. Final endpoint: 200,000 transitions.\n'
             'Each dot is a matched training seed; bands and paired-gain intervals are ±1.96 SEM across 100 seeds. All failures are retained.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .085, 1, .93))
    save(fig, output, 'policy_values.png')


def plot_predictions(data, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    actual = data['checkpoint_values'][:, 1:, :, 24]
    predicted = data['checkpoint_predictions'][..., 24]
    low = min(actual[-1].min(), predicted[-1].min())-3
    high = max(actual[-1].max(), predicted[-1].max())+3
    for m in range(2):
        axes[0].scatter(actual[-1, m], predicted[-1, m], color=COLORS[m+1],
                        s=22, alpha=.6, label=LABELS[m+1])
        band(axes[1], data['checkpoint_steps']/1000, predicted[:, m]-actual[:, m],
             COLORS[m+1], LABELS[m+1])
    axes[0].plot([low, high], [low, high], color='#3C4852', ls='--')
    axes[0].set(xlabel='C · actual return in true world', ylabel='B · original empirical-reward prediction',
                title='Final policies: B versus C', xlim=(low, high), ylim=(low, high), aspect='equal')
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(0, color='#3C4852', ls=':')
    axes[1].set(xlabel='Archived transitions per seed (thousands)', ylabel='B − C',
                title='Prediction error of chosen policies', xlim=(0, 200))
    score = data['checkpoint_internal_scores'][-1, :, 24]
    offset = .17*np.sin(np.arange(len(score))*2.39996323)
    for j, y in enumerate((score, predicted[-1, 1], actual[-1, 1])):
        axes[2].scatter(j+offset, y, s=15, color=COLORS[2], alpha=.4)
        axes[2].errorbar(j+.25, y.mean(), yerr=1.96*y.std(ddof=1)/np.sqrt(len(y)),
                         fmt='D', color='#213340', capsize=4)
    axes[2].set(xticks=[0, 1, 2], xticklabels=['A\nPenalized\nscore', 'B\nOriginal reward\nprediction', 'C\nTrue\nreturn'],
                ylabel='Value from (4,4), γ = 0.99', title='Same new policies, different objectives')
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle('A lower planning score is not evidence of better calibration', fontsize=16)
    fig.text(.5, .012, 'Prediction error always uses original empirical rewards: B − C. A includes the planning penalty.\n'
             'The empirical model is unchanged. Different policies visit different parts of it. Bands/diamonds: 95% mean intervals across seeds.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .93))
    save(fig, output, 'predictions_and_scores.png')


def plot_resources(behavior, output):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    titles = ['Stock A before acting', 'Stock B before acting', 'Either patch depleted',
              'Harvested reward per decision', 'Cumulative harvested reward', 'Rest decisions']
    for ax, k, title in zip(axes.flat, [2, 3, 4, 0, 1, 5], titles):
        scale = 100 if k in (4, 5) else 1
        x = behavior['bin_ends'] if k == 1 else behavior['bin_centers']
        for m in range(3):
            band(ax, x, scale*behavior['seed_curves'][m, :, :, k].T, COLORS[m], LABELS[m])
        ax.set(title=title, xlabel='Frozen evaluation decision', xlim=(0, 1000))
        if k in (2, 3):
            ax.set_ylim(0, 4.1)
        if k in (4, 5):
            ax.set(ylabel='Percent', ylim=(0, 100))
        ax.grid(alpha=.15)
    fig.suptitle('Frozen behavior: productive caution or avoidance of useful actions?', fontsize=16, y=.99)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, .955), ncol=3, frameon=False)
    fig.text(.5, .01, 'No learning or exploration; 20 paired 1,000-step trajectories per training seed, all from (4,4). Baseline outcomes reused.\n'
             '20-decision bins; average trajectories within seed first. Bands: 95% intervals across 100 seed means.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .91))
    save(fig, output, 'frozen_resources.png')


def plot_seed_zero(data, behavior, diagnostic, output):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    policies = [*data['checkpoint_policies'][-1, :, 0], data['oracle_policy']]
    for ax, pi, title in zip(axes[0], policies, LABELS+['Saved oracle']):
        for (x, y), row in zip(data['states'], pi):
            choices = np.flatnonzero(row)
            for j, a in enumerate(choices):
                ax.add_patch(Rectangle((x-.5+j/len(choices), y-.5), 1/len(choices), 1,
                                      facecolor=ACTION_COLORS[a], alpha=.3, edgecolor='white'))
            ax.text(x, y, '/'.join(ACTION_LETTERS[a] for a in choices), ha='center', va='center', fontsize=9)
        ax.add_patch(Rectangle((2.5, 3.5), 1, 1, fill=False, edgecolor='#7F2424', lw=2))
        stock_axes(ax)
        ax.set_title(title+' · seed 0' if title != 'Saved oracle' else title)
    for m, ax in enumerate(axes[1, :3]):
        for patch in range(2):
            ax.step(np.arange(121), behavior['trace_stocks'][m, :121, patch], where='post',
                    color=ACTION_COLORS[patch], lw=1.5, label=f'Stock {ACTION_LETTERS[patch]}')
        ax.set(title=LABELS[m]+' · trajectory 0', xlabel='Evaluation decision',
               ylabel='Stock', xlim=(0, 120), ylim=(-.15, 4.15), yticks=range(5))
        ax.legend(frameon=False, fontsize=8, loc='lower right')
        ax.grid(alpha=.15)
    ax = axes[1, 3]
    ax.axis('off')
    ax.set_title('The highlighted state: (3,4)', loc='left')
    choices = ['/'.join(ACTION_LETTERS[a] for a in np.flatnonzero(pi[19])) for pi in policies[:3]]
    d = diagnostic
    old, new = [d['policies'][name] for name in LABELS[1:]]
    counts = '/'.join(str(n) for n in d['visit_counts'])
    text = (f'Counts A/B/R: {counts}\n'
            f'Choices Q/plain/penalty: {" / ".join(choices)}\n\n'
            f'Empirical B self-loop: {d["empirical_self_loop_probabilities"][1]:.2f}\n'
            f'B planning reward: {d["planning_rewards"][1]:.2f}\n\n'
            'Values starting at (3,4):\n'
            f'Plain B prediction: {old["state_predicted_B"]:.2f}\n'
            f'Plain C actual: {old["state_actual_C"]:.2f}\n\n'
            f'Penalty A score: {d["penalized_state_score_A"]:.2f}\n'
            f'Penalty B prediction: {new["state_predicted_B"]:.2f}\n'
            f'Penalty C actual: {new["state_actual_C"]:.2f}')
    ax.text(0, .95, text, transform=ax.transAxes, va='top', fontsize=10, linespacing=1.4)
    fig.suptitle('Preselected seed 0: does the penalty avoid the apparent rewarding loop?', fontsize=16, y=.99)
    fig.legend([Patch(facecolor=c, alpha=.5) for c in ACTION_COLORS],
               ['A · harvest A', 'B · harvest B', 'R · rest'], loc='upper center',
               bbox_to_anchor=(.5, .96), ncol=3, frameon=False)
    fig.text(.5, .012, 'Final policies; split cells mix ties uniformly. Seed 0, trajectory 0 and the first 120 decisions were fixed before the experiment.\n'
             'A/B/C in value labels mean score/prediction/actual return; A/B/R in policy maps mean harvest A/harvest B/rest.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .92))
    save(fig, output, 'seed_0_policies_and_mechanism.png')


def plot_all(data, behavior, diagnostic, output):
    style()
    plot_values(data, output)
    plot_predictions(data, output)
    plot_resources(behavior, output)
    plot_seed_zero(data, behavior, diagnostic, output)
