"""Saved-data figures for the collector-by-planner continuation experiment."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

from figures import ACTION_COLORS, ACTION_LETTERS, save, stock_axes, style
from q_learning_figures import band


COLORS = ['#C56949', '#207F75', '#7960A6']
LABELS = ['A · Q collector', 'B · model collector', 'C · penalty collector']
RULES = ['Unpenalized planner', 'Count-penalized planner']


def plot_values(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for p, ax in enumerate(axes):
        for c in range(3):
            band(ax, data['additional_steps']/1000, data['checkpoint_values'][:, c, p, :, 24], COLORS[c], LABELS[c])
        ax.axhline(data['oracle_values'][24], color='#3C4852', ls='--', label='Saved oracle')
        ax.set(title=RULES[p], xlabel='Additional transitions per seed (thousands)',
               ylabel='True Vπ(4,4), γ = 0.99', xlim=(0, 50))
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle('Can model-guided interaction improve the data used for planning?', fontsize=16)
    fig.text(.5, .01, 'Each collector adds 50,000 transitions to the same 200,000-transition history. Endpoint: 250,000 total.\n'
             'Left: B versus A is the fixed primary comparison. Right and collector C are secondary. Bands: 95% across 100 seeds.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .93))
    save(fig, output, 'policy_values.png')


def plot_factorial(data, output):
    final = data['checkpoint_values'][-1, ..., 24]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    mean = final.mean(axis=-1)
    half = 1.96*final.std(axis=-1, ddof=1)/np.sqrt(final.shape[-1])
    axes[0].imshow(mean, cmap='YlGnBu', vmin=0, vmax=data['oracle_values'][24], aspect='auto')
    for c in range(3):
        for p in range(2):
            axes[0].text(p, c, f'{mean[c,p]:.2f}\n[{mean[c,p]-half[c,p]:.2f}, {mean[c,p]+half[c,p]:.2f}]',
                         ha='center', va='center', color='white' if mean[c,p] > 55 else '#213340', fontsize=10)
    axes[0].set(title='Final true value: mean [95% CI]', xticks=[0, 1],
                xticklabels=['Unpenalized', 'Penalized'], yticks=range(3), yticklabels=LABELS, xlabel='Extraction rule')
    a, b = final[0, 0], final[1, 0]
    lo, hi = 5*np.floor(min(a.min(), b.min())/5)-5, 5*np.ceil(data['oracle_values'][24]/5)
    axes[1].scatter(a, b, color=COLORS[1], s=23, alpha=.65)
    axes[1].plot([lo, hi], [lo, hi], color='#3C4852', ls=':')
    axes[1].set(title='Primary: same unpenalized extraction', xlabel='A collector · true policy value',
                ylabel='B collector · true policy value', xlim=(lo, hi), ylim=(lo, hi), aspect='equal')
    difference = b-a
    edges = np.linspace(min(0, difference.min())-1, max(0, difference.max())+1, 21)
    axes[2].hist(difference, bins=edges, color=COLORS[1], alpha=.8, edgecolor='white')
    avg, ci = difference.mean(), 1.96*difference.std(ddof=1)/10
    axes[2].axvline(0, color='#3C4852', ls=':')
    axes[2].axvline(avg, color='#213340')
    axes[2].axvspan(avg-ci, avg+ci, color='#213340', alpha=.13)
    axes[2].set(title=f'Paired gain {avg:.2f} [{avg-ci:.2f}, {avg+ci:.2f}]',
                xlabel='B minus A · same unpenalized planner', ylabel='Training seed pairs')
    for ax in axes[1:]:
        ax.grid(alpha=.15)
    fig.suptitle('Separate who collected the data from how the final policy was planned', fontsize=16)
    fig.text(.5, .01, 'Six prespecified policies per seed; no per-seed rule selection using true values. All final outcomes and failures retained.\n'
             'Historical 200,000-transition values are starting references; comparisons here use equal 250,000-transition budgets.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .93))
    save(fig, output, 'collector_by_planner.png')


def plot_predictions_coverage(data, output):
    fig, axes = plt.subplots(2, 3, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 2]})
    actual = data['checkpoint_values'][-1, ..., 24]
    predicted = data['checkpoint_predictions'][-1, ..., 24]
    lo, hi = min(actual.min(), predicted.min())-3, max(actual.max(), predicted.max())+3
    for p in range(2):
        ax = axes[0, p]
        for c in range(3):
            ax.scatter(actual[c, p], predicted[c, p], s=20, color=COLORS[c], alpha=.6, label=LABELS[c])
        ax.plot([lo, hi], [lo, hi], color='#3C4852', ls=':')
        ax.set(title=RULES[p], xlabel='Actual return in the true environment',
               ylabel='Original empirical-reward prediction', xlim=(lo, hi), ylim=(lo, hi), aspect='equal')
        ax.grid(alpha=.15)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for c in range(3):
        for p in range(2):
            e = predicted[c, p]-actual[c, p]
            axes[0, 2].errorbar(c+(p-.5)*.18, e.mean(), yerr=1.96*e.std(ddof=1)/10,
                               fmt='o' if p == 0 else 's', color=COLORS[c], capsize=4)
    axes[0, 2].axhline(0, color='#3C4852', ls=':')
    axes[0, 2].set(title='Final mean prediction − actual', xticks=range(3), xticklabels=['A', 'B', 'C'],
                   xlabel='Collector', ylabel='Error in original-reward value')
    axes[0, 2].legend([Line2D([], [], marker='o', ls='', color='#3C4852'),
                      Line2D([], [], marker='s', ls='', color='#3C4852')], RULES, frameon=False, fontsize=8)
    for c in range(3):
        sparse = 100*(data['checkpoint_visits'][-1, c] < 10).mean(axis=0)
        ax = axes[1, c]
        im = ax.imshow(sparse, cmap='YlOrRd', vmin=0, vmax=100, aspect='auto')
        for s in range(25):
            for a in range(3):
                ax.text(a, s, f'{sparse[s,a]:.0f}', ha='center', va='center', fontsize=8,
                        color='white' if sparse[s,a] > 60 else '#213340')
        ax.set(title=LABELS[c]+' · rows with N < 10 (%)', xticks=range(3), xticklabels=['Harvest A', 'Harvest B', 'Rest'],
               yticks=range(25), yticklabels=[f'({a},{b})' for a,b in data['states']], ylabel='State (stock A, stock B)')
    fig.colorbar(im, ax=axes[1].tolist(), fraction=.02, pad=.02, label='Percent of seeds with fewer than 10 observations')
    fig.suptitle('Model predictions and the parts of the world that remain poorly sampled', fontsize=16, y=.995)
    fig.text(.5, .01, 'All panels use the fixed final endpoint. Prediction error excludes the count penalty from rewards.\n'
             'Coverage threshold N < 10 fixed before the run; numbers show percentage of training seeds, not transition probabilities.', ha='center', fontsize=9)
    fig.subplots_adjust(left=.07, right=.88, bottom=.075, top=.945, wspace=.35, hspace=.28)
    save(fig, output, 'predictions_and_coverage.png')


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
            for p in range(2):
                band(ax, frozen['bin_centers'], scale*frozen['seed_curves'][c, p, :, :, metric].T,
                     COLORS[c], LABELS[c]+' / '+RULES[p])
                ax.lines[-1].set_linestyle('-' if p == 0 else '--')
        ax.set(title='Frozen · '+titles[k], xlabel='Evaluation decision', xlim=(0, 1000))
    for ax in axes.flat:
        ax.grid(alpha=.15)
    for row in range(2):
        axes[row, 1].set_ylim(0, 4.1)
        axes[row, 2].set_ylim(0, 4.1)
        axes[row, 3].set_ylim(0, 100)
    handles = [Line2D([], [], color=c, lw=2) for c in COLORS]
    handles += [Line2D([], [], color='#3C4852', ls=ls) for ls in ['-', '--']]
    fig.legend(handles, LABELS+RULES, loc='upper center', bbox_to_anchor=(.5, .95), ncol=5, frameon=False, fontsize=9)
    fig.suptitle('Collecting experience and acting with the final policy are different experiments', fontsize=16, y=.99)
    fig.text(.5, .01, 'Top: learning/model updates active, ε=0.1. Bottom: six frozen final policies; no learning or exploration, 20 paired paths per seed.\n'
             'Bands: 95% intervals across seed means. Frozen curves use 20-decision bins; trajectories are averaged within training seed first.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .91))
    save(fig, output, 'collection_and_frozen_resources.png')


def plot_seed_zero(data, diagnostic, output):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    steps = np.array(diagnostic['additional_steps'])/1000
    counts = np.array(diagnostic['visit_counts'])
    loops = np.array(diagnostic['empirical_self_loop'])
    choices = np.array(diagnostic['collector_action_probabilities'])
    for c in range(3):
        axes[0, 0].plot(steps, counts[:, c], marker='o', ms=3, color=COLORS[c], label=LABELS[c])
        axes[0, 1].plot(steps, loops[:, c], marker='o', ms=3, color=COLORS[c])
        axes[0, 2].step(steps, choices[:, c, 1], where='post', color=COLORS[c], lw=2)
    axes[0, 0].set(title='Harvest-B observations at (3,4)', ylabel='Cumulative visits (log scale)', yscale='log')
    axes[0, 1].axhline(diagnostic['true_self_loop_probability'], color='#3C4852', ls='--', label='True probability · evaluation only')
    axes[0, 1].set(title='Belief: harvest B returns to (3,4)', ylabel='Estimated self-loop probability', ylim=(-.03, 1.03))
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 2].set(title='Collector probability of harvest B', ylabel='Action probability including ε=0.1', ylim=(-.03, 1.03))
    for ax in axes[0]:
        ax.set(xlabel='Additional transitions (thousands)', xlim=(0, 50))
        ax.grid(alpha=.15)
    for c, ax in enumerate(axes[1]):
        policy = data['collector_greedy'][-1, c, 0]
        for (x,y), row in zip(data['states'], policy):
            choices = np.flatnonzero(row)
            for j,a in enumerate(choices):
                ax.add_patch(Rectangle((x-.5+j/len(choices), y-.5), 1/len(choices), 1,
                                      facecolor=ACTION_COLORS[a], alpha=.3, edgecolor='white'))
            ax.text(x,y,'/'.join(ACTION_LETTERS[a] for a in choices), ha='center', va='center', fontsize=9)
        ax.add_patch(Rectangle((2.5, 3.5), 1, 1, fill=False, edgecolor='#7F2424', lw=2))
        stock_axes(ax)
        ax.set_title(LABELS[c]+' · final greedy control')
    fig.suptitle('Preselected seed 0: can acting test the apparent rewarding loop?', fontsize=16, y=.99)
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5,.958), ncol=3, frameon=False)
    fig.text(.5, .01, 'State (3,4), harvest B, seed 0 fixed before running. Top: saved 5,000-transition checkpoints; control replans every 1,000.\n'
             'Bottom: greedy parts of the collectors, with A/B/R = harvest A/harvest B/rest. Collection adds uniform ε exploration to these policies.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.91))
    save(fig, output, 'seed_0_beliefs_and_actions.png')


def plot_all(data, frozen, diagnostic, output):
    style()
    plot_values(data, output)
    plot_factorial(data, output)
    plot_predictions_coverage(data, output)
    plot_resources(data, frozen, output)
    plot_seed_zero(data, diagnostic, output)
