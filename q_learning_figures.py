"""Readable Q-learning figures from saved checkpoints and seed-level outcomes."""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

from figures import ACTION_COLORS, ACTION_LETTERS, save, stock_axes, style


COLORS = ['#207F75', '#C56949']
LABELS = ['Q-learning · γ_train = 0.99', 'Q-learning · γ_train = 0']


def band(ax, x, y, color, label):
    mean = y.mean(axis=-1)
    half = 1.96*y.std(axis=-1, ddof=1)/np.sqrt(y.shape[-1])
    ax.plot(x, mean, color=color, label=label, lw=2)
    ax.fill_between(x, mean-half, mean+half, color=color, alpha=.15, linewidth=0)


def plot_learning(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for key, ax, title in zip(['checkpoint_values', 'checkpoint_behavior_values'], axes,
            ['Frozen greedy policy · primary evaluation', 'Frozen ε = 0.1 policy · exploration still active']):
        for m, label in enumerate(LABELS):
            band(ax, data['checkpoint_steps']/1000, data[key][:, m, :, 24], COLORS[m], label)
        for value, label, color in zip(data['reference_initial_values'],
                                      ['Random', 'Myopic', 'Optimal oracle'], ['#969EA5', '#A68675', '#3C4852']):
            ax.axhline(value, color=color, ls='--', lw=.9)
            ax.text(2, value+1, label, color=color, fontsize=8)
        ax.set(title=title, xlabel='Training transitions per seed (thousands)',
               ylabel='Exact Vπ(4,4), evaluated at γ = 0.99', xlim=(0, 200), ylim=(0, 102))
        ax.grid(alpha=.15)
    fig.suptitle('Does experience teach the value of waiting?', fontsize=16, y=1.01)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .94),
               ncol=2, frameon=False)
    fig.text(.5, -.01, '100 independent training seeds · mean ± 1.96 SEM · checkpoints fixed before training\n'
             'Q is frozen during evaluation. All conclusions use 200,000 transitions, not the best checkpoint.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .9))
    save(fig, output, 'learning_values.png')


def plot_final(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    values = data['final_values'][:, :, 24]
    oracle = data['reference_initial_values'][2]
    threshold = .9*oracle
    # Deterministic offsets: no selection of attractive seeds or random jitter.
    offsets = .18*np.sin(np.arange(100)*2.39996323)
    for m in range(2):
        axes[0].scatter(m+offsets, values[m], s=19, color=COLORS[m], alpha=.55, edgecolor='none')
        mean, half = values[m].mean(), 1.96*values[m].std(ddof=1)/10
        axes[0].errorbar(m+.29, mean, yerr=half, fmt='D', color=COLORS[m], capsize=5)
    axes[0].axhline(oracle, color='#3C4852', ls='--', label='Oracle')
    axes[0].axhline(threshold, color='#848D70', ls=':', label='90% of oracle')
    axes[0].set(xticks=[0, 1], xticklabels=['γ_train = 0.99', 'γ_train = 0'],
                ylabel='Final frozen-policy Vπ(4,4)', title='Every training seed, not just the mean', ylim=(0, 102))
    axes[0].legend(frameon=False, fontsize=9)
    difference = values[0]-values[1]
    axes[1].hist(difference, bins=np.linspace(-25, 75, 26), color=COLORS[0], alpha=.75, edgecolor='white')
    axes[1].axvline(0, color='#3C4852', ls='--')
    axes[1].set(xlabel='Paired value difference: γ_train 0.99 minus 0', ylabel='Training seed pairs',
                title='How consistently does looking ahead help?')
    for ax in axes:
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('The final learned policies vary across seeds', fontsize=16)
    fig.text(.5, .01, 'All policies evaluated at γ = 0.99 · dots: individual seeds · diamonds: mean and 95% interval\n'
             'Paired seeds share regeneration draws during training; their actions and stocks evolve separately.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .95))
    save(fig, output, 'final_seed_values.png')


def plot_policies(data, output):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    policies = [data['oracle_policy'], data['final_policies'][0, 0], data['final_policies'][1, 0]]
    means = [data['oracle_policy'], *data['final_policies'].mean(axis=1)]
    for col, (policy, average, title) in enumerate(zip(policies, means,
                ['Supplied-model oracle', 'γ_train = 0.99 · seed 0', 'γ_train = 0 · seed 0'])):
        for (x, y), row in zip(data['states'], policy):
            choices = np.flatnonzero(row)
            for j, action in enumerate(choices):
                axes[0, col].add_patch(Rectangle((x-.5+j/len(choices), y-.5), 1/len(choices), 1,
                            facecolor=ACTION_COLORS[action], alpha=.3, edgecolor='white'))
            axes[0, col].text(x, y, '/'.join(ACTION_LETTERS[a] for a in choices), ha='center', va='center')
        rest = 100*average[:, 2].reshape(5, 5).T
        im = axes[1, col].imshow(rest, origin='lower', cmap='Blues', vmin=0, vmax=100,
                               extent=(-.5, 4.5, -.5, 4.5))
        for x in range(5):
            for y in range(5):
                axes[1, col].text(x, y, f'{rest[y,x]:.0f}', ha='center', va='center',
                                 color='white' if rest[y,x]>55 else '#213340')
        for row in range(2):
            stock_axes(axes[row, col])
        axes[0, col].set_title(title)
        axes[1, col].set_title('Oracle rest probability (%)' if col == 0 else 'Mean rest probability · all 100 seeds (%)')
    fig.suptitle('Learning a policy for each pair of food stocks', fontsize=16, y=.99)
    fig.legend([Patch(facecolor=c, alpha=.5) for c in ACTION_COLORS],
               ['A · harvest A', 'B · harvest B', 'R · rest'], ncol=3, frameon=False,
               loc='upper center', bbox_to_anchor=(.5, .963))
    fig.text(.5, .008, 'Top: seed 0 was fixed in advance; split cells show uniformly mixed greedy ties.\n'
             'Bottom: average of frozen policies across seeds, not an additional policy that was trained.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .92))
    save(fig, output, 'learned_policies.png')


def plot_collection(data, output):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    titles = ['Reward per decision', 'Stock in patch A', 'Stock in patch B',
              'Either patch depleted', 'Rest decisions', 'Unsuccessful harvests / all decisions']
    x = np.arange(1, 201)
    for k, (ax, title) in enumerate(zip(axes.flat, titles)):
        scale = 100 if k >= 3 else 1
        for m in range(2):
            band(ax, x, scale*data['rollout_metrics'][m, :, :, k], COLORS[m], LABELS[m])
        ax.set(title=title, xlabel='Training rollout (1,000 transitions each)', xlim=(1, 200))
        if k >= 3:
            ax.set_ylabel('Percent')
        ax.grid(alpha=.15)
    fig.suptitle('Experience collected while ε = 0.1 exploration remains active', fontsize=16, y=1.01)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=2, frameon=False,
               loc='upper center', bbox_to_anchor=(.5, .962))
    fig.text(.5, .005, 'Mean ± 1.96 SEM across training seeds; stocks measured before acting.\n'
             'Rollouts restart at (4,4), retaining Q. These training rewards differ from frozen-policy values.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .925))
    save(fig, output, 'training_experience.png')


def plot_all(data, output):
    style()
    plot_learning(data, output)
    plot_final(data, output)
    plot_policies(data, output)
    plot_collection(data, output)
