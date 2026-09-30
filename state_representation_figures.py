"""Saved-data figures: experience, representation, and state aliasing."""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from figures import ACTION_COLORS, ACTION_LETTERS, save, stock_axes, style

COLORS = ['#C56949', '#207F75']
COLLECTORS = ['Q-controlled data', 'Model-controlled data']
REPRESENTATIONS = ['Full (A,B)', 'Compressed A+B']


def outcomes(data, output):
    values = data['actual_values'][..., 24]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    means = values.mean(axis=-1)
    half = 1.96*values.std(axis=-1, ddof=1)/10
    axes[0].imshow(means, cmap='YlGnBu', vmin=0, vmax=data['oracle_values'][24], aspect='auto')
    for c in range(2):
        for r in range(2):
            axes[0].text(r, c, f'{means[c,r]:.2f}\n[{means[c,r]-half[c,r]:.2f}, {means[c,r]+half[c,r]:.2f}]',
                         ha='center', va='center', color='white' if means[c,r]>55 else '#213340', fontsize=11)
    axes[0].set(title='True value: mean [95% interval]', xticks=range(2), xticklabels=REPRESENTATIONS,
                yticks=range(2), yticklabels=['Q data', 'Model data'], xlabel='Representation')
    full, compressed = values[1]
    lo, hi = min(full.min(), compressed.min())-2, max(full.max(), compressed.max())+2
    axes[1].scatter(compressed, full, s=28, color=COLORS[1], alpha=.55)
    axes[1].plot([lo, hi], [lo, hi], color='#213340', ls=':')
    axes[1].set(title='Primary: same model-controlled history', xlabel='Compressed-policy true value', ylabel='Full-policy true value',
                xlim=(lo, hi), ylim=(lo, hi), aspect='equal')
    keys = ['primary_full_minus_compressed', 'full_experience_benefit', 'compressed_experience_benefit', 'experience_benefit_difference']
    labels = ['Full − compressed\n(model data)', 'Model − Q data\n(full)', 'Model − Q data\n(compressed)', 'Full benefit −\ncompressed benefit']
    for i,key in enumerate(keys):
        v = data[key]
        axes[2].errorbar(v.mean(), i, xerr=1.96*v.std(ddof=1)/10, fmt='o', color=COLORS[1] if i==0 else '#7960A6', capsize=4)
    axes[2].axvline(0, color='#213340', ls=':')
    axes[2].set(title='Paired comparisons across 100 seeds', yticks=range(4), yticklabels=labels, xlabel='True-value difference', ylim=(3.5, -.5))
    for ax in axes[1:]:
        ax.grid(alpha=.15)
    fig.suptitle('Can better experience compensate for an incomplete state?', fontsize=17)
    fig.text(.5, .01, 'All policies use 250,000 archived observations per seed and the same unpenalized planner (γ=0.99). No new collection.\n'
        'Actual values are exact in the original 25-state environment. Intervals use paired training seeds; all outcomes are retained.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .94))
    save(fig, output, 'values_and_comparisons.png')


def predictions(data, output):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    actual, predicted = data['actual_values'][..., 24], data['predictions'][..., 24]
    lo, hi = min(actual.min(), predicted.min())-3, max(actual.max(), predicted.max())+3
    for r in range(2):
        for c in range(2):
            axes[r].scatter(actual[c,r], predicted[c,r], s=25, color=COLORS[c], alpha=.6, label=COLLECTORS[c])
        axes[r].plot([lo,hi], [lo,hi], color='#213340', ls=':')
        axes[r].set(title=REPRESENTATIONS[r], xlabel='Actual Vπ(4,4)', ylabel='Own-model predicted return',
                    xlim=(lo,hi), ylim=(lo,hi), aspect='equal')
    for c in range(2):
        for r in range(2):
            e = abs(predicted[c,r]-actual[c,r])
            axes[2].errorbar(c+(r-.5)*.16, e.mean(), yerr=1.96*e.std(ddof=1)/10,
                             fmt='o' if r==0 else 's', color=COLORS[c], capsize=4,
                             label=REPRESENTATIONS[r] if c==0 else None)
    axes[0].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=9)
    axes[2].set(title='Mean absolute prediction error', xticks=[0,1], xticklabels=['Q data','Model data'], ylabel='|Predicted − actual|')
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle('A fitted total-stock model can imagine futures its lifted policy does not produce', fontsize=15)
    fig.text(.5, .01, 'Full predictions start at (4,4); compressed predictions start at total stock 8. Rewards are empirical observed means.\n'
        'Actual values evaluate the selected lifted policy; the compressed observations are not assumed to form an MDP.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.10,1,.94))
    save(fig, output, 'predicted_vs_actual.png')


def policies_coverage(data, output):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for c in range(2):
        for r in range(2):
            ax = axes[c,r]
            for (x,y), row in zip(data['states'],data['policies'][c,r,0]):
                choices = np.flatnonzero(row)
                for j,a in enumerate(choices):
                    ax.add_patch(Rectangle((x-.5+j/len(choices),y-.5),1/len(choices),1,
                        facecolor=ACTION_COLORS[a],alpha=.3,edgecolor='white'))
                ax.text(x,y,'/'.join(ACTION_LETTERS[a] for a in choices),ha='center',va='center',fontsize=9)
            for x,y in [(4,3),(3,4)]:
                ax.add_patch(Rectangle((x-.5,y-.5),1,1,fill=False,edgecolor='#7F2424',lw=2))
            stock_axes(ax)
            ax.set_title(f'{COLLECTORS[c]} · {REPRESENTATIONS[r]}\nseed 0 true value {data["actual_values"][c,r,0,24]:.2f}',fontsize=10)
        for r,prefix in enumerate(['full','compressed']):
            for i,t in enumerate([1,10,100]):
                rates=100*(data[prefix+'_visits'][c]<t).mean(axis=(-2,-1))
                axes[c,2].bar(i+(r-.5)*.3,rates.mean(),width=.3,color=['#5979A5','#D69C38'][r],alpha=.8,
                    yerr=1.96*rates.std(ddof=1)/10,capsize=3,label=REPRESENTATIONS[r] if i==0 else None)
        axes[c,2].set(title=COLLECTORS[c]+' · coverage',xticks=range(3),xticklabels=['N=0','N<10','N<100'],
                       ylabel='Percent of representation’s state-action rows',ylim=(0,100))
        axes[c,2].legend(frameon=False,fontsize=8)
        axes[c,2].grid(axis='y',alpha=.15)
    fig.suptitle('Compression shares decisions and counts across hidden patch configurations',fontsize=16)
    fig.text(.5,.01,'A/B/R = harvest A / harvest B / rest. Red outlines: preselected (4,3) and (3,4), both total seven. Seed 0 fixed before analysis.\n'
        'Coverage denominators differ: full 75 rows, compressed 27. Pooling reduces sparse rows but does not restore hidden information.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.075,1,.95))
    save(fig,output,'policies_and_coverage.png')


def aliasing(data, mechanism, output):
    fig, axes = plt.subplots(2,3,figsize=(14,8))
    states = ['(4,3)','(3,4)']
    for a,ax in enumerate(axes[0]):
        for s in range(2):
            ax.bar(np.arange(6,9)+(s-.5)*.3,mechanism['true_next_observation_probabilities'][s,a,6:9],
                   width=.3,color=['#5979A5','#D69C38'][s],label=states[s],alpha=.85)
        ax.set(title=f'{data["action_names"][a]} · reward {mechanism["true_rewards"][0,a]:g} in both states',
               xticks=[6,7,8],xlabel='Next total stock',ylabel='True probability',ylim=(0,1))
        ax.grid(axis='y',alpha=.15)
    axes[0,0].legend(frameon=False)
    for c in range(2):
        ax=axes[1,c]
        for a in range(3):
            observed=mechanism['observed_rows'][c,:,a]
            w=mechanism['conditional_state_weights'][c,observed,0,a]
            mean,half=w.mean(),1.96*w.std(ddof=1)/np.sqrt(len(w))
            ax.bar(a,mean,color='#5979A5',alpha=.85,label='(4,3)' if a==0 else None)
            ax.bar(a,1-mean,bottom=mean,color='#D69C38',alpha=.85,label='(3,4)' if a==0 else None)
            ax.errorbar(a,mean,yerr=half,color='#213340',capsize=4)
            ax.text(a,mean/2,f'{mean:.1%}',ha='center',va='center',color='white',fontsize=9)
        ax.set(title=COLLECTORS[c]+' · hidden-state weights',xticks=range(3),xticklabels=['Harvest A','Harvest B','Rest'],
               ylabel='Mean P̂(full state | total 7, action)',ylim=(0,1))
    shares=mechanism['selected_compressed_actions'].mean(axis=1)
    for c in range(2):
        bottom=0
        for a in range(3):
            axes[1,2].bar(c,shares[c,a],bottom=bottom,color=ACTION_COLORS[a],alpha=.85,
                          label=str(data['action_names'][a]) if c==0 else None)
            bottom+=shares[c,a]
    axes[1,2].set(title='Compressed choices at total seven',xticks=[0,1],xticklabels=['Q data','Model data'],
                   ylabel='Mean policy action probability',ylim=(0,1))
    axes[1,2].legend(frameon=False,fontsize=8)
    fig.suptitle('Total stock seven hides different transition laws and optimal actions',fontsize=16)
    optimal=['/'.join(ACTION_LETTERS[a] for a in np.flatnonzero(row)) for row in mechanism['true_optimal_actions']]
    fig.text(.5,.01,f'True full-state optimum: (4,3) → {optimal[0]}; (3,4) → {optimal[1]}. Immediate rewards agree here; next-total distributions differ.\n'
        'Weights are action-specific visit fractions, averaged within seed then across seeds; error bars show 95% intervals. True dynamics are diagnostics only.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.075,1,.95))
    save(fig,output,'merged_states_mechanism.png')


def plot_all(data, mechanism, output):
    style()
    outcomes(data,output)
    predictions(data,output)
    policies_coverage(data,output)
    aliasing(data,mechanism,output)
