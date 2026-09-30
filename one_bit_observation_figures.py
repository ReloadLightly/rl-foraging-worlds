"""Plot-only views of the fixed one-bit follow-up; no fitting or selection."""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from figures import ACTION_COLORS, ACTION_LETTERS, save, stock_axes, style

COLORS = ['#C56949', '#207F75']
REP_COLORS = ['#5979A5', '#D69C38', '#207F75']
REPS = ['Full (A,B)', 'Total only', 'Total + bit']
COLLECTORS = ['Q-controlled data', 'Model-controlled data']


def interval(v):
    return 1.96*v.std(axis=-1, ddof=1)/np.sqrt(v.shape[-1])


def outcomes(data, output):
    values = data['actual_values'][..., 24]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    means, halves = values.mean(axis=-1), interval(values)
    axes[0].imshow(means, cmap='YlGnBu', vmin=0, vmax=data['oracle_values'][24], aspect='auto')
    for c in range(2):
        for r in range(3):
            axes[0].text(r, c, f'{means[c,r]:.2f}\n±{halves[c,r]:.2f}', ha='center', va='center',
                         color='white' if means[c,r]>55 else '#213340', fontsize=11)
    axes[0].set(title='True value: mean ± 95% half-width', xticks=range(3), xticklabels=REPS,
                yticks=range(2), yticklabels=['Q data', 'Model data'])
    total, bit = values[1, 1:]
    lo, hi = min(total.min(), bit.min())-2, max(total.max(), bit.max())+2
    axes[1].scatter(total, bit, s=28, color=COLORS[1], alpha=.6)
    axes[1].plot([lo,hi],[lo,hi],ls=':',color='#213340')
    axes[1].set(title='Primary: model-controlled history', xlabel='Total-only true value', ylabel='Total + bit true value',
                xlim=(lo,hi), ylim=(lo,hi), aspect='equal')
    contrasts = [values[1,2]-values[1,1], values[1,0]-values[1,2], values[0,2]-values[0,1], values[0,0]-values[0,2]]
    labels = ['Bit − total\n(model data)', 'Full − bit\n(model data)', 'Bit − total\n(Q data)', 'Full − bit\n(Q data)']
    for i,v in enumerate(contrasts):
        axes[2].errorbar(v.mean(), i, xerr=interval(v), fmt='o', capsize=4, color=COLORS[1 if i<2 else 0])
    axes[2].axvline(0,color='#213340',ls=':')
    axes[2].set(title='Paired value differences', yticks=range(4), yticklabels=labels, xlabel='Difference [95% interval]', ylim=(3.5,-.5))
    fig.suptitle('How much can one extra observation bit recover?',fontsize=17)
    fig.text(.5,.01,'Same 100 seeds and 250,000-observation histories. Full and total-only outcomes are reused from Experiment 8.\n'
             'The fixed 16-group representation is (A+B, int(A>B)); only this representation was fitted. No new collection.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.10,1,.94))
    save(fig,output,'values_and_comparisons.png')


def predictions(data, output):
    fig, axes = plt.subplots(2,2,figsize=(11,9))
    actual, predicted = data['actual_values'][...,24], data['predictions'][...,24]
    lo, hi = min(actual.min(),predicted.min())-3, max(actual.max(),predicted.max())+3
    for r,ax in enumerate(axes.flat):
        if r==3:
            break
        for c in range(2):
            ax.scatter(actual[c,r],predicted[c,r],s=24,color=COLORS[c],alpha=.6,label=COLLECTORS[c])
        ax.plot([lo,hi],[lo,hi],color='#213340',ls=':')
        ax.set(title=REPS[r],xlabel='Actual Vπ(4,4)',ylabel='Own-model predicted return',xlim=(lo,hi),ylim=(lo,hi),aspect='equal')
    axes[0,0].legend(frameon=False,fontsize=8)
    ax=axes[1,1]
    for r in range(3):
        errors=abs(predicted[:,r]-actual[:,r])
        ax.bar(np.arange(2)+(r-1)*.22,errors.mean(axis=-1),width=.22,yerr=interval(errors),capsize=3,
               color=REP_COLORS[r],alpha=.85,label=REPS[r])
    ax.set(title='Absolute prediction error',xticks=[0,1],xticklabels=['Q data','Model data'],ylabel='Mean |predicted − actual| [95% interval]')
    ax.legend(frameon=False,fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=.15)
    fig.suptitle('Value recovery and prediction accuracy are separate outcomes',fontsize=16)
    fig.text(.5,.01,'Predictions use each selected policy’s own empirical rewards and transitions (γ=0.99).\n'
             'True values evaluate the lifted policy in the original world; the one-bit observations are not assumed Markov.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.95))
    save(fig,output,'predictions.png')


def policies_coverage(data, output):
    fig, axes = plt.subplots(2,4,figsize=(16,8))
    for c in range(2):
        for r in range(3):
            ax=axes[c,r]
            for (x,y),row in zip(data['states'],data['policies'][c,r,0]):
                choices=np.flatnonzero(row)
                for j,a in enumerate(choices):
                    ax.add_patch(Rectangle((x-.5+j/len(choices),y-.5),1/len(choices),1,
                                 facecolor=ACTION_COLORS[a],alpha=.3,edgecolor='white'))
                ax.text(x,y,'/'.join(ACTION_LETTERS[a] for a in choices),ha='center',va='center',fontsize=9)
            for x,y in [(4,3),(3,4),(3,3),(2,4)]:
                ax.add_patch(Rectangle((x-.5,y-.5),1,1,fill=False,edgecolor='#7F2424' if x+y==7 else '#6B52A0',lw=2))
            stock_axes(ax)
            ax.set_title(f'{COLLECTORS[c]} · {REPS[r]}\nseed 0 true value {data["actual_values"][c,r,0,24]:.2f}',fontsize=10)
        ax=axes[c,3]
        for r,prefix in enumerate(['full','compressed','bit']):
            rates=np.array([100*(data[prefix+'_visits'][c]<t).mean(axis=(-2,-1)) for t in [1,10,100]])
            ax.bar(np.arange(3)+(r-1)*.23,rates.mean(axis=-1),width=.23,yerr=interval(rates),capsize=2,
                   color=REP_COLORS[r],alpha=.85,label=REPS[r])
        ax.set(title='Coverage · '+['Q data','model data'][c],xticks=range(3),xticklabels=['N=0','N<10','N<100'],
               ylabel='Percent of representation’s rows',ylim=(0,100))
        ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Seed 0: which action distinctions does the extra bit recover?',fontsize=16)
    fig.text(.5,.01,'A/B/R = harvest A / harvest B / rest. Red: (4,3) and (3,4), now separated. Purple: (3,3) and (2,4), still merged.\n'
             'Coverage denominators: full 75, total-only 27, total + bit 48 observation-action rows. Seed 0 was fixed in advance.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.075,1,.95))
    save(fig,output,'policies_and_coverage.png')


def mechanism(data, diagnostic, output):
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for pair in range(2):
        ix=np.array([2*pair,2*pair+1])
        p=diagnostic['true_next_observation_probabilities'][ix].reshape(6,16)
        columns=np.flatnonzero(p.sum(axis=0)>0)
        matrix=p[:,columns]
        ax=axes[pair,0]
        ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        labels=[f'{tuple(diagnostic["states"][i])} {ACTION_LETTERS[a]}' for i in ix for a in range(3)]
        for i in range(6):
            for j in range(len(columns)):
                ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>.55 else '#213340')
        ax.set(title=['Separated pair: true transitions','Still-merged pair: true transitions'][pair],
               yticks=range(6),yticklabels=labels,xticks=range(len(columns)),
               xticklabels=[str(tuple(diagnostic['groups'][k])) for k in columns],xlabel='Next observation (total, bit)')
        ax.tick_params(axis='x',labelrotation=40)
        ax=axes[pair,1]
        for j,i in enumerate(ix):
            q=diagnostic['fitted_action_values'][1,0,i]
            ax.bar(np.arange(3)+(j-.5)*.32,q-q.max(),width=.32,color=['#5979A5','#D69C38'][j],
                   alpha=.85,label=str(tuple(diagnostic['states'][i])))
        ax.set(title='Seed 0 · model-data fitted action values',xticks=range(3),xticklabels=['A','B','Rest'],
               ylabel='Predicted q(a) − max q (zero is selected)')
        ax.legend(frameon=False,fontsize=9)
        ax.grid(axis='y',alpha=.15)
        ax=axes[pair,2]
        names=[]
        for c in range(2):
            for j,i in enumerate(ix):
                x=2*c+j
                probs=diagnostic['selected_actions'][c,:,i].mean(axis=0)
                bottom=0
                for a in range(3):
                    ax.bar(x,probs[a],bottom=bottom,color=ACTION_COLORS[a],alpha=.85,label=ACTION_LETTERS[a] if x==0 else None)
                    bottom+=probs[a]
                optimal='/'.join(ACTION_LETTERS[a] for a in np.flatnonzero(diagnostic['true_optimal_actions'][i]))
                ax.text(x,1.025,'oracle '+optimal,ha='center',fontsize=8)
                names.append(['Q','Model'][c]+'\n'+str(tuple(diagnostic['states'][i])))
        ax.set(title='New policy choices across 100 seeds',xticks=range(4),xticklabels=names,ylabel='Mean policy action probability',ylim=(0,1.15))
        ax.legend(frameon=False,fontsize=8,loc='lower left')
    fig.suptitle('Separating one pair does not remove every hidden transition difference',fontsize=16)
    fig.text(.5,.01,'Heatmaps show all nonzero successor groups for each pair; omitted groups have probability zero. True dynamics and oracle actions are diagnostics only.\n'
             'Action values are predictions in the fitted one-bit model. Full fitted distributions, values and state-level counts are saved for every seed.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.085,1,.95))
    save(fig,output,'mechanism.png')


def plot_all(data, diagnostic, output):
    style()
    outcomes(data,output)
    predictions(data,output)
    policies_coverage(data,output)
    mechanism(data,diagnostic,output)
