# Learning to leave something for tomorrow

**When does preserving resources beat taking the largest immediate reward?**

This small NumPy project explores Sutton and Barto, second edition, **Chapter 3:
Finite Markov Decision Processes**. It follows our
[Chapter 2 bandit experiments](https://github.com/ReloadLightly/rl-changing-worlds).
Here an action changes the resources available for future decisions. Experiment 1
plans with a **supplied, exact model**. Experiment 2 now learns action values from
observed transitions using **Q-learning**, a preview of Section 6.5, in the same world.

**Latest result:** after 200,000 transitions per seed, Q-learning with training
discount 0.99 achieves mean frozen-policy value **72.10 [70.86, 73.34]**, versus
**25.58 [25.28, 25.88]** with training discount 0. Both policies are evaluated at
discount 0.99. Only **7/100** long-horizon seeds reach 90% of the oracle value:
learning helps substantially, but does not reliably recover the optimal policy.
[Jump to the learning experiment](#experiment-2-learning-from-transitions).

| Chapter connection | What we have implemented |
| --- | --- |
| Chapter 2, preceding repository | Action values, exploration, gradient policies, and contextual bandits. |
| Chapter 3, this world | Observed states, action-dependent transitions, continuing discounted returns, and Bellman equations. |
| Chapter 4 preview, experiment 1 | Exact policy evaluation and value iteration using a supplied model. |
| Section 6.5 preview, experiment 2 | Q-learning from sampled transitions; the exact model is reserved for evaluation. |

**Experiment 1 result:** myopic harvesting is optimal at discounts 0 and 0.5.
At 0.9 the optimal policy changes which patch it harvests. At 0.99 it also
waits to preserve productive stocks: its expected discounted return from
`(4,4)` is **90.2235**, versus **23.7206** for myopic harvesting. Resting is
strictly best in nine states, despite paying no immediate reward.

![Optimal policies over the two food stocks](results/first_world/optimal_policies.png)

This is **our original foraging MDP**, not a reproduction of a textbook
environment or an empirically calibrated ecological model. The environment
was fixed before the experiment and was not adjusted to obtain these results.

## Run it

From this repository, using Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python checks.py
python run_experiment.py
```

The final command solves the four discount settings and simulates all three
policies at the requested scale. It overwrites `results/first_world/`;
`--output PATH` selects a different directory. To regenerate the figures
without rerunning planning or simulation:

```bash
python run_experiment.py --plot-only
```

The recorded run used Python **3.10.12**, NumPy **1.26.4**, and Matplotlib
**3.10.9** on a WSL CPU. It took **5.15 seconds** including saving and plotting:
**0.105 s** for planning and **1.606 s** for **6 million simulated decisions**.
No GPU or paid API is used. Runtime depends on the machine.

Open this project in a separate WSL VS Code window:

```bash
code -n /home/roland/projects/rl-foraging-worlds
```

## The world: harvest first, then regenerate

The observed state is $s=(n_A,n_B)$, with each stock in $\{0,1,2,3,4\}$:
**25 states**. Every trajectory starts at **$(4,4)$**. All three actions are
available in every state:

| Action | Stock change before regeneration | Immediate reward |
| --- | --- | --- |
| Harvest A | Remove one unit from A if nonempty | 1 if successful; otherwise 0 |
| Harvest B | Remove one unit from B if nonempty | 1.5 if successful; otherwise 0 |
| Rest | Remove nothing | 0 |

There are no extra bonuses, costs, or penalties. After harvesting, let the
remaining stock in patch $i$ be $m_i$. **Both patches then regenerate
independently**, including the patch just harvested:

$$
p_i(m_i)=g_i\left(0.1+0.9\frac{m_i}{4}\right),\quad m_i<4,
\qquad g_A=0.3,\quad g_B=0.6.
$$

With this probability, add exactly one unit; otherwise add none. A full patch
stays at 4. The implementation uses probability zero for adding to a full
patch, so stocks never exceed capacity.

The **event order** is:

1. Observe $(n_A,n_B)$ and choose an action.
2. Remove a unit if possible and receive its reward.
3. Compute regeneration probabilities from the **remaining** stocks.
4. Draw the two independent regeneration outcomes to produce the next state.

For example, harvesting A at stock 1 leaves stock 0, so its regeneration
chance that step is **0.03**, not the **0.0975** chance it would have retained
at stock 1. An empty harvest pays zero even if regeneration adds a unit
afterward. At post-harvest stock 3, regeneration probabilities are **0.2325**
for A and **0.465** for B; at stock 0 they are **0.03** and **0.06**.

This is a stylized rule: depleted patches recover slowly, retaining stock
supports faster regeneration, and recovery from zero remains possible. It
does not imply that actual food patches follow this formula.

The process **continues indefinitely**. Step 1,000 is an observation cutoff,
not a terminal state, death, or reset. The state after that step remains an
ordinary stock pair that could generate further rewards.

## Chapter 3: why this is an MDP

The state has the **Markov property** under our assumptions: given current
stocks and the chosen action, the distribution of the next stocks and reward
does not require earlier history. Independent regeneration draws and fixed
parameters make the stock pair sufficient. Hidden weather, patch age, or
other unmodeled dependencies would change that claim.

The exact model stores $P(s'\mid s,a)$ in a **25 × 3 × 25** array and expected
immediate rewards $r(s,a)$ in a **25 × 3** array. Here immediate rewards are
deterministic given state and action. For each state/action pair we enumerate
the four possible combinations of patch regeneration and multiply their
independent probabilities. Zero-probability outcomes at capacity are omitted.

A **policy** $\pi(a\mid s)$ specifies action probabilities at a state. Its
discounted return is

$$
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots,
\qquad 0\leq\gamma<1.
$$

$V_\pi(s)$ is the expected return starting in state $s$ and following that
policy. $Q_\pi(s,a)$ is the expected return after taking action $a$ first and
then following the policy. Given a continuation value $V$, the action values
are

$$
Q(s,a)=r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s').
$$

The first term values today's harvest. The second values the opportunities
left after that harvest and regeneration. Averaging over the policy gives
the **Bellman expectation equation**:

$$
V_\pi=r_\pi+\gamma P_\pi V_\pi,\qquad
V_\pi=(I-\gamma P_\pi)^{-1}r_\pi.
$$

The code uses `numpy.linalg.solve`, not an explicit matrix inverse.
$P_\pi(s,s')=\sum_a\pi(a\mid s)P(s'\mid s,a)$ and
$r_\pi(s)=\sum_a\pi(a\mid s)r(s,a)$.

The **Bellman optimality equation** replaces the policy average with a maximum:

$$
V_*(s)=\max_a\left[r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V_*(s')\right].
$$

Our short value-iteration solver is a **Chapter 4 preview** used to explore
this Chapter 3 equation. It repeatedly applies the right side, extracts the
policy, and evaluates that policy with the linear solve. The largest recorded
optimality residual is **$1.43\times10^{-14}$**. These values are numerical
solutions of the supplied model, not estimates learned from the trajectories.

In the preceding bandits, actions affected rewards and observations but did
not control the next situation. Here harvesting changes both next stock and
regeneration probability. Choosing the largest current payoff can damage
future returns. That action-dependent transition is the central new ingredient.

## Policies, ties, and discount objectives

We compare uniform random actions, a **myopic** policy maximizing $r(s,a)$,
and the **optimal discounted policy** computed separately at each gamma.
The myopic policy takes B whenever B is nonempty, otherwise A if possible.

Policy extraction mixes uniformly over actions within an **absolute tolerance
of $10^{-10}$** of the largest action value; relative tolerance is zero.
Split cells in the policy maps show those ties. At `(0,0)`, all three actions
have identical rewards and transitions, so both myopic and optimal policies
mix all three. At other empty-patch states, harvesting that empty patch can
be equivalent to resting. Such ties are not counted as **strictly** optimal
rest: we require $Q_*(s,\mathrm{rest})$ to exceed **both** harvest values by
more than the tolerance.

Larger gamma weights later rewards more heavily. The sums of discount weights
$1/(1-\gamma)$ are 1, 2, 10, and 100 for our settings; these are not hard
episode lengths. **Compare policies within the same gamma.** Raw values at
different discounts represent different objectives and cannot themselves
establish that one planning horizon is better.

## Exact results: when preferences change

Expected **infinite-horizon discounted return from $(4,4)$**, with each policy
evaluated at the gamma in its row:

| Gamma | Uniform random | Myopic | Optimal at this gamma | Optimal − myopic | Strict rest states |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8333 | 1.5000 | 1.5000 | 0.0000 | 0 |
| 0.5 | 1.6644 | 2.9684 | 2.9684 | 0.0000 | 0 |
| 0.9 | 7.2331 | 9.3041 | 10.8164 | 1.5123 | 0 |
| 0.99 | 34.2907 | 23.7206 | 90.2235 | 66.5029 | 9 |

These are deterministic calculations under the specified model; sampling
confidence intervals do not apply to them.

At **gamma 0 and 0.5**, the optimal and myopic policies agree in all 25 states.
Preservation has no advantage for these discounted objectives.

At **gamma 0.9**, action sets change in **15 states**: whenever B has stock
1–3, the policy harvests A if available, sacrificing the larger immediate
payoff to preserve B. If A is empty it mixes empty-A harvest and rest. There
are no states where rest strictly beats both harvest actions. Waiting can
be optimal through a tie without being uniquely best.

At **gamma 0.99**, action sets differ from myopic in **18 states** and from
gamma 0.9 in **12 states**. Its rule can be read directly from the map:

- Harvest B when B is full.
- Otherwise harvest A when A is full.
- Otherwise wait; at empty-patch states, empty harvests may tie with rest.

Rest is strictly best at exactly **$\{1,2,3\}\times\{1,2,3\}$**. At $(3,3)$:

| First action, then follow the gamma 0.99 optimal policy | Immediate reward | $Q_*(3,3,a)$ |
| --- | ---: | ---: |
| Harvest A | 1 | 87.9964 |
| Harvest B | 1.5 | 87.6621 |
| Rest | 0 | **88.0949** |

An unrewarded rest action has positive value because harvesting remains
possible later. This is different from **resting forever**, whose value is
zero in every state. Positive value alone also does not make rest optimal:
at gamma 0.5, resting at $(3,3)$ is worth 1.4674, but harvesting B is worth
2.9054. At $(4,4)$, harvesting B remains optimal at every tested discount.

![Value advantage over myopic harvesting in every state](results/first_world/value_advantage.png)

The planning advantage is zero in every state at gamma 0 and 0.5. At 0.9 it
ranges from **0.2380 to 1.8264**; at 0.99 from **54.9056 to 67.7655**. All
25 differences per gamma are saved. A state's action can agree with myopic
while its value is higher because the policies behave differently later.

## Simulations: finite-window returns and resources

We simulate **2,000 independent trajectories per policy**, each for **1,000
decisions** from `(4,4)`, using random, myopic, and the fixed **gamma 0.99
optimal policy**. Discounted simulation returns also use **gamma 0.99**.

Each paired trajectory shares the same two independent uniform regeneration
draws per step across policies. Each policy compares those draws with the
probabilities from **its own post-harvest stocks**. Thus opportunities are
paired while realized growth and stocks can differ. Policy sampling has
separate stable random streams, independent of regeneration.

Intervals below are **mean ± 1.96 SEM across independent trajectories**.
Paired comparisons subtract corresponding trajectory outcomes first. Plot
bands use within-trajectory bin summaries before calculating SEMs; steps
within one trajectory are not treated as independent samples. Intervals are
pointwise and unadjusted for multiple comparisons.

| Policy | Simulated discounted return (95% CI) | Exact 1,000-step expectation | Exact infinite-horizon value |
| --- | ---: | ---: | ---: |
| Uniform random | 34.4244 [34.1210, 34.7278] | 34.2898 | 34.2907 |
| Myopic | 23.8809 [23.7306, 24.0312] | 23.7201 | 23.7206 |
| Planned, gamma 0.99 | 90.2000 [89.9665, 90.4336] | 90.2197 | 90.2235 |

The paired planned-minus-myopic difference is **66.3191 [66.0876, 66.5506]**.
The myopic sample mean lies **2.10 SEM** above its exact truncated expectation,
just outside its marginal 95% interval. An individual 95% interval need not
cover its expectation; the exact model supplies a useful independent check.

The simulated return is $G_0^{(T)}=\sum_{t=0}^{T-1}\gamma^tR_{t+1}$.
Since rewards are between 0 and 1.5, the omitted tail is bounded on **every
possible trajectory** by

$$
0\leq G_0-G_0^{(1000)}\leq
\frac{1.5(0.99)^{1000}}{1-0.99}=\mathbf{0.006476}.
$$

The exact expected omitted tails are **0.000912** (random), **0.000519**
(myopic), and **0.003846** (planned). The finite-window calculation truncates
a sum of rewards; it does not introduce a terminal state or optimize a policy
that knows a terminal deadline is coming.

### Harvest and resource outcomes

All resource statistics use **pre-action stocks** at decisions 0–999.
Depletion means stock zero. Rest and unsuccessful-harvest percentages use
**all decisions** as their denominator; conditional failure rates among
harvest attempts are saved separately.

| Outcome | Uniform random | Myopic | Planned, gamma 0.99 |
| --- | ---: | ---: | ---: |
| Cumulative harvested reward (95% CI) | 228.16 [226.57, 229.76] | 132.73 [132.17, 133.29] | **891.60 [890.52, 892.67]** |
| Mean reward per decision | 0.2282 | 0.1327 | 0.8916 |
| Mean stock A | 0.2117 | 0.0676 | 3.3634 |
| Mean stock B | 0.9209 | 0.0754 | 3.4651 |
| Either patch depleted | 94.79% | 99.21% | 0% |
| Both patches depleted | 55.20% | 90.00% | 0% |
| Rest decisions | 33.36% | 30.00% | 34.10% |
| Unsuccessful harvest decisions | 49.99% | 60.00% | 0% |

All outcomes have trajectory-level uncertainty in the saved summary. Planned
minus myopic cumulative reward is **758.87 [757.80, 759.94]**. Resource stocks
and average reward are **secondary outcomes**: the discounted solver does not
automatically optimize either one as a separate objective.

![Resource stocks, harvests, depletion, and rest over time](results/first_world/resource_and_harvest_curves.png)

The mechanism is visible in the stock curves. Starting at `(4,4)`, the planned
policy harvests only a full patch, leaving at least 3 units. Regeneration
cannot reduce that stock, so **both stocks stay in $\{3,4\}$ under this policy**.
Its zero depletion and failure frequencies follow from that invariant, not
only from a lucky finite sample. Myopic harvesting consumes newly recovered
units, leaving patches near zero where growth is slow.

Rest frequency alone is misleading. Random and planned policies rest at
similar overall rates, but achieve very different harvests. Myopic resting
occurs only at `(0,0)`, where all immediate rewards tie; two thirds of those
zero-payoff decisions are recorded as empty harvests instead. The planned
policy waits **before** stocks become depleted. When and where it waits
matter more than simply counting rest actions.

Random actions beat myopic harvesting at gamma 0.99, despite choosing actions
without a plan. This does not make random behavior generally superior:
myopic has higher exact value at the three shorter discounts. The optimal
policy also keeps taking immediate rewards when stocks justify doing so.

![Preselected trajectory 0, stocks and actions](results/first_world/trajectory_0.png)

Trajectory **0** and its **first 120 decisions** were selected before examining
outcomes. The full trace is saved. Red crosses show empty harvests; the stock
and action panels show how the same regeneration draws lead to different
futures. This single trajectory illustrates the mechanism, while the
2,000-trajectory summaries provide the population evidence.

## Code, results, and limits

Start with [foraging.py](foraging.py): `harvest` and `step` specify event order;
`exact_model` enumerates transitions; `policy_evaluation` solves the Bellman
system; `value_iteration` supplies the optimal reference. The explicit time
loop in [run_experiment.py](run_experiment.py) vectorizes across trajectories.
[figures.py](figures.py) only needs saved arrays. [checks.py](checks.py) checks
transition normalization, event order, recovery from zero, ties, and Bellman
calculations; it is deliberately short.

The approximately **1.0 MiB** of outputs in [results/first_world/](results/first_world/)
include:

| File | Contents |
| --- | --- |
| `model.npz` | State/action labels, exact transitions and rewards, policies and values for all four gammas, optimal action values, iteration counts and residuals. |
| `planning_summary.json` | Initial values, advantages at all 25 states, action sets, strict-rest states, and changes between discounts. |
| `simulation.npz` | `task_outcomes` indexed by policy × trajectory × outcome, labeled axes, curve means/SEMs, final stocks, and trajectory-0 states/actions/rewards/shared regeneration draws. |
| `simulation_summary.json` | Means, SEMs, intervals, trajectory variability, paired differences, exact truncated expectations, and tail bounds. |
| `manifest.json` | World parameters, configuration, seeds, library versions, source hashes, event order, tie rule, measurement definitions, and runtime. |
| Four PNG figures | Policy maps, value advantages, resource/harvest curves, and trajectory 0. |

Open archives with `numpy.load(path, allow_pickle=False)`. There is no full
population trajectory-by-time-by-state cube. Source hashes identify the code
used for the run; the regeneration-stream hash records the shared exogenous
draws. Seeds use NumPy `SeedSequence` with explicit stable stream IDs.

This model has fixed growth parameters, full observation, no movement cost,
no resource spoilage, no seasonal change, and no other foragers. Its exact
policy thresholds are consequences of these assumptions and rewards, not
ecological recommendations. Parameter uncertainty is not included in the
simulation intervals. In experiment 1, no transition model is learned and
simulated experience does not update any policy. Experiment 2 below learns Q
from experience, while still learning no transition model. The preceding
Chapter 2 repository remains intact.

**Question raised by experiment 1:** does planning still help when the supplied regeneration
rates are wrong? Keep this world fixed, plan under misspecified growth rates,
and evaluate those policies in the true world. That would separate the value
of looking ahead from the cost of trusting an inaccurate model. That proposed
model-misspecification study has not been run; our next completed study instead
adds experience-based learning to the unchanged world.

## Experiment 2: learning from transitions

**Can an agent discover the value of preserving resources without being given
the transition model?** We compare Q-learning with $\gamma_{train}=0.99$ and
$\gamma_{train}=0$. All stocks, rewards, regeneration rules, and event order
are exactly those of experiment 1. No reward shaping or parameter tuning was
used. The first experiment's source world and saved results were preserved.

### From Bellman equations to experience

The learner sees only $(S_t,A_t,R_{t+1},S_{t+1})$. It maintains a table with
one estimate per state/action pair and applies

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\left[
R_{t+1}+\gamma_{train}\max_a Q(S_{t+1},a)-Q(S_t,A_t)\right].
$$

The target combines an observed payoff with an estimate of later opportunity.
This is **bootstrapping**: one estimate helps update another. Q-learning is
**off-policy** because its target uses a greedy future action even while its
behavior explores. With $\gamma_{train}=0$, the future term disappears; the
learner only estimates immediate rewards. This follows Sutton and Barto's
[Section 6.5, Q-learning](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
(linked author-text copy is a second-edition draft).

**Alpha** controls how much each new target changes Q. **Epsilon** controls
how often an action is selected uniformly for exploration. **Gamma** controls
how much future reward matters in the target. They perform different jobs.
Unlike the first experiment's planner, the learner neither enumerates possible
successors nor knows their probabilities. It has no growth parameters,
counterfactual rewards, oracle values, or other seeds' observations.

### Fixed protocol and frozen-policy evaluation

| Setting | Value |
| --- | --- |
| Independent training seeds | 100 per learner |
| Transitions | 200,000 per learner per seed; **40 million total** |
| Collection rollouts | 200 rollouts of 1,000 transitions, each starting at `(4,4)` |
| Q initialization / learning rate | All zeros / constant $\alpha=0.1$ |
| Training exploration | Constant $\epsilon=0.1$, uniform among all three actions on exploration steps |
| Training discounts | 0.99 and 0 |
| Checkpoints | Before training, then every 10,000 transitions, fixed in advance |
| Primary evaluation | Frozen **final** Q tables; no updates; $\epsilon=0$; exact value at $\gamma_{eval}=0.99$ |
| Greedy ties | Uniform mixing within absolute tolerance $10^{-10}$, relative tolerance zero |

At every rollout cutoff, **the final real transition still bootstraps from its
actual next stock state**. Only then does collection restart at `(4,4)`,
retaining Q and all visit counts. The reset creates no transition, reward, or
terminal state. These rollouts are a data-collection procedure for the same
continuing MDP; they change which states are sampled, not its transition law.

Within a training seed, the two methods share exogenous uniform regeneration
draws. Each uses probabilities from its own post-harvest stocks. Each learner
has a separate stable action-selection stream, and seeds are independent.
We vectorize across seeds and retain an explicit time loop. Draws are generated
one rollout at a time rather than stored for every transition.

The evaluator extracts each frozen greedy policy from Q and solves
$(I-0.99P_\pi)V_\pi=r_\pi$ using the already saved exact model. The reported
value is **$V_\pi(4,4)$, not $\max_a Q(4,4,a)$**, and is an infinite-horizon
expectation. No evaluation rewards train the learner. No finite-horizon
Monte Carlo estimate or tail correction is needed for this linear solve.
Checkpoint evaluation has no effect on training or stopping. We report the
scheduled **200,000-transition endpoint**, without selecting the best earlier
checkpoint. A secondary evaluation freezes the same tables while retaining
$\epsilon=0.1$ to show behavior with continued exploration.

### Final outcomes: a substantial but incomplete improvement

All rows below use the **same evaluation discount, 0.99**. Mean intervals are
mean ± 1.96 SEM over **100 independent training seeds**; a seed, not a transition
or checkpoint, is the sampling unit. Exact evaluation removes rollout sampling
noise, but learned policies still vary with training experience.

| Training discount | Mean final frozen-policy value (95% CI) | Seed SD | Mean share of oracle | Seeds ≥90% of oracle (Wilson 95% CI) |
| --- | ---: | ---: | ---: | ---: |
| 0.99 | **72.1011 [70.8602, 73.3420]** | 6.3311 | 79.91% | **7/100 = 7% [3.43%, 13.75%]** |
| 0 | **25.5764 [25.2768, 25.8759]** | 1.5283 | 28.35% | **0/100 = 0% [0%, 3.70%]** |

The supplied-model oracle is **90.2235**; 90% of it is **81.2011**. The myopic
reference is **23.7206**, and uniform random actions yield **34.2907**. These
reference values come from the saved first experiment and were not rerun.

The paired value improvement for training discount 0.99 over 0 is
**46.5247 [45.2247, 47.8247]**. Every one of the 100 matched seed pairs improved,
but high performance was not reliable: **93/100** long-horizon seeds fell below
the 90% threshold. Their final values range from **52.6203 to 90.2235**; the
10th, 50th, and 90th percentiles are **64.0113, 73.7827, and 79.0947**. The best
seed is included in the range, not used as the reported policy or as a reason
to change the protocol.

![Frozen-policy values during training, with and without exploration](results/q_learning/learning_values.png)

The learning curve shows gradual improvement for discount 0.99 and a plateau
well below the oracle. The small late fluctuations are retained. With frozen
$\epsilon=0.1$ policies, mean final value is **67.9862 [66.9521, 69.0204]** for
the long-horizon learner and **25.8744 [25.5932, 26.1556]** for the immediate
learner. Continuing random exploration can damage productive stocks, although
for the poorer policy it can also interrupt damaging repeated harvesting.
These are evaluations of the same tables, not another training experiment.

![Individual seed outcomes and paired differences](results/q_learning/final_seed_values.png)

The discount-zero learner is **not exactly the known myopic policy** after a
finite run: some state/action pairs have little or no experience. Its final
value differs from 23.7206 accordingly. Immediate rewards in this world are
deterministic, so its estimate after $N(s,a)$ visits satisfies
$Q(s,a)=r(s,a)[1-0.9^{N(s,a)}]$. We checked that identity against the saved
tables. Underestimating rarely tried harvest actions can change which action
looks best; an incomplete immediate-reward table can incidentally wait or
harvest another patch. Its modest advantage over exact myopic harvesting
therefore does **not** show that $\gamma_{train}=0$ learned delayed value.

### What was learned, and what was missed?

![Seed-0 policy maps and rest probabilities across all seeds](results/q_learning/learned_policies.png)

The top row uses **seed 0, chosen before training**, whose final values are
65.3646 and 23.7334. It is an illustration, not a representative seed selected
afterward. The bottom row shows the average resting probability in each state
across all seeds. Empty harvests can be equivalent to rest, so resting
probability alone is not a complete measure of waiting or policy quality.

The learned long-horizon policies often wait when A is depleted and B is
recovering, rather than consuming every new B unit. But they rarely reproduce
the oracle's full pattern of waiting while **both** patches still have stock.
For example, at `(3,3)` the oracle always rests, whereas only **15%** of final
long-horizon policies rest. It is more accurate to say that they learned some
valuable delayed consequences than that they learned the optimal conservation
strategy.

![Rewards and stocks while collecting training experience](results/q_learning/training_experience.png)

The final ten training rollouts, with exploration and learning still active,
make this partial success concrete:

| Training measurement, last 10 rollouts | $\gamma_{train}=0.99$ | $\gamma_{train}=0$ |
| --- | ---: | ---: |
| Reward per decision | 0.6734 [0.6693, 0.6775] | 0.1367 [0.1359, 0.1376] |
| Mean stock A | 0.1642 [0.1229, 0.2056] | 0.0688 [0.0672, 0.0703] |
| Mean stock B | 3.2757 [3.2574, 3.2940] | 0.0972 [0.0950, 0.0994] |
| Either patch depleted | 89.06% [86.75%, 91.37%] | 98.81% [98.75%, 98.86%] |

These are measurements during training, not frozen-policy simulations or
values. The long-horizon agent maintains B far more successfully, while A
usually remains depleted. Better discounted performance does not imply that
both resources were preserved. Average reward and resource stocks remain
secondary outcomes.

Coverage is uneven despite 200,000 transitions per seed: the long-horizon
learner leaves a mean **5.20 of 75 state/action pairs unvisited**, and only
**8/100 seeds** visit all 75 pairs. The immediate learner leaves **8.77 pairs**
unvisited on average; none of its seeds visits every pair. Epsilon explores
actions **in encountered states**; it does not guarantee good coverage of
states that the policy rarely reaches. This is evidence of a coverage problem,
but does not establish that coverage alone caused the performance gap.

Constant alpha, finite data, bootstrapped targets, and state visitation could
all matter. This run does not isolate their contributions or demonstrate
convergence. No settings were tuned after seeing the shortfall. The exact-model
oracle has access to information the learner lacks and serves as an upper
reference, not an equal-information competitor. Q-learning learned values and
a policy, **not a transition model or a planning system**.

### Reproduction and saved learning outputs

After installing the same requirements above:

```bash
source .venv/bin/activate
python check_q_learning.py
OPENBLAS_NUM_THREADS=1 python run_q_learning.py
# Rebuild the four figures using existing results, without training:
python run_q_learning.py --plot-only
```

The runner reuses completed results when present, and resumes from the latest
checkpoint when a run is incomplete. To intentionally repeat the experiment,
use `--output results/q_learning_repeat`; this keeps the published run intact.
The model is read from `results/first_world/model.npz` **only by evaluation**.
Neither its arrays nor its reference values enter the learner's updates.

The measured run took **56.53 seconds for training** (collection and updates)
and **61.78 seconds for the completion invocation**, including evaluation,
checkpoint saving, and plotting. It used the same Python 3.10.12, NumPy 1.26.4,
and Matplotlib 3.10.9 WSL CPU environment, with one OpenBLAS thread. These are
wall-clock measurements, not estimates from transition counts.

[results/q_learning/](results/q_learning/) contains approximately **10.3 MiB**:

| File | Contents |
| --- | --- |
| `checkpoint_000000.npz` through `checkpoint_200000.npz` | All Q tables and visit counts, exact greedy and ε=0.1 values for all states, rollout summaries so far, training time, and RNG states; every 10,000 transitions. |
| `results.npz` | Final Q/counts/policies/values, all checkpoint value curves, and per-rollout measurements for each training seed; oracle and reference values for plotting. |
| `summary.json` | Seed-level uncertainty, paired differences, coverage statistics, final training-window summaries, and 90%-oracle fractions with Wilson intervals. |
| `manifest.json` | Fixed configuration, complete per-seed stream IDs, world parameters, source/model hashes, versions, runtime, and evaluation protocol. |
| `run.log` and four PNGs | Recorded checkpoint progress and regenerated scientific figures. |

In `results.npz`, final tables use **method × seed × state × action**, checkpoint
values use **checkpoint × method × seed × state**, and rollout measurements
use **method × rollout × seed × metric**. Methods are training discounts
`[0.99, 0]`; metric names are saved. There is no full transition history cube.
Open with `numpy.load(path, allow_pickle=False)`.

Read [q_learning.py](q_learning.py) for the learner, [run_q_learning.py](run_q_learning.py)
for collection and frozen evaluation, and [q_learning_figures.py](q_learning_figures.py)
for plots. Brief checks cover the update equation, gamma zero, uniform ties,
and the actual-successor bootstrap. Saved visit totals, the discount-zero
closed form, and agreement with the existing single-policy evaluator were
also checked. The experiment was run once, with no rerun of experiment 1.

**Next scientific question:** would deliberately broader state coverage during
training improve the final policy at the same transition budget? A future
matched comparison could vary collection starting states while holding the
world, update rule, and evaluation from `(4,4)` fixed. That study has not been run.

## Textbook connection

We consulted Sutton and Barto, *Reinforcement Learning: An Introduction*,
second edition, **Sections 3.1, 3.3, 3.5, and 3.6** for the agent–environment
interface, discounted returns, policies, state/action values, and Bellman
equations ([book text](https://studylib.net/doc/27814306/reinforcement-learning--an-introduction)).
Value iteration previews **Section 4.4**. The numerical world and experiments
in this repository are original; they are not the book's recycling-robot or
gridworld examples.
