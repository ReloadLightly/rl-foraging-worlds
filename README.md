# Learning to leave something for tomorrow

**When does preserving resources beat taking the largest immediate reward?**

This small NumPy project explores Sutton and Barto, second edition, **Chapter 3:
Finite Markov Decision Processes**. It follows our
[Chapter 2 bandit experiments](https://github.com/ReloadLightly/rl-changing-worlds).
Here an action changes the resources available for future decisions. The first
experiment plans with a **supplied, exact model**; it does not learn dynamics
or policies from experience. Later chapters will add those learning algorithms.

**Measured result:** myopic harvesting is optimal at discounts 0 and 0.5.
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
simulation intervals. No transition model is learned, and simulated experience
does not update any policy. The preceding Chapter 2 repository remains intact.

**Next question:** does planning still help when the supplied regeneration
rates are wrong? Keep this world fixed, plan under misspecified growth rates,
and evaluate those policies in the true world. That would separate the value
of looking ahead from the cost of trusting an inaccurate model, before adding
experience-based learning.

## Textbook connection

We consulted Sutton and Barto, *Reinforcement Learning: An Introduction*,
second edition, **Sections 3.1, 3.3, 3.5, and 3.6** for the agent–environment
interface, discounted returns, policies, state/action values, and Bellman
equations ([book text](https://studylib.net/doc/27814306/reinforcement-learning--an-introduction)).
Value iteration previews **Section 4.4**. The numerical world and experiments
in this repository are original; they are not the book's recycling-robot or
gridworld examples.
