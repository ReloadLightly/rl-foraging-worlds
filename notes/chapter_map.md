# From Chapter 3 concepts to our experiments

This repository studies a small foraging world while reading Sutton and Barto,
second edition. The experimental extensions are ours; they are not textbook
replications or evidence that any algorithm always wins.

| Concept | Concrete demonstration | Scope |
| --- | --- | --- |
| Agent, environment, states and actions | Stocks `(A,B)`; harvest A, harvest B, rest; harvest then regeneration | Chapter 3 framework |
| Dynamics and rewards, `p(s',r | s,a)` | An action changes stocks before stochastic regeneration; rewards differ between patches | Chapter 3 framework; Experiment 1 supplies the model, later experiments estimate it |
| Policy `π(a | s)` | Uniform choice among greedy ties; ε exploration only during collection | Chapter 3 policy concept; greedy and exploratory policies are distinct |
| Return and discount factor | Immediate harvest can compete with future stock; fixed γ=0.99 in later comparisons | Chapter 3 return; γ changes the objective, not the environment |
| `v_π`, `q_π`, optimal value and Bellman relations | Exact discounted policy values and an oracle reference | Chapter 3 objects; computational planning algorithms preview Chapter 4 |
| Continuing versus episodic tasks | Our world continues; collection stops and resets administratively | Chapter 3 task distinction; worked example below |
| A policy changes the experience distribution | Different collectors visit different state-action pairs, even at equal transition budgets | Experiments 5–6; model-guided interaction previews Chapter 8 |
| Markov state representation | `(4,3)` and `(3,4)` have equal totals but different next-total laws and optimal actions; adding `A>B` separates them while leaving other conflicts | Experiments 8–9: useful decisions can improve without making the fitted surrogate an MDP |
| Stationary dynamics | Each fixed regeneration regime has its own MDP; across an unobserved change, stocks alone lack one stationary law | Experiment 7 extension; different from Experiment 8’s fixed world with incomplete observation |

A **transition model** estimates successor probabilities and immediate rewards
for each state-action pair. Planning combines those estimates into predicted
long-term returns. A **Q table** directly estimates action returns through
updates from experience. Neither a well-populated table nor a large number of
observations ensures that the supplied state representation is sufficient.

**Previews:** exact policy evaluation, value iteration and policy iteration belong
to Chapter 4. Our Q-learning update previews Chapter 6 §6.5. Online learning of a
model followed by replanning previews Chapter 8. Our implementation does not
reproduce Dyna-Q: its empirical-model planning uses policy iteration, rather than
Dyna-Q’s interleaving of real and simulated Q-learning updates.

**Research extensions:** matched-data comparisons (Experiment 3), a fixed heuristic
count penalty (4), changes to the collector (5–6), forgetting in a changing world
(7), and compressed observations with a fixed extra bit (8–9). These use the book’s language to pose new,
restricted questions; they do not establish general rankings of learning methods.

## Worked example: termination is not a collection cutoff

Our foraging task has no terminal stock state. Depletion is not death or failure
termination; regeneration can continue. The 1,000-step rollout length controls how
we collect experience, not how long the environment’s return lasts.

Suppose the 1,000th transition starts at `(4,4)`, harvests A for reward **1**, and
actually ends at `(3,4)` with no regeneration. For illustration, suppose the
current maximum Q value at that successor is **80**, with γ=0.99.

| Meaning of the boundary | Continuation value | Illustrative Q-learning target |
| --- | --- | ---: |
| A genuinely terminal transition in a hypothetical episodic task | No future rewards; terminal value is zero | `1` |
| Our administrative collection cutoff in this continuing task | The actual successor still has a future | `1 + 0.99 × 80 = 80.2` |

The numbers 80 and 80.2 are a worked example, not measured policy values. The
Q-learning target previews Chapter 6; the reason to retain or remove continuation
comes from Chapter 3’s definition of return. For α=0.1, the update moves the current
Q entry one tenth of the way toward the appropriate target.

In our experiment, count the observed transition
`N((4,4), harvest A, (3,4)) += 1` and its reward. Start the next collection rollout
at `(4,4)`, without inserting a fictitious `(3,4) → (4,4)` transition, labeling the
observed successor terminal, or bootstrapping from the reset state. No new
survival environment is introduced by this comparison.

## Questions the evidence leaves open

Experiment 8 motivated **total stock plus the bit `A>B`** on the same saved
histories. [Experiment 9](one_bit_observation.md) now finds that this fixed,
hand-designed bit recovers 85.2% of the mean model-history value gap and greatly
reduces return prediction error. It still merges `(3,3)` and `(2,4)`, which have
different transition laws and optimal actions. The next question is whether a
short observation/action history can resolve that remaining ambiguity; it has
not been tested. These reused-history results are not an independent replication.

Other unresolved questions include learning useful memory from observation
histories, collecting experience while the agent itself has incomplete
observations, and deciding when to forget without a privileged change signal.
The current results test none of those capabilities. They also do not optimize
over all memoryless or history-dependent policies under partial observation.
