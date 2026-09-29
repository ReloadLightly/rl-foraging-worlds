"""Four scientific figures, generated entirely from saved numerical results."""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np


ACTION_COLORS = ["#24857A", "#D69C38", "#5979A5"]
ACTION_LETTERS = ["A", "B", "R"]
POLICY_COLORS = ["#79828B", "#C56949", "#207F75"]
POLICY_LABELS = ["Uniform random", "Myopic", "Planned, γ=0.99"]


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "figure.facecolor": "#FAFAF7", "axes.facecolor": "#FAFAF7",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#69747C", "text.color": "#213340",
                         "axes.labelcolor": "#213340", "xtick.color": "#465660",
                         "ytick.color": "#465660", "savefig.facecolor": "#FAFAF7"})


def save(fig, output, filename):
    fig.savefig(output/filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def stock_axes(ax):
    ax.set(xlim=(-.5, 4.5), ylim=(-.5, 4.5), xticks=range(5), yticks=range(5),
           xlabel="Stock in patch A", ylabel="Stock in patch B", aspect="equal")


def plot_policies(model, output):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for g, ax in enumerate(axes.flat):
        for stocks, policy in zip(model["states"], model["policies"][g, 2]):
            x, y = stocks
            choices = np.flatnonzero(policy)
            for j, action in enumerate(choices):
                ax.add_patch(Rectangle((x-.5+j/len(choices), y-.5), 1/len(choices), 1,
                                       facecolor=ACTION_COLORS[action], alpha=.25, edgecolor="none"))
            ax.add_patch(Rectangle((x-.5, y-.5), 1, 1, fill=False, edgecolor="white", linewidth=2))
            ax.text(x, y, "/".join(ACTION_LETTERS[a] for a in choices),
                    ha="center", va="center", fontsize=11, fontweight="bold")
        stock_axes(ax)
        ax.set_title(f"γ = {model['gammas'][g]:g}", loc="left", fontweight="bold")
    fig.suptitle("When does waiting become worth more than harvesting?", fontsize=17, y=.99)
    fig.legend([Patch(facecolor=c, alpha=.55) for c in ACTION_COLORS],
               ["A · harvest patch A", "B · harvest patch B", "R · rest"],
               ncol=3, loc="upper center", bbox_to_anchor=(.5, .958), frameon=False)
    fig.text(.5, .014, "Supplied-model optimal policies. Split cells mix tied actions uniformly (absolute tolerance 10⁻¹⁰).\n"
             "R alone means rest strictly beats both harvest actions. Empty harvests may be equivalent to rest.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .91))
    save(fig, output, "optimal_policies.png")


def plot_advantage(model, output):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for g, ax in enumerate(axes.flat):
        difference = model["values"][g, 2] - model["values"][g, 1]
        matrix = difference.reshape(5, 5).T
        maximum = max(0., float(matrix.max()))
        im = ax.imshow(matrix, origin="lower", extent=(-.5, 4.5, -.5, 4.5),
                       cmap="YlGnBu", vmin=0, vmax=maximum if maximum > 1e-9 else 1)
        for x in range(5):
            for y in range(5):
                value = matrix[y, x]
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if maximum > 1e-9 and value > maximum*.58 else "#213340")
        stock_axes(ax)
        ax.set_title(f"γ = {model['gammas'][g]:g}" + (" · no advantage" if maximum < 1e-9 else ""),
                     loc="left", fontweight="bold")
        if maximum > 1e-9:
            fig.colorbar(im, ax=ax, shrink=.78, label="V* − V(myopic)")
    fig.suptitle("The value of preserving future harvesting opportunities", fontsize=17, y=.99)
    fig.text(.5, .015, "Exact policy evaluation at each state's own starting stocks; no Monte Carlo uncertainty.\n"
             "Panels use separate color scales. Compare policies within a discount: different γ values define different objectives.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .95))
    save(fig, output, "value_advantage.png")


def plot_simulation(data, output):
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    titles = ["Harvested reward per decision", "Cumulative harvested reward",
              "Stock in patch A", "Stock in patch B", "Either patch depleted", "Rest decisions"]
    for k, ax in enumerate(axes.flat):
        x = data["bin_ends"]+1 if k == 1 else data["bin_centers"]
        scale = 100 if k in (4, 5) else 1
        for m, label in enumerate(POLICY_LABELS):
            mean = data["bin_mean"][m, :, k]*scale
            half = 1.96*data["bin_sem"][m, :, k]*scale
            ax.plot(x, mean, color=POLICY_COLORS[m], label=label, lw=1.8)
            ax.fill_between(x, mean-half, mean+half, color=POLICY_COLORS[m], alpha=.13, linewidth=0)
        ax.set_title(titles[k], loc="left", fontweight="bold")
        ax.set(xlim=(0, 1000), xlabel="Decision" if k != 1 else "Decisions completed")
        ax.grid(alpha=.16)
        if k in (2, 3):
            ax.set_ylim(0, 4.1)
        if k in (4, 5):
            ax.set(ylabel="Percent", ylim=(0, 100))
    fig.suptitle("Short-term harvests change the resources available tomorrow", fontsize=17, y=.99)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=3, loc="upper center",
               bbox_to_anchor=(.5, .963), frameon=False)
    fig.text(.5, .01, "2,000 paired trajectories per policy, all starting at (4,4). Bands: pointwise 95% trajectory intervals.\n"
             "20-decision bins; cumulative reward at bin endpoints. Stocks and depletion measured before acting.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .92))
    save(fig, output, "resource_and_harvest_curves.png")


def plot_trace(data, output):
    stop = 120  # Selected before seeing the trajectory.
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex=True)
    for m, label in enumerate(POLICY_LABELS):
        left, right = axes[m]
        for patch in range(2):
            left.step(np.arange(stop+1), data["trace_stocks"][m, :stop+1, patch],
                      where="post", color=ACTION_COLORS[patch], lw=1.7, label=f"Patch {ACTION_LETTERS[patch]}")
        left.set(title=label + " · stocks before acting", ylim=(-.2, 4.2), yticks=range(5), ylabel="Stock")
        actions, rewards = data["trace_actions"][m, :stop], data["trace_rewards"][m, :stop]
        for a in range(3):
            hit = actions == a
            right.scatter(np.arange(stop)[hit], actions[hit], c=ACTION_COLORS[a], s=15)
        failed = (actions < 2) & (rewards == 0)
        right.scatter(np.arange(stop)[failed], actions[failed], color="#7F2424", s=28, marker="x", linewidths=.9)
        right.set(title=label + " · selected actions", ylim=(-.4, 2.4),
                  yticks=range(3), yticklabels=["Harvest A", "Harvest B", "Rest"])
        for ax in (left, right):
            ax.set_xlim(0, stop)
            ax.grid(alpha=.16)
    axes[0, 0].legend(loc="lower right", ncol=2, fontsize=8, framealpha=.85)
    for ax in axes[-1]:
        ax.set_xlabel("Decision (zero-based)")
    fig.suptitle("Trajectory 0 · the same regeneration draws, different choices", fontsize=17, y=.99)
    fig.text(.5, .014, "Preselected trajectory and first 120 decisions; the complete 1,000-step trace is saved.\n"
             "A successful harvest pays 1 (A) or 1.5 (B). Red × marks an empty harvest; rest pays zero. Regeneration follows the action.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .945))
    save(fig, output, "trajectory_0.png")


def plot_all(model, simulation, output):
    style()
    plot_policies(model, output)
    plot_advantage(model, output)
    plot_simulation(simulation, output)
    plot_trace(simulation, output)
