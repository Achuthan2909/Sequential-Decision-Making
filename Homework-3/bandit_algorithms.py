"""
DS 592 – Homework 3 Programming Assignment
Empirical Analysis of Multi-Armed Bandit Algorithms

Two-armed Gaussian bandit
  Arm 1 (optimal):    mu1 = 0,    sigma^2 = 1
  Arm 2 (suboptimal): mu2 = -Delta, sigma^2 = 1
  Horizon: n = 1000
  Delta values: {0.05, 0.1, 0.2, ..., 1.0}

Algorithms implemented
  1. Explore-Then-Commit (ETC)  -- m = ceil(n^{2/3})
  2. Successive Elimination     -- bonus b_i(t) = sqrt(2 log(n) / N_i(t))
  3. eps-Greedy                 -- eps_t = min(1, c/t), c = 50
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt
import os

#seeded for reproducibility
RNG_SEED = 42

# Bandit helpers

def sample_reward(arm: int, delta: float, rng: np.random.Generator) -> float:
    """Return one reward sample.  arm=0 is optimal (mu=0), arm=1 is suboptimal (mu=-delta)."""
    mu = 0.0 if arm == 0 else -delta
    return rng.normal(mu, 1.0)


def compute_regret(actions: np.ndarray, delta: float) -> float:
    """
    Given the sequence of arm choices (0 or 1) over n rounds,
    return the total pseudo-regret: sum_t (mu* - mu_{a_t}) = delta * #{times arm 1 pulled}.
    """
    return delta * np.sum(actions == 1)


# Algorithm 1: Explore-Then-Commit (ETC)

def run_etc(delta: float, n: int, rng: np.random.Generator) -> float:
    """
    ETC with exploration phase length m = ceil(n^{2/3}).
    Each arm is explored for m rounds (round-robin), then the empirically
    best arm is committed to for the remaining n - 2m rounds.

    Returns total pseudo-regret for this single run.
    """
    m = math.ceil(n ** (2 / 3))

    # --- Exploration phase: pull each arm m times ---
    rewards = np.zeros(2)
    actions = []
    for _ in range(m):
        for arm in range(2):
            r = sample_reward(arm, delta, rng)
            rewards[arm] += r
            actions.append(arm)

    # --- Commit phase ---
    best_arm = int(np.argmax(rewards))          # arm with highest cumulative reward
    rounds_remaining = n - 2 * m
    if rounds_remaining > 0:
        actions.extend([best_arm] * rounds_remaining)

    return compute_regret(np.array(actions[:n]), delta)


# Algorithm 2: Successive Elimination

def run_successive_elimination(delta: float, n: int, rng: np.random.Generator) -> float:
    """
    Successive Elimination.
    In each round, play every active arm once, then eliminate arm i if
      exists arm i' such that UCB_i(t) < LCB_{i'}(t)
    where UCB/LCB use bonus b_i(t) = sqrt(2 log(n) / N_i(t)).

    Returns total pseudo-regret.
    """
    K = 2
    active = [True, True]
    counts = np.zeros(K, dtype=int)
    means = np.zeros(K)
    actions = []
    t = 0                                       # rounds elapsed

    while t < n:
        active_arms = [i for i in range(K) if active[i]]
        if len(active_arms) == 0:
            break

        # Pull each active arm once (one full round)
        for arm in active_arms:
            if t >= n:
                break
            r = sample_reward(arm, delta, rng)
            counts[arm] += 1
            means[arm] += (r - means[arm]) / counts[arm]   # incremental mean
            actions.append(arm)
            t += 1

        # Elimination step
        bonuses = np.array([
            math.sqrt(2 * math.log(n) / counts[i]) if counts[i] > 0 else float('inf')
            for i in range(K)
        ])
        ucbs = means + bonuses
        lcbs = means - bonuses

        best_lcb = max(lcbs[i] for i in range(K) if active[i])
        for i in range(K):
            if active[i] and ucbs[i] < best_lcb:
                active[i] = False

        # If only one arm remains, commit to it
        active_arms = [i for i in range(K) if active[i]]
        if len(active_arms) == 1:
            remaining_arm = active_arms[0]
            while t < n:
                actions.append(remaining_arm)
                t += 1
            break

    return compute_regret(np.array(actions[:n]), delta)


# Algorithm 3: Epsilon-Greedy
def run_eps_greedy(delta: float, n: int, rng: np.random.Generator, c: float = 50.0) -> float:
    """
    eps-Greedy with eps_t = min(1, c/t), c = 50.

    At each round t (1-indexed):
      - With probability eps_t: pull a uniformly random arm
      - With probability 1 - eps_t: pull the arm with the highest empirical mean
        (break ties uniformly at random)

    Returns total pseudo-regret.
    """
    K = 2
    counts = np.zeros(K, dtype=int)
    means = np.zeros(K)
    actions = []

    for t in range(1, n + 1):
        eps_t = min(1.0, c / t)
        if rng.random() < eps_t:
            arm = rng.integers(0, K)            # explore: uniform random
        else:
            # Exploit: pick best arm; break ties randomly
            best_val = np.max(means)
            best_arms = np.where(means == best_val)[0]
            arm = rng.choice(best_arms)

        r = sample_reward(int(arm), delta, rng)
        counts[arm] += 1
        means[arm] += (r - means[arm]) / counts[arm]
        actions.append(int(arm))

    return compute_regret(np.array(actions), delta)


# Monte Carlo simulation wrapper

def simulate(
    algorithm,
    delta: float,
    n: int = 1000,
    num_simulations: int = 100,
    seed: int = RNG_SEED,
    **kwargs,
):
    """
    Run `num_simulations` independent trials of `algorithm`.

    Returns
    -------
    mean_regret : float
    std_error   : float   (standard error of the mean)
    """
    rng = np.random.default_rng(seed)
    regrets = np.array([algorithm(delta, n, rng, **kwargs) for _ in range(num_simulations)])
    return regrets.mean(), regrets.std(ddof=1) / math.sqrt(num_simulations)


# Theoretical upper bounds

def bound_etc(delta: float, n: int) -> float:
    """R(n) <= Delta * m + Delta*(n-m)*exp(-m*Delta^2/4)"""
    m = math.ceil(n ** (2 / 3))
    return delta * m + delta * max(n - 2 * m, 0) * math.exp(-m * delta ** 2 / 4)


def bound_successive_elimination(delta: float, n: int, K: int = 2) -> float:
    """R(n) <= sqrt(K * n * log(n))"""
    return math.sqrt(K * n * math.log(n))


def bound_eps_greedy(delta: float, n: int, c: float = 50.0) -> float:
    """R(n) <= c*Delta + Delta*n/c"""
    return c * delta + delta * n / c


# Main: run experiments and produce plots

def main():
    n = 1000
    num_simulations = 100
    deltas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    c = 50.0

    algorithms = {
        "ETC":                   (run_etc,                   {}),
        "Successive Elimination": (run_successive_elimination, {}),
        "ε-Greedy (c=50)":       (run_eps_greedy,             {"c": c}),
    }
    
    #Collect empirical results  
    print("Running Monte Carlo simulations …")
    results = {}          # {alg_name: (mean_list, se_list)}
    for name, (fn, kwargs) in algorithms.items():
        means, ses = [], []
        for delta in deltas:
            m, se = simulate(fn, delta, n=n, num_simulations=num_simulations, **kwargs)
            means.append(m)
            ses.append(se)
            print(f"  {name:30s}  delta={delta:.2f}  regret={m:.2f} ± {se:.2f}")
        results[name] = (np.array(means), np.array(ses))
    print("Done.\n")

    # Theoretical bounds
    theory = {
        "ETC (theory)":                  [bound_etc(d, n) for d in deltas],
        "Succ. Elim. (theory)":          [bound_successive_elimination(d, n) for d in deltas],
        "ε-Greedy (theory, c=50)":       [bound_eps_greedy(d, n, c) for d in deltas],
    }

    # Save intermediate results
    os.makedirs("results", exist_ok=True)
    np.save("results/empirical_results.npy", results, allow_pickle=True)
    np.save("results/theoretical_bounds.npy", theory, allow_pickle=True)

    # Plot
    colors = {
        "ETC":                    "tab:blue",
        "Successive Elimination": "tab:orange",
        "ε-Greedy (c=50)":        "tab:green",
    }
    theory_colors = {
        "ETC (theory)":             "tab:blue",
        "Succ. Elim. (theory)":     "tab:orange",
        "ε-Greedy (theory, c=50)":  "tab:green",
    }
    theory_alg_map = {
        "ETC (theory)":             "ETC",
        "Succ. Elim. (theory)":     "Successive Elimination",
        "ε-Greedy (theory, c=50)":  "ε-Greedy (c=50)",
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    delta_arr = np.array(deltas)

    # Empirical curves with error bars
    for name, (means, ses) in results.items():
        ax.errorbar(
            delta_arr, means, yerr=2 * ses,
            label=f"{name} (empirical)",
            color=colors[name],
            marker="o", markersize=5,
            linewidth=2, capsize=4,
        )

    # Theoretical bound curves (dashed)
    for tname, tvals in theory.items():
        alg_name = theory_alg_map[tname]
        ax.plot(
            delta_arr, tvals,
            label=tname,
            color=theory_colors[tname],
            linestyle="--", linewidth=1.5,
        )

    ax.set_xlabel(r"Gap parameter $\Delta$", fontsize=13)
    ax.set_ylabel("Expected Regret", fontsize=13)
    ax.set_title(
        f"Bandit Algorithms: Empirical Regret vs. Theoretical Bounds\n"
        f"(n={n}, {num_simulations} simulations per point)",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/regret_vs_delta.png", dpi=150)
    print("Plot saved to results/regret_vs_delta.png")

    # Separate per-algorithm plots 
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    algo_list = list(algorithms.keys())
    theory_list = list(theory.keys())

    for ax2, name, tname in zip(axes, algo_list, theory_list):
        means, ses = results[name]
        tvals = theory[tname]
        color = colors[name]

        ax2.errorbar(
            delta_arr, means, yerr=2 * ses,
            label="Empirical",
            color=color, marker="o", markersize=5,
            linewidth=2, capsize=4,
        )
        ax2.plot(
            delta_arr, tvals,
            label="Theoretical UB",
            color=color, linestyle="--", linewidth=1.5,
        )
        ax2.set_title(name, fontsize=11)
        ax2.set_xlabel(r"$\Delta$", fontsize=12)
        ax2.set_ylabel("Expected Regret", fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, linestyle=":", alpha=0.6)

    fig2.suptitle(f"Per-Algorithm: Empirical vs. Theoretical Regret  (n={n})", fontsize=13)
    plt.tight_layout()
    plt.savefig("results/regret_per_algorithm.png", dpi=150)
    print("Plot saved to results/regret_per_algorithm.png")


if __name__ == "__main__":
    main()
