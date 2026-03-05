# HW3 – DS 592: Sequential Decision Making

## Contents

| File | Description |
|------|-------------|
| `HW3.pdf` | Original problem set |
| `theory_solutions.pdf` | Written solutions: Doubling Trick (a, b, c) + Successive Elimination (Q1, Q2) |
| `report.pdf` | Programming assignment report: empirical analysis, theoretical bounds, discussion |
| `bandit_algorithms.py` | All algorithm implementations, Monte Carlo simulations, and plots |
| `results/` | Output plots and saved arrays (created on first run) |

## Theory Solutions

`theory_solutions.pdf` covers two problems from the theory section:

**Problem 1: The Doubling Trick**
- (a) Regret bound for the meta-algorithm over ℓ_max phases
- (b) Anytime regret bound O(√(T log(KT))) with n_ℓ = 2^{ℓ-1}
- (c) Anytime version of ETC with bound O(T^{2/3} (K log(KT))^{1/3})

**Problem 2: Successive Elimination**
- Q1: Why UCB_i(t) ≥ LCB★(t) at the last active moment, and what it implies for empirical means
- Q2: Key inequality Δ_i ≤ 2[b★(t) + b_i(t)] and why N_i(t) = N★(t)

## Programming Assignment

Setup — no external installation beyond a standard scientific Python stack.

**Required packages:** `numpy`, `matplotlib` (both ship with Anaconda / any standard environment).

```bash
pip install numpy matplotlib   # skip if already installed
```

Run:

```bash
python bandit_algorithms.py
```

This will:
1. Run 100 Monte Carlo simulations for each of the 3 algorithms × 11 Δ values.
2. Print a table of empirical regret ± standard error to stdout.
3. Save to `results/`:
   - `empirical_results.npy` — raw arrays for re-use.
   - `theoretical_bounds.npy` — raw arrays for re-use.
   - `regret_vs_delta.png` — combined plot (all algorithms + theory bounds).
   - `regret_per_algorithm.png` — 1×3 subplot grid (one panel per algorithm).

Typical runtime: ~20 seconds on a modern laptop.

## Algorithm Summary

| Algorithm | Key parameter | Exploration schedule |
|-----------|--------------|----------------------|
| Explore-Then-Commit (ETC) | m = ⌈n^{2/3}⌉ = 100 | Fixed m rounds per arm, then commit |
| Successive Elimination | b_i(t) = √(2 log n / N_i(t)) | Adaptive; eliminate arms whose UCB < best LCB |
| ε-Greedy | c = 50, εₜ = min(1, c/t) | Decaying random exploration |

## Reproducibility

The random number generator is seeded with `RNG_SEED = 42` at the top of `bandit_algorithms.py`. Change this constant or pass a different `seed` argument to `simulate()` to explore variability across seeds.
