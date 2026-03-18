# HW4 Programming Assignment

This folder contains a full implementation for:

- Part I: Bayesian regret of Thompson Sampling (well-specified vs misspecified prior)
- Part II: Frequentist regret comparison (UCB vs Thompson Sampling)

## Files

- `hw4_bandit_analysis.py` - main code for both parts
- `output/` - generated plots and `.npy` result arrays after running

## How to run

From repo root:

```bash
python Homework-4/hw4_bandit_analysis.py
```

## Posterior sampling rules used

### 1) TS with true prior

Prior (per arm):

$$
\mu_i \sim \mathcal{N}(0,1)
$$

Reward model:

$$
r_t \mid \mu_i \sim \mathcal{N}(\mu_i,1)
$$

If arm $i$ has been pulled $N_i$ times and reward sum is $S_i$, posterior is:

$$
\mu_i \mid \text{data} \sim \mathcal{N}\!\left(\frac{S_i}{1+N_i}, \frac{1}{1+N_i}\right)
$$

At each round sample one $\theta_i$ from each posterior and play

$$
\arg\max_i \theta_i.
$$

### 2) TS with misspecified prior

Prior (per arm):

$$
\mu_i \sim \mathrm{Uniform}([-1,1])
$$

Reward model stays Gaussian with variance $1$.

For $N_i > 0$, the posterior is proportional to:

$$
\mathbf{1}_{[-1,1]}(\mu_i)\exp\!\left(-\frac{N_i}{2}(\mu_i-\bar{x}_i)^2\right)
$$

where $\bar{x}_i = S_i/N_i$. This is a truncated normal on $[-1,1]$ with center $\bar{x}_i$ and sd $1/\sqrt{N_i}$.

The implementation samples from this rule via rejection sampling with a safe fallback when acceptance is very low.

## Notes

- Random seed is fixed for reproducibility (`SEED = 42`).
- Part I uses horizons $n \in \{100,1000,10000\}$ and $50$ simulations each.
- Part II uses $n = 2000$, $10$ simulations per $\Delta$, $\Delta \in \{0.05, 0.1, \ldots, 1.0\}$.
