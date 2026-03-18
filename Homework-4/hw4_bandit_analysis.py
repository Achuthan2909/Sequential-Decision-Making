import math
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 42

# sample_reward: sample a reward from the normal distribution
def sample_reward(mu, arm, rng):
    return rng.normal(mu[arm], 1.0)


# run_ts_gaussian: run Thompson Sampling with a Gaussian prior
def run_ts_gaussian(mu, n, rng):
    k = len(mu)
    c = np.zeros(k, dtype=int)
    s = np.zeros(k)
    r = 0.0
    mu_star = float(np.max(mu))

    for _ in range(n):
        post_m = s / (1.0 + c)
        post_sd = np.sqrt(1.0 / (1.0 + c))
        th = rng.normal(post_m, post_sd)
        a = int(np.argmax(th))

        x = sample_reward(mu, a, rng)
        c[a] += 1
        s[a] += x
        r += mu_star - mu[a]
    return r


# trunc_normal_sample: sample from a truncated normal distribution
def trunc_normal_sample(mean, sd, low, high, rng, tries=20):
    for _ in range(tries):
        x = rng.normal(mean, sd)
        if low <= x <= high:
            return x
    return min(max(mean, low), high)


# run_ts_uniform_prior: run Thompson Sampling with a uniform prior
def run_ts_uniform_prior(mu, n, rng):
    k = len(mu)
    c = np.zeros(k, dtype=int)
    s = np.zeros(k)
    r = 0.0
    mu_star = float(np.max(mu))

    for _ in range(n):
        th = np.zeros(k)
        for i in range(k):
            if c[i] == 0:
                th[i] = rng.uniform(-1.0, 1.0)
            else:
                m = s[i] / c[i]
                sd = 1.0 / math.sqrt(c[i])
                th[i] = trunc_normal_sample(m, sd, -1.0, 1.0, rng)
        a = int(np.argmax(th))

        x = sample_reward(mu, a, rng)
        c[a] += 1
        s[a] += x
        r += mu_star - mu[a]
    return r


# bayes_regret_curve: compute the Bayes regret curve
def bayes_regret_curve(n_vals, sims=50, k=10, seed=SEED):
    rng = np.random.default_rng(seed)
    out_true = []
    out_miss = []

    for n in n_vals:
        rt = []
        rm = []
        for _ in range(sims):
            mu = rng.normal(0.0, 1.0, size=k)
            rt.append(run_ts_gaussian(mu, n, rng))
            rm.append(run_ts_uniform_prior(mu, n, rng))
        out_true.append(float(np.mean(rt)))
        out_miss.append(float(np.mean(rm)))
    return np.array(out_true), np.array(out_miss)


# make_means: create a vector of means for the bandit arms
def make_means(delta, k=10):
    mu = np.full(k, -0.5)
    mu[0] = 0.5
    mu[1] = 0.5 - delta
    return mu


# run_ucb: run UCB
def run_ucb(mu, n, rng):
    k = len(mu)
    c = np.zeros(k, dtype=int)
    m = np.zeros(k)
    r = 0.0
    mu_star = float(np.max(mu))

    for i in range(k):
        x = sample_reward(mu, i, rng)
        c[i] = 1
        m[i] = x
        r += mu_star - mu[i]

    for t in range(k + 1, n + 1):
        b = np.sqrt(2.0 * math.log(t) / c)
        a = int(np.argmax(m + b))
        x = sample_reward(mu, a, rng)
        c[a] += 1
        m[a] += (x - m[a]) / c[a]
        r += mu_star - mu[a]
    return r


# frequentist_eval: evaluate the frequentist regret of a given algorithm
def frequentist_eval(alg, deltas, n=2000, sims=10, seed=SEED):
    rng = np.random.default_rng(seed)
    means = []
    ses = []
    for d in deltas:
        vals = []
        mu = make_means(d, 10)
        for _ in range(sims):
            vals.append(alg(mu, n, rng))
        vals = np.array(vals)
        means.append(float(np.mean(vals)))
        ses.append(float(np.std(vals, ddof=1) / math.sqrt(sims)))
    return np.array(means), np.array(ses)


# save_bayes_plot: save the Bayes regret plot
def save_bayes_plot(n_vals, reg_true, reg_miss, out_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(n_vals, reg_true, marker="o", linewidth=2, label="TS (true Gaussian prior)")
    plt.plot(n_vals, reg_miss, marker="s", linewidth=2, label="TS (misspecified Uniform prior)")
    plt.xscale("log")
    plt.xlabel("Time horizon n")
    plt.ylabel("Bayes Regret")
    plt.title("Part I: Bayesian Regret")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "part1_bayes_regret.png"), dpi=150)
    plt.close()


# save_freq_plot: save the frequentist regret plot
def save_freq_plot(deltas, m_ucb, se_ucb, m_ts, se_ts, out_dir):
    x = np.array(deltas)
    plt.figure(figsize=(9, 5))
    plt.errorbar(x, m_ucb, yerr=2.0 * se_ucb, marker="o", capsize=3, linewidth=2, label="UCB")
    plt.errorbar(x, m_ts, yerr=2.0 * se_ts, marker="s", capsize=3, linewidth=2, label="TS (Gaussian prior)")
    plt.xlabel("Gap delta")
    plt.ylabel("Expected Regret")
    plt.title("Part II: Frequentist Regret (n=2000)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "part2_frequentist_regret.png"), dpi=150)
    plt.close()


# main: main function
def main():
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    # Part I: Bayesian regret
    n_vals = [100, 1000, 10000]
    reg_true, reg_miss = bayes_regret_curve(n_vals=n_vals, sims=50, k=10, seed=SEED)
    save_bayes_plot(n_vals, reg_true, reg_miss, out_dir)

    # Part II: Frequentist regret
    deltas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    m_ucb, se_ucb = frequentist_eval(run_ucb, deltas=deltas, n=2000, sims=10, seed=SEED)
    m_ts, se_ts = frequentist_eval(run_ts_gaussian, deltas=deltas, n=2000, sims=10, seed=SEED + 1)
    save_freq_plot(deltas, m_ucb, se_ucb, m_ts, se_ts, out_dir)

    np.save(os.path.join(out_dir, "part1_regret_true.npy"), reg_true)
    np.save(os.path.join(out_dir, "part1_regret_misspecified.npy"), reg_miss)
    np.save(os.path.join(out_dir, "part2_ucb_mean.npy"), m_ucb)
    np.save(os.path.join(out_dir, "part2_ucb_se.npy"), se_ucb)
    np.save(os.path.join(out_dir, "part2_ts_mean.npy"), m_ts)
    np.save(os.path.join(out_dir, "part2_ts_se.npy"), se_ts)

    print("Part I Bayes regret (true prior):", reg_true)
    print("Part I Bayes regret (misspecified prior):", reg_miss)
    print("Part II UCB mean regret:", m_ucb)
    print("Part II TS mean regret:", m_ts)
    print("Saved outputs in:", out_dir)


if __name__ == "__main__":
    main()
