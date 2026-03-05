import numpy as np
import matplotlib.pyplot as plt
def build_transition_matrix():
    nodes = ["11", "12", "21", "22", "23", "31", "32", "41", "42", "43"]
    idx = {n: i for i, n in enumerate(nodes)}

    neighbors = {
        "11": ["21", "22", "23"],
        "12": ["21", "22", "23"],
        "21": ["31", "32"],
        "22": ["31", "32"],
        "23": ["31", "32"],
        "31": ["41", "42", "43"],
        "32": ["41", "42", "43"],
        "41": [],
        "42": [],
        "43": [],
    }

    n = len(nodes)
    P = np.zeros((n, n), dtype=float)
    for s, outs in neighbors.items():
        i = idx[s]
        if outs:
            p = 1.0 / len(outs)
            for t in outs:
                j = idx[t]
                P[i, j] = p
        else:
            P[i, i] = 1.0

    return nodes, idx, P


def compute_pi4(pi1_two_nodes):
    nodes, idx, P = build_transition_matrix()
    pi1_full = np.zeros(len(nodes))
    pi1_full[idx["11"]] = pi1_two_nodes[0]
    pi1_full[idx["12"]] = pi1_two_nodes[1]
    P3 = np.linalg.matrix_power(P, 3)
    pi4_full = pi1_full @ P3
    pi4 = np.array(
        [
            pi4_full[idx["41"]],
            pi4_full[idx["42"]],
            pi4_full[idx["43"]],
        ]
    )
    return pi4

def simulate_walk(start):
    neighbors = {
        "11": ["21", "22", "23"],
        "12": ["21", "22", "23"],
        "21": ["31", "32"],
        "22": ["31", "32"],
        "23": ["31", "32"],
        "31": ["41", "42", "43"],
        "32": ["41", "42", "43"],
        "41": [],
        "42": [],
        "43": [],
    }
    state = start
    for _ in range(3):
        outs = neighbors[state]
        if not outs:
            break
        state = np.random.choice(outs)
    return state
    
def monte_carlo_errors(pi1, trials_list):
    true_pi4 = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    terminals = ["41", "42", "43"]
    errors = []

    for N in trials_list:
        counts = {s: 0 for s in terminals}
        for _ in range(N):
            start = np.random.choice(["11", "12"], p=pi1)
            end = simulate_walk(start)
            counts[end] += 1
        hat_pi4 = np.array([counts[s] / N for s in terminals])
        err = np.linalg.norm(true_pi4 - hat_pi4)
        errors.append(err)
    return np.array(errors)

def main():
    cases = {
        "pi1 = [1/2, 1/2]": np.array([0.5, 0.5]),
        "pi1 = [0.4, 0.6]": np.array([0.4, 0.6]),
        "pi1 = [0.1, 0.9]": np.array([0.1, 0.9]),
    }

    for name, pi1 in cases.items():
        pi4 = compute_pi4(pi1)
        print(f"{name} -> pi4 = [P(41), P(42), P(43)] = {pi4}")
        print(f"  sum(pi4) = {pi4.sum():.6f}")
        print()

    pi1 = np.array([0.5, 0.5])
    trials_list = [100, 500, 1000, 5000, 20000]
    errors = monte_carlo_errors(pi1, trials_list)

    plt.figure()
    plt.plot(trials_list, errors, marker="o")
    plt.xscale("log")
    plt.xlabel("Number of simulations N")
    plt.ylabel("||pi4 - hat_pi4||_2")
    plt.title("Monte Carlo error vs. number of samples")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("hw1_bonus_error.png", dpi=200)


if __name__ == "__main__":
    main()

