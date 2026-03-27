import numpy as np

from inference import HiddenMarkovModel
from utils import load_data, sim_data


def learn_all_observed(X_list, Z_list, C_list):
    """MLEs when X, Z, C are all observed.

    Returns in the order:
        alpha, beta, gamma, lam0, lam1
    """
    sum_x0, count_x0 = 0.0, 0
    sum_x1, count_x1 = 0.0, 0

    alpha_matches, alpha_total = 0, 0
    beta_num, beta_den = 0, 0
    gamma_num, gamma_den = 0, 0

    for X, Z, C in zip(X_list, Z_list, C_list):
        # lam0, lam1
        sum_x0 += np.sum(X[Z == 0])
        count_x0 += np.sum(Z == 0)

        sum_x1 += np.sum(X[Z == 1])
        count_x1 += np.sum(Z == 1)

        # alpha:
        # in serial state 0 we expect Z=0
        # in serial state 1 we expect Z=1
        serial0 = C[:, None] == 0
        serial1 = C[:, None] == 1

        matches = ((Z == 0) & serial0) | ((Z == 1) & serial1)
        serial_mask = serial0 | serial1

        alpha_matches += np.sum(matches)
        alpha_total += np.sum(serial_mask)

        # beta, gamma from transitions in C
        c_curr = C[:-1]
        c_next = C[1:]

        beta_num += np.sum((c_curr == 2) & (c_next != 2))
        beta_den += np.sum(c_curr == 2)

        gamma_num += np.sum((c_curr != 2) & (c_next == 2))
        gamma_den += np.sum(c_curr != 2)

    eps = 1e-8

    lam0 = sum_x0 / count_x0 if count_x0 > 0 else eps
    lam1 = sum_x1 / count_x1 if count_x1 > 0 else eps

    alpha = alpha_matches / alpha_total if alpha_total > 0 else 0.5
    alpha = max(alpha, 1 - alpha)  # enforce alpha > 0.5
    alpha = min(max(alpha, 0.500001), 0.999999)

    beta = beta_num / beta_den if beta_den > 0 else eps
    gamma = gamma_num / gamma_den if gamma_den > 0 else eps

    beta = min(max(beta, eps), 1 - eps)
    gamma = min(max(gamma, eps), 1 - eps)

    return alpha, beta, gamma, lam0, lam1


def learn_only_x(X_list, initial_params, max_iter=50, tolerance=1e-4):
    """Hard EM: infer hard C,Z from X, then re-estimate parameters.

    Parameter order is always:
        alpha, beta, gamma, lam0, lam1
    """
    params = initial_params.copy()

    for i in range(max_iter):
        print(f"EM Iteration {i + 1}")

        hmm = HiddenMarkovModel(**params)

        C_hard_list = []
        Z_hard_list = []

        for X_seq in X_list:
            post_c, post_z = hmm.inference(X_seq)
            C_hard_list.append(np.argmax(post_c, axis=1))
            Z_hard_list.append((post_z >= 0.5).astype(int))

        new_alpha, new_beta, new_gamma, new_lam0, new_lam1 = learn_all_observed(X_list, Z_hard_list, C_hard_list)

        new_params = {
            "alpha": new_alpha,
            "beta": new_beta,
            "gamma": new_gamma,
            "lam0": new_lam0,
            "lam1": new_lam1,
        }

        diff = sum(abs(params[k] - new_params[k]) for k in ["alpha", "beta", "gamma", "lam0", "lam1"])

        params = new_params

        if diff < tolerance:
            break

    return params


if __name__ == "__main__":
    initial_params = {
        "alpha": 0.9,
        "beta": 0.2,
        "gamma": 0.1,
        "lam0": 1.0,
        "lam1": 5.0,
    }
    true_hmm = HiddenMarkovModel(**initial_params)

    X_list, Z_list, C_list = sim_data(T=10, n=10, N=10, HMM=true_hmm)

    # fully observed check
    alpha, beta, gamma, lam0, lam1 = learn_all_observed(X_list, Z_list, C_list)
    print("Fully observed estimates:")
    print(f"alpha={alpha:.4f}, beta={beta:.4f}, gamma={gamma:.4f}, lam0={lam0:.4f}, lam1={lam1:.4f}")

    # hard EM
    Xlist = []
    for i in range(10):
        filepath = f"data/Ex_{i + 1}.csv"
        _, _, X, _, _ = load_data(filepath)
        Xlist.append(X)

    estimated_params = learn_only_x(Xlist, initial_params)
    print("Estimated parameters across all files:")
    for param in ["alpha", "beta", "gamma", "lam0", "lam1"]:
        print(f"  {param}: {estimated_params[param]:.4f}")
