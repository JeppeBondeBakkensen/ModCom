import numpy as np

from inference import HiddenMarkovModel


def test_inference_calibration(HMM, T, n, num_trials=1000):
    # Storage for residuals (Indicator - Probability)
    # We track all 3 states for C and the attention for Z
    diffs_c = []
    diffs_z = []

    print(f"Running {num_trials} trials for calibration testing...")
    for _ in range(num_trials):
        # 1. Generate ground truth using part I simulation
        # C_true: (T,), Z_true: (T, n), X_sim: (T, n)
        C_true, Z_true, X_sim = HMM.simulate(T, n)

        # 2. Run the Part II inference algorithm
        # post_c: (T, 3), post_z: (T, n)
        post_c, post_z = HMM.inference(X_sim)

        # 3. Compute C residuals: 1(Ct = c) - P(Ct = c | X)
        # We create an indicator matrix (T, 3) where the true state is 1 and others are 0
        c_indicator = np.zeros((T, 3))
        for t in range(T):
            c_indicator[t, C_true[t]] = 1

        diffs_c.append(c_indicator - post_c)

        # 4. Compute Z residuals: 1(Zti = 1) - P(Zti = 1 | X)
        # Z_true is already 0s and 1s, acting as its own indicator
        diffs_z.append(Z_true - post_z)

    # Convert lists to arrays for global averaging
    diffs_c = np.array(diffs_c)  # Shape: (num_trials, T, 3)
    diffs_z = np.array(diffs_z)  # Shape: (num_trials, T, n)

    # Calculate mean bias across all trials, time steps, and neurons
    bias_c = np.mean(diffs_c, axis=(0, 1))
    bias_z = np.mean(diffs_z)

    print("\n--- Calibration Results (Expected: ~0.0000) ---")
    print(f"Bias for C=0 (Serial 0): {bias_c[0]:.5f}")
    print(f"Bias for C=1 (Serial 1): {bias_c[1]:.5f}")
    print(f"Bias for C=2 (Parallel): {bias_c[2]:.5f}")
    print(f"Bias for Z (Attention):  {bias_z:.5f}")

    # Sanity check: Total bias should be extremely low
    if np.all(np.abs(bias_c) < 0.05) and np.abs(bias_z) < 0.05:
        print("\nResult: SUCCESS. The inference algorithm is unbiased.")
    else:
        print("\nResult: WARNING. Significant bias detected. Check Gamma or Local Evidence.")


def brute_force(HMM, T, n, num_trails=1000):
    pass


if __name__ == "__main__":
    T, n = 3, 2
    params = {"alpha": 0.9, "beta": 0.2, "gamma": 0.1, "lam0": 1, "lam1": 5}
    HMM = HiddenMarkovModel(**params)
    test_inference_calibration(HMM, T, n)
