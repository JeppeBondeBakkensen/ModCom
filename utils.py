import numpy as np
import pandas as pd


# Load data
def load_data(filepath):
    """Load data_file and split columns accordingly"""
    df = pd.read_csv(filepath, sep=",")
    t = df["t"].to_numpy()  # time
    X = df.filter(regex=r"^X\d+$").to_numpy()
    T, n = X.shape

    # Test of t and X
    assert np.array_equal(t, np.arange(1, T + 1))  # t = 1, ..., T
    assert (X >= 0).all()  # Check all X >= 0
    return df, t, X, T, n


def sim_data(T, n, N, HMM):
    allC = []
    allZ = []
    allX = []

    for _ in range(N):
        C_sim, Z_sim, X_sim = HMM.simulate(T, n)
        allC.append(C_sim)
        allZ.append(Z_sim)
        allX.append(X_sim)

    return allX, allZ, allC
