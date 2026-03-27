"""Hidden Markov Model with the 2-TBN structure:

    C_1 → C_2 → ... → C_T           (hidden Markov chain, 3 states)
     |     |            |
    Z_t   Z_t          Z_t          (attention variables, per neuron)
     |     |            |
    X_t   X_t          X_t          (observed spike counts, Poisson)

where Z_{t,1}, ..., Z_{t,n} are conditionally independent given C_t.
"""

import numpy as np
from scipy.stats import poisson


class HiddenMarkovModel:
    """Hidden Markov Model of Visual Attention (Figure 2, project description).

    Parameters
    ----------
    alpha  : float in (0.5, 1)   P(Z_{t,i}=1 | C_t=1); attention bias
    beta   : float in (0, 1)     transition rate out of parallel state (C=2)
    gamma  : float in (0, 1)     transition rate into parallel state from serial states
    lam0   : float > 0           Poisson rate when Z_{t,i} = 0
    lam1   : float > 0           Poisson rate when Z_{t,i} = 1
    """

    def __init__(self, alpha, beta, gamma, lam0, lam1):
        self._validate(alpha, beta, gamma, lam0, lam1)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lam0 = lam0
        self.lam1 = lam1

    # ──────────────────────────────────────────────────────────────────
    # Simulation  (Part I)
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _validate(alpha, beta, gamma, lam0, lam1):
        if not (0 < beta < 1):
            raise ValueError(f"Beta must be between 0 and 1: Your input: {beta}")
        if not (0 < gamma < 1):
            raise ValueError(f"Gamma must be between 0 and 1. Your input: {gamma}")
        if not (0.5 < alpha < 1):
            raise ValueError(f"Alpha must be a value between 0.5 and 1: Your input {alpha}")
        if lam0 < 0 or lam1 < 0:
            raise ValueError("Both lambda0 and lambda1 most be bigger than 0")

    @property
    def prob_z1_given_c(self):
        """P(Z=1 | C=c) for c in {0,1,2}."""
        return np.array([1 - self.alpha, self.alpha, 0.5])

    @property
    def Gamma(self):
        """Transition matrix Γ[i, j] = P(C_{t+1}=j | C_t=i)
        states 0 and 1 are serial, state 2 is parallel processing
        """
        beta, gamma = self.beta, self.gamma
        Gamma = np.array(
            [
                [1 - gamma, 0, gamma],
                [0, 1 - gamma, gamma],
                [beta / 2, beta / 2, 1 - beta],
            ]
        )
        return Gamma

    def _simulate_C(self, T) -> list[int]:
        """Simulate the hidden Markov chain C_1, ..., C_T.

        P(C_1 = 2) = 1.  For t > 1:
            C_t | C_{t-1} = c  ~  Categorical(Γ[c, :])
        """
        C = np.empty(T, dtype=int)
        C[0] = 2  # P(C_1 = 2) = 1
        for t in range(1, T):
            # p is the probabilities associated with each entry
            C[t] = np.random.choice(3, p=self.Gamma[C[t - 1]])
        return C

    def _simulate_Z(self, C, T, n):
        """Simulate attention variables Z_{t,i} | C_t ~ Bernoulli(p_c).

        P(Z_{t,i}=1 | C_t=0) = 1 - alpha
        P(Z_{t,i}=1 | C_t=1) = alpha
        P(Z_{t,i}=1 | C_t=2) = 0.5
        """
        p = np.empty(T, dtype=float)  # shape(T,)
        p[C == 0] = 1 - self.alpha
        p[C == 1] = self.alpha
        p[C == 2] = 0.5
        return np.random.binomial(n=1, p=p[:, None], size=(T, n))

    def _simulate_X(self, Z):
        """Simulate spike counts X_{t,i} | Z_{t,i} ~ Poisson(λ_{Z_{t,i}})."""
        rates = np.where(Z == 0, self.lam0, self.lam1)
        return np.random.poisson(rates)

    def simulate(self, T, n):
        """Simulate one sequence from the HMM.

        Parameters
        ----------
        T : int   number of time steps
        n : int   number of neurons

        Returns
        -------
        C : ndarray(T,)    hidden state sequence, values in {0, 1, 2}
        Z : ndarray(T, n)  attention variables, values in {0, 1}
        X : ndarray(T, n)  observed spike counts
        """
        C = self._simulate_C(T)
        Z = self._simulate_Z(C, T, n)
        X = self._simulate_X(Z)
        return C, Z, X

    # ──────────────────────────────────────────────────────────────────
    # Inference  (Part II)
    # ──────────────────────────────────────────────────────────────────

    def inference(self, X):
        """Compute P(C_t | X) and P(Z_{t,i} = 1 | X) via forward-backward.

        Returns
        -------
        posterior_c : ndarray(T, 3)   P(C_t = c | X)
        posterior_z : ndarray(T, n)   P(Z_{t,i} = 1 | X)
        """
        self.T, self.n = X.shape

        psi = np.array([self._local_evidence(X[t]) for t in range(self.T)])  # (T, 3)
        forward = self._forward_pass(psi)
        backward = self._backward_pass(psi)

        # P(C_t = c | X) ∝ forward_t(c) * backward_t(c)
        posterior_c = forward * backward
        posterior_c /= posterior_c.sum(axis=1, keepdims=True)

        # P(Z_{t,i} = 1 | X)
        posterior_z = self._posterior_Z(X, posterior_c)

        return posterior_c, posterior_z

    def _local_evidence(self, X):
        """Compute the local evidence ψ_t(c) = P(X_t | C^(t) = c) for all t"""
        prob_X_given_z0 = poisson.pmf(X, self.lam0)  # P(X^(t,i) | Z=0)
        prob_X_given_z1 = poisson.pmf(X, self.lam1)  # P(X^(t,i) | Z=1)

        psi = np.ones(3)  # (T, 3)

        for c in range(3):
            # Σ_z P(X^(t,i)|Z=z) · P(Z=z|C=c)
            term_z0 = (1 - self.prob_z1_given_c[c]) * prob_X_given_z0
            term_z1 = self.prob_z1_given_c[c] * prob_X_given_z1

            # ψ_t(c) = ∏_i [ term_z1_i + term_z0_i ]
            psi[c] = np.prod(term_z1 + term_z0)  # (T, )
        return psi  # (T, 3)

    def _forward_pass(self, psi):
        """Compute the normalized forward messages delta_t(c) = P(C^(t) = c | X^(1:t)) for all t"""
        delta_fwd = np.zeros((self.T, 3))
        prior = np.array([0.0, 0.0, 1.0])

        # delta_{1->2}(c_1) = P(C_1=c_1) * psi_1(c_1)
        delta_fwd[0] = prior * psi[0]
        delta_fwd[0] /= delta_fwd[0].sum()  # Normalize

        # Message/update:
        # psi_t(C_t) = The likelihood of the new observation 𝑋_t ​ for each possible state C_t ​
        # self.Gamma.T @ delta_fwd[t - 1] = The prior/predicted probability of each state C_t ​ before seeing X_t ​
        # delta_{t->t+1}(C_t) = psi_t(C_t) * sum_{C_{t-1}} P(C_t | C_{t-1}) delta_{t-1->t}(C_{t-1})
        for t in range(1, self.T):
            delta_fwd[t] = psi[t] * (self.Gamma.T @ delta_fwd[t - 1])
            delta_fwd[t] /= delta_fwd[t].sum()
        return delta_fwd

    def _backward_pass(self, psi):
        """Compute the normalized backward messages
        beta_t(c_t) ∝ P(X_{t+1:T} | C_t = c_t) for all t
        """
        backward = np.zeros((self.T, 3))

        # Boundary condition:
        # beta_T(C_T) = 1
        backward[-1] = np.ones(3)

        for t in range(self.T - 2, -1, -1):
            # Backward update:
            # beta_t(C_t) = sum_{C_{t+1}} P(C_{t+1} | C_t) * psi_{t+1}(C_{t+1}) * beta_{t+1}(C_{t+1})
            backward[t] = self.Gamma @ (psi[t + 1] * backward[t + 1])
            backward[t] /= backward[t].sum()  # Normalize for numerical stability
        return backward

    def _posterior_Z(self, X, posterior_c):
        T, n = X.shape
        posterior_z = np.zeros((T, n))

        for t in range(T):
            # Probability of observed spikes given Z=1 and Z=0
            prob_x_z0 = poisson.pmf(X[t], self.lam0)
            prob_x_z1 = poisson.pmf(X[t], self.lam1)

            for c in range(3):
                # Local probability P(Z_{t,i}=1 | X_{t,i}, C_t=c)
                num = prob_x_z1 * self.prob_z1_given_c[c]
                den = (prob_x_z1 * self.prob_z1_given_c[c]) + (prob_x_z0 * (1 - self.prob_z1_given_c[c]))
                p_z_given_x_c = num / den

                # Marginalize over the posterior of C_t
                posterior_z[t] += p_z_given_x_c * posterior_c[t, c]
        return posterior_z
