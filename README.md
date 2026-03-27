# Hidden Markov Model of Visual Attention

This project implements a Hidden Markov Model (HMM) for visual attention as part of the **Models for Complex Systems** exam project.

The model is based on a 2-time-slice Bayesian network with:

- a hidden Markov chain `C_1, ..., C_T` with three states
- latent attention variables `Z_{t,i}` for each neuron and time step
- observed spike counts `X_{t,i}` modeled with Poisson distributions

The implementation in this repository focuses on:

1. **Forward simulation** from the HMM
2. **Inference** of hidden states using a forward-backward style message passing algorithm
3. **Testing and validation** of the inference procedure

---

## Model overview

The hidden state sequence `C_t` takes values in `{0, 1, 2}`:

- `0`: serial processing with preference toward stimulus 0
- `1`: serial processing with preference toward stimulus 1
- `2`: parallel processing

For each time step `t` and neuron `i`:

- `Z_{t,i} in {0,1}` indicates which stimulus neuron `i` attends to
- `X_{t,i}` is the observed spike count

The model parameters are:

- `alpha`: attention bias in serial states
- `beta`: transition probability from parallel to serial processing
- `gamma`: transition probability from serial to parallel processing
- `lam0`: Poisson rate for `Z=0`
- `lam1`: Poisson rate for `Z=1`

The transition matrix is

```text
Γ = [[1-γ,   0,   γ],
     [  0, 1-γ,   γ],
     [β/2, β/2, 1-β]]
```

---

## Repository structure

```text
.
├── inference.py          # Main HMM implementation
├── test_inference.py     # Calibration and brute-force testing
├── del1.ipynb            # Notebook work for the project
└── del2.ipynb            # Notebook work for the project
```

---

## Requirements

Install the required Python packages before running the code:

### Standard library

- `itertools`
- `math`
- `os`

### External packages

- `matplotlib`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`

---

## Main implementation

### `HiddenMarkovModel` class

The main code is implemented in `inference.py` in the class `HiddenMarkovModel`.

### Constructor

```python
from inference import HiddenMarkovModel

hmm = HiddenMarkovModel(
    alpha=0.9,
    beta=0.2,
    gamma=0.1,
    lam0=1,
    lam1=5,
)
```

### Methods

#### `simulate(T, n)`

Simulates one full sequence from the model.

Returns:

- `C`: shape `(T,)` — hidden state sequence
- `Z`: shape `(T, n)` — latent attention variables
- `X`: shape `(T, n)` — observed spike counts

Example:

```python
C, Z, X = hmm.simulate(T=100, n=10)
```

#### `inference(X)`

Runs inference on observed spike counts and computes posterior probabilities.

Returns:

- `posterior_c`: shape `(T, 3)` with `P(C_t = c | X)`
- `posterior_z`: shape `(T, n)` with `P(Z_{t,i} = 1 | X)`

Example:

```python
posterior_c, posterior_z = hmm.inference(X)
```

---

## How the inference works

The implementation computes:

- local evidence `P(X_t | C_t)`
- forward messages over the hidden state chain
- backward messages over the hidden state chain
- smoothed posterior probabilities for `C_t`
- posterior probabilities for each latent attention variable `Z_{t,i}`

This corresponds to the message passing / forward-backward procedure described in Part II of the project.

---

## Running the tests

The file `test_inference.py` contains two main validation approaches:

### 1. Calibration test

This checks whether

```text
E[1(C_t = c) - P(C_t = c | X)] = 0
```

and similarly for `Z_{t,i}`.

The script simulates data, runs inference, and measures empirical bias over many trials.

### 2. Brute-force comparison

For very small values of `T`, the posterior over `C_t` can be computed by enumerating all `3^T` hidden state sequences.
The brute-force posterior is then compared to the posterior from the forward-backward implementation.

### Run the test script

```bash
python test_inference.py
```

---

## Example usage

```python
from inference import HiddenMarkovModel

params = {
    "alpha": 0.9,
    "beta": 0.2,
    "gamma": 0.1,
    "lam0": 1,
    "lam1": 5,
}

hmm = HiddenMarkovModel(**params)

# Simulate data
C, Z, X = hmm.simulate(T=100, n=10)

# Infer hidden variables from observations
posterior_c, posterior_z = hmm.inference(X)

print("Posterior over C at first time step:")
print(posterior_c[0])

print("Posterior probability that neuron 0 attends stimulus 1 at first time step:")
print(posterior_z[0, 0])
```

---

## Notes on the current code

- The code currently focuses on **simulation** and **inference**.
- Testing is implemented in a separate script.
- The broader project report also discusses:
  - visual illustrations of simulation
  - logistic regression as an approximation to `P(C_t | X)`
  - learning model parameters
- Those parts may exist in the notebooks or report draft, but the core reusable Python implementation in this repository is currently centered around the HMM class and its inference validation.

---

## Academic context

This repository is part of the exam project **“A Hidden Markov Model of Visual Attention”** for the course **Models for Complex Systems**.

According to the project specification, the full project is divided into three parts:

1. simulation from the Bayesian network
2. inference of hidden nodes
3. learning model parameters from data

The report is expected to document algorithms, results, and tests, while code is placed in an appendix or supplementary material.

---

## Authors

Group 10

- Jeppe Bakkensen
- Felix Nielsen
- Thomas Schelde
- Gustav Saarup
- Jonas Godskesen
- Mads Damgaard

---
