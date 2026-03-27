# Hidden Markov Model of Visual Attention

This project implements a Hidden Markov Model (HMM) for visual attention as part of the _Models for Complex Systems_ exam project.

---

## Files

- `inference.py` – main HMM implementation
- `test_inference.py` – tests and validation of the inference method
- `del1.ipynb`, `del2.ipynb` – notebook work for the project

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

## Usage

Run the test script with:

```bash
python test_inference.py
```

Example:

```python
from inference import HiddenMarkovModel

hmm = HiddenMarkovModel(
    alpha=0.9,
    beta=0.2,
    gamma=0.1,
    lam0=1,
    lam1=5,
)

C, Z, X = hmm.simulate(T=100, n=10)
posterior_c, posterior_z = hmm.inference(X)
```

---

## Project focus

The repository focuses on:

1. simulation from the HMM
2. inference of hidden states
3. testing and validation of the inference algorithm

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
