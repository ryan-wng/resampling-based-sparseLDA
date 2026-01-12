# Resampling-Based Sparse LDA

## Using a Python implementation of Adaptive and Elastic-Net Sparse Discriminant Analysis (accSDA)  
Developed by referencing the original **R `accSDA` package**.

This repository provides an original Python version of the core algorithms in `accSDA`, including sparse discriminant analysis, elastic-net regularization, ADMM solvers, accelerated proximal gradient methods, and ordinal extensions.

---

## Overview

Sparse Discriminant Analysis (SDA) is a supervised classification framework designed for **high-dimensional, low-sample-size** data (e.g., genomics, proteomics, imaging).  
The `accSDA` framework extends SDA with:

- Elastic-Net regularization
- Adaptive feature selection
- Efficient optimization (APG, ADMM, SMW tricks)
- Ordinal classification support
- Cross-validation for tuning regularization parameters

This repository implements these ideas **natively in Python**, while preserving the **mathematical structure and convergence behavior** of the original R code.

## Notes

- Algorithms are line-by-line translations of the original R code where possible
- Numerical linear algebra mirrors R behavior (backsolve, forwardsolve)
- Convergence criteria and stopping rules are preserved
- Feature selection behavior matches accSDA outputs under identical seeds

## Testing

- Individual solvers validated against R outputs
- Shape and sparsity consistency confirmed
- Floating-point differences expected at ~1e−6 level
- Full large-scale benchmarks ongoing
---

## Features

- **SDA variants**
  - `SDAAP` – Accelerated Proximal Gradient (Elastic Net)
  - `SDAP` – Proximal Gradient formulation
  - `SDAD` – ADMM-based formulation

- **Cross-validation**
  - `SDAAPcv`
  - `SDAPcv`
  - `SDADcv`

- **Ordinal classification**
  - `ordASDA` (ordinal SDA via data augmentation)

- **Efficient solvers**
  - Accelerated Proximal Gradient (APG)
  - Backtracking line search
  - ADMM with Sherman–Morrison–Woodbury optimization

- **High-dimensional ready**
  - Designed for \( p \gg n \) settings
  - Feature masking and rank-reduced updates

- **Sklearn-compatible postprocessing**
  - Final classification performed via LDA on learned discriminant scores

---

## Repository Structure

```text
accSDA-python/
│
├── ASDA.py                # Main user-facing API (matches accSDA::ASDA)
├── ordASDA.py             # Ordinal sparse discriminant analysis
│
├── SDAAP.py               # Accelerated proximal gradient SDA
├── SDAP.py                # Proximal gradient SDA
├── SDAD.py                # ADMM-based SDA
│
├── SDAAPcv.py             # Cross-validation (SDAAP)
├── SDAPcv.py              # Cross-validation (SDAP)
├── SDADcv.py              # Cross-validation (SDAD)
│
├── APG_EN2.py              # Accelerated proximal gradient (Elastic Net)
├── APG_EN2bt.py            # APG with backtracking
├── APG_EN2rr.py            # Rank-reduced APG
│
├── ADMM_EN2.py             # ADMM solver
├── ADMM_EN_SMW.py          # ADMM with SMW optimization
│
├── prox_EN.py              # Proximal gradient elastic net
├── prox_ENbt.py            # Proximal gradient with backtracking
│
├── normalize.py            # Training data normalization
├── normalizetest.py        # Test data normalization
│
└── README.md
```

## Basic Usage

Clone the repository and install dependencies.
#### To use **Binary or Multiclass SDA**:
```bash
from ASDA import ASDA
import numpy as np

X = np.random.randn(100, 500)     # n x p data
y = np.random.choice([0, 1, 2], size=100)

res = ASDA(
    Xt=X,
    Yt=y,
    gam=1e-3,
    lam=1e-4,
    method="SDAAP"
)

B = res["beta"]       # sparse coefficients
lda = res["fit"]      # trained LDA model
```
#### To use **Ordinal SDA**:
```bash
from ordASDA import ordASDA

res = ordASDA(
    Xt=X,
    Yt=y,      # ordinal labels
    s=1,
    gam=0,
    lam=0.05
)

selected_features = res["n_selected"]
```
#### To use **Cross-Validation for λ**:
```bash
res = ASDA(
    Xt=X,
    Yt=y,
    gam=1e-3,
    lam=np.logspace(-5, -1, 20),
    method="SDAAP",
    control={"CV": True, "folds": 5}
)

best_lambda = res["lambda"]
```

## Key Parameters

- `gam` - Ridge / elastic-net quadratic penalty     
- `lam` - L1 sparsity penalty (or grid for CV)      
- `q` - Number of discriminant directions (≤ K−1) 
- `PGsteps` - Max proximal gradient iterations          
- `PGtol` - Proximal gradient tolerance               
- `maxits` - Max outer iterations                      
- `tol` - Convergence tolerance                    
- `bt` - Use backtracking line search              
- `eta` - Backtracking multiplier                   

---

## Reference

If you use this code in academic work, please cite the original accSDA paper/package:
> Clemmensen, L., Hastie, T., Witten, D., & Ersbøll, B. (2011).  
> Sparse Discriminant Analysis. Technometrics.

---

## Disclaimer

This is a research-grade implementation, not a drop-in replacement for scikit-learn classifiers.  
It is intended for:
- Method development
- Reproducibility
- High-dimensional statistical learning research