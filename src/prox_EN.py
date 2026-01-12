import numpy as np
from numpy.linalg import norm
import pandas as pd

def prox_EN(A, d, x0, lam, alpha, maxits, tol):
    """
    Proximal gradient algorithm for elastic net problem.

    Parameters:
    - A: numpy array (n,n) positive semi-definite matrix
    - d: numpy array (n,1)
    - x0: numpy array (n,1) initial guess
    - lam: float, lambda regularization parameter
    - alpha: float, step size
    - maxits: int, max number of iterations
    - tol: float, tolerance for stopping criterion

    Returns:
    - dict with keys 'x' (solution), 'k' (iterations)
    """

    x = x0.copy()
    n = x.shape[0]

    for k in range(maxits + 1):
        df = A @ x - d  # Gradient of differentiable part

        # Compute error vector on support
        err = np.zeros_like(x)
        support = np.abs(x) > 1e-12
        err[support] = -df[support] - lam * np.sign(x[support])

        # Check stopping criteria
        if max(np.linalg.norm(df, ord=np.inf) - lam, np.linalg.norm(err, ord=np.inf)) < tol * n:
            return {'x': x, 'k': k}

        # Soft-thresholding update
        x = np.sign(x - alpha * df) * np.maximum(np.abs(x - alpha * df) - lam * alpha, 0)

    return {'x': x, 'k': k}