import numpy as np
from sklearn.linear_model import ElasticNet
import pandas as pd

def APG_EN2rr(A, d, x0, lam, alpha, maxits, tol, selector=None):
    """
    Python translation of R's APG_EN2rr.
    Accelerated Proximal Gradient for the rank-reduced (A$flag != 1) case.

    Parameters
    ----------
    A : dict
        Expected keys:
            'X'   : (n x p) matrix
            'gom' : (p x r) factor matrix (or p-vector)
            'A'   : (p x p) matrix (may be present but not used here)
    d : ndarray, shape (p,) or (p,1)
        Vector d
    x0 : ndarray, shape (p,) or (p,1)
        Initial vector
    lam : float
        L1 regularization parameter
    alpha : float
        Step size
    maxits : int
    tol : float
    selector : ndarray, shape (p,), optional
        Mask vector (1 = update with proximal, 0 = plain gradient step)

    Returns
    -------
    dict
      {'x': x (ndarray), 'k': k (int iterations used)}
    """

    # Ensure 1D arrays for simplicity
    x = np.asarray(x0).astype(float).reshape(-1)
    xold = x.copy()
    p = x.shape[0]

    if selector is None:
        selector = np.ones(p, dtype=float)
    else:
        selector = np.asarray(selector, dtype=float).reshape(-1)

    # Momentum init
    t = 1.0
    told = 1.0

    # Define gradient for rank-reduced case:
    # df(x) = 2*( X^T (X x) + gom (gom^T x) ) - d
    def df(v):
        v = v.reshape(-1)
        term1 = A['X'].T @ (A['X'] @ v)           # p-vector
        # A['gom'] may be p x r or p-vector. Handle both:
        gom = np.asarray(A['gom'])
        if gom.ndim == 1:
            term2 = gom * (gom @ v)              # elementwise * scalar
        else:
            # gom @ (gom.T @ v) -> p-vector
            term2 = gom @ (gom.T @ v)
        return 2.0 * (term1 + term2) - d.reshape(-1)

    # Main loop
    for k in range(maxits + 1):
        dfx = df(x)  # gradient at current x, shape (p,)

        # Compute error on support (vectorized)
        supp_mask = np.abs(x) > 1e-12
        # compute err only on support; elsewhere 0
        err = np.zeros(p)
        if np.any(supp_mask):
            err[supp_mask] = (-dfx[supp_mask] - lam * np.sign(x[supp_mask])) * selector[supp_mask]

        # Stopping criterion (match R logic)
        cond1 = np.linalg.norm(dfx, ord=np.inf) - lam
        cond2 = np.linalg.norm(err, ord=np.inf)
        if max(cond1, cond2) < tol * p:
            break
        else:
            # APG extrapolation
            told = t
            t = (1.0 + np.sqrt(1.0 + 4.0 * told**2)) / 2.0

            # Extrapolate
            y = x + (told - 1.0) / t * (x - xold)

            # gradient at extrapolated point
            dfy = df(y)

            # proximal gradient step (soft-threshold)
            xold = x.copy()
            z = y - alpha * dfy  # gradient step
            # shrinkage: sign(z) * max(|z| - lam*alpha, 0)
            shrink = np.sign(z) * np.maximum(np.abs(z) - lam * alpha, 0.0)

            # selector: where selector == 1 use shrink, where 0 use plain gradient step (z)
            x = selector * shrink + (1.0 - selector) * z

    return {'x': x, 'k': k}