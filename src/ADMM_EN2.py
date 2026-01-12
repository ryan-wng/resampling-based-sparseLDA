import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve_triangular

def ADMM_EN2(R, d, x0, lam, mu, maxits, tol, quiet=False, selector=None):
    """
    Exact translation of the provided R ADMM_EN2.

    Parameters
    ----------
    R : (p x p) ndarray
        Upper-triangular Cholesky-like factor. In R: backsolve(R, forwardsolve(t(R), b))
    d, x0 : (p,1) ndarrays
    lam, mu : floats
    maxits : int
    tol : tuple/list of two floats -> (abs_tol, rel_tol)
    quiet : bool
    selector : (p,1) ndarray of 0/1. If None, all ones.

    Returns
    -------
    dict: {'x','y','z','k'}
    """

    # Ensure column vectors
    R = np.asarray(R, dtype=float)
    d = np.asarray(d, dtype=float).reshape(-1, 1)
    x0 = np.asarray(x0, dtype=float).reshape(-1, 1)

    x = x0.copy()
    y = x0.copy()
    p = x.shape[0]
    z = np.zeros((p, 1), dtype=float)

    if selector is None:
        selector = np.ones((p, 1), dtype=float)
    else:
        selector = np.asarray(selector, dtype=float).reshape(p, 1)

    abs_tol, rel_tol = tol

    # Loop: for(k in 0:maxits)  -> inclusive in R
    for k in range(maxits + 1):
        # Update x
        # b <- d + mu*y - z
        b = d + mu * y - z

        # Rx <- forwardsolve(t(R), b)   (R' is lower)
        Rx = solve_triangular(R.T, b, lower=True, check_finite=False)

        # x <- backsolve(R, Rx)         (R is upper)
        x = solve_triangular(R, Rx, lower=False, check_finite=False)

        # Update y (soft-threshold)
        yold = y.copy()
        tmp = x + z / mu
        thresh = (lam / mu)
        yy = np.sign(tmp) * np.maximum(np.abs(tmp) - thresh, 0.0)
        y = selector * yy + np.abs(selector - 1.0) * tmp

        # Update z
        z = z + mu * (x - y)

        # Residuals
        r = x - y
        dr = norm(r, 2)
        s = mu * (y - yold)
        ds = norm(s, 2)

        # Tolerances
        ep = np.sqrt(p) * abs_tol + rel_tol * max(norm(x, 2), norm(y, 2))
        es = np.sqrt(p) * abs_tol + rel_tol * norm(y, 2)

        if not quiet:
            print(f"it = {k}, primal_viol = {dr - ep:.6e}, dual_viol = {ds - es:.6e}, "
                  f"norm_y = {max(norm(x, 2), norm(y, 2)):.6e}")

        # Stopping test (same as R): if(dr < ep & ds < es)
        if (dr < ep) and (ds < es):
            break

    return {'x': x, 'y': y, 'z': z, 'k': k}