import numpy as np
from scipy.linalg import solve_triangular
from numpy.linalg import norm
import scipy.io


def ADMM_EN_SMW(Ainv, V, R, d, x0, lam, mu, maxits, tol, quiet=False, selector=None):
    """
    ADMM solver with Sherman-Morrison-Woodbury update (Python translation of R function)

    Parameters
    ----------
    Ainv     : numpy array, shape (p,) diagonal of A^-1
    V        : numpy array, shape (n, p)
    R        : numpy array, shape (n, n) upper-triangular from Cholesky
    d        : numpy array, shape (p, 1)
    x0       : numpy array, shape (p, 1)
    lam      : float, lambda regularization
    mu       : float, ADMM penalty
    maxits   : int, maximum iterations
    tol      : list or tuple, [abs_tol, rel_tol]
    quiet    : bool, suppress print
    selector : numpy array, shape (p, 1) of 0/1 (default all ones)

    Returns
    -------
    dict with keys 'x', 'y', 'z', 'k'
    """
    x = x0.copy()
    y = x0.copy()
    p = x.shape[0]
    n = V.shape[0]
    z = np.zeros_like(x)

    if selector is None:
        selector = np.ones_like(x)

    abs_tol, rel_tol = tol

    for k in range(maxits + 1):
        # ------------------------------
        # Update x using SMW formula
        # ------------------------------
        b = d + mu * y - z
        # ensure column vector
        b = b.reshape(-1, 1)
        tmp_vec = (Ainv.reshape(-1, 1) * b)       # shape (p,1)
        btmp = (V @ tmp_vec) / n                   # shape (n,1)

        # Solve R^T temp = btmp  (forwardsolve)
        temp = solve_triangular(R.T, btmp, lower=True)
        # Solve R x2 = temp  (backsolve)
        x = tmp_vec - 2 * (Ainv.reshape(-1, 1) * (V.T @ solve_triangular(R, temp, lower=False)))

        # ------------------------------
        # Update y using soft-thresholding
        # ------------------------------
        yold = y.copy()
        tmp = x + z / mu
        yy = np.sign(tmp) * np.maximum(np.abs(tmp) - lam / mu, 0)
        y = selector * yy + np.abs(selector - 1) * tmp

        # ------------------------------
        # Update z
        # ------------------------------
        z = z + mu * (x - y)

        # ------------------------------
        # Check convergence
        # ------------------------------
        r = x - y
        dr = norm(r, 2)

        s = mu * (y - yold)
        ds = norm(s, 2)

        ep = np.sqrt(p) * abs_tol + rel_tol * max(norm(x, 2), norm(y, 2))
        es = np.sqrt(p) * abs_tol + rel_tol * norm(y, 2)

        if not quiet:
            print(f"it = {k}, primal_viol = {dr - ep:.4e}, dual_viol = {ds - es:.4e}, norm_y = {max(norm(x,2), norm(y,2)):.4e}")

        if dr < ep and ds < es:
            break

    return {"x": x, "y": y, "z": z, "k": k}