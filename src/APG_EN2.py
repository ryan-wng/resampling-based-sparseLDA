import numpy as np
from sklearn.linear_model import ElasticNet
import pandas as pd


def APG_EN2(A, d, x0, lam, alpha, maxits, tol, selector=None):
    # Flatten all inputs to 1D
    x = np.asarray(x0).flatten()
    d = np.asarray(d).flatten()
    if selector is None:
        selector = np.ones_like(x)
    else:
        selector = np.asarray(selector).flatten()
    if A['flag'] == 1:
        A['gom'] = np.asarray(A['gom']).flatten()

    p = x.shape[0]
    xold = x.copy()
    t = 1.0
    told = 1.0

    # Gradient function
    if A['flag'] == 1:
        def df(x):
            return 2 * (A['gom'] * x + A['X'].T @ (A['X'] @ (x / A['n']))) - d
    else:
        def df(x):
            return (A['A'] @ x).flatten() - d  # ensure 1D output

    for k in range(maxits + 1):
        dfx = df(x)

        # Ensure dfx is 1D
        dfx = np.asarray(dfx).flatten()

        err = np.zeros(p)
        for i in range(p):
            if abs(x[i]) > 1e-12:
                err[i] = (-dfx[i] - lam * np.sign(x[i])) * selector[i]

        if max(np.linalg.norm(dfx, ord=np.inf) - lam, np.linalg.norm(err, ord=np.inf)) < tol * p:
            break
        else:
            told = t
            t = (1 + np.sqrt(1 + 4 * told ** 2)) / 2
            y = x + ((told - 1) / t) * (x - xold)
            dfy = df(y).flatten()
            xold = x.copy()
            shrink = np.sign(y - alpha * dfy) * np.maximum(np.abs(y - alpha * dfy) - lam * alpha, 0)
            x = selector * shrink + (1 - selector) * (y - alpha * dfy)

    return {'x': x, 'k': k}