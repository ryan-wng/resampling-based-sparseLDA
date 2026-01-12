import numpy as np
from sklearn.linear_model import ElasticNet
import pandas as pd

def APG_EN2bt(A, Xt, Om, gamma, d, x0, lam, L, eta, maxits, tol, selector=None):
    """
    Python translation of R function APG_EN2bt
    """
    x = x0.copy()
    y = x.copy()
    p = x.shape[0]

    if selector is None:
        selector = np.ones((p,1), dtype=float)
    else:
        selector = selector.astype(float)

    # Check if Om is diagonal
    ifDiag = np.linalg.norm(np.diag(np.diag(Om)) - Om, 'fro') < 1e-15
    if not ifDiag:
        # Factorize Omega
        R = np.linalg.cholesky(gamma * Om)

    oneMat = np.ones((p,1))
    zeroMat = np.zeros((p,1))

    t = 1.0
    origL = L

    # Define gradient function
    if getattr(A,'flag',0) == 1:
        gom = getattr(A,'gom')
        n = getattr(A,'n')
        df = lambda x: 2*(gom * x + Xt.T @ (Xt @ (x / n))) - d
    else:
        AA = getattr(A,'A')
        df = lambda x: AA @ x - d

    for k in range(maxits+1):
        dfx = df(x)

        # Compute error on support
        err = np.zeros((p,1))
        for i in range(p):
            if abs(x[i,0]) > 1e-12:
                err[i,0] = (-dfx[i,0] - lam*np.sign(x[i,0])) * selector[i,0]

        if max(np.linalg.norm(dfx, ord=np.inf) - lam, np.linalg.norm(err, ord=np.inf)) < tol * p:
            break

        # Backtracking step
        alpha = 1.0 / L
        dfy = df(y)
        pLyy = np.sign(y - alpha*dfy) * np.maximum(np.abs(y - alpha*dfy) - lam*alpha*oneMat, zeroMat)
        pLy = selector * pLyy + (1 - selector) * (y - alpha*dfy)
        pTilde = pLy - y

        if ifDiag:
            n = getattr(A,'n')
            gom = getattr(A,'gom')
            QminusF = 0.5*L*np.linalg.norm(pTilde)**2 - (1/n)*np.linalg.norm(Xt @ pTilde)**2 - np.sum(pTilde * (gom * pTilde))
        else:
            QminusF = 0.5*(L*np.linalg.norm(pTilde)**2 - float(pTilde.T @ AA @ pTilde))

        while QminusF < -tol:
            L = eta*L
            alpha = 1.0 / L
            pLyy = np.sign(y - alpha*dfy) * np.maximum(np.abs(y - alpha*dfy) - lam*alpha*oneMat, zeroMat)
            pLy = selector * pLyy + (1 - selector) * (y - alpha*dfy)
            pTilde = pLy - y
            if ifDiag:
                QminusF = 0.5*L*np.linalg.norm(pTilde)**2 - (1/n)*np.linalg.norm(Xt @ pTilde)**2 - np.sum(pTilde * (gom * pTilde))
            else:
                QminusF = 0.5*(L*np.linalg.norm(pTilde)**2 - float(pTilde.T @ AA @ pTilde))

        # Update step
        xold = x.copy()
        x = pLy.copy()
        told = t
        t = (1 + np.sqrt(1 + 4*told**2)) / 2
        y = x + ((told - 1)/t) * (x - xold)

    return {'x': x, 'k': k, 'L': L}