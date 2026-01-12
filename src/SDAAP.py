import numpy as np
from numpy.linalg import norm, cholesky, solve
from scipy.linalg import solve_triangular
from .APG_EN2 import APG_EN2
from .APG_EN2rr import APG_EN2rr
from .APG_EN2bt import APG_EN2bt

def SDAAP(Xt, Yt, Om, gam, lam, q, PGsteps, PGtol, maxits, tol, selector=None, initTheta=None,
                  bt=False, L=None, eta=None, rankRed=False):
    nt, p = Xt.shape
    K = Yt.shape[1]

    subits = 0
    totalits = np.repeat(maxits, q)

    if selector is None:
        selector = np.ones(p)
    elif len(selector) != p:
        raise ValueError("The length of selector must be the same as that of Xt")

    # Structure A
    A = {
        "flag": None,
        "gom": None,
        "X": None,
        "n": None,
        "A": None
    }

    # Check if Omega is diagonal
    if rankRed:
        A["flag"] = 0
        A["X"] = np.sqrt(1 / nt) * Xt
        A["gom"] = np.sqrt(gam) * Om
        A["n"] = nt
        alpha = 1 / (2 * (norm(Xt, 1) * norm(Xt, np.inf) / nt +
                          norm(A["gom"], 1) * norm(A["gom"], np.inf)))
    elif norm(np.diag(np.diag(Om)) - Om, 'fro') < 1e-15:
        A["flag"] = 1
        A["gom"] = gam * np.diag(Om)
        A["X"] = Xt
        A["n"] = nt
        A["A"] = 2 * (Xt.T @ Xt / nt + gam * Om)
        alpha = 1 / (2 * (norm(Xt, 1) * norm(Xt, np.inf) / nt +
                          norm(np.diag(A["gom"]), np.inf)))
    else:
        A["flag"] = 0
        A["A"] = 2 * (Xt.T @ Xt / nt + gam * Om)
        alpha = 1 / norm(A["A"], 'fro')

    L = 1 / alpha
    L = norm(np.diag(np.diag(Om * gam)), np.inf) + norm(Xt, 'fro') ** 2
    origL = L

    D = (1 / nt) * (Yt.T @ Yt)
    R = cholesky(D)

    Q = np.ones((K, q))
    B = np.zeros((p, q))

    for j in range(q):
        L = origL

        Qj = Q[:, :j+1]  # Python slice is exclusive

        def Mj(u):
            return u - Qj @ (Qj.T @ (D @ u))

        # Initialize theta
        theta = np.random.rand(K, 1)
        theta = Mj(theta)
        if j == 0 and initTheta is not None:
            theta = initTheta
        theta = theta / np.sqrt((theta.T @ (D @ theta))[0, 0])

        # Initialize beta
        beta = np.zeros((p, 1))
        if norm(np.diag(np.diag(Om)) - Om, 'fro') < 1e-15 and np.sum(selector) == len(selector):
            ominv = 1 / np.diag(Om)
            rhs0 = Xt.T @ (Yt @ (theta / nt))
            rhs = Xt @ ((ominv / nt)[:, None] * rhs0)
            tmp_partial = solve(np.eye(nt) + Xt @ ((ominv / (gam * nt))[:, None] * Xt.T), rhs)
            beta = (ominv / gam)[:, None] * rhs0 - (1 / gam**2) * ominv[:, None] * (Xt.T @ tmp_partial)

        for its in range(maxits):
            d = 2 * Xt.T @ (Yt @ (theta / nt))

            b_old = beta.copy()

            if not bt and not rankRed:
                betaOb = APG_EN2(A, d, beta, lam, alpha, PGsteps, PGtol, selector)
            elif rankRed:
                betaOb = APG_EN2rr(A, d, beta, lam, alpha, PGsteps, PGtol, selector)
            else:
                betaOb = APG_EN2bt(A, Xt, Om, gam, d, beta, lam, L, eta, PGsteps, PGtol, selector)

            beta = betaOb["x"]
            if "k" in betaOb:
                subits += betaOb["k"]

            if norm(beta, 2) > 1e-12:
                b = Yt.T @ (Xt @ beta)
                y = solve_triangular(R.T, b, lower=True)
                z = solve_triangular(R, y, lower=False)
                tt = Mj(z)
                t_old = theta.copy()
                theta = tt / np.sqrt((tt.T @ D @ tt))
                db = norm(beta - b_old, 2) / norm(beta, 2)
                dt = norm(theta - t_old, 2) / norm(theta, 2)
            else:
                beta *= 0
                theta *= 0
                db = 0
                dt = 0

            if max(db, dt) < tol:
                totalits[j] = its + 1
                break

        if theta[0] < 0:
            theta *= -1
            beta *= -1

        Q[:, j] = theta.flatten()
        B[:, j] = beta.flatten()

    totalits = np.sum(totalits)

    return {
        "B": B,
        "Q": Q,
        "subits": subits,
        "totalits": totalits
    }