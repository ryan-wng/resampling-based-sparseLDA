import numpy as np
from .ASDA import ASDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def ordASDA(Xt, Yt, s=1, Om=None, gam=0, lam=1e-6, method="SDAAP", control=None, h=1):
    """
    Python translation of ordASDA.default from R.
    Calls ASDA() internally.
    """

    if Yt is None:
        raise ValueError("We need the ordinal labels Yt to run this function!")

    # number of classes
    unique_classes = np.unique(Yt)
    K = len(unique_classes)
    if K == 2:
        raise ValueError("Only two types of labels, just use a binary classifier!")

    # control settings
    if control is None:
        control = {"ordinal": True}
    else:
        control["ordinal"] = True

    n, p = Xt.shape

    # Regularization matrix Om
    if Om is None:
        Om = np.zeros((p + K - 1, p + K - 1))
        for i in range(p):
            Om[i, i] = 1
    else:
        if Om.shape[0] != p:
            raise ValueError("Om must be of dimension p by p, where p is the number of predictors!")
        regMat = np.zeros((p + K - 1, p + K - 1))
        regMat[:p, :p] = Om
        Om = regMat

    # Augment the data
    augX = np.zeros((0, p + K - 1))
    augY = []

    for qq in range(1, K):  # 1:(K-1) in R
        # Class 1
        inds = range(max(1, qq - s + 1), qq + 1)
        X1 = np.zeros((0, p + K - 1))
        Y1 = []
        for j in inds:
            inds1 = np.where(Yt == j)[0]
            if len(inds1) > 0:
                block = np.hstack([Xt[inds1, :], np.zeros((len(inds1), K - 1))])
                X1 = np.vstack([X1, block])
                Y1.extend([1] * len(inds1))

        # Class 2
        inds = range(qq + 1, min(K, qq + s) + 1)
        X2 = np.zeros((0, p + K - 1))
        Y2 = []
        for j in inds:
            inds2 = np.where(Yt == j)[0]
            if len(inds2) > 0:
                block = np.hstack([Xt[inds2, :], np.zeros((len(inds2), K - 1))])
                X2 = np.vstack([X2, block])
                Y2.extend([2] * len(inds2))

        # Merge X1 and X2
        X = np.vstack([X1, X2])
        if X.shape[0] > 0:
            X[:, p + qq - 1] = h  # add bias
        Y = Y1 + Y2

        augX = np.vstack([augX, X])
        augY.extend(Y)

    augY = np.array(augY)

    # Train a discriminant model (stand-in for accSDA::ASDA)
    lda_model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda_model.fit(augX, augY)

    # Train the model using ASDA
    res = ASDA(
        Xt=augX,
        Yt=augY,
        Om=Om,
        gam=gam,
        lam=lam,
        q=1,
        method=method,
        control=control,
    )

    # Add extra info like R function
    res["varNames"][-(K - 1) :] = [f"bias{i}" for i in range(1, K)]
    res["class"] = "ordASDA"
    res["K"] = K
    res["h"] = h
    res["call"] = "ordASDA(...)"
    res["lda_model"] = lda_model
    res["augX"] = augX
    res["augY"] = augY
    res["n_selected"] = np.count_nonzero(res["beta"][:p])

    return res