import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelBinarizer
from .SDAAP import SDAAP

def class_ind(cl):
    lb = LabelBinarizer()
    Y_bin = lb.fit_transform(cl)
    if Y_bin.shape[1] == 1:  # binary case
        Y_bin = np.hstack((1 - Y_bin, Y_bin))
    return Y_bin, lb.classes_

def ASDA(Xt, Yt, Om=None, gam=1e-3, lam=1e-6, q=None, method='SDAAP', control=None, **kwargs):
    if Yt is None:
        raise ValueError("You must specify labels/classes Yt to use this function!")

    n, p = Xt.shape
    Xt = np.array(Xt)

    # Convert factor to one-hot encoding
    if isinstance(Yt[0], str) or isinstance(Yt[0], (np.str_, np.object_)):
        # case 1: string labels
        Y_bin, class_labels = class_ind(Yt)
        factorY = np.array(Yt)
    elif len(Yt.shape) == 2:
        # case 2: already one-hot
        Y_bin = Yt
        class_labels = [str(i+1) for i in range(Yt.shape[1])]
        factorY = np.array([class_labels[np.argmax(row)] for row in Yt])
    elif np.issubdtype(Yt.dtype, np.integer):
        # case 3: integer labels
        Y_bin, class_labels = class_ind([str(y) for y in Yt])
        factorY = np.array([str(y) for y in Yt])
    else:
        raise ValueError("Yt format not recognized.")

    K = len(class_labels)
    if q is None:
        q = K - 1
    if q > K - 1:
        raise ValueError("q is too high, at most K-1 variates allowed")

    # Default control options
    default_control = {
        'PGsteps': 1000,
        'PGtol': 1e-5,
        'maxits': 250,
        'tol': 1e-3,
        'mu': None,
        'CV': False,
        'folds': 5,
        'feat': 0.15,
        'quiet': True,
        'ordinal': False,
        'initTheta': np.tile(np.arange(1, K+1).reshape(K, 1), (1, 1)),
        'bt': False,
        'L': 0.25,
        'eta': 2,
        'rankRed': False,
    }

    control = control or {}
    for key in control:
        if key not in default_control:
            print(f"Warning: Unknown control option '{key}' ignored")
    control = {**default_control, **control}

    # Validations
    if control['initTheta'].shape != (K, 1):
        raise ValueError("initTheta must be a K x 1 matrix")

    if control['eta'] < 0:
        raise ValueError("Backtracking multiplier must be positive")

    if Om is None:
        Om = np.eye(p)
    elif Om.shape != (p, p) and not control['rankRed']:
        raise ValueError("Om must be p x p")

    if gam < 0 or np.any(np.array(lam) < 0):
        raise ValueError("gam and lam must be positive")

    if method not in ['SDAAP', 'SDAP', 'SDAD']:
        raise ValueError(f"{method} is not a valid method")

    if method == 'SDAD' and control['mu'] is None:
        control['mu'] = 1 

    if control['CV'] and len(np.atleast_1d(lam)) < 2:
        raise ValueError("If using CV, lam must be a vector with >=2 elements")

    if control['CV'] and not (0 < control['feat'] <= 1):
        raise ValueError("feat must be between 0 and 1")

    if not control['quiet']:
        print("Input validation complete. Using method:", method)

    ###
    # Model Training
    ###

    if not control['CV']:
        selector = np.ones(p)
        if control['ordinal']:
            selector[-(K - 1):] = 0

        # Dispatch to correct training method
        if method == 'SDAAP':
            if control['bt']:
                res = SDAAP(Xt, Y_bin, Om, gam, lam, q, control['PGsteps'], control['PGtol'],
                            control['maxits'], control['tol'], selector, control['initTheta'],
                            bt=True, L=control['L'], eta=control['eta'])
            else:
                res = SDAAP(Xt, Y_bin, Om, gam, lam, q, control['PGsteps'], control['PGtol'],
                            control['maxits'], control['tol'], selector, control['initTheta'],
                            rankRed=control['rankRed'])

    ###
    # LDA Postprocessing
    ###

    B = res['B']
    if np.all(B == 0):
        print("Warning: All coefficients are zero. Try different regularization parameters.")
        lda_model = None
    else:
        sl = Xt @ B
        lda_model = LDA()
        lda_model.fit(sl, factorY)

    # Return result
    return {
        'call': 'ASDA',
        'beta': B,
        'theta': res.get('Q'),
        'varNames': kwargs.get('varNames', [f'X{i}' for i in range(1, p+1)]),
        'origP': p,
        'fit': lda_model,
        'classes': class_labels,
        'lambda': res.get('lambest', lam),
        'CV': control['CV'],
    }
