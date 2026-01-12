import numpy as np

def predict_asda_ordinal(model, X_new):
    """
    Works for ordinal classification with K-1 binary LDA problems.

    model: dict returned by Python port of ordASDA:
        model["beta"]      → sparse coefficient vector (p + K-1)
        model["fit"]       → sklearn LDA object
        model["K"]         → number of ordinal classes
        model["h"]         → bias height
        model["varNames"]  → variable names
    
    X_new: (n × p) matrix of predictors (before augmenting)
    """
    beta = model["beta"].ravel()
    lda_fit = model["fit"]          
    means = lda_fit.means_.ravel()  
    priors = lda_fit.priors_        
    scaling = lda_fit.scalings_.ravel()[0]  
    
    K = model["K"]
    h = model["h"]
    p = X_new.shape[1]

    n = X_new.shape[0]
    y_pred = np.zeros(n, dtype=int)

    # ------------------------------
    # helper: MASS::predict.lda scoring
    # ------------------------------
    def lda_binary_score(score_value):
        zLD = score_value * scaling

        delta_raw = np.zeros_like(means)
        for k in range(len(means)):
            mk = means[k]
            delta_raw[k] = zLD * mk - 0.5 * mk**2 + np.log(priors[k])

        delta = delta_raw - np.max(delta_raw)

        return 1 if delta[0] > delta[1] else 2


    # ------------------------------
    # ordinal prediction: do K-1 binary LDA calls
    # ------------------------------
    for i in range(n):
        class_votes = []
        x_raw = X_new[i, :]

        for j in range(K - 1):
            extra = np.zeros(K - 1)
            extra[j] = h

            obs = np.hstack([x_raw, extra]) 
            score = np.dot(obs, beta)

            # Apply EXACT lda scoring (MASS)
            c = lda_binary_score(score)
            class_votes.append(c)

        # Final ordinal class:
        # number of binary classifiers predicting "2" + 1
        y_pred[i] = sum(v == 2 for v in class_votes) + 1

    return y_pred
