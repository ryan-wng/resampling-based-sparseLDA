import numpy as np
import pandas as pd
from ordinal.ordinalFunctions import ordASDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
from datetime import datetime
import math
from funs import predict_asda_ordinal
from sklearn.model_selection import GroupKFold

# -----------------------------
# Lambda CV with Grouping
# -----------------------------
def cv_lambda_grouped(X, y, groups, lambda_grid, n_folds=4):
    """
    Ensures duplicates of the same patient stay in the same fold.
    This forces a higher (sparser) lambda by preventing 'cheating' via leakage.
    """
    gkf = GroupKFold(n_splits=n_folds)
    avg_maes = []

    for lam in lambda_grid:
        fold_maes = []

        # Split based on the original row indices (groups)
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Scale and stabilize
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            # Filter near-zero variance
            stds = X_train_s.std(axis=0)
            keep = stds > 1e-8
            X_train_s = X_train_s[:, keep]
            X_val_s = X_val_s[:, keep]

            # Add jitter
            X_train_s += 1e-8 * np.random.randn(*X_train_s.shape)

            try:
                # Use gam=1e-4 for CV stability
                model = ordASDA(X_train_s, y_train, s=2, h=1, gam=1e-4, lam=float(lam))
                y_pred = predict_asda_ordinal(model, X_val_s)
                mae = mean_absolute_error(y_val, y_pred)
            except Exception:
                mae = np.inf

            fold_maes.append(mae)

        avg_maes.append(np.nanmean(fold_maes))

    best_idx = int(np.nanargmin(avg_maes))
    return float(lambda_grid[best_idx])

# -----------------------------
# Train subspace
# -----------------------------
def train_subspace(s0, training, testing, trainY, testY, train_groups, 
                   varnames_full, cv_folds=4, rng_seed=None):
    
    sub_varnames = [varnames_full[j] for j in s0]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(training[:, s0])
    X_test_s = scaler.transform(testing[:, s0])


    y_centered = trainY - np.mean(trainY)
    n_samples = X_train_s.shape[0]
    
    # Calculate lambda_max: max(abs(X.T @ y)) / n
    # This identifies the "entry point" where the first variable would become non-zero
    dot_products = np.abs(np.dot(X_train_s.T, y_centered))
    lambda_max_sub = np.max(dot_products) / n_samples
    lambda_grid = np.logspace(np.log10(lambda_max_sub), 
                              np.log10(lambda_max_sub * 0.001), 
                              num=20)
    # Pass the bootstrap sample indices as groups for CV
    optimal_lambda = cv_lambda_grouped(X_train_s, trainY, train_groups, 
                                       lambda_grid, n_folds=cv_folds)

    # Final fit on this subspace
    ord_model = ordASDA(X_train_s, trainY, s=2, h=1, gam=1e-4, lam=optimal_lambda)
    ord_model["varNames"] = sub_varnames

    y_pred = predict_asda_ordinal(ord_model, X_test_s)
    mae = mean_absolute_error(testY, y_pred)
    acc = float(np.mean(testY == y_pred))

    # Clean and filter zero betas
    beta = np.abs(ord_model["beta"])
    if beta.ndim == 1: beta = beta.reshape(-1, 1)
    nonzero_mask = ~(np.all(beta == 0, axis=1))
    
    vars_filtered = [v for v, k in zip(sub_varnames, nonzero_mask) 
                     if k and v not in ["bias1", "bias2"]]

    return {
        "ord_model": ord_model,
        "optimal_lambda": optimal_lambda,
        "lambda_max": lambda_max_sub,
        "mae": mae,
        "accuracy": acc,
        "selected_vars": vars_filtered
    }

# -----------------------------
# Run one bootstrap
# -----------------------------
def run_bootstrap(i, X, Y, varnames_full,
                  boot_scale=0.9, predictor_subset=800, subspace_size=5,
                  cv_folds=4, base_seed=42):
    
    rng = np.random.default_rng(base_seed + int(i))
    n_samples, n_features = X.shape
    n_boot = int(round(boot_scale * n_samples))

    # 's' contains indices (duplicates allowed). These are our groups for CV.
    s = rng.choice(n_samples, size=n_boot, replace=True)
    unique_s = np.unique(s)
    oob_idx = np.setdiff1d(np.arange(n_samples), unique_s)

    training = X[s, :]
    trainY = Y[s]
    
    # OOB handling
    if oob_idx.size > 0:
        testing, testY = X[oob_idx, :], Y[oob_idx]
    else:
        # Fallback if no OOB samples exist (rare with boot_scale 1.6)
        holdout_idx = unique_s[:max(1, int(0.1 * len(unique_s)))]
        testing, testY = X[holdout_idx, :], Y[holdout_idx]

    sub_results = []
    for _ in range(subspace_size):
        s0 = rng.choice(n_features, size=predictor_subset, replace=False)
        res = train_subspace(
            s0=s0, training=training, testing=testing, trainY=trainY, testY=testY,
            train_groups=s,
            varnames_full=varnames_full,
            cv_folds=cv_folds, rng_seed=int(rng.integers(0, 2**31 - 1))
        )
        sub_results.append(res)

    # Pick best subspace by MAE
    maes = [r["mae"] for r in sub_results]
    best = sub_results[int(np.nanargmin(maes))]

    return {
        "beta": best["ord_model"]["beta"],
        "varNames": best["selected_vars"],
        "optimal_lambda": best["optimal_lambda"],
        "lambda_max": best["lambda_max"],
        "accuracy": best["accuracy"],
        "mae": best["mae"]
    }

# -----------------------------
# Main pipeline (Parallel over bootstraps)
# -----------------------------
def main():

    # ------------------
    # Load data
    # ------------------
    x = pd.read_csv("use_glio_data_filter1000.csv")
    y = pd.read_csv("use_glio_dataY_filter1000.csv")

    X_new = x.values
    Y = y.values.squeeze()
    varnames_full = x.columns.tolist()
    n_cols = X_new.shape[1]

    # ------------------
    # Hyperparameters
    # ------------------
    subspace_size = 5
    predictor_subset = int(round(n_cols * 0.8))
    target_unique_prob =  0.8
    boot_scale = -math.log(1-target_unique_prob)
    #lambda_grid = 10 ** np.linspace(-2.5, -1, 10)
    n_bootstraps = 200
    cv_folds = 4
    n_jobs = -1    
    base_seed = 42 

    # For reproducibility of top-level RNG
    random.seed(base_seed)
    np.random.seed(base_seed)

    start = time.time()

    # Parallel over bootstraps with nice progress bar
    with parallel_backend("loky"):
        with tqdm_joblib(tqdm(total=n_bootstraps, desc="Bootstraps", ncols=80)):
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_bootstrap)(
                    i,
                    X=X_new,
                    Y=Y,
                    varnames_full=varnames_full,
                   # lambda_grid=lambda_grid,
                    boot_scale=boot_scale,
                    predictor_subset=predictor_subset,
                    subspace_size=subspace_size,
                    cv_folds=cv_folds,
                    base_seed=base_seed
                )
                for i in range(n_bootstraps)
            )

    end = time.time()
    print("total time:", round(end - start, 2), "sec")

    # ------------------
    # Collect results
    # ------------------
    models_beta = [r["beta"] for r in results]
    models_cvm = [r["varNames"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    maes = [r["mae"] for r in results]
    lambdas = [r["optimal_lambda"] for r in results]

    best_result = max(results, key=lambda x: x["accuracy"])
    print("Best model accuracy:", round(best_result["accuracy"], 4))

    # ------------------
    # Save model summary
    # ------------------
    df = pd.DataFrame({
        "Beta": models_beta,
        "Variables": models_cvm,
        "optimal_lambda": lambdas,
        "Accuracy": accuracies,
        "MAE": maes
    })
    date_str = datetime.now().strftime("%Y%m%d")
    rand_num = random.randint(1000, 9999)
    filename = f"BS_Glios_group_{date_str}_{rand_num}.csv"
    df.to_csv(filename, index=False)
    print("Saved model results to:", filename)

    # ------------------
    # Variable frequency
    # ------------------
    all_selected = sum([r["varNames"] for r in results], [])
    freq = pd.Series(all_selected).value_counts().sort_values(ascending=False)
    freq_df = freq.reset_index()
    freq_df.columns = ["Variable", "Frequency"]
    rand_num = random.randint(1000, 9999)
    freq_filename = f"BS_Glios_gr_VIP_{date_str}_{rand_num}.csv"
    freq_df.to_csv(freq_filename, index=False)
    print("Saved variable frequency to:", freq_filename)

    # Plot frequency
    plt.figure(figsize=(12, 6))
    sns.barplot(data=freq_df, x="Variable", y="Frequency", order=freq_df["Variable"])
    plt.xticks(rotation=90)
    plt.title("Frequency of Each Selected Variable (Ordered)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
