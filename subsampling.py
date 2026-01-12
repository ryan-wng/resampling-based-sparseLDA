import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from ordinal.ordinalFunctions import ordASDA
from tqdm import tqdm
import random
from datetime import datetime
from funs import predict_asda_ordinal

# =====================================================
# --- Cross-validation for lambda tuning ---
# =====================================================
def cv_lambda(data, response, lambda_grid, n_folds=4):

    data = np.asarray(data)
    response = np.asarray(response)
    kf = KFold(n_splits=n_folds, shuffle=True)

    def evaluate_lambda(lam):
        fold_mae = []
        for train_idx, test_idx in kf.split(data):
            X_train, X_val = data[train_idx], data[test_idx]
            y_train, y_val = response[train_idx], response[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            try:
                # NOTE: gam=1e-8 as in original cv_lambda
                model = ordASDA(X_train_s, y_train, s=2, h=1, gam=1e-8, lam=lam)
                preds = predict_asda_ordinal(model, X_val_s)
                fold_mae.append(np.mean(np.abs(preds - y_val)))
            except:
                fold_mae.append(np.nan)

        return np.nanmean(fold_mae)

    avg_mae = []
    for lam in tqdm(lambda_grid, desc="CV lambda", leave=False):
        avg_mae.append(evaluate_lambda(lam))
    avg_mae = np.array(avg_mae)

    # NOTE: The SE calculation here is highly questionable for cross-validation.
    # It calculates the standard error of the mean over the lambda sequence, not the folds.
    # It should ideally be calculated across the folds for the chosen lambda, 
    # but maintaining original logic:
    se_mae = np.full_like(avg_mae, np.nanstd(avg_mae, ddof=1) / np.sqrt(n_folds))
    
    idx_min = np.nanargmin(avg_mae)
    lambda_min = lambda_grid[idx_min]
    threshold = avg_mae[idx_min] + se_mae[idx_min]
    valid_idx = np.where(avg_mae <= threshold)[0]
    lambda_1se = lambda_grid[valid_idx[-1]] if len(valid_idx) > 0 else lambda_min

    return {
        'avg_mae': avg_mae,
        'se_mae': se_mae,
        'lambda_seq': lambda_grid,
        'lambda_min': lambda_min,
        'lambda_1se': lambda_1se
    }

# =====================================================
# --- Subspace Search Functions ---
# =====================================================

def train_single_subspace_cv(X_train_sub, y_train, lambda_grid, sub_varnames, 
                             n_folds_cv, rng_seed=None):
    """
    Train ASDA on a single, already-selected subspace:
      - scale
      - tune lambda by CV (using cv_lambda)
      - fit final model
    """
    
    cv_results = cv_lambda(X_train_sub, y_train, lambda_grid, n_folds=n_folds_cv)
    optimal_lambda = cv_results['lambda_min']
    
    scaler = StandardScaler()
    X_train_sub_s = scaler.fit_transform(X_train_sub)
    
    stds = X_train_sub_s.std(axis=0)
    keep = stds > 1e-8
    X_train_sub_s = X_train_sub_s[:, keep]
    
    varNames_filtered = [name for name, k in zip(sub_varnames, keep) if k]
    
    X_train_sub_s = X_train_sub_s + 1e-8 * np.random.randn(*X_train_sub_s.shape)
    
    # NOTE: Using fixed gam=1e-4 as requested
    model = ordASDA(X_train_sub_s, y_train, s=2, h=1, gam=1e-4, lam=optimal_lambda) 
    
    beta = model['beta']
    vars_selected = [v for v, b in zip(varNames_filtered, beta.flatten())
                     if abs(b) > 1e-6] 
    
    return {
        'model': model,
        'scaler': scaler,
        'keep_mask': keep,
        'optimal_lambda': optimal_lambda,
        'selected_vars': vars_selected,
        'varNames_filtered': varNames_filtered
    }
    
def find_best_subspace_and_lambda(X_train, y_train, varnames_full, 
                                  predictor_subset=800, n_subspaces=5, 
                                  n_folds_cv=4, iter_idx=0):
    n_features = X_train.shape[1]
    n_samples = X_train.shape[0]
    best_mae_avg = np.inf
    best_results = None
    
    rng = np.random.default_rng(iter_idx) 
    y_centered = y_train - np.mean(y_train)

    for sub_i in range(n_subspaces):
        # 1. Random Subspace selection
        current_subset_size = min(n_features, predictor_subset)
        subspace_indices = rng.choice(n_features, size=current_subset_size, replace=False)
        
        X_train_sub = X_train[:, subspace_indices]
        sub_varnames = [varnames_full[j] for j in subspace_indices]

        # 2. Scaling
        scaler = StandardScaler()
        X_train_sub_scaled = scaler.fit_transform(X_train_sub)

        # 3. Dynamic Lambda Grid (Increased depth to 1e-5 to avoid all-zero models)
        dot_products = np.abs(np.dot(X_train_sub_scaled.T, y_centered))
        l_max = np.max(dot_products) / n_samples
        
        # We use a deeper and denser grid (50 points) to ensure signal capture
        dynamic_lambda_grid = np.logspace(np.log10(l_max), 
                                          np.log10(l_max * 1e-5), 
                                          num=20)

        # 4. Safety-Wrapped Cross-Validation
        try:
            cv_res_sub = cv_lambda(X_train_sub_scaled, y_train, dynamic_lambda_grid, n_folds=n_folds_cv)
            
            # Check if we got valid results
            if np.all(np.isnan(cv_res_sub['avg_mae'])):
                continue
                
            avg_mae_for_subspace = np.nanmin(cv_res_sub['avg_mae'])
            
            if avg_mae_for_subspace < best_mae_avg:
                best_mae_avg = avg_mae_for_subspace
                best_results = train_single_subspace_cv(
                    X_train_sub_scaled, y_train, dynamic_lambda_grid, sub_varnames, 
                    n_folds_cv, rng_seed=iter_idx
                )
                best_results['subspace_indices'] = subspace_indices
                best_results['scaler'] = scaler 
        except Exception as e:
            # If a specific subspace is mathematically unstable, skip it and keep the loop running
            continue
            
    return best_results

def run_iteration_tts_subspace(iter_idx, X_new, Y, varnames_full, 
                               test_ratio=0.33, n_folds_cv=5, 
                               predictor_subset=800, n_subspaces=10):
    """
    Performs a single train-test split, then uses random subspace search 
    with cross-validation to select the best model and lambda on the training set.
    """

    # --- 1. Perform a single Train-Test Split ---
    x_train, x_test, y_train, y_test = train_test_split(
        X_new, Y, test_size=test_ratio, shuffle=True, random_state=iter_idx
    )

    # --- 2. Inner Subspace Search and CV for Lambda Selection on the TRAINING set ---
    best_subspace_res = find_best_subspace_and_lambda(
        X_train=x_train, 
        y_train=y_train, 
        #lambda_grid=lambda_grid, 
        varnames_full=varnames_full, 
        predictor_subset=predictor_subset,
        n_subspaces=n_subspaces,
        n_folds_cv=n_folds_cv,
        iter_idx=iter_idx
    )
    
    optimal_lambda = best_subspace_res['optimal_lambda']
    model = best_subspace_res['model']
    scaler = best_subspace_res['scaler']
    keep = best_subspace_res['keep_mask']
    subspace_indices = best_subspace_res['subspace_indices']

    # --- 3. Model Evaluation on the TEST set ---
    x_test_sub = x_test[:, subspace_indices]
    X_test_s = scaler.transform(x_test_sub)
    X_test_s = X_test_s[:, keep]
    X_test_s = X_test_s + 1e-8 * np.random.randn(*X_test_s.shape)
    
    preds = predict_asda_ordinal(model, X_test_s)
    
    mae = np.mean(np.abs(preds - y_test))
    accuracy = np.mean(preds == y_test)

    results_vars_str = ", ".join(best_subspace_res['selected_vars'])

    # --- 4. Return Results ---
    df_iter = pd.DataFrame({
        'iteration': iter_idx + 1,
        'Selected_Variables': [results_vars_str],
        'optimal_lambda': [optimal_lambda],
        'Subspace_Size': [len(subspace_indices)],
        'Subspaces_Searched': [n_subspaces],
        'MAE': [mae],
        'Accuracy': [accuracy]
    })
    return df_iter


# =====================================================
# --- Updated Parallel Execution (Sequential wrapper) ---
# =====================================================
def run_parallel_subspace(iters, X_new, Y, varnames_full,  
                          test_ratio, n_folds_cv, predictor_subset, n_subspaces):
    
    # n_jobs=-1 automatically detects all your CPU cores (e.g., 8, 12, or 16)
    # backend="loky" is the most robust for Windows
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(run_iteration_tts_subspace)(
            i, X_new, Y, varnames_full,  
            test_ratio, n_folds_cv, predictor_subset, n_subspaces
        ) for i in tqdm(range(iters), desc="Parallel Execution")
    )
    
    # Filter out any None results if an entire iteration failed
    return [r for r in results if r is not None]

# =============
# --- MAIN ---
# =============
if __name__ == '__main__':
    x = pd.read_csv("use_glio_data_filter1000.csv")
    y = pd.read_csv("use_glio_dataY_filter1000.csv")
    X_new = x.values
    Y = y.values.squeeze()
    varnames_full = x.columns.tolist()

    # Configuration
    iters = 200
    test_ratio = 0.20
    n_folds_cv = 4
    
    # Subspace Configuration
    n_cols = X_new.shape[1]
    predictor_subset = int(round(n_cols * 0.8))
    n_subspaces = 5
    start = time.time()
    outputs = run_parallel_subspace(
        iters, 
        X_new, 
        Y, 
        varnames_full,
        #lambda_grid, 
        test_ratio=test_ratio, 
        n_folds_cv=n_folds_cv,
        predictor_subset=predictor_subset, 
        n_subspaces=n_subspaces          
    )

    final_df = pd.concat(outputs, ignore_index=True)
    out_prefix="CVlam_Glio_Subspace" 
    date_str = datetime.now().strftime("%Y%m%d")
    rand_num = random.randint(1000, 9999)
    filename = f"{out_prefix}_{date_str}_{rand_num}.csv"
    final_df.to_csv(filename, index=False)

    print(f"total time: {round(time.time() - start, 2)} sec")