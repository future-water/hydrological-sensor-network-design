import numpy as np
from scipy.linalg import qr

def sensor_placement_qr(X, r, weights=None, fixed_indices=None):
    """
    Perform sensor placement using weighted QR decomposition with column pivoting.
    """
    if weights is not None:
        W = np.diag(weights)
        X_w = X @ W
    else:
        X_w = X

    if fixed_indices:
        A_F = X_w[:, fixed_indices]
        free_indices = [i for i in range(X.shape[1]) if i not in fixed_indices]
        A_R = X_w[:, free_indices]

        Q_F, R_F = np.linalg.qr(A_F)
        projection = Q_F @ (Q_F.T @ A_R)
        A_R_prime = A_R - projection
        Q_R, R_R, pivots_R = qr(A_R_prime, pivoting=True)
        pivots = fixed_indices + [free_indices[i] for i in pivots_R]
    else:
        Q, R, pivots = qr(X_w, pivoting=True)

    J = pivots[:r]
    return J

def reconstruction_evaluation(X_train, X_test, sensor_location, n_sensors):
    """
    Evaluate reconstruction performance for given sensor locations.
    """
    N_sensors = X_test.shape[1]
    all_sensors = np.arange(N_sensors)
    selected_sensors = sensor_location[:n_sensors]
    non_selected_sensors = np.setdiff1d(all_sensors, selected_sensors)

    X_train_selected = X_train[:, selected_sensors]  
    X_test_selected = X_test[:, selected_sensors]

    solution = np.linalg.lstsq(X_train_selected.T, X_test_selected.T, rcond=None)[0]
    X_test_reconstructed = solution.T @ X_train
    X_test_reconstructed = np.maximum(X_test_reconstructed, 1e-10)

    rmse = np.sqrt(np.mean((X_test - X_test_reconstructed) ** 2, axis=0))
    relative_error = np.linalg.norm(X_test_reconstructed - X_test,'fro') / np.linalg.norm(X_test,'fro')

    return X_test_selected, X_test_reconstructed, selected_sensors, non_selected_sensors, rmse, relative_error

def calculate_performance_metrics(true_values, pred_values):
    """
    Calculate R-squared and NSE metrics.
    """
    ss_res = np.sum((true_values - pred_values) ** 2, axis=0)
    ss_tot = np.sum((true_values - np.mean(true_values, axis=0)) ** 2, axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        r_squared = np.where(ss_tot != 0, 1 - (ss_res / ss_tot), np.nan)
        nse = np.where(ss_tot != 0, 1 - (ss_res / ss_tot), np.nan)
        nnse = 1 / (2 - nse)
    
    return r_squared, nse, nnse 