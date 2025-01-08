# Standard library imports
import numpy as np
import pandas as pd

# Geospatial imports
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as feature

# Visualization imports
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# Scientific computing imports
from scipy.linalg import qr
from sklearn.metrics import r2_score

def sensor_placement_qr(X, r, weights=None, fixed_indices=None):
    """
    Perform sensor placement using weighted QR decomposition with column pivoting.

    Parameters:
    - X: np.ndarray, data matrix (n x m).
    - r: int, the number of sensors to select.
    - weights: np.ndarray or list, optional, weights for each column (default: None).
    - fixed_indices: list, optional, indices of fixed columns (default: None).

    Returns:
    - J: List of indices corresponding to selected sensor locations.
    """
    if weights is not None:
        # Step 1: Scale the columns by weights
        W = np.diag(weights)
        X_w = X @ W
    else:
        X_w = X

    if fixed_indices:
        # Step 2: Partition the matrix into fixed and free columns
        A_F = X_w[:, fixed_indices]
        free_indices = [i for i in range(X.shape[1]) if i not in fixed_indices]
        A_R = X_w[:, free_indices]

        # Step 3: QR on fixed columns
        Q_F, R_F = np.linalg.qr(A_F)

        # Step 4: Orthogonalize free columns with respect to Q_F
        projection = Q_F @ (Q_F.T @ A_R)
        A_R_prime = A_R - projection

        # Step 5: Pivoted QR on orthogonalized free columns
        Q_R, R_R, pivots_R = qr(A_R_prime, pivoting=True)

        # Combine fixed and pivoted columns
        pivots = fixed_indices + [free_indices[i] for i in pivots_R]
    else:
        # Step 5: Pivoted QR directly on weighted matrix
        Q, R, pivots = qr(X_w, pivoting=True)

    # Step 6: Select the first r columns from the permutation
    J = pivots[:r]
    return J

def load_data(flowdata_paths, gauge_shapefile, flowlines_shapefile, gauge_index_file):
    """Load and combine all necessary datasets"""
    # Load streamflow data
    streamflow_dfs = [pd.read_csv(path, parse_dates=['Unnamed: 0'], index_col='Unnamed: 0') 
                     for path in flowdata_paths]
    streamflow_df = pd.concat(streamflow_dfs, axis=0)
    df_cleaned = streamflow_df[1:].dropna(axis=1)
    
    # Load geographical data
    gauges_gdf = gpd.read_file(gauge_shapefile)
    flowlines_gdf = gpd.read_file(flowlines_shapefile)
    usgs_index_df = pd.read_csv(gauge_index_file)
    
    return df_cleaned, gauges_gdf, flowlines_gdf, usgs_index_df

def prepare_usgs_indices(df_cleaned, usgs_index_df):
    """Prepare USGS indices and locations"""
    columns_to_keep = usgs_index_df['COMID'].astype(int).astype(str).values
    site_indices = {idx: list(df_cleaned.columns).index(idx) 
                   for idx in list(df_cleaned.columns) if idx in columns_to_keep}
    usgs_location = np.array(list(site_indices.values()))
    usgs_number = len(site_indices)
    
    return usgs_location, usgs_number, columns_to_keep

def split_data(data, train_frac=0.7):
    """Split data into training and testing sets"""
    num_samples = data.shape[0]
    train_samples = int(num_samples * train_frac)
    return data[:train_samples,:], data[train_samples:,:]

def reconstruction_evaluation(X_train, X_test, sensor_location, n_sensors):
    """Evaluate reconstruction performance for given sensor locations"""
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
    """Calculate R-squared and NSE metrics"""
    ss_res = np.sum((true_values - pred_values) ** 2, axis=0)
    ss_tot = np.sum((true_values - np.mean(true_values, axis=0)) ** 2, axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        r_squared = np.where(ss_tot != 0, 1 - (ss_res / ss_tot), np.nan)
        nse = np.where(ss_tot != 0, 1 - (ss_res / ss_tot), np.nan)
        nnse = 1 / (2 - nse)
    
    return r_squared, nse, nnse

def prepare_visualization_data(flowlines_gdf, df_cleaned, diff_nnse, sensor_locations, n_sensors):
    """Prepare data for visualization"""
    comid_rmse_df = pd.DataFrame({
        'COMID': df_cleaned.columns,
        'RMSE': diff_nnse
    })
    
    flowlines_gdf['COMID'] = flowlines_gdf['COMID'].astype(str)
    comid_rmse_df['COMID'] = comid_rmse_df['COMID'].astype(str)
    flowlines_gdf_with_rmse = flowlines_gdf.merge(comid_rmse_df, on='COMID', how='left')
    
    def prepare_sensor_centroids(sensor_loc):
        selected_sensors_gdf = flowlines_gdf_with_rmse[
            flowlines_gdf_with_rmse['COMID'].isin(df_cleaned.columns[sensor_loc].values)
        ].copy()
        selected_sensors_gdf = selected_sensors_gdf.to_crs(epsg=5070)
        selected_sensors_gdf['centroid'] = selected_sensors_gdf.geometry.centroid
        return gpd.GeoDataFrame(selected_sensors_gdf, geometry='centroid', crs=selected_sensors_gdf.crs)
    
    centroids = {name: prepare_sensor_centroids(loc[:n_sensors]) 
                for name, loc in sensor_locations.items()}
    
    return flowlines_gdf_with_rmse, centroids

def plot_sensor_network(flowlines_gdf_with_rmse, centroids, save_path=None):
    """Create and save the sensor network visualization"""
    proj = ccrs.LambertConformal(central_latitude=33, central_longitude=-96, 
                                standard_parallels=(33.0, 45.0))
    
    # Project all data to the same CRS
    flowlines_gdf_with_rmse = flowlines_gdf_with_rmse.to_crs(proj.proj4_params)
    centroids = {k: v.to_crs(proj.proj4_params) for k, v in centroids.items()}
    
    fig, ax = plt.subplots(figsize=(7, 5), dpi=600, subplot_kw={'projection': proj})
    ax.set_extent([-106.65, -93.0, 25.0, 36.5], crs=ccrs.PlateCarree())
    ax.spines['geo'].set_visible(False)
    
    # Plot flowlines
    lines = LineCollection([np.array(geometry.xy).T for geometry in flowlines_gdf_with_rmse.geometry],
                         linewidths=1, alpha=1, zorder=1)
    norm = Normalize(vmin=-1, vmax=1)
    lines.set_array(flowlines_gdf_with_rmse['RMSE'])
    lines.set_cmap('bwr_r')
    lines.set_norm(norm)
    ax.add_collection(lines)
    
    # Add colorbar
    cb_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])
    cb = fig.colorbar(lines, cax=cb_ax, orientation='vertical', label=r'$\Delta \mathrm{NNSE}$')
    
    # Plot sensors
    scatter_props = {
        'usgs': {'color': 'k', 'label': 'USGS gauges'},
        'opt': {'color': 'green', 'label': 'Reconfigured sensors'}
    }
    
    for name, gdf in centroids.items():
        ax.scatter(gdf.geometry.x, gdf.geometry.y, 
                  edgecolor='white', linewidths=0.6, alpha=0.8, s=7,
                  **scatter_props[name])
    
    # Add map features and legend
    ax.add_feature(feature.BORDERS, linestyle='-', alpha=.2)
    ax.add_feature(feature.STATES, linestyle=':', alpha=.2)
    ax.legend(frameon=False, loc='best')
    
    plt.subplots_adjust(left=0.05, right=0.8, top=0.95, bottom=0.1)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
    
    plt.show()
