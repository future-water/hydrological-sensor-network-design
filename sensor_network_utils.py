# Standard library imports
import numpy as np
import pandas as pd
import xarray as xr  

# Geospatial imports
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as feature

# Visualization imports
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm

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
        'opt': {'color': 'green', 'label': 'Reconfigured sensors'},
        'risk': {'color': 'darkorange', 'label': 'Risk-weighted sensors'}
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

def plot_sensor_ranking(flowlines_gdf_with_rmse, save_path=None):
    """Create and save the sensor network visualization"""
    proj = ccrs.LambertConformal(central_latitude=33, central_longitude=-96, 
                                standard_parallels=(33.0, 45.0))
    
    # Project all data to the same CRS
    flowlines_gdf_with_rmse = flowlines_gdf_with_rmse.to_crs(proj.proj4_params)
    
    fig, ax = plt.subplots(figsize=(7, 5), dpi=600, subplot_kw={'projection': proj})
    ax.set_extent([-106.65, -93.0, 25.0, 36.5], crs=ccrs.PlateCarree())
    ax.spines['geo'].set_visible(False)
    
    # Plot flowlines
    lines = LineCollection([np.array(geometry.xy).T for geometry in flowlines_gdf_with_rmse.geometry],
                            linewidths=0.025, alpha=0.8, color='black', zorder=1)
    ax.add_collection(lines)

    lines = LineCollection([np.array(geometry.xy).T for geometry in flowlines_gdf_with_rmse.geometry],
                         linewidths=1, alpha=1, zorder=1)
    lines.set_array(flowlines_gdf_with_rmse['Median Rank'])
    lines.set_cmap('viridis_r')
    ax.add_collection(lines)
    
    # Add colorbar
    cb_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])
    cb = fig.colorbar(lines, cax=cb_ax, orientation='vertical', label='Sensor Rank')
    
    # Add map features and legend
    ax.add_feature(feature.BORDERS, linestyle='-', alpha=.2)
    ax.add_feature(feature.STATES, linestyle=':', alpha=.2)
    ax.legend(frameon=False, loc='best')
    
    plt.subplots_adjust(left=0.05, right=0.8, top=0.95, bottom=0.1)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
    
    plt.show()

def plot_sensor_network_expansion(
    flowlines_gdf,
    df_cleaned,
    sensor_locations_dict,
    sensor_labels=None,
    save_path=None,
    figsize=(10, 8),
    dpi=300
):
    """
    Plot sensor network with multiple configurations.
    
    Parameters:
    -----------
    flowlines_gdf : GeoDataFrame
        GeoDataFrame containing flowline geometries.
    df_cleaned : DataFrame
        Cleaned streamflow data.
    sensor_locations_dict : dict
        Dictionary with sensor counts as keys and sensor location arrays as values.
    sensor_labels : dict, optional
        Dictionary mapping sensor counts to labels.
    save_path : str, optional
        Path to save the figure.
    figsize : tuple, optional
        Figure size (width, height).
    dpi : int, optional
        Figure resolution.
    """
    # Ensure df_cleaned columns are integers
    df_cleaned.columns = df_cleaned.columns.astype(int)
    
    # Set up projection
    proj = ccrs.LambertConformal(
        central_latitude=33,
        central_longitude=-96,
        standard_parallels=(33.0, 45.0)
    )
    
    # Create figure
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={'projection': proj},
        dpi=dpi
    )
    
    # Reproject flowlines to plotting CRS
    if flowlines_gdf.crs.is_geographic:
        flowlines_gdf = flowlines_gdf.to_crs(proj.proj4_params)
    
    # Plot flowlines
    lines = LineCollection(
        [np.array(geometry.xy).T for geometry in flowlines_gdf.geometry],
        linewidths=0.05,
        alpha=1,
        color='black',
        zorder=1
    )
    ax.add_collection(lines)
    
    # Style settings for different configurations
    sensor_counts = sorted(sensor_locations_dict.keys())
    styles = {
        'colors': ['k', 'green', 'darkorange', 'darkviolet'],
        'markers': ['o', 's', 'D', '^'],
        'sizes': [10, 10, 10, 15],
        'alphas': [0.9, 1, 1, 0.5]
    }
    
    # Generate default labels if not provided
    if sensor_labels is None:
        base_count = sensor_counts[0]
        sensor_labels = {
            count: f'{count} sensors ({int(count / base_count * 100)}%)'
            for count in sensor_counts
        }
    
    # Track plotted points to avoid overlap
    previous_points = set()
    
    # Plot each sensor configuration
    for i, n_sensors in enumerate(sensor_counts):
        selected_sensors = sensor_locations_dict[n_sensors]
        
        # Filter flowlines to sensor locations
        selected_sensors_gdf = flowlines_gdf[
            flowlines_gdf['COMID'].isin(df_cleaned.columns[selected_sensors].values)
        ].copy()
        
        # Reproject to a suitable projected CRS for centroid calculation
        selected_sensors_gdf = selected_sensors_gdf.to_crs("EPSG:5070") 
        selected_sensors_gdf['centroid'] = selected_sensors_gdf.geometry.centroid
        
        # Reproject centroids back to the plotting CRS
        centroids_gdf = gpd.GeoDataFrame(
            selected_sensors_gdf,
            geometry='centroid',
            crs=selected_sensors_gdf.crs
        ).to_crs(proj.proj4_params)
        
        # Identify new points
        current_points = set(centroids_gdf.geometry)
        new_points = current_points - previous_points
        previous_points.update(current_points)
        
        # Create GeoDataFrame for new points
        new_points_gdf = gpd.GeoDataFrame(
            geometry=list(new_points),
            crs=centroids_gdf.crs
        )
        
        # Plot sensors
        ax.scatter(
            new_points_gdf.geometry.x,
            new_points_gdf.geometry.y,
            color=styles['colors'][i % len(styles['colors'])],
            s=styles['sizes'][i % len(styles['sizes'])],
            marker=styles['markers'][i % len(styles['markers'])],
            alpha=styles['alphas'][i % len(styles['alphas'])],
            edgecolor='white',  
            linewidth=0.5, 
            label=sensor_labels[n_sensors],
            zorder=2
        )
    
    # Set map extent
    ax.set_extent([-106.65, -93.0, 25.0, 36.5], crs=ccrs.PlateCarree())
    ax.spines['geo'].set_visible(False)
    
    # Add legend
    ax.add_feature(feature.BORDERS, linestyle='-', alpha=.2)
    ax.add_feature(feature.STATES, linestyle=':', alpha=.2)
    ax.legend(frameon=False, loc='best', fontsize=16)
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    
    return fig, ax


def calculate_cost_index(flowlines_with_mean_risk, df_cleaned):
    """
    Calculate cost index based on flood risk scores.
    
    Parameters:
    -----------
    flowlines_with_mean_risk : GeoDataFrame
        GeoDataFrame containing flowlines with risk scores
    df_cleaned : DataFrame
        Cleaned streamflow data with COMID columns
        
    Returns:
    --------
    numpy.ndarray
        Array of cost indices corresponding to each COMID in df_cleaned
    """
    # Create mapping of COMID to flood risk
    comid_to_mean_rfld_afreq = dict(
        zip(flowlines_with_mean_risk['COMID'], 
            flowlines_with_mean_risk['RFLD_RISKS'])
    )
    
    # Get cost index for each COMID in df_cleaned
    comids = [int(comid) for comid in df_cleaned.columns]
    cost_index = np.array([
        comid_to_mean_rfld_afreq.get(str(comid), 1e-10) 
        for comid in comids
    ])
    
    # Replace NaN values with small number
    cost_index = np.where(np.isnan(cost_index), 1e-10, cost_index)
    
    return cost_index

def plot_flood_risk_map(
    gdb_path,
    flowlines_gdf,
    df_cleaned,
    save_path=None,
    figsize=(4, 3),
    dpi=300
):
    """
    Plot flood risk map with census tract data and calculate cost index.
    
    Parameters:
    -----------
    gdb_path : str
        Path to the NRI GDB file containing census tract data
    flowlines_gdf : GeoDataFrame
        GeoDataFrame containing flowline geometries
    df_cleaned : DataFrame
        Cleaned streamflow data with COMID columns
    save_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height)
    dpi : int, optional
        Figure resolution
        
    Returns:
    --------
    tuple
        (figure, axis, flowlines_with_risk_df, cost_index)
        - flowlines_with_risk_df contains the flowlines with normalized risk scores
        - cost_index is an array of flood risk scores for each COMID in df_cleaned
    """
    # Load and prepare census tract data
    gdf = gpd.read_file(gdb_path, layer='NRI_CensusTracts')
    flowlines_gdf = flowlines_gdf.copy()
    flowlines_gdf['COMID'] = flowlines_gdf['COMID'].astype(str)
    
    # Setup projection
    proj = ccrs.LambertConformal(
        central_latitude=33,
        central_longitude=-96,
        standard_parallels=(33.0, 45.0)
    )
    
    # Create figure
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={'projection': proj},
        dpi=dpi
    )
    fig.patch.set_alpha(0)
    
    # Set map extent
    ax.set_extent([-106.65, -93.0, 25.0, 36.5], crs=ccrs.PlateCarree())
    
    # Project and plot risk data
    gdf = gdf.to_crs(proj.proj4_params)
    gdf.plot(
        column='RFLD_RISKS',
        cmap='Reds',
        ax=ax,
        linewidth=0.
    )
    
    # Add map features
    ax.add_feature(feature.BORDERS, linestyle='-', alpha=0.2)
    ax.add_feature(feature.STATES, linestyle=':', alpha=0.2)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap='Reds',
        norm=plt.Normalize(
            vmin=gdf['RFLD_RISKS'].min(),
            vmax=gdf['RFLD_RISKS'].max()
        )
    )
    sm.set_array([])
    
    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation='vertical',
        shrink=1,
        pad=0.03
    )
    cbar.set_label('Flood Risk Index')
    
    # Calculate flowline risk scores
    gdf = gdf.to_crs(flowlines_gdf.crs)
    flowlines_with_risk = gpd.sjoin(
        flowlines_gdf,
        gdf[['RFLD_RISKS', 'geometry']],
        how="left",
        predicate='intersects'
    )
    
    # Calculate mean risk per flowline
    flowlines_with_mean_risk = flowlines_with_risk.groupby('COMID').agg({
        'RFLD_RISKS': 'mean',
        'geometry': 'first'
    }).reset_index()
    
    flowlines_with_mean_risk = gpd.GeoDataFrame(
        flowlines_with_mean_risk,
        geometry='geometry',
        crs=flowlines_gdf.crs
    )
    
    # Normalize risk scores
    min_val = flowlines_with_mean_risk['RFLD_RISKS'].min()
    max_val = flowlines_with_mean_risk['RFLD_RISKS'].max()
    flowlines_with_mean_risk['Normalized_RFLD_RISKS'] = (
        (flowlines_with_mean_risk['RFLD_RISKS'] - min_val) /
        (max_val - min_val)
    )
    
    # Calculate cost index
    cost_index = calculate_cost_index(flowlines_with_mean_risk, df_cleaned)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    
    return fig, ax, flowlines_with_mean_risk, cost_index 

def plot_flood_risk_map_clipped(
    gdb_path,
    flowlines_gdf,
    df_cleaned,
    huc2_boundary_gdf=None,
    save_path=None,
    figsize=(4, 3),
    dpi=300
):
    """
    Plot flood risk map with census tract data clipped to HUC2 Texas boundary.
    """
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as feature
    import numpy as np
    
    # Load and prepare census tract data
    gdf = gpd.read_file(gdb_path, layer='NRI_CensusTracts')
    flowlines_gdf = flowlines_gdf.copy()
    flowlines_gdf['COMID'] = flowlines_gdf['COMID'].astype(str)
    
    # Setup projection
    proj = ccrs.LambertConformal(
        central_latitude=33,
        central_longitude=-96,
        standard_parallels=(33.0, 45.0)
    )
    
    # Create figure
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={'projection': proj},
        dpi=dpi
    )
    fig.patch.set_alpha(0)
    
    # Clip data to HUC2 boundary if provided
    if huc2_boundary_gdf is not None:
        huc2_boundary = huc2_boundary_gdf.to_crs(gdf.crs)
    
        if huc2_boundary.geometry.isna().any() or huc2_boundary.is_empty.any():
            print("Warning: Invalid geometries found in HUC2 boundary")
            huc2_boundary = huc2_boundary[~huc2_boundary.geometry.isna() & ~huc2_boundary.is_empty]
        
        gdf_clipped = gpd.clip(gdf, huc2_boundary)
        flowlines_clipped = gpd.clip(flowlines_gdf, huc2_boundary.to_crs(flowlines_gdf.crs))
        
        huc2_wgs84 = huc2_boundary.to_crs('EPSG:4326')
        bounds = huc2_wgs84.total_bounds
        
        if np.any(np.isnan(bounds)) or np.any(np.isinf(bounds)):
            print("Invalid bounds detected, using default Texas extent")
            ax.set_extent([-106.65, -93.0, 25.0, 36.5], crs=ccrs.PlateCarree())
        else:
            # Add small buffer to bounds (in degrees)
            buffer = 0.1  # degrees
            extent = [
                bounds[0] - buffer, bounds[2] + buffer,  # lon_min, lon_max
                bounds[1] - buffer, bounds[3] + buffer   # lat_min, lat_max
            ]
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        gdf_to_plot = gdf_clipped
        flowlines_to_use = flowlines_clipped
        
    else:
        ax.set_extent([-106.65, -93.0, 25.0, 36.5], crs=ccrs.PlateCarree())
        gdf_to_plot = gdf
        flowlines_to_use = flowlines_gdf
        huc2_boundary = None

    gdf_to_plot = gdf_to_plot.to_crs(proj.proj4_params)
    

    if len(gdf_to_plot) > 0:
        valid_risks = gdf_to_plot['RFLD_RISKS'].dropna()
        if len(valid_risks) > 0:
            gdf_to_plot.plot(
                column='RFLD_RISKS',
                cmap='Reds',
                ax=ax,
                linewidth=0.,
                missing_kwds={'color': 'lightgray'} 
            )
            
            sm = plt.cm.ScalarMappable(
                cmap='Reds',
                norm=plt.Normalize(vmin=valid_risks.min(), vmax=valid_risks.max())
            )
            sm.set_array([])
            
            cbar = fig.colorbar(
                sm,
                ax=ax,
                orientation='vertical',
                shrink=1,
                pad=0.03
            )
            cbar.set_label('Flood Risk Index')
        else:
            gdf_to_plot.plot(ax=ax, color='lightgray', linewidth=0.)
    else:
        print("No census tract data to plot after clipping")
    
    # Add HUC2 boundary outline if provided
    if huc2_boundary_gdf is not None and huc2_boundary is not None:
        huc2_proj = huc2_boundary.to_crs(proj.proj4_params)
        huc2_proj.boundary.plot(ax=ax, color='black', linewidth=0.5, alpha=0.8)

    ax.add_feature(feature.BORDERS, linestyle='-', alpha=0.2)
    ax.add_feature(feature.STATES, linestyle=':', alpha=0.2)
    
    if len(gdf_to_plot) > 0 and len(flowlines_to_use) > 0:
        print("Calculating flowline risk scores...")
        gdf_for_analysis = gdf_to_plot.to_crs(flowlines_to_use.crs)
        flowlines_with_risk = gpd.sjoin(
            flowlines_to_use,
            gdf_for_analysis[['RFLD_RISKS', 'geometry']],
            how="left",
            predicate='intersects'
        )
        
        flowlines_with_mean_risk = flowlines_with_risk.groupby('COMID').agg({
            'RFLD_RISKS': 'mean',
            'geometry': 'first'
        }).reset_index()
        
        flowlines_with_mean_risk = gpd.GeoDataFrame(
            flowlines_with_mean_risk,
            geometry='geometry',
            crs=flowlines_to_use.crs
        )
        
        if len(flowlines_with_mean_risk) > 0 and not flowlines_with_mean_risk['RFLD_RISKS'].isna().all():
            min_val = flowlines_with_mean_risk['RFLD_RISKS'].min()
            max_val = flowlines_with_mean_risk['RFLD_RISKS'].max()
            if min_val != max_val:  # Avoid division by zero
                flowlines_with_mean_risk['Normalized_RFLD_RISKS'] = (
                    (flowlines_with_mean_risk['RFLD_RISKS'] - min_val) /
                    (max_val - min_val)
                )
            else:
                flowlines_with_mean_risk['Normalized_RFLD_RISKS'] = 0.5
        else:
            flowlines_with_mean_risk['Normalized_RFLD_RISKS'] = 0
    else:
        print("No data available for flowline risk calculation")
        flowlines_with_mean_risk = gpd.GeoDataFrame(columns=['COMID', 'RFLD_RISKS', 'Normalized_RFLD_RISKS', 'geometry'])
    
    try:
        cost_index = calculate_cost_index(flowlines_with_mean_risk, df_cleaned)
    except Exception as e:
        print(f"Error calculating cost index: {e}")
        cost_index = None
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"Figure saved to {save_path}")
    
    return fig, ax, flowlines_with_mean_risk, cost_index

def plot_correlations_bar(correlations, p_values, significance_threshold=0.05, figsize=(12, 6), dpi=300):
    """
    Create a bar chart of correlations with significance highlighting.
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Create bar plot
    bars = ax.bar(np.arange(len(correlations)), correlations)
    
    # Color the bars based on significance
    for i, (bar, p_val) in enumerate(zip(bars, p_values)):
        bar.set_color('dodgerblue' if p_val < significance_threshold else 'lightgray')
    
    # Add zero reference line
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    # Add labels
    # ax.set_xlabel('Variables')
    ax.set_ylabel('Spearman correlation')
    
    # Set x-axis ticks and labels
    ax.set_xticks(np.arange(len(correlations)))
    ax.set_xticklabels(correlations.index, rotation=90, ha='right', fontsize = 8)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Remove the figure box by hiding the spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    
    return fig

def process_region(glofas_data, boundary, region_extent, sensor_number=50):
    # Clip GLOFAS data to region boundary
    glofas_data = glofas_data.sel(
        latitude=slice(region_extent[3], region_extent[2]),  # Latitudes in descending order
        longitude=slice(region_extent[0], region_extent[1])
    )
    
    glofas_data_clipped = glofas_data.rio.clip(
    boundary.geometry,
    boundary.crs,
    drop=True
    )

    # Align cropped data with original grid
    glofas_data_clipped = glofas_data_clipped.interp(
        latitude=glofas_data.latitude,
        longitude=glofas_data.longitude
    )
    
    # Mask NaN values in 'dis24'
    glofas_data_clipped_masked = glofas_data_clipped.where(
        ~xr.ufuncs.isnan(glofas_data_clipped['dis24']), drop=True
    )
    
    # Create latitude-longitude pairs
    lat_lon = xr.DataArray(
        [
            f"({lat}, {lon})"
            for lat in glofas_data_clipped.latitude.values
            for lon in glofas_data_clipped.longitude.values
        ],
        dims="lat_lon",
        name="lat_lon"
    )
    
    # Reshape data for QR decomposition
    dis24_matrix = glofas_data_clipped['dis24'].stack(lat_lon=("latitude", "longitude"))
    dis24_matrix = dis24_matrix.drop_vars(['lat_lon', 'latitude', 'longitude'], errors='ignore')
    dis24_matrix = dis24_matrix.assign_coords(lat_lon=lat_lon).dropna(dim='lat_lon', how='all')
    valid_lat_lon = dis24_matrix.lat_lon.values
    matrix = dis24_matrix.values
    
    # Sensor placement
    sensor_location = sensor_placement_qr(matrix, sensor_number)
    selected = valid_lat_lon[sensor_location]
    selected_points = [eval(point) for point in selected]
    
    # Calculate mean discharge data
    mean_data = glofas_data_clipped['dis24'].mean(dim='time')
    mean_data = mean_data.assign_attrs(**glofas_data_clipped['dis24'].attrs)
    
    return mean_data, selected_points

def plot_map_glofas(plot_data, selected_points, region_name, cbar_label="Maximum discharge (cms)"):
    selected_latitudes = [point[0] for point in selected_points]
    selected_longitudes = [point[1] for point in selected_points]
    
    fig, ax = plt.subplots(
        1, 1, figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    
    # Plot the mean data
    im = ax.pcolormesh(
        plot_data.longitude,
        plot_data.latitude,
        plot_data,
        cmap="Blues",
        norm=LogNorm(vmin=1, vmax=1e4),
        transform=ccrs.PlateCarree()
    )
    
    # Plot sensor locations
    ax.scatter(
        selected_longitudes, selected_latitudes,
        color="orangered", edgecolors="black", s=30,
        label="Sensor Locations", transform=ccrs.PlateCarree()
    )
    
    # Add features
    ax.coastlines()
    ax.add_feature(feature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    ax.legend(frameon=False, loc="upper right")
    
    # Add color bar
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.05)
    cbar.set_label(cbar_label, fontsize=12)
    
    plt.show()
