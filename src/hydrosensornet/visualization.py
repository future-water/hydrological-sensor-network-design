"""Visualization functions for sensor network analysis."""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as feature
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LogNorm
import numpy as np

def plot_sensor_network(flowlines_gdf_with_rmse, centroids, save_path=None):
    """
    Create and save the sensor network visualization.
    
    Parameters:
    -----------
    flowlines_gdf_with_rmse : GeoDataFrame
        GeoDataFrame containing flowlines with RMSE values
    centroids : dict
        Dictionary of sensor centroids for different configurations
    save_path : str, optional
        Path to save the figure
    """
    proj = ccrs.LambertConformal(
        central_latitude=33, 
        central_longitude=-96, 
        standard_parallels=(33.0, 45.0)
    )
    
    # Project all data to the same CRS
    flowlines_gdf_with_rmse = flowlines_gdf_with_rmse.to_crs(proj.proj4_params)
    centroids = {k: v.to_crs(proj.proj4_params) for k, v in centroids.items()}
    
    fig, ax = plt.subplots(figsize=(7, 5), dpi=600, subplot_kw={'projection': proj})
    ax.set_extent([-106.65, -93.0, 25.0, 36.5], crs=ccrs.PlateCarree())
    ax.spines['geo'].set_visible(False)
    
    # Plot flowlines
    lines = LineCollection(
        [np.array(geometry.xy).T for geometry in flowlines_gdf_with_rmse.geometry],
        linewidths=1, 
        alpha=1, 
        zorder=1
    )
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
        'risk': {'color': 'dodgerblue', 'label': 'Risk-weighted sensors'}
    }
    
    for name, gdf in centroids.items():
        ax.scatter(
            gdf.geometry.x, 
            gdf.geometry.y, 
            edgecolor='white', 
            linewidths=0.6, 
            alpha=0.8, 
            s=7,
            **scatter_props[name]
        )
    
    # Add map features and legend
    ax.add_feature(feature.BORDERS, linestyle='-', alpha=.2)
    ax.add_feature(feature.STATES, linestyle=':', alpha=.2)
    ax.legend(frameon=False, loc='best')
    
    plt.subplots_adjust(left=0.05, right=0.8, top=0.95, bottom=0.1)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
    
    plt.show()

def plot_sensor_ranking(flowlines_gdf_with_rmse, save_path=None):
    """
    Create and save the sensor ranking visualization.
    
    Parameters:
    -----------
    flowlines_gdf_with_rmse : GeoDataFrame
        GeoDataFrame containing flowlines with ranking values
    save_path : str, optional
        Path to save the figure
    """
    proj = ccrs.LambertConformal(
        central_latitude=33, 
        central_longitude=-96, 
        standard_parallels=(33.0, 45.0)
    )
    
    flowlines_gdf_with_rmse = flowlines_gdf_with_rmse.to_crs(proj.proj4_params)
    
    fig, ax = plt.subplots(figsize=(7, 5), dpi=600, subplot_kw={'projection': proj})
    ax.set_extent([-106.65, -93.0, 25.0, 36.5], crs=ccrs.PlateCarree())
    ax.spines['geo'].set_visible(False)
    
    # Plot flowlines
    lines = LineCollection(
        [np.array(geometry.xy).T for geometry in flowlines_gdf_with_rmse.geometry],
        linewidths=0.025, 
        alpha=0.8, 
        color='black', 
        zorder=1
    )
    ax.add_collection(lines)

    lines = LineCollection(
        [np.array(geometry.xy).T for geometry in flowlines_gdf_with_rmse.geometry],
        linewidths=1, 
        alpha=1, 
        zorder=1
    )
    lines.set_array(flowlines_gdf_with_rmse['Median Rank'])
    lines.set_cmap('viridis_r')
    ax.add_collection(lines)
    
    # Add colorbar
    cb_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])
    cb = fig.colorbar(lines, cax=cb_ax, orientation='vertical', label='Sensor Rank')
    
    # Add map features
    ax.add_feature(feature.BORDERS, linestyle='-', alpha=.2)
    ax.add_feature(feature.STATES, linestyle=':', alpha=.2)
    ax.legend(frameon=False, loc='best')
    
    plt.subplots_adjust(left=0.05, right=0.8, top=0.95, bottom=0.1)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
    
    plt.show()

def plot_correlations_bar(correlations, p_values, significance_threshold=0.05, figsize=(12, 6), dpi=300):
    """
    Create a bar chart of correlations with significance highlighting.
    
    Parameters:
    -----------
    correlations : pandas.Series
        Series containing correlation values
    p_values : pandas.Series
        Series containing p-values
    significance_threshold : float, optional
        Threshold for statistical significance
    figsize : tuple, optional
        Figure size (width, height)
    dpi : int, optional
        Figure resolution
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Create bar plot
    bars = ax.bar(np.arange(len(correlations)), correlations)
    
    # Color the bars based on significance
    for i, (bar, p_val) in enumerate(zip(bars, p_values)):
        bar.set_color('dodgerblue' if p_val < significance_threshold else 'lightgray')
    
    # Add zero reference line
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    # Add labels
    ax.set_ylabel('Spearman correlation')
    
    # Set x-axis ticks and labels
    ax.set_xticks(np.arange(len(correlations)))
    ax.set_xticklabels(correlations.index, rotation=90, ha='right', fontsize=8)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    
    return fig 