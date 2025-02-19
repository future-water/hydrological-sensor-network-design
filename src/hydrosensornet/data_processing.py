"""Data processing utilities for sensor network analysis."""

import pandas as pd
import geopandas as gpd
import xarray as xr

def load_data(flowdata_paths, gauge_shapefile, flowlines_shapefile, gauge_index_file):
    """
    Load and combine all necessary datasets.
    
    Parameters:
    -----------
    flowdata_paths : list
        List of paths to flow data files
    gauge_shapefile : str
        Path to gauge shapefile
    flowlines_shapefile : str
        Path to flowlines shapefile
    gauge_index_file : str
        Path to gauge index file
        
    Returns:
    --------
    tuple
        (df_cleaned, gauges_gdf, flowlines_gdf, usgs_index_df)
    """
    # ... existing implementation ...

# ... other data processing functions ... 