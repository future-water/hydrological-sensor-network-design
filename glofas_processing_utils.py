import glob
from pathlib import Path
import warnings
import urllib3
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import xarray.ufuncs as xu
import geopandas as gpd
from shapely.geometry import Point
from scipy import linalg

warnings.filterwarnings("ignore", category=RuntimeWarning)
urllib3.disable_warnings()

def load_boundary_shapefile(shapefile_path: Union[str, Path], dissolve: bool = True) -> any:
    """
    Load and process boundary shapefile.
    
    Parameters:
    - shapefile_path: Path to shapefile
    - dissolve: Whether to dissolve all features into one polygon
    
    Returns:
    - Shapely geometry object
    """
    gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    if dissolve:
        return gdf.dissolve().geometry.values[0]
    return gdf

def prepare_gauge_geodataframe(gauge_data: Union[str, Path, pd.DataFrame],
                              lat_col: str = "gauge_lat", 
                              lon_col: str = "gauge_lon",
                              id_col: str = "gauge_id") -> gpd.GeoDataFrame:
    """
    Prepare gauge data as GeoDataFrame.
    
    Parameters:
    - gauge_data: Path to CSV file or DataFrame
    - lat_col: Name of latitude column
    - lon_col: Name of longitude column  
    - id_col: Name of ID column
    
    Returns:
    - GeoDataFrame with gauge locations
    """
    if isinstance(gauge_data, (str, Path)):
        df = pd.read_csv(gauge_data)
    else:
        df = gauge_data.copy()
    
    # Clean and rename columns
    df = df.dropna(subset=[lat_col, lon_col])
    df = df.rename(columns={lat_col: "gauge_lat", lon_col: "gauge_lon"})
    
    # Create ID column if it doesn't exist
    if id_col not in df.columns:
        df["gauge_id"] = np.arange(len(df))
    else:
        df = df.rename(columns={id_col: "gauge_id"})
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.gauge_lon, df.gauge_lat),
        crs="EPSG:4326",
    )
    
    return gdf

def load_glofas_data(glofas_files: List[str]) -> xr.Dataset:
    """
    Load and concatenate GloFAS GRIB files.
    
    Parameters:
    - glofas_files: List of paths to GRIB files
    
    Returns:
    - xarray Dataset with concatenated data
    """
    datasets = [xr.open_dataset(f, engine="cfgrib", backend_kwargs={'indexpath': ''}, decode_timedelta=False) for f in glofas_files]
    glofas = xr.concat(datasets, dim="time").sortby("time")
    return glofas


def clip_to_region(glofas: xr.Dataset, boundary_poly, extent: List[float]) -> xr.Dataset:
    """
    Clip GloFAS data to a specific geographic region.
    
    Parameters:
    - glofas: xarray Dataset
    - boundary_poly: Shapely polygon for clipping
    - extent: [west, east, south, north] bounding box
    
    Returns:
    - Clipped xarray Dataset
    """
    # Fix longitude coordinates
    glofas = glofas.assign_coords(longitude=((glofas.longitude + 180) % 360) - 180)
    glofas = glofas.rio.write_crs("EPSG:4326", inplace=True)
    
    # Apply bounding box
    glofas = glofas.sel(
        longitude=slice(extent[0], extent[1]),
        latitude=slice(extent[3], extent[2])
    )

    # Clip to polygon
    glofas = glofas.rio.clip([boundary_poly], "EPSG:4326", drop=True)
    return glofas


def prepare_matrix(glofas: xr.Dataset, variable: str = "dis24") -> Tuple[np.ndarray, List[str]]:
    """
    Convert xarray Dataset to 2D matrix (time × cells) and handle NaNs.
    
    Parameters:
    - glofas: xarray Dataset
    - variable: Variable name to extract (default: "dis24")
    
    Returns:
    - Tuple of (matrix, valid_lat_lon_labels)
    """
    # Mask NaNs
    glofas_masked = glofas.where(~xu.isnan(glofas[variable]), drop=True)
    
    # Get coordinate values
    lat_vals = glofas_masked.latitude.values
    lon_vals = glofas_masked.longitude.values
    lat_lon_labels = [f"({lat:.4f}, {lon:.4f})" for lat in lat_vals for lon in lon_vals]
    
    # Stack and reshape
    dis24_da = (
        glofas_masked[variable]
        .stack(lat_lon=("latitude", "longitude"))
        .reset_index("lat_lon", drop=True)  # removes the MultiIndex
        .assign_coords(lat_lon=lat_lon_labels)
        .dropna("lat_lon", how="all")
    )
    
    matrix = dis24_da.values.astype(np.float32)
    valid_lat_lon = dis24_da.lat_lon.values
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Valid columns: {len(valid_lat_lon):,}")
    
    return dis24_da, matrix, valid_lat_lon


def align_gauges_to_grid(gauge_gdf: gpd.GeoDataFrame, 
                        lat_vals: np.ndarray, 
                        lon_vals: np.ndarray,
                        valid_lat_lon: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Align gauge locations to the nearest grid cells.
    
    Parameters:
    - gauge_gdf: GeoDataFrame with gauge locations
    - lat_vals: Array of latitude values from the grid
    - lon_vals: Array of longitude values from the grid
    - valid_lat_lon: List of valid lat/lon labels from prepare_matrix
    
    Returns:
    - Tuple of (sensor_columns_df, original_indices)
    """
    def nearest_idx(axis_vals, pts):
        return np.abs(axis_vals[:, None] - pts).argmin(axis=0)
    
    # Create lookup table
    lookup = pd.DataFrame({
        "lat_c": [float(s.split(',')[0][1:]) for s in valid_lat_lon],
        "lon_c": [float(s.split(',')[1][:-1]) for s in valid_lat_lon],
        "matrix_col": np.arange(len(valid_lat_lon)),
    }).round(4)
    
    # Find nearest grid cells for gauges
    lat_idx = nearest_idx(lat_vals, gauge_gdf.gauge_lat.values)
    lon_idx = nearest_idx(lon_vals, gauge_gdf.gauge_lon.values)
    
    gauges = pd.DataFrame({
        "gauge_id": gauge_gdf.gauge_id.values,
        "lat_c": lat_vals[lat_idx],
        "lon_c": lon_vals[lon_idx],
    }).round(4)
    
    sensor_cols = gauges.merge(lookup, on=["lat_c", "lon_c"], how="inner")
    sensor_column_indices_orig = sensor_cols.matrix_col.to_numpy()
    
    print(f"{sensor_column_indices_orig.size} gauges matched to grid cells")
    return gauges, sensor_cols, sensor_column_indices_orig


def train_test_split_and_filter(matrix: np.ndarray, 
                               train_split: float = 0.7) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Split data and filter out columns with NaN/infinite values in training set.
    
    Parameters:
    - matrix: Input matrix (time × cells)
    - train_split: Fraction of data to use for training (default: 0.7)
    
    Returns:
    - Tuple of (X_train, X_test, mapping_dict)
    """
    n_train = int(train_split * matrix.shape[0])
    X_train, X_test = matrix[:n_train, :], matrix[n_train:, :]
    
    # Filter out columns with NaN/inf in training
    finite_mask = np.isfinite(X_train).all(axis=0)
    good_cols = np.where(finite_mask)[0]
    X_train = X_train[:, good_cols]
    X_test = X_test[:, good_cols]
    
    print(f"Columns before filter: {matrix.shape[1]:,}")
    print(f"Columns after filter: {X_train.shape[1]:,}")
    
    # Create mapping from old to new column indices
    old_to_new = {old: new for new, old in enumerate(good_cols)}
    
    return X_train, X_test, {"old_to_new": old_to_new, "good_cols": good_cols}


def create_points_geodataframe(valid_lat_lon: List[str], 
                              mapping_dict: Dict) -> gpd.GeoDataFrame:
    """
    Create GeoDataFrame of grid cell centroids that survived filtering.
    
    Parameters:
    - valid_lat_lon: List of valid lat/lon labels from prepare_matrix
    - mapping_dict: Dictionary with column mappings from train_test_split_and_filter
    
    Returns:
    - GeoDataFrame with point geometries
    """
    coords = np.column_stack((
        [float(s.split(',')[0][1:]) for s in valid_lat_lon],
        [float(s.split(',')[1][:-1]) for s in valid_lat_lon],
    ))
    
    points_gdf = gpd.GeoDataFrame(
        {
            "matrix_col": np.arange(len(valid_lat_lon)),
            "lat": coords[:, 0],
            "lon": coords[:, 1],
        },
        geometry=[Point(lon, lat) for lat, lon in coords],
        crs="EPSG:4326",
    )
    
    # Keep only cells that survived filtering
    good_cols = mapping_dict["good_cols"]
    old_to_new = mapping_dict["old_to_new"]
    
    points_gdf = points_gdf[points_gdf.matrix_col.isin(good_cols)].copy()
    points_gdf["col_pos"] = points_gdf.matrix_col.map(old_to_new).astype(int)
    
    return points_gdf

def assign_basins(points_gdf: gpd.GeoDataFrame, 
                 basin_gdf: Optional[gpd.GeoDataFrame] = None,
                 country_name: str = "Country") -> gpd.GeoDataFrame:
    """
    Assign basin information to grid points.
    
    Parameters:
    - points_gdf: GeoDataFrame of grid points
    - basin_gdf: GeoDataFrame with basin polygons (optional)
    - country_name: Name to use if treating whole area as one basin
    
    Returns:
    - GeoDataFrame with basin assignments
    """
    if basin_gdf is not None:
        # First, ensure basin_gdf has the required columns
        if "RHI_CD" not in basin_gdf.columns or "RHI_NM" not in basin_gdf.columns:
            raise ValueError("basin_gdf must have 'RHI_CD' and 'RHI_NM' columns")
        
        # Use left join to keep all points
        points_with_basin = gpd.sjoin(
            points_gdf,
            basin_gdf[["RHI_CD", "RHI_NM", "geometry"]],
            how="left",
            predicate="within",
        )
        
        # Initialize columns if they don't exist
        if "RHI_CD" not in points_with_basin.columns:
            points_with_basin["RHI_CD"] = np.nan
        if "RHI_NM" not in points_with_basin.columns:
            points_with_basin["RHI_NM"] = None
            
        # For points that didn't match any basin, assign them to the nearest basin
        unassigned_mask = points_with_basin["RHI_CD"].isna()
        n_unassigned = unassigned_mask.sum()
        
        if n_unassigned > 0:
            print(f"Found {n_unassigned} points not within any basin. Assigning to nearest basin...")
    
            crs_projected = "EPSG:5880"
            
            unassigned_indices = points_with_basin[unassigned_mask].index
            
            # Project geometries for distance calculation
            points_proj = points_with_basin.loc[unassigned_mask].to_crs(crs_projected)
            basins_proj = basin_gdf.to_crs(crs_projected)
            
            # For each unassigned point, find nearest basin
            for idx in unassigned_indices:
                point_geom = points_proj.loc[idx, "geometry"]
                distances = basins_proj.geometry.distance(point_geom)
                nearest_idx = distances.idxmin()
                
                points_with_basin.at[idx, "RHI_CD"] = basin_gdf.at[nearest_idx, "RHI_CD"]
                points_with_basin.at[idx, "RHI_NM"] = basin_gdf.at[nearest_idx, "RHI_NM"]
        
        if "index_right" in points_with_basin.columns:
            points_with_basin = points_with_basin.drop(columns=["index_right"])
            
    else:
        points_with_basin = points_gdf.copy()
        points_with_basin["RHI_CD"] = 0
        points_with_basin["RHI_NM"] = country_name
        
    return points_with_basin

def count_gauges_per_basin(gauge_gdf: gpd.GeoDataFrame,
                          basin_gdf: Optional[gpd.GeoDataFrame] = None,
                          country_name: str = "Country",
                          total_gauges: int = None) -> Dict:
    """
    Count the number of gauges in each basin.
    
    Parameters:
    - gauge_gdf: GeoDataFrame with gauge locations
    - basin_gdf: GeoDataFrame with basin polygons (optional)
    - country_name: Name to use if treating whole area as one basin
    - total_gauges: Total number of gauges (used when basin_gdf is None)
    
    Returns:
    - Dictionary mapping basin names to gauge counts
    """
    if basin_gdf is not None:
        if "RHI_CD" not in basin_gdf.columns or "RHI_NM" not in basin_gdf.columns:
            raise ValueError("basin_gdf must have 'RHI_CD' and 'RHI_NM' columns")
    
        gauges_with_basin = gpd.sjoin(
            gauge_gdf[["gauge_id", "geometry"]],
            basin_gdf[["RHI_CD", "RHI_NM", "geometry"]],
            how="left",
            predicate="within",
        )
        
        if "RHI_CD" not in gauges_with_basin.columns:
            gauges_with_basin["RHI_CD"] = np.nan
        if "RHI_NM" not in gauges_with_basin.columns:
            gauges_with_basin["RHI_NM"] = None
            
        unassigned_mask = gauges_with_basin["RHI_CD"].isna()
        n_unassigned = unassigned_mask.sum()
        
        if n_unassigned > 0:
            print(f"Found {n_unassigned} gauges not within any basin. Assigning to nearest basin...")
        
            # Get unassigned indices
            unassigned_indices = gauges_with_basin[unassigned_mask].index
            
            crs_projected = "EPSG:5880"
            gauges_proj = gauges_with_basin.loc[unassigned_mask].to_crs(crs_projected)
            basins_proj = basin_gdf.to_crs(crs_projected)
            
            # Find nearest basin for each unassigned gauge
            for idx in unassigned_indices:
                gauge_geom = gauges_proj.loc[idx, "geometry"]
                distances = basins_proj.geometry.distance(gauge_geom)
                nearest_idx = distances.idxmin()
                
                gauges_with_basin.at[idx, "RHI_CD"] = basin_gdf.at[nearest_idx, "RHI_CD"]
                gauges_with_basin.at[idx, "RHI_NM"] = basin_gdf.at[nearest_idx, "RHI_NM"]
        
        if "index_right" in gauges_with_basin.columns:
            gauges_with_basin = gauges_with_basin.drop(columns=["index_right"])
            
        gauge_counts = gauges_with_basin.groupby("RHI_NM").size().to_dict()
        print(f"Total gauges assigned: {sum(gauge_counts.values())}")
    else:
        gauge_counts = {country_name: total_gauges or len(gauge_gdf)}
        
    return gauge_counts

def qr_pivot_selection(X_train: np.ndarray,
                      points_with_basin: gpd.GeoDataFrame,
                      gauge_counts: Dict) -> Tuple[pd.DataFrame, List[int]]:
    """
    Perform QR-pivot sensor selection within each basin.
    
    Parameters:
    - X_train: Training data matrix
    - points_with_basin: GeoDataFrame with basin assignments
    - gauge_counts: Dictionary of gauge counts per basin
    
    Returns:
    - Tuple of (selected_sensors_df, selected_indices_list)
    """
    selected_rows = []
    selected_sensor_indices = []
    
    for basin, grp in points_with_basin.groupby("RHI_NM"):
        k = gauge_counts.get(basin, 0)
        if k == 0:
            continue
            
        pos = grp.col_pos.to_numpy()
        if len(pos) == 0:
            continue
            
        k = min(k, len(pos))
    
        _, _, piv = linalg.qr(X_train[:, pos], mode="economic", pivoting=True)
        chosen_pos = pos[piv[:k]]
        selected_sensor_indices.extend(chosen_pos)
    
        sel = grp.loc[grp.col_pos.isin(chosen_pos),
                     ["RHI_CD", "RHI_NM", "matrix_col", "lat", "lon"]]
        selected_rows.append(sel)
    
    selected_sensors = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    
    print(f"\nSelected {len(selected_sensor_indices)} optimal sensor locations")
    return selected_sensors, selected_sensor_indices

def create_diff_nnse_map(diff_nnse, dis24_matrix):
    """
    Convert diff_nnse array back to spatial coordinates for mapping
    
    Parameters:
    -----------
    diff_nnse : numpy.array
        Array of ΔNNSE values
    dis24_matrix : xarray.DataArray
        The stacked discharge matrix with lat_lon coordinates
    glofas_data_clipped : xarray.Dataset
        The clipped GloFAS data for coordinate reference
        
    Returns:
    --------
    diff_nnse_spatial : xarray.DataArray
        ΔNNSE values mapped back to spatial coordinates
    """
    coords = []
    for coord_str in dis24_matrix.lat_lon.values:
        clean_str = coord_str.strip('()')
        lat, lon = map(float, clean_str.split(', '))
        coords.append([lat, lon])
    coords = np.array(coords)
    
    lats = coords[:, 0]
    lons = coords[:, 1]

    unique_lats = np.unique(lats)
    unique_lons = np.unique(lons)
    
    diff_nnse_grid = np.full((len(unique_lats), len(unique_lons)), np.nan)
    
    for i, (lat, lon) in enumerate(coords):
        if i < len(diff_nnse): 
            lat_idx = np.where(unique_lats == lat)[0][0]
            lon_idx = np.where(unique_lons == lon)[0][0]
            diff_nnse_grid[lat_idx, lon_idx] = diff_nnse[i]
    
    diff_nnse_spatial = xr.DataArray(
        diff_nnse_grid,
        coords={'latitude': unique_lats, 'longitude': unique_lons},
        dims=['latitude', 'longitude'],
        name='diff_nnse',
        attrs={'long_name': 'Change in NNSE', 'units': 'ΔNNSE'}
    )
    
    return diff_nnse_spatial