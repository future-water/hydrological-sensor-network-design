"""
Sensor Network Package

A package for sensor network analysis and optimization.
"""

from .core import (
    sensor_placement_qr,
    reconstruction_evaluation,
    calculate_performance_metrics
)

from .visualization import (
    plot_sensor_network,
    plot_sensor_ranking,
    # plot_sensor_network_expansion,
    # plot_flood_risk_map
)

from .data_processing import (
    load_data,
    # prepare_usgs_indices,
    # split_data,
    # process_region
)

__version__ = "0.1.0" 