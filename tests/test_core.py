import numpy as np
import pytest
from hydrosensornet import sensor_placement_qr

def test_sensor_placement_qr():
    # Create a simple test matrix
    X = np.random.rand(100, 20)  # 100 timesteps, 20 locations
    r = 5  # number of sensors to select
    
    # Test basic functionality
    selected_sensors = sensor_placement_qr(X, r)
    assert len(selected_sensors) == r
    assert all(isinstance(i, (int, np.integer)) for i in selected_sensors)
    assert max(selected_sensors) < X.shape[1]
