from hydrosensornet import sensor_placement_qr
import numpy as np

# Create test data
test_data = np.random.rand(100, 20)
n_sensors = 5

# Test the function
selected_sensors = sensor_placement_qr(test_data, n_sensors)
print(f"Selected sensor locations: {selected_sensors}")