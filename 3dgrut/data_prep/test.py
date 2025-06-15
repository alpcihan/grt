import numpy as np
import struct
import random

N = 5  # or any other value

# positions (N, 3)
positions = [[x - (N / 2), 0, 0] for x in range(N)]
position_np = np.array(positions, dtype=np.float32)

# scales (N, 3)
scales = [[100, 100, 100]] * N
scale_np = np.array(scales, dtype=np.float32)

# rotations (N, 4)
rotations = [[0, 0, 0, 1]] * N
rotate_np = np.array(rotations, dtype=np.float32)

# random albedos (N, 3)
albedos = [[random.random(), random.random(), random.random()] for _ in range(N)]
features_albedo_np = np.array(albedos, dtype=np.float32)

# densities (N, 1)
densities = [[0.1]] * N 
density_np = np.array(densities, dtype=np.float32)

# Random specular features (N, 45)
speculars = [[1] * 45] * N
features_specular_np = np.array(speculars, dtype=np.float32)

# Print shape info
print("Shape info:")
print("N (number of points):", position_np.shape[0])
print("pos_dim:", position_np.shape[1])
print("scale_dim:", scale_np.shape[1])
print("rotate_dim:", rotate_np.shape[1])
print("features_albedo_dim:", features_albedo_np.shape[1])
print("features_specular_dim:", features_specular_np.shape[1])
print("density_dim:", density_np.shape[1])
print()

def print_first_last(name, arr):
    print(f"{name} first element:", arr[0])
    print(f"{name} last element:", arr[-1])
    print()

print_first_last("position", position_np)
print_first_last("scale", scale_np)
print_first_last("rotate", rotate_np)
print_first_last("features_albedo", features_albedo_np)
print_first_last("features_specular", features_specular_np)
print_first_last("density", density_np)

# Write to data.bin
with open("data_test.bin", "wb") as f:
    # Write header (7 integers)
    header = (
        position_np.shape[0],  # N
        position_np.shape[1],  # pos_dim
        scale_np.shape[1],     # scale_dim
        rotate_np.shape[1],    # rotate_dim
        features_albedo_np.shape[1],  # albedo_dim
        features_specular_np.shape[1], # specular_dim
        density_np.shape[1],    # density_dim
    )
    f.write(struct.pack('<7i', *header))

    # Write all data sequentially
    f.write(position_np.tobytes())
    f.write(scale_np.tobytes())
    f.write(rotate_np.tobytes())
    f.write(features_albedo_np.tobytes())
    f.write(features_specular_np.tobytes())
    f.write(density_np.tobytes())

print("Data and header written to data.bin successfully.")
