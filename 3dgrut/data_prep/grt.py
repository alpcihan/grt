import torch
import numpy as np
import struct

# Path to your .pt file
pt_path = "/home/alp/Desktop/3dgrut/runs/lego-1306_174837/ckpt_last.pt"

# Load the .pt file
data = torch.load(pt_path, map_location='cpu')

# Extract tensors
position = data['positions']        # (N, 3)
scale    = data['scale']            # (N, 3)
rotate   = data['rotation']         # (N, 4)
features_albedo   = data['features_albedo']    # (N, 3)
features_specular = data['features_specular']  # (N, 45)
density  = data['density']          # (N, 1)

# Detach and convert to numpy float32
position_np       = position.detach().cpu().numpy().astype(np.float32)
scale_np          = scale.detach().cpu().numpy().astype(np.float32)
rotate_np         = rotate.detach().cpu().numpy().astype(np.float32)
features_albedo_np   = features_albedo.detach().cpu().numpy().astype(np.float32)
features_specular_np = features_specular.detach().cpu().numpy().astype(np.float32)
density_np        = density.detach().cpu().numpy().astype(np.float32)

# Print keys and types
print("Model keys and types:")
for key, value in data.items():
    print(f"  {key}: {(value)}")
print()

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

def print_stats(name, arr):
    print(f"{name} first element: {arr[0]}")
    print(f"{name} last element:  {arr[-1]}")
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    for i in range(arr.shape[1]):
        print(f"{name} dim {i}: min = {mins[i]:.6f}, max = {maxs[i]:.6f}")
    print()

print_stats("position", position_np)
print_stats("scale", scale_np)
print_stats("rotate", rotate_np)
print_stats("features_albedo", features_albedo_np)
print_stats("features_specular", features_specular_np)
print_stats("density", density_np)

# Write to data.bin
with open("data.bin", "wb") as f:
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
