import os
import matplotlib.pyplot as plt
import numpy as np


os.makedirs("2_plots", exist_ok=True)

############## LayerNorm ##############

def generate_data(num_points):
    np.random.seed(42)
    return np.random.rand(num_points, 3) * 6 - 3  # Random points in 3 dimensions, scaled up

def layer_normalize(data):
    mu = np.mean(data, axis=1, keepdims=True)
    sigma = np.std(data, axis=1, keepdims=True)
    return (data - mu) / sigma

# Generate random 3D data and do layer norm
data = generate_data(100)
normalized_data = layer_normalize(data)

# Plane perpendicular to the [1, 1, 1] vector
normal = np.array([1, 1, 1])
point = np.array([0, 0, 0])  # Point through which the plane passes
xx, yy = np.meshgrid(range(-3, 4), range(-3, 4))
# Calculate the corresponding z values, using general equation of a plane
d = -point.dot(normal)
zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]

fig = plt.figure(figsize=(12, 6.5))
ax1 = fig.add_subplot(121, projection="3d")
ax1.scatter(
    data[:, 0], data[:, 1], data[:, 2], c=np.linspace(0, 1, data.shape[0]), cmap="Dark2"
)
ax1.set_title("Original Points")
ax1.set_box_aspect([1, 1, 1])
ax1.view_init(azim=-45)

ax2 = fig.add_subplot(122, projection="3d")
scatter = ax2.scatter(normalized_data[:, 0], normalized_data[:, 1], normalized_data[:, 2],
    c=np.linspace(0, 1, normalized_data.shape[0]), cmap="Dark2"
)
ax2.plot_surface(xx, yy, zz, alpha=0.2, color="turquoise")
ax2.quiver(point[0], point[1], point[2], normal[0], normal[1], normal[2],
    length=1.0, color="black", label="Normal Vector", arrow_length_ratio=0.1,
)
ax2.set_title("Normalized Points")
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)
ax2.set_zlim(-3, 3)
ax2.set_box_aspect([1, 1, 1])
ax2.view_init(azim=-45)
ax2.legend()
plt.tight_layout()
plt.savefig("2_plots/01_layer_normalization_3d.jpg", dpi=250)
