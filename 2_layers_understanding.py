import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Model


os.makedirs("2_plots", exist_ok=True)

model = GPT2Model.from_pretrained('gpt2')
model.eval()
position_embeddings = model.wpe.weight.detach()  # Word Position Embeddings
log_std_pos = torch.log10(position_embeddings.std(dim=0))

############## LayerNorm ##############

def generate_data(num_points, seed=42):
    torch.manual_seed(seed)
    return torch.rand(num_points, 3) * 6 - 3  # Random points in 3 dimensions, scaled up

# Generate random 3D data and do layer norm
data = generate_data(100)
layer_norm = nn.LayerNorm(normalized_shape=data.size()[1])
normalized_data = layer_norm(data).detach().numpy()

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


gammas_ln1 = []  # Before attention layers
gammas_ln2 = []  # Before MLPs

# Collect gamma parameters for each LayerNorm layer, take abs and log
for name, module in model.named_modules():
    if isinstance(module, torch.nn.LayerNorm):
        gamma = torch.log10(torch.abs(module.weight)).detach().numpy()
        if 'ln_1' in name:
            gammas_ln1.append(gamma)
        elif 'ln_2' in name:
            gammas_ln2.append(gamma)

fig, axs = plt.subplots(4, 3, figsize=(12, 8))
layer_count = 0
for i, gamma in enumerate(gammas_ln1):
    ax = axs[i // 3, i % 3]
    ax.scatter(log_std_pos, gamma, alpha=0.6, color='salmon')
    ax.set_title(f'Layer {i+1} LN1 Weights')
    ax.set_xlabel('Log Std of Position Embeddings')
    ax.set_ylabel('Log |Gamma| values')
    layer_count = i
plt.tight_layout()
plt.savefig('2_plots/02_ln1_gammas.png', dpi=250)

fig, axs = plt.subplots(4, 3, figsize=(12, 8))
layer_count = 0
for i, gamma in enumerate(gammas_ln2):
    ax = axs[i // 3, i % 3]
    ax.scatter(log_std_pos, gamma, alpha=0.6, color='mediumseagreen')
    ax.set_title(f'Layer {i+1} LN2 Weights')
    ax.set_xlabel('Log Std of Position Embeddings')
    ax.set_ylabel('Log |Gamma| values')
    layer_count = i
plt.tight_layout()
plt.savefig('2_plots/03_ln2_gammas.png', dpi=250)
