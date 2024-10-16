import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import GPT2Model, GPT2Tokenizer


os.makedirs("2_plots", exist_ok=True)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_attentions=True)
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
    length=1.0, color="black", label="Normal Vector", arrow_length_ratio=0.1
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
        if "ln_1" in name:
            gammas_ln1.append(gamma)
        elif "ln_2" in name:
            gammas_ln2.append(gamma)

fig, axs = plt.subplots(4, 3, figsize=(12, 8))
layer_count = 0
for i, gamma in enumerate(gammas_ln1):
    ax = axs[i // 3, i % 3]
    ax.scatter(log_std_pos, gamma, alpha=0.6, color="salmon")
    ax.set_title(f"Layer {i+1} LN1 Weights")
    ax.set_xlabel("Log Std of Position Embeddings")
    ax.set_ylabel("Log |Gamma| values")
    layer_count = i
plt.tight_layout()
plt.savefig("2_plots/02_ln1_gammas.jpg", dpi=250)

fig, axs = plt.subplots(4, 3, figsize=(12, 8))
layer_count = 0
for i, gamma in enumerate(gammas_ln2):
    ax = axs[i // 3, i % 3]
    ax.scatter(log_std_pos, gamma, alpha=0.6, color="mediumseagreen")
    ax.set_title(f"Layer {i+1} LN2 Weights")
    ax.set_xlabel("Log Std of Position Embeddings")
    ax.set_ylabel("Log |Gamma| values")
    layer_count = i
plt.tight_layout()
plt.savefig("2_plots/03_ln2_gammas.jpg", dpi=250)


############## Attention ##############

text = "Alice gave Bob a goose, and he gave her a duck."

inputs = tokenizer(text, return_tensors="pt")
with torch.inference_mode():
    outputs = model(**inputs)
attentions = (
    outputs.attentions
)  # (num_layers, batch_size, num_heads, seq_length, seq_length)

num_layers = len(attentions)  # 12 for GPT-2 base model
num_heads = attentions[0].shape[1]  # 12 heads per layer
seq_length = attentions[0].shape[2]

print(f"Number of layers: {num_layers}")
print(f"Number of heads per layer: {num_heads}")
print(f"Shape of attention tensor for the first layer: {attentions[0][0].shape}")

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

fig, axes = plt.subplots(3, 4, figsize=(12, 8))
for layer in range(num_layers):
    # Get the first attention head for the current layer
    attention_matrix = attentions[layer][0, 0].numpy()  # (seq_length, seq_length)

    # Create a mask to black out the top-right triangle
    mask = np.triu(np.ones(attention_matrix.shape), k=1)
    attention_matrix_masked = np.where(mask == 1, np.nan, attention_matrix)

    ax = axes[layer // 4, layer % 4]
    im = ax.imshow(attention_matrix_masked, cmap="viridis")
    ax.set_title(f"Layer {layer + 1}, Head 1")
    ax.set_xticks(range(len(tokens)), tokens, rotation=90, fontsize=8)
    ax.set_yticks(range(len(tokens)), tokens, fontsize=8)
plt.tight_layout()
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
plt.savefig("2_plots/04_attention_heatmaps.jpg", dpi=250)


blue_gradient = [plt.cm.Blues(i / num_layers) for i in range(num_layers)]
red_gradient = [plt.cm.Reds(i / num_layers) for i in range(num_layers)]

# Loop through each layer to compute and plot the aggregate attention layer by layer
fig, ax = plt.subplots(figsize=(12, 6))
for layer in range(num_layers):
    aggregate_first_token_attention = np.zeros(seq_length)
    aggregate_last_token_attention = np.zeros(seq_length)

    for head in range(num_heads):
        attention_matrix = attentions[layer][0, head].numpy()
        aggregate_first_token_attention += attention_matrix[
            :, 0
        ]  # Attention of all tokens to the first token
        aggregate_last_token_attention += attention_matrix[
            -1, :
        ]  # Attention of the last token to previous tokens
    # Average over all heads in the layer
    aggregate_first_token_attention /= num_heads
    aggregate_last_token_attention /= num_heads

    ax.plot(
        range(seq_length),
        aggregate_first_token_attention,
        label=f"Layer {layer + 1} First Token",
        color=blue_gradient[layer],
    )
    ax.plot(
        range(seq_length),
        aggregate_last_token_attention,
        label=f"Layer {layer + 1} Last Token",
        linestyle="--",
        color=red_gradient[layer],
    )

sm_blue = plt.cm.ScalarMappable(
    cmap="Blues", norm=plt.Normalize(vmin=1, vmax=num_layers)
)
sm_red = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(vmin=1, vmax=num_layers))
cbar_blue = plt.colorbar(
    sm_blue,
    ax=ax,
    label="Layer Number (Attention to First Token)",
    fraction=0.05,
    pad=0.07,
)
cbar_red = plt.colorbar(
    sm_red, ax=ax, label="Layer Number (Attention of Last Token)", fraction=0.05
)

ax.set_title("Layer-wise Attention to First Tokens and and of Last Token")
ax.set_xlabel("Token Index")
ax.set_ylabel("Average Attention Weight")
ax.grid(True)
plt.tight_layout()
plt.savefig("2_plots/05_layer_wise_attention.jpg", dpi=250)


############## MLP ##############

mlp = model.h[0].mlp
hidden_layer_size = mlp.c_fc.weight.shape[1]  # 3072
print(f"The hidden layer size in the MLP of GPT-2 is: {hidden_layer_size}")

# test superposition of vectors
num_vectors = 1000
vector_ndim = 100
vect_matrix = torch.randn(num_vectors, vector_ndim)
vect_matrix /= vect_matrix.norm(p=2, dim=1, keepdim=True)  # Normalize
vect_matrix.requires_grad_(True)

# Optimization loop to create nearly-perpendicular vectors
optimizer = torch.optim.Adam([vect_matrix], lr=0.01)
num_steps = 300
dot_diff_cutoff = 0.01

losses = []
big_id = torch.eye(num_vectors, num_vectors)

for step_num in tqdm(range(num_steps), desc="Superposition experiment..."):
    optimizer.zero_grad()

    dot_products = vect_matrix @ vect_matrix.T
    # Punish deviation from orthogonal
    diff = dot_products - big_id
    loss = (diff.abs() - dot_diff_cutoff).relu().sum()
    # Extra incentive to keep rows normalized
    loss += num_vectors * diff.diag().pow(2).sum()

    loss.backward()
    optimizer.step()
    losses.append(loss.item())

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Loss Curve")
plt.grid(True)

# Angle Distribution
dot_products = vect_matrix @ vect_matrix.T
norms = torch.sqrt(torch.diag(dot_products))
normed_dot_products = dot_products / torch.outer(norms, norms)
clamped_dot_products = torch.clamp(normed_dot_products, -1, 1)
angles_degrees = torch.rad2deg(torch.acos(clamped_dot_products.detach()))

self_orthogonality_mask = ~torch.eye(num_vectors, num_vectors).bool()  # Ignore self-orthogonality
plt.subplot(1, 2, 2)
plt.hist(angles_degrees[self_orthogonality_mask].numpy().ravel(), bins=1000)
plt.title("Angle Distribution")
plt.xlim((89, 91))
plt.grid(True)
plt.tight_layout()
plt.savefig("2_plots/06_angle_distribution.jpg", dpi=250)
