import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from transformers import GPT2Model, GPT2Tokenizer


os.makedirs("1_plots", exist_ok=True)

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()
print("Loaded GPT-2 model!")

# Count total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params // 10**6} Millions")

# Extract embedding matrices from the model
word_embeddings = model.wte.weight.detach()  # Word Token Embeddings
position_embeddings = model.wpe.weight.detach()  # Word Position Embeddings

print(
    f"Word Embeddings shape (vocab_size, embedding_dim): {tuple(word_embeddings.shape)}"
)
print(
    f"Positional Embeddings shape (context_size, embedding_dim): {tuple(position_embeddings.shape)}"
)

vocab_size, embedding_dim = word_embeddings.shape  # (50257, 768)
context_length, embedding_dim = position_embeddings.shape  # (1024, 768)

# Inspect the first 300 tokens
print(f"First 300 tokens: {tokenizer.convert_ids_to_tokens(range(300))}")


############## Word embeddings exploration ##############

g = torch.Generator().manual_seed(42)
# Select 500 random word indices and 150 random dimension indices
random_indices = torch.randperm(vocab_size, generator=g)[:500]
random_dimensions = torch.randperm(embedding_dim, generator=g)[:150]
random_embeddings_chunk = word_embeddings[random_indices][:, random_dimensions]  # (500, 150)

# Plot a random chunk of the token embedding matrix
plt.figure(figsize=(12, 5))
plt.imshow(random_embeddings_chunk.T, aspect="auto", cmap="viridis")
plt.xlabel("Randomly Selected Tokens")
plt.ylabel("Randomly Selected Dimensions")
plt.title("Chunk of the Token Embedding Matrix")
plt.colorbar(label="Embedding Value")
plt.tight_layout()
plt.savefig("1_plots/01_token_embedding_chunk.jpg", dpi=250, bbox_inches="tight", pad_inches=0.1)


# Calculate mean and standard deviation along each embedding dimension
mean_values = word_embeddings.mean(dim=0)
std_values = word_embeddings.std(dim=0)

# Plot the histogram for the mean and standard deviation values
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(mean_values, bins=50, color="deepskyblue")
axes[0].set_xlabel("Mean Values")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Distribution of Mean Across Dimensions")

axes[1].hist(std_values, bins=50, color="salmon")
axes[1].set_xlabel("Standard Deviation Values")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Distribution of Standard Deviation Across Dimensions")
plt.tight_layout()
plt.savefig("1_plots/02_mean_std_distributions.jpg", dpi=250)


# Extract the first 30 embedding dimensions
first_30_embeddings = word_embeddings[:, :30]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Plot the distribution of values in this subset
for i in range(30):
    axes[0].hist(first_30_embeddings[:, i], bins=100, alpha=0.3)
axes[0].set_xlabel("Embedding Value")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Distribution of Values for the First 30 Embedding Dimensions")

# Plot mean vs std
axes[1].scatter(mean_values, std_values, alpha=0.7)
axes[1].set_xlabel("Mean Values")
axes[1].set_ylabel("Standard Deviation Values")
axes[1].set_title("Mean vs Standard Deviation for all Dimensions")
plt.tight_layout()
plt.savefig("1_plots/03_embedding_distributions_and_scatter.jpg", dpi=250)


############## Position embeddings exploration ##############

# Extract the selected dimensions from the position embedding matrix
selected_position_embeddings = position_embeddings[:, random_dimensions]  # (1024, 150)
# Plot a random chunk of the position embedding matrix
plt.figure(figsize=(12, 5))
plt.imshow(selected_position_embeddings.T, aspect="auto", cmap="viridis")
plt.xlabel("Position (Context Length)")
plt.ylabel("Randomly Selected Dimensions")
plt.title("Chunk of the Position Embedding Matrix")
plt.colorbar(label="Embedding Value")
plt.tight_layout()
plt.savefig("1_plots/04_position_embedding_chunk.jpg", dpi=250, bbox_inches="tight", pad_inches=0.1)


# Plot position embeddings as functions of context position
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i in range(embedding_dim):
    axes[0].plot(position_embeddings[:, i])
axes[0].set_xlabel("Context Position")
axes[0].set_ylabel("Embedding Value")
axes[0].set_title("Position Embeddings Across Context Positions")

# Plot standard deviation by dimension
axes[1].hist(torch.log10(position_embeddings.std(axis=0)), bins=50, color="mediumseagreen")
axes[1].set_xlabel("Log10 of Standard Deviation")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Distribution of PE Standard Deviation Across Dimensions")
plt.tight_layout()
plt.savefig("1_plots/05_position_embedding_distributions.jpg", dpi=250)


############## Interactions exploration ##############

mean_tok = torch.abs(word_embeddings.mean(dim=0))
std_tok = word_embeddings.std(dim=0)
mean_pos = torch.abs(position_embeddings.mean(dim=0))
std_pos = position_embeddings.std(dim=0)

# Plot means and standard deviations of token and position embeddings across dimensions
plt.figure(figsize=(12, 5))
plt.scatter(
    std_pos,
    std_tok,
    c="green",
    alpha=0.5,
    label="Standard Deviation of Token Embeddings",
)
plt.scatter(std_pos, mean_tok, c="blue", alpha=0.5, label="|Mean of Token Embeddings|")
plt.scatter(
    std_pos,
    std_pos,
    c="orange",
    alpha=0.5,
    label="Standard Deviation of Position Embeddings",
)
plt.scatter(std_pos, mean_pos, c="red", alpha=0.5, label="|Mean of Position Embeddings|")
plt.xlabel("Standard Deviation of Position Embeddings (Log Scale)")
plt.ylabel("Summary Statistics (Log Scale)")
plt.title("Summary Stats of Position and Token Embeddings")
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-5, 1)
plt.legend()
plt.tight_layout()
plt.savefig("1_plots/06_summary_stats.jpg", dpi=250)


# Plot word embeddings and positional embeddings for dimension pairs
pairs = [(87, 138), (68, 85), (361, 724)]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (dim1, dim2) in enumerate(pairs):
    # Plot word embeddings
    axes[i].scatter(
        word_embeddings[:, dim1],
        word_embeddings[:, dim2],
        c="black",
        s=1,
        alpha=0.5,
        label="Words",
    )
    # Plot position embeddings
    axes[i].scatter(
        position_embeddings[:, dim1],
        position_embeddings[:, dim2],
        c="red",
        s=1,
        alpha=0.5,
        label="Positions",
    )
    axes[i].set_title(f"Dimensions {dim1} vs {dim2}")
    axes[i].set_xlabel(f"Dimension {dim1}")
    axes[i].set_ylabel(f"Dimension {dim2}")
    axes[i].legend()
plt.tight_layout()
plt.savefig("1_plots/07_word_position_scatter.jpg", dpi=250)


############## Dimensionality analysis ##############

# PCA on the entire global set of word embeddings
pca = PCA()
pca.fit(word_embeddings)
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Multiple PCAs applied to local subsets of 1000 neighbors
num_local_pcas = 10
k_neighbors = 1000

# Fit the nearest neighbors model once on the whole dataset
neighbors_model = NearestNeighbors(n_neighbors=k_neighbors)
neighbors_model.fit(word_embeddings)

local_cumulative_variances = []
np.random.seed(42)

for _ in range(num_local_pcas):
    # Choose a random point to define the local neighborhood
    random_idx = np.random.randint(0, word_embeddings.shape[0])
    word_embedding = word_embeddings[random_idx]

    # Find the k-nearest neighbors of the selected embedding
    distances, indices = neighbors_model.kneighbors(word_embedding.reshape(1, -1))

    # Get the embeddings of the k-nearest neighbors
    local_embeddings = word_embeddings[indices[0]]

    # Apply PCA on the local neighborhood
    local_pca = PCA()
    local_pca.fit(local_embeddings)

    # Get the cumulative explained variance for this local subset
    local_cumulative_variance = np.cumsum(local_pca.explained_variance_ratio_)
    local_cumulative_variances.append(local_cumulative_variance)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# PCA on all word embeddings
axes[0].plot(cumulative_explained_variance, marker="o")
axes[0].set_xlabel("Number of Principal Components")
axes[0].set_ylabel("Cumulative Explained Variance")
axes[0].set_title("Cumulative Explained Variance (Word Embeddings)")
axes[0].grid(True)

# Plot cumulative explained variance for each local subset
for i, local_cumulative_variance in enumerate(local_cumulative_variances):
    axes[1].plot(
        np.arange(1, len(local_cumulative_variance) + 1),
        local_cumulative_variance,
        marker="o",
        linestyle="-",
        alpha=0.5,
        label=f"Subset {i+1}",
    )
axes[1].set_xlabel("Number of Principal Components")
axes[1].set_ylabel("Cumulative Explained Variance")
axes[1].set_title("Cumulative Explained Variance (Local Subsets of 1000 WE)")
axes[1].grid(True)
axes[1].legend(loc="lower right", fontsize="small")
plt.tight_layout()
plt.savefig("1_plots/08_pca_analysis_word_embeddings.jpg", dpi=250)

# Determine intrinsic dimensionality (90% variance explained) for whole dataset
intrinsic_dimensionality = np.argmax(cumulative_explained_variance >= 0.90) + 1
print(
    f"Intrinsic dimensionality for whole dataset (90% variance explained): {intrinsic_dimensionality}"
)

# Print intrinsic dimensionality for each local PCA subset
local_intrinsic_dims = []
for i, local_cumulative_variance in enumerate(local_cumulative_variances):
    local_intrinsic_dimensionality = np.argmax(local_cumulative_variance >= 0.90) + 1
    local_intrinsic_dims.append(local_intrinsic_dimensionality)

print(
    f"Intrinsic dimensionalities for {num_local_pcas} local subsets (90% variance explained): {local_intrinsic_dims}"
)


# Perform PCA on position embeddings
pca = PCA()
pca.fit(position_embeddings)
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Determine intrinsic dimensionality (90% variance explained)
intrinsic_dimensionality = np.argmax(cumulative_explained_variance >= 0.90) + 1
print(
    f"Intrinsic dimensionality for position embeddings (90% variance explained): {intrinsic_dimensionality}"
)

# Perform PCA to reduce to 3 dimensions
pca_3d = PCA(n_components=3)
reduced_pos_embeddings = pca_3d.fit_transform(position_embeddings)
position = np.arange(len(reduced_pos_embeddings))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Plot cumulative explained variance
axes[0].plot(cumulative_explained_variance, marker="o")
axes[0].set_xlabel("Number of Principal Components")
axes[0].set_ylabel("Cumulative Explained Variance")
axes[0].set_title("Cumulative Explained Variance (Position Embeddings)")
axes[0].grid(True)

# Plot 3D scatter plot of PCA projection
ax_3d = fig.add_subplot(122, projection="3d")
scatter = ax_3d.scatter(
    reduced_pos_embeddings[:, 0],
    reduced_pos_embeddings[:, 1],
    reduced_pos_embeddings[:, 2],
    c=position,
    marker="o",
    cmap="Spectral",
)
ax_3d.set_xlabel("Principal Component 1")
ax_3d.set_ylabel("Principal Component 2")
ax_3d.set_zlabel("Principal Component 3")
ax_3d.set_title("3D Scatter Plot of Reduced Position Embeddings")
ax_3d.view_init(elev=20, azim=45)
colorbar = plt.colorbar(scatter, ax=ax_3d)
colorbar.set_label("Position Index")
axes[1].axis("off")
plt.tight_layout()
plt.savefig("1_plots/09_position_embedding_pca.jpg", dpi=250)


# Cosine similarity matrix for positional embeddings
cosine_sim_matrix = cosine_similarity(position_embeddings)  # (1024, 1024)

# Transform word embeddings using the same PCA transformation used for positional embeddings
reduced_word_embeddings = pca_3d.transform(word_embeddings)

fig = plt.figure(figsize=(12, 5))
# Cosine similarity heatmap
ax1 = fig.add_subplot(121)
sns.heatmap(cosine_sim_matrix, cmap="coolwarm", ax=ax1)
tick_positions = np.arange(0, 1025, 256)
ax1.set_xticks(tick_positions)
ax1.set_yticks(tick_positions)
ax1.set_xticklabels(tick_positions)
ax1.set_yticklabels(tick_positions)
ax1.set_title("Cosine Similarity of Positional Embeddings")
ax1.set_xlabel("Position Index")
ax1.set_ylabel("Position Index")

# 3D PCA of positional and word embeddings
ax2 = fig.add_subplot(122, projection="3d")
# Plot positional embeddings
ax2.scatter(
    reduced_pos_embeddings[:, 0],
    reduced_pos_embeddings[:, 1],
    reduced_pos_embeddings[:, 2],
    c="blue",
    label="Positional Embeddings",
)
# Plot word embeddings
ax2.scatter(
    reduced_word_embeddings[:, 0],
    reduced_word_embeddings[:, 1],
    reduced_word_embeddings[:, 2],
    c="green",
    label="Word Embeddings",
)
ax2.view_init(elev=20, azim=45)
ax2.set_xlabel("Principal Component 1")
ax2.set_ylabel("Principal Component 2")
ax2.set_zlabel("Principal Component 3")
ax2.set_title("3D PCA of Positional and Word Embeddings (Fitted on Positional)")
ax2.legend()
plt.tight_layout()
plt.savefig("1_plots/10_similarity_and_combined_pca.jpg", dpi=250)
