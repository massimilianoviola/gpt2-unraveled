import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Model, GPT2Tokenizer


class UMAPWithLaplacianInit:
    def __init__(self, n_neighbors=15, min_dist=0.1, n_components=3):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.umap_model = None

    def fit(self, data):
        # Compute Laplacian Eigenmap embedding
        laplacian_embedding = SpectralEmbedding(
            n_components=self.n_components
        ).fit_transform(data)

        # Use Laplacian embedding as initialization for UMAP
        self.umap_model = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=self.n_components,
            init=laplacian_embedding,
        ).fit(data)

        return self.umap_model.embedding_

    def transform(self, new_data):
        if self.umap_model is None:
            raise ValueError("Model needs to be fitted before transforming data.")

        return self.umap_model.transform(new_data)

os.makedirs("3_plots", exist_ok=True)

model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
embedding_matrix = model.wte.weight.detach().cpu().numpy()

sentence = "He killed two birds with one stone by taking a quick action"
inputs = tokenizer(sentence, return_tensors="pt")
input_ids = inputs["input_ids"]

with torch.inference_mode():
    output = model(**inputs)
hidden_states = output.hidden_states  # Hidden states for each layer

umap_reducer = UMAPWithLaplacianInit(n_components=3)
print("Fitting UMAP...")
umap_reducer.fit(embedding_matrix[:10000])  # Fit UMAP to a subset of the embedding matrix

emb = model.wte(input_ids).detach()  # Extract non-contextualized word embeddings for the input
emb_3d = umap_reducer.transform(emb.squeeze(0).cpu().numpy())

labels = [tokenizer.decode([i]) for i in input_ids.squeeze().tolist()]
# Choose a specific token to track
token_index = 5

# Collect embeddings for the specified token across all layers
token_embeddings = np.stack(
    [
        hidden_states[layer][0][token_index].cpu().numpy()
        for layer in range(len(hidden_states))
    ]
)

# Transform all embeddings at once using UMAP
token_trajectory = umap_reducer.transform(token_embeddings)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
# Scatter plot for initial embeddings of all tokens
ax.scatter(*emb_3d.T, c="blue", label="Initial Embeddings", alpha=0.5)
# Plot the trajectory for the selected token across layers
ax.plot(token_trajectory[:, 0], token_trajectory[:, 1], token_trajectory[:, 2],
    color="red", label="Trajectory"
)
# Final latent feature of the selected token only
ax.scatter(*token_trajectory[-1], c="purple",
    label='Final Latent Feature of "{}"'.format(labels[token_index]), alpha=0.7
)
for i, label in enumerate(labels):
    ax.text(*emb_3d[i], label, color="deepskyblue", fontsize=9)
ax.text(*token_trajectory[-1], labels[token_index], color="darkviolet", fontsize=12)
ax.set_xlabel("UMAP Component 1")
ax.set_ylabel("UMAP Component 2")
ax.set_zlabel("UMAP Component 3")
ax.set_title('Token Trajectory and Final Latent Feature for "{}"'.format(labels[token_index]))
ax.legend()
ax.set_box_aspect([1, 1, 1])
ax.view_init(azim=-45)
plt.savefig("3_plots/01_journey.jpg", dpi=250)


n_layers = len(hidden_states)  # Number of layers
n_tokens = input_ids.size(1)  # Number of tokens in the input

# Initialize matrices to store cosine similarities with current and next token embeddings
cosine_similarities_to_next = np.zeros((n_layers, n_tokens - 1))
cosine_similarities_to_self = np.zeros((n_layers, n_tokens))

# Iterate through each layer
for layer in range(n_layers):
    layer_hidden_states = (
        hidden_states[layer][0].cpu().numpy()
    )  # (n_tokens, hidden_dim)

    # Iterate through each token to compute similarity to own embedding
    for i in range(n_tokens):
        current_token_hidden_state = layer_hidden_states[i]
        current_token_id = input_ids[0, i].item()
        current_token_embedding = embedding_matrix[current_token_id]  # Non-contextualized

        # Compute cosine similarity between the current hidden state and the current token's embedding
        cosine_sim_self = cosine_similarity([current_token_hidden_state], [current_token_embedding])[0, 0]
        cosine_similarities_to_self[layer, i] = cosine_sim_self

    # Iterate through each token (except the last one) to compute similarity to the next token
    for i in range(n_tokens - 1):
        current_token_hidden_state = layer_hidden_states[i]
        next_token_id = input_ids[0, i + 1].item()
        next_token_embedding = embedding_matrix[next_token_id]  # Non-contextualized embedding of the next token

        # Compute cosine similarity between the current hidden state and the next token's embedding
        cosine_sim_next = cosine_similarity([current_token_hidden_state], [next_token_embedding])[0, 0]
        cosine_similarities_to_next[layer, i] = cosine_sim_next

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Plot similarity to next token embedding
for i in range(n_tokens - 1):  # Exclude the last token
    axs[0].plot(
        range(n_layers),
        cosine_similarities_to_next[:, i],
        label=f"{tokenizer.decode([input_ids[0, i]])} → {tokenizer.decode([input_ids[0, i + 1]])}",
    )
axs[0].set_xlabel("Layer")
axs[0].set_ylabel("Cosine Similarity to Next Token Embedding")
axs[0].set_title("Cosine Similarity to Next Token Embedding")
axs[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# Plot similarity to own token embedding
for i in range(n_tokens):
    axs[1].plot(
        range(n_layers),
        cosine_similarities_to_self[:, i],
        label=f"{tokenizer.decode([input_ids[0, i]])}",
    )
axs[1].set_xlabel("Layer")
axs[1].set_ylabel("Cosine Similarity to Own Token Embedding")
axs[1].set_title("Cosine Similarity to Own Token Embedding")
axs[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.savefig("3_plots/02_cosine_similarities.jpg", dpi=250)


# Load GPT-2 model with language modeling head for predictions
model_lm = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True)

with torch.inference_mode():
    output = model_lm(**inputs)
hidden_states = output.hidden_states  # Hidden states for each layer
logits = output.logits  # Logits for next-token prediction

# Get the predicted next token by taking the last token's logits
predicted_token_id = torch.argmax(logits[0, -1, :]).item()  # ID of the predicted token
predicted_token = tokenizer.decode([predicted_token_id])  # Decode the predicted token

# Non-contextualized embedding of the predicted token
predicted_token_embedding = (model_lm.transformer.wte.weight[predicted_token_id].detach().cpu().numpy())

n_layers = len(hidden_states)  # Number of layers
n_tokens = input_ids.size(1)  # Number of tokens in the input

# Matrices to store cosine similarities with the last token's hidden state and predicted token's embedding
cosine_similarities_to_last_token = np.zeros((n_layers, n_tokens))
cosine_similarities_to_predicted = np.zeros((n_layers, n_tokens))

for layer in range(n_layers):
    layer_hidden_states = (hidden_states[layer][0].cpu().numpy())  # (n_tokens, hidden_dim)
    last_token_hidden_state = layer_hidden_states[-1]  # Last token hidden state

    # Compute similarity to the last token's hidden state and predicted token's embedding
    for i in range(n_tokens):
        current_token_hidden_state = layer_hidden_states[i]

        # Compute cosine similarity to the last token's hidden state
        cosine_sim_last = cosine_similarity([current_token_hidden_state], [last_token_hidden_state])[0, 0]
        cosine_similarities_to_last_token[layer, i] = cosine_sim_last

        # Compute cosine similarity to the predicted token's non-contextualized embedding
        cosine_sim_predicted = cosine_similarity([current_token_hidden_state], [predicted_token_embedding])[0, 0]
        cosine_similarities_to_predicted[layer, i] = cosine_sim_predicted

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
# Plot similarity to last token hidden state
for i in range(n_tokens):
    axs[0].plot(range(n_layers), cosine_similarities_to_last_token[:, i],
        label=f"{tokenizer.decode([input_ids[0, i]])}"
    )
axs[0].set_xlabel("Layer")
axs[0].set_ylabel("Cosine Similarity to Last Token Hidden State")
axs[0].set_title("Cosine Similarity to Last Token Hidden State")
axs[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# Plot similarity to predicted token's non-contextualized embedding
for i in range(n_tokens):
    axs[1].plot(
        range(n_layers),
        cosine_similarities_to_predicted[:, i],
        label=f"{tokenizer.decode([input_ids[0, i]])}",
    )
axs[1].set_xlabel("Layer")
axs[1].set_ylabel("Cosine Similarity to Predicted Token")
axs[1].set_title(f'Cosine Similarity to Predicted Token "{predicted_token}"')
axs[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("3_plots/03_cosine_similarities_next.jpg", dpi=250)

print(f'Sentence: "{sentence}"')
print(f'Predicted next token: "{predicted_token}"')


# Set pad token to EOS token (GPT-2 does not have a pad token by default)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# List of sentences to process
sentences = [
    "He killed two birds with one stone by taking a quick action",
    "She won the race and broke the record in her age group by ten seconds",
    "A rolling stone gathers no moss, it moves through hill and dale \
without a single pause",
    "Despite the stormy weather and the warning from the news, \
we ventured out into the dense forest, hoping to catch a glimpse \
of the rare wildlife species",
    "In the quiet moments of the evening, as the sun began to set \
over the distant hills, casting a golden glow on the shimmering lake, \
I found myself lost in thought, reflecting on the decisions of the past \
and how they had led me to this particular point in my life"
]

# Tokenize input sentences and pad them to have the same length
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Pass inputs through the model to get hidden states
with torch.inference_mode():
    output = model(input_ids, attention_mask=attention_mask)
hidden_states = output.hidden_states

n_sentences = len(sentences)
colors = plt.cm.viridis(np.linspace(0, 1, n_sentences))

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
# Iterate through each layer and sentence to compute mean and covariance
for idx, sentence in enumerate(sentences):
    mean_distances = []
    cov_traces = []
    for layer in range(n_layers):
        layer_hidden_states = (hidden_states[layer][idx].cpu().numpy())  # (n_tokens, hidden_dim)

        # Filter out the pad tokens using the attention_mask
        active_hidden_states = layer_hidden_states[attention_mask[idx] == 1]

        mean_embedding = np.mean(active_hidden_states, axis=0)

        # Compute distances from the mean embedding and take the average
        distances = np.linalg.norm(active_hidden_states - mean_embedding, axis=1)
        mean_distance = np.mean(distances)
        mean_distances.append(mean_distance)

        # Compute the covariance matrix of the token embeddings if more than one active token
        if active_hidden_states.shape[0] > 1:
            cov_matrix = np.cov(active_hidden_states.T)
            cov_trace = np.trace(cov_matrix)
        else:
            cov_trace = 0  # Default to 0 if not enough tokens to compute covariance
        cov_traces.append(cov_trace)
    # Plot mean distances across layers for this sentence
    axs[0].plot(range(n_layers), mean_distances, marker="o",
        color=colors[idx], label=f"Sentence {idx+1}"
    )
    axs[1].plot(
        range(n_layers), cov_traces, marker="o",
        color=colors[idx], label=f"Sentence {idx+1}"
    )

axs[0].set_ylabel("Mean Embedding Distance from Centroid")
axs[0].set_title("Mean Distance of Token Embeddings Across Layers")
axs[0].set_xlabel("Layer")
axs[0].legend()
axs[1].set_xlabel("Layer")
axs[1].set_ylabel("Covariance Trace")
axs[1].set_title("Covariance Trace of Token Embeddings Across Layers")
axs[1].legend()
plt.tight_layout()
plt.savefig("3_plots/04_mean_distance_across_layers.jpg", dpi=250)
