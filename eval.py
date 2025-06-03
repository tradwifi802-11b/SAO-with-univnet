import torch
import torchaudio
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import librosa
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load wav2vec2 pretrained model
bundle = torchaudio.pipelines.WAV2VEC2_BASE
model = bundle.get_model().to(DEVICE)
model.eval()

def extract_embedding(file_path):
    waveform, sr = torchaudio.load(file_path)
    if sr != bundle.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=bundle.sample_rate)
        waveform = resampler(waveform)

    with torch.inference_mode():
        features = model(waveform.to(DEVICE))[0]  # shape: (1, time, feature)
        embedding = features.mean(dim=1).squeeze().cpu().numpy()  # (768,)
    return embedding

def compute_spectral_entropy(y, sr, n_fft=2048, hop_length=512):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    ps = S / (np.sum(S, axis=0, keepdims=True) + 1e-10)
    entropy = -np.sum(ps * np.log2(ps + 1e-10), axis=0)
    return np.mean(entropy)

def extract_all(folder):
    embeddings = []
    entropies = []
    filenames = []
    for fname in tqdm(sorted(os.listdir(folder))):
        if fname.lower().endswith(".wav"):
            path = os.path.join(folder, fname)
            emb = extract_embedding(path)
            y, sr = librosa.load(path, sr=24000)
            entropy = compute_spectral_entropy(y, sr)

            embeddings.append(emb)
            entropies.append(entropy)
            filenames.append(fname)
    return np.stack(embeddings), np.array(entropies), filenames

def compute_diversity_score(embeddings):
    dist_matrix = cosine_distances(embeddings)
    return np.mean(dist_matrix[np.triu_indices_from(dist_matrix, k=1)])

def plot_tsne(embeddings, filenames, output_path="tsne.png"):
    n_samples = embeddings.shape[0]
    perplexity = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity)
    proj = tsne.fit_transform(embeddings)

    plt.figure(figsize=(7, 6))
    plt.scatter(proj[:, 0], proj[:, 1], c='blue', alpha=0.7)
    for i, name in enumerate(filenames):
        plt.annotate(name, (proj[i, 0], proj[i, 1]), fontsize=6, alpha=0.5)
    plt.title(f"t-SNE of Wav2Vec2 Embeddings (perplexity={perplexity})")
    plt.savefig(output_path)
    plt.close()
    print(f"t-SNE saved to: {output_path}")

def save_csv(folder, filenames, entropies, diversity_score, output_file="results.csv"):
    df = pd.DataFrame({
        "filename": filenames,
        "spectral_entropy": entropies
    })
    df["diversity_score"] = diversity_score
    df["folder"] = folder
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")

# -------- RUN EVALUATION -------- #
folder = "./output"

embeddings, entropies, filenames = extract_all(folder)
div_score = compute_diversity_score(embeddings)
entropy_mean = np.mean(entropies)
entropy_std = np.std(entropies)

print(f"Diversity Score: {div_score:.4f}")
print(f"Spectral Entropy: mean = {entropy_mean:.4f}, std = {entropy_std:.4f}")

plot_tsne(embeddings, filenames, output_path="tsne_output.png")
save_csv(folder, filenames, entropies, div_score, output_file="audio_eval_results.csv")
