import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import open_clip
from sklearn.decomposition import PCA

# ============================================
# 1. LOAD MODEL
# ============================================
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
model = model.to(device)

# ============================================
# 2. SETTINGS
# ============================================
dataset_path = "/Users/milin/Downloads/COGS118B_FA25_Project-main_2/bird-species-220"
train_root = os.path.join(dataset_path, "Train")
test_root  = os.path.join(dataset_path, "Test")

n_components = 50      # PCA dimension
max_remove   = 10      # max PCs to remove

species_list = sorted(os.listdir(test_root))

# ============================================
# 3. HELPERS
# ============================================
def load_images(folder):
    out = []
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    for p in files:
        try:
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            out.append(img)
        except:
            pass
    return out

def embed_images(images):
    embs = []
    for img in images:
        with torch.no_grad():
            e = model.encode_image(img)
            e = e / e.norm(dim=-1, keepdim=True)
        embs.append(e.cpu().numpy())
    return np.vstack(embs)

# ============================================
# 4. LOAD ALL TRAIN AND TEST EMBEDDINGS
# ============================================
train_embeddings = []
train_labels = []

test_embeddings = []
test_labels = []

print("\nLoading embeddings...")

for species in tqdm(species_list):
    train_dir = os.path.join(train_root, species)
    test_dir  = os.path.join(test_root, species)

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        continue

    # train
    imgs = load_images(train_dir)
    if len(imgs) > 0:
        emb = embed_images(imgs)
        train_embeddings.append(emb)
        train_labels.extend([species] * len(emb))

    # test
    imgs = load_images(test_dir)
    if len(imgs) > 0:
        emb = embed_images(imgs)
        test_embeddings.append(emb)
        test_labels.extend([species] * len(emb))

train_embeddings = np.vstack(train_embeddings)
test_embeddings  = np.vstack(test_embeddings)

train_labels = np.array(train_labels)
test_labels  = np.array(test_labels)

print("Train embeddings:", train_embeddings.shape)
print("Test embeddings:",  test_embeddings.shape)

# ============================================
# 5. PCA TRAINED ON *ALL TRAIN DATA*
# ============================================
pca = PCA(n_components=n_components)
pca.fit(train_embeddings)

Z_train = pca.transform(train_embeddings)
Z_test  = pca.transform(test_embeddings)

# ============================================
# 6. RUN EXPERIMENT FOR N_remove = 0..10
# ============================================
results = {}  # species -> list of accuracies

for species in species_list:
    results[species] = []

for N_remove in range(max_remove + 1):
    print(f"\n=== Removing top {N_remove} PCs ===")

    Z_train_f = Z_train.copy()
    Z_test_f  = Z_test.copy()

    if N_remove > 0:
        Z_train_f[:, :N_remove] = 0
        Z_test_f[:, :N_remove]  = 0

    # reconstruct → normalize
    X_train_pca = pca.inverse_transform(Z_train_f)
    X_test_pca  = pca.inverse_transform(Z_test_f)

    X_train_pca /= np.linalg.norm(X_train_pca, axis=1, keepdims=True)
    X_test_pca  /= np.linalg.norm(X_test_pca, axis=1, keepdims=True)

    # ============================================
    # TRUE CLASSIFICATION FOR EACH TEST IMAGE
    # ============================================
    correct = np.zeros(len(test_labels))

    for i in range(len(test_embeddings)):
        sims = X_train_pca @ X_test_pca[i]
        nn_idx = sims.argmax()
        pred_label = train_labels[nn_idx]
        correct[i] = (pred_label == test_labels[i])

    # store per-species accuracy
    for species in species_list:
        mask = (test_labels == species)
        if mask.sum() > 0:
            acc = correct[mask].mean()
            results[species].append(acc)

# ============================================
# 7. PLOT RESULTS
# ============================================
plt.figure(figsize=(18, 8))
xs = np.arange(max_remove + 1)
width = 0.03

species_list_plot = [s for s in species_list if len(results[s]) > 0]
num_species = len(species_list_plot)

for i, species in enumerate(species_list_plot):
    acc_list = results[species]
    plt.bar(xs + i * width, acc_list, width=width, label=species)

plt.xlabel("Number of Top PCA Components Removed")
plt.ylabel("Classification Accuracy")
plt.title("Accuracy per Species vs. Removed PCA Components")
plt.grid(alpha=0.3, axis='y')
plt.xticks(xs + width * num_species / 2, xs)  # center x-ticks
plt.ylim(0, 1)
plt.legend(fontsize=6, ncol=4)
plt.show()