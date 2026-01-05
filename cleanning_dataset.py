import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_PATH = "task2/train_data/annotations.csv" # Ton chemin
IMG_FOLDER = "task2/train_data/" # Dossier contenant les images
OUTPUT_CSV = "task2/train_data/clean_annotations.csv"
SAVE_DIR = "checkpoints/task2_cleaned_training"
NOISE_RATE = 0.2
KEEP_RATIO = 1 - NOISE_RATE  # On garde ~60% des données les plus sûres

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_cleaning_results(all_feats, all_labels, keep_indices):
    # On prend un échantillon pour que ce soit rapide (ex: 2000 points)
    n_samples = min(len(all_feats), 2000)
    tsne = TSNE(n_components=2, perplexity=30)
    feats_2d = tsne.fit_transform(all_feats[:n_samples])
    
    plt.figure(figsize=(10, 8))
    # Points supprimés (bruit probable)
    mask_kept = np.isin(np.arange(len(all_feats)), keep_indices)[:n_samples]
    
    plt.scatter(feats_2d[~mask_kept, 0], feats_2d[~mask_kept, 1], c='red', label='Supprimés (Bruit)', alpha=0.5, s=10)
    plt.scatter(feats_2d[mask_kept, 0], feats_2d[mask_kept, 1], c='green', label='Gardés (Propres)', alpha=0.3, s=10)
    
    plt.title("Visualisation t-SNE du nettoyage des données")
    plt.legend()
    plt.savefig(f"{SAVE_DIR}/cleaning_visual.png")
    plt.show()

# 1. Dataset simple pour l'extraction
class FeatureDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        # On resize en 224 JUSTE pour l'extraction de features
        # Cela aide le modèle pré-entrainé à "comprendre" l'image 64x64 upscalée
        self.transform = T.Compose([
            T.Resize((224, 224)), 
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["renamed_path"] # Assure-toi que le chemin est bon
        label = int(row["label_idx"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label, idx  # On retourne l'index pour filtrer le DataFrame

def main():
    # 2. Modèle Extracteur (ResNet50 puissant)
    print("Load extractor...")
    extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(DEVICE)
    extractor.fc = nn.Identity() # On enlève la classification pour avoir les vecteurs (2048 dim)
    extractor.eval()

    # 3. Extraction
    dataset = FeatureDataset(CSV_PATH)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    all_feats = []
    all_labels = []
    all_indices = []

    print("Extracting features...")
    with torch.no_grad():
        for imgs, lbls, idxs in tqdm(loader):
            imgs = imgs.to(DEVICE)
            # Extraction
            feats = extractor(imgs)
            # Normalisation L2 (Crucial pour la similarité cosinus !)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            
            all_feats.append(feats.cpu().numpy())
            all_labels.extend(lbls.numpy())
            all_indices.extend(idxs.numpy())

    all_feats = np.concatenate(all_feats)
    all_labels = np.array(all_labels)
    all_indices = np.array(all_indices)

    # 4. Filtrage : On garde les points proches du centre de leur classe
    print("Filtering noisy labels...")
    final_indices = []

    for c in range(100): # 100 classes
        # Récupérer tous les samples de la classe c
        class_mask = (all_labels == c)
        class_idxs = all_indices[class_mask]
        class_feats = all_feats[class_mask]

        if len(class_feats) == 0: continue

        # Calculer le "Centre" de la classe (Prototype)
        center = np.mean(class_feats, axis=0, keepdims=True)
        center = center / np.linalg.norm(center) # Re-normaliser le centre

        # Calculer la similarité de chaque image avec le centre
        # Plus c'est proche de 1, plus l'image est "typique" de la classe
        similarities = cosine_similarity(class_feats, center).flatten()

        # On trie du plus similaire au moins similaire
        sorted_local_idx = np.argsort(similarities)[::-1] # Descending

        # On ne garde que le TOP (1 - noise_rate)
        # Ex: si 100 images, 40% bruit -> on garde les 60 meilleures
        n_keep = int(len(class_feats) * KEEP_RATIO)
        # Sécurité: garder au moins 5 images par classe
        n_keep = max(n_keep, 5) 
        
        # Récupérer les index globaux des meilleurs samples
        best_local_idxs = sorted_local_idx[:n_keep]
        best_global_idxs = class_idxs[best_local_idxs]
        
        final_indices.extend(best_global_idxs)

    # 5. Sauvegarde
    df_orig = pd.read_csv(CSV_PATH)
    df_clean = df_orig.iloc[final_indices]
    df_clean.to_csv(OUTPUT_CSV, index=False)
    # 6. Visualisation du nettoyage
    plot_cleaning_results(all_feats, all_labels, final_indices)

    print(f"✅ DONE! Dataset cleaned.")
    print(f"Original size: {len(df_orig)}")
    print(f"Cleaned size:  {len(df_clean)} (Removed {len(df_orig) - len(df_clean)} suspicious images)")

if __name__ == "__main__":
    main()