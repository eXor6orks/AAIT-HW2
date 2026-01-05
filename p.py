import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import ResNet34_Weights

# =========================
# CONFIG
# =========================
NUM_CLASSES = 100
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
NOISE_RATE = 0.4     # ε - taux de bruit estimé
Tk = 25              # Epoch threshold (augmenté pour décroissance plus douce)
R_MIN = 0.65         # Plancher minimum pour R(T) (65% > 60% de données propres)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "checkpoints/task2_co_teaching_v2"

os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Config: NOISE_RATE={NOISE_RATE}, Tk={Tk}, R_MIN={R_MIN}")

# =========================
# DATASET
# =========================
class ImageCSVdataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["renamed_path"]).convert("RGB")
        label = int(row["label_idx"])

        if self.transform:
            img = self.transform(img)

        return img, label

class TestDataset(Dataset):
    """Dataset pour les prédictions sans labels - lit directement les images d'un dossier"""
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        
        # Récupérer tous les fichiers images
        self.image_files = sorted([
            f for f in os.listdir(image_folder) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)

        # Retourner le chemin complet pour le CSV final
        return img, img_path

# =========================
# TRANSFORMS
# =========================
train_tf = T.Compose([
    T.Resize((64, 64)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_tf = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# LOADERS
# =========================
train_dataset = ImageCSVdataset(
    "task2/train_data/annotations.csv",
    train_tf
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# =========================
# MODEL
# =========================
def get_model():
    model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

# =========================
# PREDICTION FUNCTION
# =========================
def generate_predictions(model, image_folder, output_path):
    """Génère les prédictions sur le dataset de test"""
    model.eval()
    
    test_dataset = TestDataset(image_folder, val_tf)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    predictions = []
    image_ids = []
    
    with torch.no_grad():
        for images, image_paths in tqdm(test_loader, desc="Generating predictions"):
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            
            predictions.extend(preds)
            image_ids.extend(image_paths)
    
    # Créer le DataFrame de soumission au format Kaggle
    submission_df = pd.DataFrame({
        'ID': image_ids,
        'label': predictions
    })
    
    submission_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")
    print(f"   Format: {len(submission_df)} images with columns ['ID', 'label']")
    return submission_df

# =========================
# MAIN TRAINING
# =========================
def main():
    model1 = get_model().to(DEVICE)
    model2 = get_model().to(DEVICE)

    optimizer1 = optim.Adam(model1.parameters(), lr=LR, weight_decay=1e-4)
    optimizer2 = optim.Adam(model2.parameters(), lr=LR, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=EPOCHS)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=EPOCHS)

    criterion = nn.CrossEntropyLoss(reduction="none")

    # Tracking
    history = {
        "loss1": [],
        "loss2": [],
        "loss_avg": [],
        "R_t": [],
        "lr": []
    }

    best_avg_loss = float('inf')
    best_epoch = 0

    # =========================
    # TRAINING LOOP
    # =========================
    for epoch in range(EPOCHS):
        model1.train()
        model2.train()

        # R(T) schedule avec plancher
        R_t_raw = 1 - min((epoch / Tk) * NOISE_RATE, NOISE_RATE)
        R_t = max(R_t_raw, R_MIN)  # Ne jamais descendre sous R_MIN
        history["R_t"].append(R_t)
        history["lr"].append(optimizer1.param_groups[0]['lr'])

        loss_sum_1 = 0.0
        loss_sum_2 = 0.0
        n_samples = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            batch_size = labels.size(0)
            n_samples += batch_size

            # Forward pass des deux modèles
            logits1 = model1(images)
            logits2 = model2(images)

            # Calcul des pertes individuelles
            loss1 = criterion(logits1, labels)
            loss2 = criterion(logits2, labels)

            # Sélection des k plus petites pertes
            k = max(1, int(R_t * batch_size))  # Au moins 1 sample

            # Modèle 1 apprend sur les exemples que Modèle 2 trouve faciles
            idx2 = torch.argsort(loss2)[:k]
            optimizer1.zero_grad()
            loss1[idx2].mean().backward()
            torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=5.0)
            optimizer1.step()

            # Modèle 2 apprend sur les exemples que Modèle 1 trouve faciles
            idx1 = torch.argsort(loss1.detach())[:k]
            optimizer2.zero_grad()
            loss2[idx1].mean().backward()
            torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=5.0)
            optimizer2.step()

            loss_sum_1 += loss1.detach().sum().item()
            loss_sum_2 += loss2.detach().sum().item()

        # Epoch stats
        avg_loss1 = loss_sum_1 / n_samples
        avg_loss2 = loss_sum_2 / n_samples
        avg_loss = (avg_loss1 + avg_loss2) / 2

        history["loss1"].append(avg_loss1)
        history["loss2"].append(avg_loss2)
        history["loss_avg"].append(avg_loss)

        # Step schedulers
        scheduler1.step()
        scheduler2.step()

        print(
            f"Epoch {epoch+1}/{EPOCHS}: "
            f"Loss1={avg_loss1:.4f}, "
            f"Loss2={avg_loss2:.4f}, "
            f"Avg={avg_loss:.4f}, "
            f"R(T)={R_t:.3f}, "
            f"LR={history['lr'][-1]:.6f}"
        )

        # Sauvegarde du meilleur modèle basé sur la perte moyenne
        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            best_epoch = epoch + 1
            torch.save(model1.state_dict(), f"{SAVE_DIR}/best_model1.pth")
            torch.save(model2.state_dict(), f"{SAVE_DIR}/best_model2.pth")
            print(f"  ✨ New best! Saved models at epoch {best_epoch}")

        # Sauvegarde régulière tous les 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model1.state_dict(), f"{SAVE_DIR}/model1_epoch{epoch+1}.pth")
            torch.save(model2.state_dict(), f"{SAVE_DIR}/model2_epoch{epoch+1}.pth")

    # =========================
    # SAVE FINAL MODELS
    # =========================
    torch.save(model1.state_dict(), f"{SAVE_DIR}/final_model1.pth")
    torch.save(model2.state_dict(), f"{SAVE_DIR}/final_model2.pth")

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best epoch: {best_epoch} (Avg Loss: {best_avg_loss:.4f})")
    print(f"{'='*60}\n")

    # =========================
    # PLOTS
    # =========================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curves
    axes[0, 0].plot(history["loss1"], label="Model 1", alpha=0.7)
    axes[0, 0].plot(history["loss2"], label="Model 2", alpha=0.7)
    axes[0, 0].plot(history["loss_avg"], label="Average", linewidth=2, color='black')
    axes[0, 0].axvline(x=best_epoch-1, color='red', linestyle='--', label=f'Best (epoch {best_epoch})')
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # R(T) schedule
    axes[0, 1].plot(history["R_t"], linewidth=2, color='green')
    axes[0, 1].axhline(y=R_MIN, color='red', linestyle='--', label=f'R_MIN={R_MIN}')
    axes[0, 1].axhline(y=1-NOISE_RATE, color='orange', linestyle='--', label=f'1-ε={1-NOISE_RATE}')
    axes[0, 1].set_title("R(T) Schedule")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Keep Ratio")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 0].plot(history["lr"], linewidth=2, color='purple')
    axes[1, 0].set_title("Learning Rate Schedule")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("LR")
    axes[1, 0].grid(True, alpha=0.3)

    # Loss difference
    loss_diff = [abs(l1 - l2) for l1, l2 in zip(history["loss1"], history["loss2"])]
    axes[1, 1].plot(loss_diff, linewidth=2, color='red')
    axes[1, 1].set_title("Loss Difference |Loss1 - Loss2|")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Absolute Difference")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/training_summary.png", dpi=150)
    plt.close()

    print(f"📊 Training plots saved to {SAVE_DIR}/training_summary.png")

    # =========================
    # GENERATE PREDICTIONS
    # =========================
    print("\n" + "="*60)
    print("Generating predictions on validation set...")
    print("="*60 + "\n")

    # Chemin du dossier de validation
    VAL_FOLDER = "task2/val_data"

    # Charger le meilleur modèle
    model1.load_state_dict(torch.load(f"{SAVE_DIR}/best_model1.pth"))
    model2.load_state_dict(torch.load(f"{SAVE_DIR}/best_model2.pth"))

    # Générer les prédictions (moyenne des deux modèles)
    print("🔮 Using Model 1 (best)...")
    generate_predictions(model1, VAL_FOLDER, f"{SAVE_DIR}/submission_model1.csv")
    
    print("🔮 Using Model 2 (best)...")
    generate_predictions(model2, VAL_FOLDER, f"{SAVE_DIR}/submission_model2.csv")

    # Ensemble: moyenne des prédictions
    print("\n🎯 Creating ensemble predictions...")
    model1.eval()
    model2.eval()
    
    test_dataset = TestDataset(VAL_FOLDER, val_tf)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    predictions = []
    image_ids = []
    
    with torch.no_grad():
        for images, image_paths in tqdm(test_loader, desc="Ensemble prediction"):
            images = images.to(DEVICE)
            
            logits1 = model1(images)
            logits2 = model2(images)
            
            # Moyenne des logits
            avg_logits = (logits1 + logits2) / 2
            preds = avg_logits.argmax(1).cpu().numpy()
            
            predictions.extend(preds)
            image_ids.extend(image_paths)
    
    submission_df = pd.DataFrame({
        'ID': image_ids,
        'label': predictions
    })
    
    submission_df.to_csv(f"{SAVE_DIR}/submission_ensemble.csv", index=False)
    print(f"✅ Ensemble predictions saved to {SAVE_DIR}/submission_ensemble.csv")
    print(f"   Format: {len(submission_df)} images with columns ['ID', 'label']")

    print("\n" + "="*60)
    print("🎉 ALL DONE! You can submit:")
    print(f"   - {SAVE_DIR}/submission_model1.csv")
    print(f"   - {SAVE_DIR}/submission_model2.csv")
    print(f"   - {SAVE_DIR}/submission_ensemble.csv  (RECOMMENDED)")
    print("="*60)

if __name__ == "__main__":
    main()