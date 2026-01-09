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
import torch.nn.functional as F
import argparse

# =========================
# CONFIG
# =========================
NUM_CLASSES = 100
BATCH_SIZE = 32

LR = 1e-3  
Tk = 25              
R_MIN = 0.65 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "checkpoints/task2_co_teaching_v2"
os.makedirs(SAVE_DIR, exist_ok=True)

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

        return img, img_path

class SymmetricCrossEntropy(nn.Module):
    """
    Symmetric Cross Entropy pour l'apprentissage avec labels bruités.
    Reference: Wang et al. ICCV 2019
    """
    def __init__(self, alpha=0.1, beta=1.0, num_classes=10):
        super(SymmetricCrossEntropy, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)

        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)

        loss = self.alpha * ce + self.beta * rce
        return loss

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
    parser = argparse.ArgumentParser(description="AAIT Task 2 - Co-Teaching")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--noisy_rate", type=float, default=0.4)
    args = parser.parse_args()

    EPOCHS = args.epochs
    NOISE_RATE = args.noisy_rate

    print(f"Device: {DEVICE}")
    print(f"Config: NOISE_RATE={NOISE_RATE}, Tk={Tk}, R_MIN={R_MIN}")

    model1 = get_model().to(DEVICE)
    model2 = get_model().to(DEVICE)

    optimizer1 = optim.Adam(model1.parameters(), lr=LR, weight_decay=1e-4)
    optimizer2 = optim.Adam(model2.parameters(), lr=LR, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=EPOCHS)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=EPOCHS)

    criterion = SymmetricCrossEntropy(alpha=0.1, beta=1.0, num_classes=NUM_CLASSES)

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

    train_dataset = ImageCSVdataset(
        csv_path=os.path.join(args.dataset, "train_data/annotations.csv"),
        transform=train_tf
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # =========================
    # TRAINING LOOP
    # =========================
    for epoch in range(EPOCHS):
        model1.train()
        model2.train()

        R_t_raw = 1 - min((epoch / Tk) * NOISE_RATE, NOISE_RATE)
        R_t = max(R_t_raw, R_MIN)
        history["R_t"].append(R_t)
        history["lr"].append(optimizer1.param_groups[0]['lr'])

        loss_sum_1 = 0.0
        loss_sum_2 = 0.0
        n_samples = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            batch_size = labels.size(0)
            n_samples += batch_size

            logits1 = model1(images)
            logits2 = model2(images)

            loss1 = criterion(logits1, labels)
            loss2 = criterion(logits2, labels)

            k = max(1, int(R_t * batch_size)) 

            idx2 = torch.argsort(loss2)[:k]
            optimizer1.zero_grad()
            loss1[idx2].mean().backward()
            torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=5.0)
            optimizer1.step()

            idx1 = torch.argsort(loss1.detach())[:k]
            optimizer2.zero_grad()
            loss2[idx1].mean().backward()
            torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=5.0)
            optimizer2.step()

            loss_sum_1 += loss1.detach().sum().item()
            loss_sum_2 += loss2.detach().sum().item()

        avg_loss1 = loss_sum_1 / n_samples
        avg_loss2 = loss_sum_2 / n_samples
        avg_loss = (avg_loss1 + avg_loss2) / 2

        history["loss1"].append(avg_loss1)
        history["loss2"].append(avg_loss2)
        history["loss_avg"].append(avg_loss)

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

        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            best_epoch = epoch + 1
            torch.save(model1.state_dict(), f"{SAVE_DIR}/best_model1.pth")
            torch.save(model2.state_dict(), f"{SAVE_DIR}/best_model2.pth")
            print(f"  ✨ New best! Saved models at epoch {best_epoch}")

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