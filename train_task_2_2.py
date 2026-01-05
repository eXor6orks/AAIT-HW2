import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_PATH = "task2/train_data/clean_annotations.csv"
SAVE_DIR = "checkpoints/task2_cleaned_training"

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# VISUALIZATION FUNCTIONS
# =========================
def plot_training_history(history):
    """Génère des graphiques pour la perte et la précision."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history['train_loss'], label='Train Loss', color='#1f77b4')
    ax1.plot(history['val_loss'], label='Val Loss', color='#ff7f0e')
    ax1.set_title('Évolution de la Perte (Loss)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(history['train_acc'], label='Train Acc', color='#2ca02c')
    ax2.plot(history['val_acc'], label='Val Acc', color='#d62728')
    ax2.set_title('Évolution de la Précision (Accuracy)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/training_curves.png")
    print(f"📊 Courbes d'apprentissage sauvegardées dans {SAVE_DIR}/training_curves.png")
    plt.show()

def plot_confusion_matrix(model, val_loader):
    """Génère une heatmap de la matrice de confusion sur 100 classes."""
    print("🧠 Calcul de la matrice de confusion...")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Evaluation CM"):
            outputs = model(imgs.to(DEVICE))
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
    
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(18, 14))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title("Matrice de Confusion (100 classes)")
    plt.xlabel("Prédictions")
    plt.ylabel("Vrais Labels")
    
    plt.savefig(f"{SAVE_DIR}/confusion_matrix.png")
    print(f"🖼️ Matrice de confusion sauvegardée dans {SAVE_DIR}/confusion_matrix.png")
    plt.show()

# =========================
# DATASET & MODEL (Identiques à ton code)
# =========================
class ImageCSVdataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["renamed_path"]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, int(row["label_idx"])

class TestDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, img_path

def get_model(num_classes):
    model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# =========================
# TRANSFORMS
# =========================
train_tf = T.Compose([
    T.Resize((64, 64)),
    T.RandomCrop(64, padding=4),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_tf = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
    print(f"🚀 Device: {DEVICE}")
    full_dataset = ImageCSVdataset(CSV_PATH, transform=train_tf)
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    val_set.dataset.transform = val_tf 

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = get_model(NUM_CLASSES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0

    for epoch in range(EPOCHS):
        # TRAIN
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_total += labels.size(0)
            train_correct += outputs.argmax(1).eq(labels).sum().item()
            pbar.set_postfix({"Acc": 100.*train_correct/train_total})

        # VALIDATION
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_total += labels.size(0)
                val_correct += outputs.argmax(1).eq(labels).sum().item()
        
        # Enregistrement des stats
        epoch_val_acc = 100 * val_correct / val_total
        epoch_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(100. * train_correct / train_total)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"   📝 Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pth")
            print("   💾 Best model saved!")

    # Affichage des graphiques après l'entraînement
    plot_training_history(history)
    
    # Matrice de confusion avec le meilleur modèle
    model.load_state_dict(torch.load(f"{SAVE_DIR}/best_model.pth"))
    plot_confusion_matrix(model, val_loader)

    print("\n🔮 Generating final submission with TTA...")
    
    model.load_state_dict(torch.load(f"{SAVE_DIR}/best_model.pth"))    
    submission_df = generate_predictions(model, "task2/test_data/", f"{SAVE_DIR}/submission.csv")

if __name__ == "__main__":
    main()