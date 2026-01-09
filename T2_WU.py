import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms
from torchvision.transforms import RandAugment
from collections import Counter, defaultdict
import json
from pathlib import Path
from itertools import cycle
from tqdm import tqdm
from torch.utils.data import random_split
from src.utils.utils import set_seed
import pandas as pd
from src.losses.sce import SymmetricCrossEntropy
import numpy as np

import argparse


# Importations de tes modules locaux
from src.dataset.dataset import (
    LabeledImageDataset,
    UnlabeledImageDataset
)
from src.model.model import get_model_ResNet_50 as get_model

import torch.multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# =========================
# CONFIGURATION
# =========================
NUM_CLASSES = 100
BATCH_SIZE = 32
LR = 1e-3
NUM_WORKERS = 4

MIXUP_ALPHA = 0.3
SCE_ALPHA = 0.1
SCE_BETA = 1.0

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# --- MIXUP FUNCTIONS ---
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Faible augmentation : pour les labels réels et la génération de pseudo-labels
strong_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_warmup_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)
        
        images, targets_a, targets_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
        logits = model(images)
        loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
        # loss = loss.mean()
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total

def load_annotations(csv_path):
    df = pd.read_csv(csv_path)
    return df["renamed_path"].values, df["label_idx"].values

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="AAIT Task 2 - Noisy Labels")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--checkpoints", type=str, default="checkpoints")

    args = parser.parse_args()

    LABELED_CSV = os.path.join(args.dataset, "train_data/annotations.csv")
    CHECKPOINT_PATH = os.path.join(args.checkpoints, "task2/best_resnet.pth")
    EPOCHS = args.epochs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = get_model(num_classes=NUM_CLASSES).to(device)

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    criterion = SymmetricCrossEntropy(SCE_ALPHA, SCE_BETA, NUM_CLASSES)

    # Data
    samples, labels_attr = load_annotations(LABELED_CSV)

    generator = torch.Generator().manual_seed(42)

    indices = torch.randperm(len(samples), generator=generator)
    val_size = int(0.1 * len(indices))

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_dataset = LabeledImageDataset(
        [samples[i] for i in train_idx],
        [labels_attr[i] for i in train_idx],
        transform=strong_transform
    )

    val_dataset = LabeledImageDataset(
        [samples[i] for i in val_idx],
        [labels_attr[i] for i in val_idx],
        transform=eval_transform
    )

    labeled_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("\n===== Phase 1: Supervised Warm-up =====")
    metrics = {
        "warmup": [],
        "pseudo_stages": []
    }
    best_acc = 0.0

    for epoch in range(EPOCHS):
        loss = train_warmup_epoch(model, labeled_loader, optimizer, criterion, device)
        val_acc = evaluate_accuracy(model, val_loader, device)

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

        metrics["warmup"].append({
            "epoch": epoch + 1,
            "loss": loss,
            "val_acc": val_acc,
        })

        print(
            f"Warm-up [{epoch+1}] "
            f"Loss={loss:.4f} "
            f"ValAcc={val_acc:.3f} "
        )

    with open(log_dir / "training_metrics_T2.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    set_seed(42)
    main()