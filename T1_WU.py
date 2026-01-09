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

import argparse


# Importations de tes modules locaux
from src.dataset.dataset import (
    LabeledImageDataset,
    UnlabeledImageDataset,
    load_annotations
)
from src.model.model import get_model_ResNet_34

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
LR = 1e-4
NUM_WORKERS = 4

LABELED_CSV = "task1/train_data/annotations.csv"
CHECKPOINT_PATH = "checkpoints/best_resnet.pth"

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Faible augmentation : pour les labels réels et la génération de pseudo-labels
weak_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(64, padding=4),
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
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
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

@torch.no_grad()
def evaluate_accuracy_tta(model, loader, device, tta_runs=5):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        labels = labels.to(device)
        logits_sum = 0

        for _ in range(tta_runs):
            imgs = images.to(device)
            logits_sum += model(imgs)

        logits_avg = logits_sum / tta_runs
        preds = logits_avg.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="AAIT Task 1 - Pseudo Labels")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--checkpoints", type=str, default="checkpoints")

    args = parser.parse_args()

    LABELED_CSV = os.path.join(args.dataset, "train_data/annotations.csv")
    CHECKPOINT_PATH = os.path.join(args.checkpoints, "best_resnet.pth")
    WARMUP_EPOCHS = args.epochs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = get_model_ResNet_34(num_classes=NUM_CLASSES).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    # Data
    samples, labels_attr = load_annotations(LABELED_CSV)

    indices = torch.randperm(len(samples))
    val_size = int(0.1 * len(indices))

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_dataset = LabeledImageDataset(
        [samples[i] for i in train_idx],
        [labels_attr[i] for i in train_idx],
        transform=weak_transform
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

    for epoch in range(WARMUP_EPOCHS):
        loss = train_warmup_epoch(model, labeled_loader, optimizer, criterion, device)
        val_acc = evaluate_accuracy(model, val_loader, device)
        val_acc_tta = evaluate_accuracy_tta(model, val_loader, device)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

        metrics["warmup"].append({
            "epoch": epoch + 1,
            "loss": loss,
            "val_acc": val_acc,
            "val_acc_tta": val_acc_tta
        })

        print(
            f"Warm-up [{epoch+1}] "
            f"Loss={loss:.4f} "
            f"ValAcc={val_acc:.3f} "
            f"TTA={val_acc_tta:.3f}"
        )

    with open(log_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    set_seed(42)
    main()