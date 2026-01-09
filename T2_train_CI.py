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
from src.model.model import get_model_ResNeXt_50

import torch.multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

NUM_CLASSES = 100
BATCH_SIZE = 32
LR = 1e-3
MIXUP_ALPHA = 0.4
SCE_ALPHA = 0.5
SCE_BETA = 0.5

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

def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def load_annotations(csv_path):
    df = pd.read_csv(csv_path)
    return df["sample"].values, df["label"].values

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

def main():
    parser = argparse.ArgumentParser(description="AAIT Task 2 - Noisy Labels")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--clean_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)

    args = parser.parse_args()

    PSEUDO_LABELED_CSV = os.path.join(args.dataset, args.clean_file)
    EPOCHS = args.epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    student = get_model_ResNeXt_50(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.SGD(student.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    criterion = SymmetricCrossEntropy(SCE_ALPHA, SCE_BETA, NUM_CLASSES)

    samples, labels_attr = load_annotations(PSEUDO_LABELED_CSV)

    indices = torch.randperm(len(samples))
    val_size = int(0.1 * len(indices))

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_dataset = LabeledImageDataset(samples[train_idx], labels_attr[train_idx], transform=strong_transform)
    val_dataset = LabeledImageDataset(samples[val_idx], labels_attr[val_idx], transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("\n===== Retraining on Pseudo-Labels =====")
    metrics = {
        "epoch": [],
        "train_loss": [],
        "val_accuracy": []
    }
    best_acc = 0.0
    for epoch in range(EPOCHS):
        student.train()
        total_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)

            images, targets_a, targets_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
            logits = student(images)
            loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
        scheduler.step()
        avg_loss = total_loss / len(train_loader.dataset)
        val_acc = evaluate_accuracy(student, val_loader, device)

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), "checkpoints/task2/retrained_student.pth")

        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(avg_loss)
        metrics["val_accuracy"].append(val_acc)

    with open("retrain_metrics_T2.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()