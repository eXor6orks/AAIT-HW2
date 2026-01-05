import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms
import torch.nn.functional as F

from src.dataset.dataset import LabeledImageDataset
from src.model.model import get_model
from src.losses.sce import SymmetricCrossEntropy

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

# =========================
# CONFIG
# =========================
NUM_CLASSES = 100
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 30
NUM_WORKERS = 4

DROP_PERCENT = 0.1
WARMUP_EPOCHS = 5
DROP_RAMP_EPOCHS = 10   # montée progressive du drop

LOSS_ALPHA = 1.0
LOSS_BETA = 1.0

LABELED_CSV = "task2/train_data/annotations.csv"

def load_annotations(csv_path):
    df = pd.read_csv(csv_path)
    return df["renamed_path"].values, df["label_idx"].values

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def select_samples_per_class(
    losses,
    labels,
    drop_rate
):
    """
    losses: tensor [B]
    labels: tensor [B]
    """
    selected_idx = []

    for c in labels.unique():
        class_mask = labels == c
        class_losses = losses[class_mask]

        if class_losses.numel() <= 1:
            selected_idx.append(
                torch.where(class_mask)[0]
            )
            continue

        num_keep = max(
            1,
            int((1 - drop_rate) * class_losses.numel())
        )

        _, idx_sorted = torch.sort(class_losses)
        keep_idx = idx_sorted[:num_keep]

        selected_idx.append(
            torch.where(class_mask)[0][keep_idx]
        )

    return torch.cat(selected_idx)

def train_epoch_coteaching(
    model1,
    model2,
    loader,
    optimizer1,
    optimizer2,
    criterion,
    device,
    epoch
):
    model1.train()
    model2.train()

    total_loss = 0.0
    total_kept = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits1 = model1(images)
        logits2 = model2(images)

        # ===== Warmup =====
        if epoch < WARMUP_EPOCHS:
            loss1 = criterion(logits1, labels)
            loss2 = criterion(logits2, labels)

            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            total_loss += (loss1.item() + loss2.item()) / 2
            total_kept += labels.size(0)
            continue

        # ===== Drop rate progressif =====
        drop_rate = min(
            DROP_PERCENT,
            DROP_PERCENT * (epoch - WARMUP_EPOCHS + 1) / DROP_RAMP_EPOCHS
        )

        # Per-sample losses
        losses1 = F.cross_entropy(logits1, labels, reduction="none")
        losses2 = F.cross_entropy(logits2, labels, reduction="none")

        # Sélection par classe
        idx1 = select_samples_per_class(losses1, labels, drop_rate)
        idx2 = select_samples_per_class(losses2, labels, drop_rate)

        # 🔁 Échange croisé
        loss1 = criterion(logits1[idx2], labels[idx2])
        loss2 = criterion(logits2[idx1], labels[idx1])

        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        total_loss += (loss1.item() + loss2.item()) / 2
        total_kept += idx1.numel()

    avg_loss = total_loss / len(loader)
    avg_kept_ratio = total_kept / len(loader.dataset)

    return avg_loss, avg_kept_ratio

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    samples, labels = load_annotations(LABELED_CSV)

    dataset = LabeledImageDataset(
        samples, labels, transform=train_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model1 = get_model(NUM_CLASSES).to(device)
    model2 = get_model(NUM_CLASSES).to(device)

    criterion = SymmetricCrossEntropy(
        alpha=LOSS_ALPHA,
        beta=LOSS_BETA,
        num_classes=NUM_CLASSES
    )

    optimizer1 = optim.Adam(model1.parameters(), lr=LR)
    optimizer2 = optim.Adam(model2.parameters(), lr=LR)

    best_loss = float("inf")

    print("\n===== Co-Teaching + Class-wise Drop =====")

    for epoch in range(EPOCHS):
        loss, kept_ratio = train_epoch_coteaching(
            model1,
            model2,
            loader,
            optimizer1,
            optimizer2,
            criterion,
            device,
            epoch
        )

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Loss: {loss:.4f} | "
            f"Kept: {kept_ratio*100:.1f}%"
        )

        if loss < best_loss:
            best_loss = loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                model1.state_dict(),
                "checkpoints/best_model_coteaching_classdrop.pth"
            )

    print("Training finished.")

if __name__ == "__main__":
    main()