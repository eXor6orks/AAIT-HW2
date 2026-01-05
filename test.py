import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms

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

LOSS_ALPHA = 1.0
LOSS_BETA = 1.0

DROP_PERCENT = 0.1          # α ≈ 10%
WARMUP_EPOCHS = 5           # No drop at beginning
EARLY_STOP_PATIENCE = 6    # Soft early stopping

LABELED_CSV = "task2/train_data/annotations.csv"

# =========================
# DATA
# =========================
def load_annotations(csv_path):
    df = pd.read_csv(csv_path)
    return df["renamed_path"].values, df["label_idx"].values

# =========================
# TRANSFORMS
# =========================
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

# =========================
# TRAINING FUNCTION
# =========================
def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch
):
    model.train()
    total_loss = 0.0
    total_kept = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)

        # Warm-up: no filtering
        if epoch < WARMUP_EPOCHS:
            loss = criterion(logits, labels)
            kept = labels.size(0)

        else:
            # Per-sample CE loss
            losses = torch.nn.functional.cross_entropy(
                logits, labels, reduction="none"
            )

            threshold = torch.quantile(losses, 1 - DROP_PERCENT)
            mask = losses < threshold

            if mask.sum() == 0:
                continue

            loss = criterion(logits[mask], labels[mask])
            kept = mask.sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_kept += kept

    avg_loss = total_loss / len(loader)
    avg_kept_ratio = total_kept / (len(loader.dataset))

    return avg_loss, avg_kept_ratio

# =========================
# MAIN
# =========================
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

    model = get_model(NUM_CLASSES).to(device)

    criterion = SymmetricCrossEntropy(
        alpha=LOSS_ALPHA,
        beta=LOSS_BETA,
        num_classes=NUM_CLASSES
    )

    optimizer = optim.Adam(model.parameters(), lr=LR)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_loss = float("inf")
    patience_counter = 0

    print("\n===== Training Task 2 (Noisy Labels) =====")

    for epoch in range(EPOCHS):
        loss, kept_ratio = train_epoch(
            model, loader, optimizer, criterion, device, epoch
        )

        scheduler.step(loss)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Loss: {loss:.4f} | "
            f"Kept samples: {kept_ratio*100:.1f}%"
        )

        # Early stopping
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
            torch.save(
                model.state_dict(),
                "checkpoints/best_model.pth"
            )
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        model.state_dict(),
        "checkpoints/resnet18_task2_noisylabels_final.pth"
    )
    print("Model saved.")

if __name__ == "__main__":
    main()
