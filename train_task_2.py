import os
import torch
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms

from src.dataset.dataset import LabeledImageDataset, load_annotations
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
DROP_PERCENT = 0.1  # 10% worst samples

LABELED_CSV = "task2/train_data/annotations.csv"

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
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        # Loss par sample
        losses = torch.nn.functional.cross_entropy(
            logits, labels, reduction="none"
        )

        # Drop high-loss samples
        threshold = torch.quantile(losses, 1 - DROP_PERCENT)
        mask = losses < threshold

        if mask.sum() == 0:
            continue

        loss = criterion(logits[mask], labels[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

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
        num_workers=NUM_WORKERS
    )

    model = get_model(NUM_CLASSES).to(device)

    criterion = SymmetricCrossEntropy(
        alpha=LOSS_ALPHA,
        beta=LOSS_BETA,
        num_classes=NUM_CLASSES
    )

    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("\n===== Training Task 2 (Noisy Labels) =====")
    for epoch in range(EPOCHS):
        loss = train_epoch(
            model, loader, optimizer, criterion, device
        )
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        model.state_dict(),
        "checkpoints/resnet18_task2_noisylabels.pth"
    )
    print("Model saved.")

if __name__ == "__main__":
    main()
