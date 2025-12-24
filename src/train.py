import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms

from src.dataset.dataset import (
    LabeledImageDataset,
    UnlabeledImageDataset,
    load_annotations
)
from src.model.model import get_model


# =========================
# CONFIG
# =========================
NUM_CLASSES = 99
BATCH_SIZE = 32
LR = 1e-4
WARMUP_EPOCHS = 10
RETRAIN_EPOCHS = 20
PSEUDO_THRESHOLD = 0.8
PSEUDO_WEIGHT = 0.3
NUM_WORKERS = 4

LABELED_CSV = "task1/train_data/images/annotations.csv"
UNLABELED_DIR = "task1/train_data/images/unlabeled"


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

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# DEVICE & MODEL
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = get_model(num_classes=NUM_CLASSES).to(device)


# =========================
# DATASETS & LOADERS
# =========================
samples, labels = load_annotations(LABELED_CSV)

labeled_dataset = LabeledImageDataset(
    samples,
    labels,
    transform=train_transform
)

labeled_loader = DataLoader(
    labeled_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

unlabeled_dataset = UnlabeledImageDataset(
    UNLABELED_DIR,
    transform=eval_transform
)

unlabeled_loader = DataLoader(
    unlabeled_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# =========================
# TRAINING FUNCTIONS
# =========================
def train_epoch(model, loader, optimizer, criterion, device, weight=1.0):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        loss = weight * loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def generate_pseudo_labels(model, dataloader, threshold, device):
    model.eval()
    pseudo_samples = []
    pseudo_labels = []

    with torch.no_grad():
        for images, paths in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            max_probs, preds = probs.max(dim=1)
            mask = max_probs > threshold

            for i in range(len(mask)):
                if mask[i]:
                    pseudo_samples.append(paths[i])
                    pseudo_labels.append(preds[i].item())

    return pseudo_samples, pseudo_labels


# =========================
# OPTIMIZER & LOSS
# =========================
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()


# =========================
# 1️⃣ SUPERVISED WARM-UP
# =========================
print("\n===== Supervised warm-up =====")
for epoch in range(WARMUP_EPOCHS):
    loss = train_epoch(
        model,
        labeled_loader,
        optimizer,
        criterion,
        device,
        weight=1.0
    )
    print(f"Warm-up Epoch [{epoch+1}/{WARMUP_EPOCHS}] - Loss: {loss:.4f}")


# =========================
# 2️⃣ PSEUDO-LABELING
# =========================
print("\n===== Generating pseudo-labels =====")
pseudo_samples, pseudo_labels = generate_pseudo_labels(
    model,
    unlabeled_loader,
    threshold=PSEUDO_THRESHOLD,
    device=device
)

print(f"Pseudo-labeled samples: {len(pseudo_samples)} / {len(unlabeled_dataset)}")


# =========================
# 3️⃣ RETRAINING
# =========================
pseudo_dataset = LabeledImageDataset(
    pseudo_samples,
    pseudo_labels,
    transform=train_transform
)

pseudo_loader = DataLoader(
    pseudo_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

print("\n===== Retraining with pseudo-labels =====")
for epoch in range(RETRAIN_EPOCHS):
    loss_labeled = train_epoch(
        model,
        labeled_loader,
        optimizer,
        criterion,
        device,
        weight=1.0
    )

    loss_pseudo = train_epoch(
        model,
        pseudo_loader,
        optimizer,
        criterion,
        device,
        weight=PSEUDO_WEIGHT
    )

    print(
        f"Epoch [{epoch+1}/{RETRAIN_EPOCHS}] "
        f"Labeled Loss: {loss_labeled:.4f} | "
        f"Pseudo Loss: {loss_pseudo:.4f}"
    )


# =========================
# SAVE MODEL
# =========================
os.makedirs("checkpoints", exist_ok=True)
ckpt_path = f"checkpoints/resnet18_task1_pseudo{PSEUDO_THRESHOLD}_w{PSEUDO_WEIGHT}.pth"
torch.save(model.state_dict(), ckpt_path)
print(f"\nModel saved to {ckpt_path}")
