import os
import copy
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

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

# =========================
# CONFIG
# =========================
NUM_CLASSES = 100
BATCH_SIZE = 32
LR = 1e-4

WARMUP_EPOCHS = 50
PSEUDO_THRESHOLDS = [0.9, 0.85, 0.8]
RETRAIN_EPOCHS_PER_STAGE = 10

PSEUDO_WEIGHT = 0.3
EMA_ALPHA = 0.999

NUM_WORKERS = 4

LABELED_CSV = "task1/train_data/annotations.csv"
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
# DEVICE & MODELS
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

student_model = get_model(num_classes=NUM_CLASSES).to(device)

teacher_model = copy.deepcopy(student_model)
teacher_model.to(device)
teacher_model.eval()
for p in teacher_model.parameters():
    p.requires_grad = False

# =========================
# DATASETS & LOADERS
# =========================
samples, labels = load_annotations(LABELED_CSV)



labeled_dataset = LabeledImageDataset(
    samples, labels, transform=train_transform
)

labeled_loader = DataLoader(
    labeled_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

unlabeled_dataset = UnlabeledImageDataset(
    UNLABELED_DIR, transform=eval_transform
)

unlabeled_loader = DataLoader(
    unlabeled_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# =========================
# UTILS
# =========================
def update_teacher(student, teacher, alpha=0.99):
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(alpha).add_(s_param.data * (1 - alpha))


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


def generate_pseudo_labels(teacher_model, dataloader, threshold, device):
    teacher_model.eval()
    pseudo_samples = []
    pseudo_labels = []

    with torch.no_grad():
        for images, paths in dataloader:
            images = images.to(device)
            logits = teacher_model(images)
            probs = F.softmax(logits / 0.5, dim=1)

            max_probs, preds = probs.max(dim=1)
            print(
                f"Max prob stats - Min: {max_probs.min().item():.4f}, "
                f"Max: {max_probs.max().item():.4f}, "
                f"Mean: {max_probs.mean().item():.4f}"
            )
            mask = max_probs > threshold

            for i in range(len(mask)):
                if mask[i]:
                    pseudo_samples.append(paths[i])
                    pseudo_labels.append(preds[i].item())

    return pseudo_samples, pseudo_labels

# =========================
# MAIN
# =========================
def main():
    optimizer = optim.Adam(student_model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    # =========================
    # 1️⃣ SUPERVISED WARM-UP
    # =========================
    print("\n===== Supervised warm-up =====")
    for epoch in range(WARMUP_EPOCHS):
        loss = train_epoch(
            student_model,
            labeled_loader,
            optimizer,
            criterion,
            device,
            weight=1.0
        )
        if epoch < 25:
            alpha = 0.99
        else:
            alpha = 0.999
        update_teacher(student_model, teacher_model, alpha)

        print(f"Warm-up Epoch [{epoch+1}/{WARMUP_EPOCHS}] - Loss: {loss:.4f}")

    # =========================
    # 2️⃣ ITERATIVE MEAN TEACHER PSEUDO-LABELING
    # =========================
    for stage, threshold in enumerate(PSEUDO_THRESHOLDS):
        print(f"\n===== Stage {stage+1} | threshold={threshold} =====")

        pseudo_samples, pseudo_labels = generate_pseudo_labels(
            teacher_model,
            unlabeled_loader,
            threshold,
            device
        )

        print(
            f"Accepted pseudo-labels: "
            f"{len(pseudo_samples)} / {len(unlabeled_dataset)}"
        )

        if len(pseudo_samples) == 0:
            continue

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

        for epoch in range(RETRAIN_EPOCHS_PER_STAGE):
            loss_labeled = train_epoch(
                student_model,
                labeled_loader,
                optimizer,
                criterion,
                device,
                weight=1.0
            )

            loss_pseudo = train_epoch(
                student_model,
                pseudo_loader,
                optimizer,
                criterion,
                device,
                weight=PSEUDO_WEIGHT
            )

            update_teacher(student_model, teacher_model, EMA_ALPHA)

            print(
                f"Stage {stage+1} | Epoch [{epoch+1}/{RETRAIN_EPOCHS_PER_STAGE}] "
                f"Labeled: {loss_labeled:.4f} | Pseudo: {loss_pseudo:.4f}"
            )

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(
            teacher_model.state_dict(),
            f"checkpoints/teacher_stage{stage+1}_thr{threshold}.pth"
        )

    # =========================
    # SAVE FINAL TEACHER
    # =========================
    torch.save(
        teacher_model.state_dict(),
        "checkpoints/resnet18_mean_teacher_final.pth"
    )
    print("\nFinal Mean Teacher model saved.")


if __name__ == "__main__":
    main()
