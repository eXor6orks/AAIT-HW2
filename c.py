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
    load_annotations,
    SoftLabeledImageDataset
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
WARMUP_EPOCHS = 40
RETRAIN_EPOCHS_PER_STAGE = 10
PSEUDO_THRESHOLDS = [0.95, 0.9, 0.85]
PSEUDO_WEIGHT = 0.3
EMA_ALPHA = 0.999
TEMPERATURE = 0.7
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
teacher_model = copy.deepcopy(student_model).to(device)

# =========================
# DATASETS
# =========================
samples, labels = load_annotations(LABELED_CSV)

# from collections import Counter

# label_counts = Counter(labels)

# print("Nombre de classes :", len(label_counts))
# print("Nombre total d’images labellisées :", len(labels))

# # Affichage trié par fréquence croissante
# for label, count in sorted(label_counts.items(), key=lambda x: x[1]):
#     print(f"Classe {label:3d} : {count:4d} images")

# input("...")

labeled_dataset = LabeledImageDataset(samples, labels, transform=train_transform)
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
# UTILS
# =========================
def update_ema(teacher, student, alpha):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.data.mul_(alpha).add_(s.data * (1 - alpha))


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def generate_pseudo_labels_soft(teacher, dataloader, threshold):
    teacher.eval()
    pseudo_samples = []
    pseudo_labels = []

    with torch.no_grad():
        for images, paths in dataloader:
            images = images.to(device)
            logits = teacher(images)
            probs = F.softmax(logits / TEMPERATURE, dim=1)

            max_probs, _ = probs.max(dim=1)
            mask = max_probs > threshold

            for i in range(len(mask)):
                if mask[i]:
                    pseudo_samples.append(paths[i])
                    pseudo_labels.append(probs[i].cpu())

    return pseudo_samples, pseudo_labels


def train_student_with_pseudo(student, loader, optimizer):
    student.train()
    total_loss = 0.0

    for images, soft_labels in loader:
        images = images.to(device)
        soft_labels = soft_labels.to(device)

        logits = student(images)
        log_probs = F.log_softmax(logits, dim=1)

        loss = -(soft_labels * log_probs).sum(dim=1).mean()
        loss = loss * PSEUDO_WEIGHT

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# =========================
# MAIN
# =========================
def main():
    optimizer_student = optim.Adam(student_model.parameters(), lr=LR)
    optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    print("\n===== Supervised warm-up =====")
    for epoch in range(WARMUP_EPOCHS):
        ls = train_epoch(student_model, labeled_loader, optimizer_student, criterion)
        lt = train_epoch(teacher_model, labeled_loader, optimizer_teacher, criterion)
        update_ema(teacher_model, student_model, EMA_ALPHA)
        print(f"Epoch [{epoch+1}/{WARMUP_EPOCHS}] | Student: {ls:.4f} | Teacher: {lt:.4f}")

    for stage, threshold in enumerate(PSEUDO_THRESHOLDS):
        print(f"\n===== Stage {stage+1} | threshold={threshold} =====")

        pseudo_samples, pseudo_labels = generate_pseudo_labels_soft(
            teacher_model, unlabeled_loader, threshold
        )

        print(f"Accepted pseudo-labels: {len(pseudo_samples)}")

        if len(pseudo_samples) == 0:
            continue

        pseudo_dataset = SoftLabeledImageDataset(
            pseudo_samples, pseudo_labels, transform=train_transform
        )

        pseudo_loader = DataLoader(
            pseudo_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS
        )

        for epoch in range(RETRAIN_EPOCHS_PER_STAGE):
            ls = train_epoch(student_model, labeled_loader, optimizer_student, criterion)
            lp = train_student_with_pseudo(student_model, pseudo_loader, optimizer_student)
            update_ema(teacher_model, student_model, EMA_ALPHA)

            print(
                f"Stage {stage+1} | Epoch [{epoch+1}] "
                f"Labeled: {ls:.4f} | Pseudo: {lp:.4f}"
            )

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(
            teacher_model.state_dict(),
            f"checkpoints/teacher_stage{stage+1}.pth"
        )

    torch.save(student_model.state_dict(), "checkpoints/student_final_soft.pth")
    torch.save(teacher_model.state_dict(), "checkpoints/teacher_final_soft.pth")
    print("\nTraining finished.")

if __name__ == "__main__":
    main()
