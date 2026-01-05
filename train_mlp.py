import os
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms
from collections import Counter, defaultdict

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
LR_STUDENT = 1e-4
LR_TEACHER = 5e-5

WARMUP_EPOCHS = 20
MPL_EPOCHS = 30

PSEUDO_THRESHOLD = 0.9
PSEUDO_WEIGHT = 1.0
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
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
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

student = get_model(NUM_CLASSES).to(device)
teacher = copy.deepcopy(student).to(device)

# =========================
# DATASETS
# =========================
samples, labels = load_annotations(LABELED_CSV)

label_counts = Counter(labels)
class_weights = torch.tensor(
    [1.0 / label_counts.get(i, 1) for i in range(NUM_CLASSES)],
    device=device
)
class_weights = class_weights / class_weights.mean()

labeled_ds = LabeledImageDataset(samples, labels, transform=train_transform)
labeled_loader = DataLoader(
    labeled_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

unlabeled_ds = UnlabeledImageDataset(
    UNLABELED_DIR, transform=eval_transform
)
unlabeled_loader = DataLoader(
    unlabeled_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

# =========================
# UTILS
# =========================
def update_ema(teacher, student, alpha):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.data.mul_(alpha).add_(s.data * (1 - alpha))

def train_supervised(model, loader, optimizer, criterion):
    model.train()
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)

# =========================
# MAIN
# =========================
def main():
    opt_student = optim.Adam(student.parameters(), lr=LR_STUDENT)
    opt_teacher = optim.Adam(teacher.parameters(), lr=LR_TEACHER)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    print("\n===== SUPERVISED WARM-UP =====")
    for epoch in range(WARMUP_EPOCHS):
        ls = train_supervised(student, labeled_loader, opt_student, criterion)
        lt = train_supervised(teacher, labeled_loader, opt_teacher, criterion)
        update_ema(teacher, student, EMA_ALPHA)
        print(f"[{epoch+1}/{WARMUP_EPOCHS}] Student {ls:.4f} | Teacher {lt:.4f}")

    print("\n===== META PSEUDO LABEL TRAINING =====")
    for epoch in range(MPL_EPOCHS):
        student.train()
        teacher.train()

        stats = defaultdict(int)
        total_loss = 0

        for (x_l, y_l), (x_u, _) in zip(labeled_loader, unlabeled_loader):
            x_l, y_l = x_l.to(device), y_l.to(device)
            x_u = x_u.to(device)

            # --- Teacher pseudo-labels ---
            with torch.no_grad():
                logits_u = teacher(x_u)
                probs_u = F.softmax(logits_u, dim=1)
                max_probs, pseudo_y = probs_u.max(dim=1)
                mask = max_probs > PSEUDO_THRESHOLD

            if mask.sum() == 0:
                continue

            x_u = x_u[mask]
            pseudo_y = pseudo_y[mask]

            for c in pseudo_y.tolist():
                stats[c] += 1

            # --- Student update ---
            loss_u = F.cross_entropy(student(x_u), pseudo_y)
            loss_l = criterion(student(x_l), y_l)
            loss = loss_l + PSEUDO_WEIGHT * loss_u

            opt_student.zero_grad()
            loss.backward()
            opt_student.step()

            # --- Teacher meta-update ---
            meta_loss = criterion(student(x_l).detach(), y_l)
            opt_teacher.zero_grad()
            meta_loss.backward()
            opt_teacher.step()

            update_ema(teacher, student, EMA_ALPHA)

            total_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{MPL_EPOCHS}] "
            f"Loss: {total_loss:.4f} | "
            f"Pseudo-labels used: {sum(stats.values())}"
        )

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(student.state_dict(), "checkpoints/student_mpl.pth")
    torch.save(teacher.state_dict(), "checkpoints/teacher_mpl.pth")
    print("\n✅ Meta Pseudo Label training finished.")

if __name__ == "__main__":
    main()
