import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from itertools import cycle
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import argparse

from torch.amp import GradScaler, autocast

from src.dataset.dataset import (
    LabeledImageDataset,
    UnlabeledImageDataset,
    load_annotations
)
from src.visual.visual_task_1 import (
    plot_training_curves,
    plot_pseudo_distribution,
    plot_distribution_alignment
)
from src.model.model import get_model_ResNet_34

# =========================
# CONFIG
# =========================
NUM_CLASSES = 100
BATCH_SIZE = 32
LR_STUDENT = 2e-4

# Pseudo-label threshold adaptatif
PSEUDO_THRESHOLD_START = 0.95 
PSEUDO_THRESHOLD_END = 0.75 
PSEUDO_THRESHOLD_WARMUP = 15

EMA_DECAY = 0.995
DA_DECAY = 0.99  
MIXUP_ALPHA = 0.6  
T_SHARPEN = 0.5  

RANDAUG_N = 2  
RANDAUG_M = 10   

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# =========================
# DIRECTORIES
# =========================
EXP_DIR = Path("experiments/remixmatch_optimized")
VIZ_DIR = EXP_DIR / "visualizations"
EXP_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# TRANSFORMS
# =========================
class OptimizedTransform:
    def __init__(self):
        self.weak = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.strong = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=RANDAUG_N, magnitude=RANDAUG_M),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.weak(x), self.strong(x)

# =========================
# UTILS
# =========================
@torch.no_grad()
def update_teacher_ema(student, teacher, alpha):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(alpha).add_(ps.data, alpha=1 - alpha)

def mixup_data(x, y, alpha):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def get_adaptive_threshold(epoch, total_epoch, start, end, warmup):
    """Threshold adaptatif qui décroît progressivement"""
    if epoch < warmup:
        return start
    progress = (epoch - warmup) / (total_epoch - warmup)
    return start - (start - end) * progress

def get_lambda_u(epoch, max_epochs):
    """Pondération progressive de la loss unsupervised"""
    return min(1.0, epoch / 10.0)



# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="AAIT Task 1 - Meta pseudo labels")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--checkpoints", type=str, default="checkpoints")
    parser.add_argument("--wu_checkpoints", type=str, default="checkpoints")
    args = parser.parse_args()

    EPOCHS = args.epochs
    LABELED_CSV = os.path.join(args.dataset, "train_data/annotations.csv")
    UNLABELED_DIR = os.path.join(args.dataset, "train_data/images/unlabeled")
    WARMUP_CHECKPOINT = os.path.join(args.wu_checkpoints, "resnet_warmup.pth")

    print(f"🚀 ReMixMatch CONSERVATIVE Training on {device}")
    print(f"📊 Config: BS={BATCH_SIZE}, LR={LR_STUDENT}, Epochs={EPOCHS}")
    print(f"📊 Threshold: {PSEUDO_THRESHOLD_START} → {PSEUDO_THRESHOLD_END}")
    print(f"📊 RandAugment: N={RANDAUG_N}, M={RANDAUG_M}, MixUp α={MIXUP_ALPHA}")
    print(f"📊 Weight Decay: {WEIGHT_DECAY}, DA Decay: {DA_DECAY}")

    # Models
    student = get_model_ResNet_34(NUM_CLASSES).to(device)
    teacher = get_model_ResNet_34(NUM_CLASSES).to(device)

    if os.path.exists(WARMUP_CHECKPOINT):
        print("🔥 Loading warmup checkpoint")
        state = torch.load(WARMUP_CHECKPOINT, map_location=device)
        student.load_state_dict(state)
        teacher.load_state_dict(state)
        student.train()
    else:
        print("⚠️  No warmup checkpoint found, starting from scratch")

    for p in teacher.parameters():
        p.requires_grad = False

    samples, labels = load_annotations(LABELED_CSV)
    print(f"📊 Loaded {len(samples)} labeled samples")
    
    tf = OptimizedTransform()

    labeled_loader = DataLoader(
        LabeledImageDataset(samples, labels, transform=tf.weak),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    unlabeled_loader = DataLoader(
        UnlabeledImageDataset(UNLABELED_DIR, transform=tf),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    print(f"📊 Unlabeled batches per epoch: {len(unlabeled_loader)}")

    labeled_iter = cycle(labeled_loader)
    
    # AdamW avec weight decay
    optimizer = optim.AdamW(student.parameters(), lr=LR_STUDENT, weight_decay=WEIGHT_DECAY)
    
    # Cosine Annealing Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    scaler = GradScaler("cuda")
    
    # Gradient clipping pour stabilité
    MAX_GRAD_NORM = 1.0

    # Distribution alignment
    target_probs = torch.ones(NUM_CLASSES, device=device) / NUM_CLASSES

    # History tracking
    history = {
        "loss_total": [],
        "loss_supervised": [],
        "loss_unsupervised": [],
        "mask_rate": [],
        "learning_rate": [],
        "threshold": []
    }
    all_target_distributions = []
    all_pseudo_counts = []

    # Training loop
    for epoch in range(EPOCHS):
        student.train()
        teacher.train()

        # ✅ Adaptive threshold
        current_threshold = get_adaptive_threshold(
            epoch, EPOCHS, PSEUDO_THRESHOLD_START, PSEUDO_THRESHOLD_END, PSEUDO_THRESHOLD_WARMUP
        )
        
        # ✅ Progressive lambda_u
        lambda_u = get_lambda_u(epoch, EPOCHS)

        epoch_loss = epoch_ls = epoch_lu = epoch_mask = 0.0
        pseudo_class_counter = torch.zeros(NUM_CLASSES, device=device)

        pbar = tqdm(unlabeled_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch_idx, ((x_u_w, x_u_s), _) in enumerate(pbar):
            x_l, y_l = next(labeled_iter)

            x_u_w = x_u_w.to(device, non_blocking=True)
            x_u_s = x_u_s.to(device, non_blocking=True)
            x_l = x_l.to(device, non_blocking=True)
            y_l = y_l.to(device, non_blocking=True)

            with autocast("cuda"):
                # ===========================
                # PSEUDO-LABELS (TEACHER)
                # ===========================
                with torch.no_grad():
                    logits_u = teacher(x_u_w)
                    probs_u = F.softmax(logits_u, dim=1)

                    # Distribution Alignment
                    probs_u = probs_u * (target_probs + 1e-6)
                    probs_u = probs_u / probs_u.sum(dim=1, keepdim=True)
                    target_probs = DA_DECAY * target_probs + (1 - DA_DECAY) * probs_u.mean(0)

                    # ✅ Temperature Sharpening
                    probs_u = probs_u ** (1 / T_SHARPEN)
                    probs_u = probs_u / probs_u.sum(dim=1, keepdim=True)

                    max_probs, pseudo_labels = probs_u.max(1)
                    mask = max_probs.ge(current_threshold).float()

                    # Count pseudo-labels
                    for c in pseudo_labels[mask.bool()]:
                        pseudo_class_counter[c] += 1

                # ===========================
                # SUPERVISED LOSS (avec MixUp)
                # ===========================
                mixed_x, y_a, y_b, lam = mixup_data(x_l, y_l, MIXUP_ALPHA)
                logits_mix = student(mixed_x)
                loss_sup = lam * F.cross_entropy(logits_mix, y_a) + \
                           (1 - lam) * F.cross_entropy(logits_mix, y_b)

                # ===========================
                # UNSUPERVISED LOSS
                # ===========================
                logits_u_s = student(x_u_s)
                loss_unsup = (F.cross_entropy(
                    logits_u_s, pseudo_labels, reduction="none") * mask).mean()

                # ✅ Loss pondérée avec lambda_u progressif
                loss = loss_sup + lambda_u * loss_unsup

            # Optimization
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            
            # Gradient clipping pour stabilité
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), MAX_GRAD_NORM)
            
            scaler.step(optimizer)
            scaler.update()

            # EMA update
            update_teacher_ema(student, teacher, EMA_DECAY)

            # Metrics
            epoch_loss += loss.item()
            epoch_ls += loss_sup.item()
            epoch_lu += loss_unsup.item()
            epoch_mask += mask.mean().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.3f}',
                'L_s': f'{loss_sup.item():.3f}',
                'L_u': f'{loss_unsup.item():.3f}',
                'Mask': f'{mask.mean().item():.2f}',
                'Thr': f'{current_threshold:.2f}'
            })

        # End of epoch
        scheduler.step()
        
        n = len(unlabeled_loader)
        avg_loss = epoch_loss / n
        avg_ls = epoch_ls / n
        avg_lu = epoch_lu / n
        avg_mask = epoch_mask / n
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history["loss_total"].append(avg_loss)
        history["loss_supervised"].append(avg_ls)
        history["loss_unsupervised"].append(avg_lu)
        history["mask_rate"].append(avg_mask)
        history["learning_rate"].append(current_lr)
        history["threshold"].append(current_threshold)
        
        all_target_distributions.append(target_probs.cpu().numpy())
        all_pseudo_counts.append(pseudo_class_counter.cpu().numpy())

        # Save pseudo stats
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            with open(EXP_DIR / f"pseudo_stats_epoch_{epoch+1}.json", "w") as f:
                json.dump({
                    "epoch": epoch + 1,
                    "total": int(pseudo_class_counter.sum().item()),
                    "threshold": current_threshold,
                    "lambda_u": lambda_u,
                    "per_class": {str(i): int(pseudo_class_counter[i].item()) for i in range(NUM_CLASSES)}
                }, f, indent=2)

        print(f"\n[Epoch {epoch+1}/{EPOCHS}] "
              f"Loss={avg_loss:.4f} (L_s={avg_ls:.4f}, L_u={avg_lu:.4f}) | "
              f"Mask={avg_mask:.3f} | LR={current_lr:.2e} | Thr={current_threshold:.2f}")

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            print("📊 Generating visualizations...")
            plot_training_curves(history, epoch + 1)
            plot_pseudo_distribution(pseudo_class_counter.cpu().numpy(), epoch + 1)
            plot_distribution_alignment(all_target_distributions, epoch + 1)

        # Save checkpoints
        torch.save({
            'epoch': epoch + 1,
            'student_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history
        }, EXP_DIR / "checkpoint_latest.pth")
        
        # Best model (based on mask rate > 0.3 and lowest loss)
        if avg_mask > 0.3:
            if not hasattr(main, 'best_loss') or avg_loss < main.best_loss:
                main.best_loss = avg_loss
                torch.save(student.state_dict(), EXP_DIR / "student_best.pth")
                torch.save(teacher.state_dict(), EXP_DIR / "teacher_best.pth")
                print(f"✅ New best model saved! (Loss: {avg_loss:.4f})")

    # Save final results
    with open(EXP_DIR / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    np.save(EXP_DIR / "target_distributions.npy", np.stack(all_target_distributions))
    np.save(EXP_DIR / "pseudo_counts.npy", np.stack(all_pseudo_counts))

    torch.save(student.state_dict(), EXP_DIR / "student_final.pth")
    torch.save(teacher.state_dict(), EXP_DIR / "teacher_final.pth")

    print("\n✅ ReMixMatch OPTIMIZED Training Finished!")
    print(f"📁 Results saved in: {EXP_DIR}")
    print(f"📊 Visualizations in: {VIZ_DIR}")

if __name__ == "__main__":
    main()