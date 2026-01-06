import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split

import torchvision.transforms as T
import torchvision.models as models

from sklearn.mixture import GaussianMixture

NUM_CLASSES = 100
BATCH_SIZE = 64
EPOCHS = 50
WARMUP_EPOCHS = 5
VAL_SPLIT = 0.1

LR = 3e-4
LAMBDA_U = 25
T_SHARPEN = 0.5
M = 2  # Nombre d'augmentations pour unlabeled data
TAU = 0.5

LOSS_EMA_ALPHA = 0.9
RAMPUP_EPOCHS = 10
CLIP_GRAD = 5.0
USE_AMP = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DivideMixDataset(Dataset):
    """Dataset qui retourne weak ET strong augmentations"""
    def __init__(self, csv_path, mode='train'):
        self.df = pd.read_csv(csv_path)
        self.mode = mode
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["renamed_path"]).convert("RGB")
        label = int(row["label_idx"])
        
        if self.mode == 'train':
            # Retourne weak ET strong augmentations
            return weak_tf(img), strong_tf(img), label, idx
        elif self.mode == 'eval':
            # Pour calcul des pertes (pas d'augmentation forte)
            return weak_tf(img), label, idx
        else:  # validation
            # Pas d'augmentation pour validation
            return eval_tf(img), label, idx

# Transformations faibles (pour labeled data et eval)
weak_tf = T.Compose([
    T.Resize((64,64)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Transformations fortes (pour unlabeled data dans MixMatch)
strong_tf = T.Compose([
    T.Resize((64,64)),
    T.RandomHorizontalFlip(),
    T.RandomCrop(64, padding=8),
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Transformations pour validation (pas d'augmentation)
eval_tf = T.Compose([
    T.Resize((64,64)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def get_model():
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(3,64,3,1,1,bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

def sharpen(p, T):
    """Sharpen les prédictions pour pseudo-labeling"""
    p = p ** (1.0 / T)
    return p / p.sum(dim=1, keepdim=True)

def mixmatch(x, y, alpha=0.75):
    """MixUp pour données et labels"""
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1-lam)  # Assure que lam >= 0.5
    idx = torch.randperm(x.size(0))
    return lam*x + (1-lam)*x[idx], lam*y + (1-lam)*y[idx]

def gmm_split_ema(current_losses, loss_ema):
    """GMM pour séparer clean/noisy avec EMA des pertes"""
    current_losses = np.array(current_losses)

    if loss_ema is None:
        loss_ema = current_losses.copy()
    else:
        loss_ema = LOSS_EMA_ALPHA * loss_ema + (1 - LOSS_EMA_ALPHA) * current_losses

    gmm = GaussianMixture(2, max_iter=20, random_state=0)
    gmm.fit(loss_ema.reshape(-1,1))
    prob = gmm.predict_proba(loss_ema.reshape(-1,1))

    # Probabilité d'être dans le cluster avec la moyenne la plus faible (clean)
    clean_cluster = np.argmin(gmm.means_)
    return prob[:, clean_cluster], loss_ema

def warmup_epoch(loader, model, optimizer):
    """Entraînement standard pendant warmup"""
    model.train()
    ce = nn.CrossEntropyLoss()

    correct, total = 0, 0
    for x_weak, x_strong, y, _ in tqdm(loader, desc="Warmup"):
        x, y = x_weak.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        loss = ce(model(x), y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)

    return correct / total

def evaluate(loader, model):
    """Évaluation sur validation set"""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)

    return correct / total

scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

def train_dividemix_epoch(
    epoch, model1, model2, opt1, opt2, train_ds, loss_ema_dict
):
    """
    DivideMix avec co-division:
    - model1 utilise prob2 pour diviser
    - model2 utilise prob1 pour diviser
    """
    model1.train()
    model2.train()
    ce = nn.CrossEntropyLoss(reduction="none")

    # Créer un subset avec mode 'eval' pour calcul des pertes
    train_indices = train_ds.indices if isinstance(train_ds, Subset) else range(len(train_ds))
    base_dataset = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    
    # Dataset temporaire pour évaluation (mode eval = weak transform seulement)
    eval_dataset = DivideMixDataset(base_dataset.df.iloc[train_indices].to_csv(index=False), mode='eval')
    eval_loader = DataLoader(eval_dataset, BATCH_SIZE, shuffle=False)

    # ---- Calcul des pertes par échantillon
    losses1, losses2 = [], []
    with torch.no_grad():
        for x, y, _ in eval_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            losses1.extend(ce(model1(x), y).cpu().numpy())
            losses2.extend(ce(model2(x), y).cpu().numpy())

    # ---- GMM pour identifier clean/noisy
    prob1, loss_ema_dict['ema1'] = gmm_split_ema(losses1, loss_ema_dict['ema1'])
    prob2, loss_ema_dict['ema2'] = gmm_split_ema(losses2, loss_ema_dict['ema2'])

    # ---- CO-DIVISION: model1 utilise prob2, model2 utilise prob1
    clean_idx_1 = np.where(prob2 > TAU)[0]
    noisy_idx_1 = np.where(prob2 <= TAU)[0]

    clean_idx_2 = np.where(prob1 > TAU)[0]
    noisy_idx_2 = np.where(prob1 <= TAU)[0]

    # Pour simplifier, on utilise l'intersection pour clean et l'union pour noisy
    clean_idx = np.intersect1d(clean_idx_1, clean_idx_2)
    noisy_idx = np.union1d(noisy_idx_1, noisy_idx_2)

    if len(clean_idx) == 0:
        print("⚠️  Aucun échantillon clean, utilise tout le dataset")
        clean_idx = np.arange(len(eval_dataset))
        noisy_idx = np.array([])

    clean_ratio = len(clean_idx) / len(eval_dataset)
    print(f"Clean ratio: {clean_ratio:.2%} ({len(clean_idx)}/{len(eval_dataset)})")

    # Créer des subsets en mode 'train' (weak + strong transforms)
    train_dataset_full = DivideMixDataset(base_dataset.df.iloc[train_indices].to_csv(index=False), mode='train')
    
    clean_ds = Subset(train_dataset_full, clean_idx)
    clean_loader = DataLoader(clean_ds, BATCH_SIZE, shuffle=True, drop_last=True)

    if len(noisy_idx) > 0:
        noisy_ds = Subset(train_dataset_full, noisy_idx)
        noisy_loader = DataLoader(noisy_ds, BATCH_SIZE, shuffle=True, drop_last=True)
        noisy_iter = iter(noisy_loader)
    else:
        noisy_iter = None

    # ---- Lambda ramp-up
    lambda_u = LAMBDA_U * min(1.0, (epoch - WARMUP_EPOCHS) / RAMPUP_EPOCHS)

    for x_weak, x_strong, y, idx in tqdm(clean_loader, desc=f"Epoch {epoch}"):
        x_weak = x_weak.to(DEVICE)
        x_strong = x_strong.to(DEVICE)
        y = y.to(DEVICE)

        # Obtenir un batch d'échantillons bruités (unlabeled)
        if noisy_iter is not None:
            try:
                u_weak, u_strong, _, _ = next(noisy_iter)
            except StopIteration:
                noisy_iter = iter(noisy_loader)
                u_weak, u_strong, _, _ = next(noisy_iter)
            
            u_weak = u_weak.to(DEVICE)
            u_strong = u_strong.to(DEVICE)
        else:
            # Pas d'échantillons bruités, utiliser les clean
            u_weak, u_strong = x_weak, x_strong

        with torch.amp.autocast('cuda', enabled=USE_AMP):
            # ---- Label refinement (pour labeled data)
            with torch.no_grad():
                # Moyenne des prédictions sur weak augmentation
                px = (F.softmax(model1(x_weak), 1) + F.softmax(model2(x_weak), 1)) / 2
                
                # One-hot encoding
                y_onehot = F.one_hot(y, NUM_CLASSES).float()
                
                # Pondération par prob d'être clean
                idx_np = idx.cpu().numpy()
                w1 = torch.tensor([prob2[i] if i < len(prob2) else 1.0 for i in idx_np]).to(DEVICE)
                w2 = torch.tensor([prob1[i] if i < len(prob1) else 1.0 for i in idx_np]).to(DEVICE)
                
                # Labels raffinés
                y_hat1 = sharpen(w1[:,None]*y_onehot + (1-w1[:,None])*px, T_SHARPEN)
                y_hat2 = sharpen(w2[:,None]*y_onehot + (1-w2[:,None])*px, T_SHARPEN)

            # ---- Label co-guessing (pour unlabeled data)
            with torch.no_grad():
                # Prédictions sur weak augmentation
                pu = (F.softmax(model1(u_weak), 1) + F.softmax(model2(u_weak), 1)) / 2
                q_u = sharpen(pu, T_SHARPEN)
                # Répéter M fois pour avoir M augmentations
                q_u_repeated = q_u.repeat(M, 1)

            # ---- MixMatch pour model1 (utilise STRONG augmentation)
            # Répéter les unlabeled M fois avec leurs M augmentations fortes
            u_strong_repeated = u_strong.repeat(M, 1, 1, 1)
            
            X1 = torch.cat([x_strong, u_strong_repeated])
            Y1 = torch.cat([y_hat1, q_u_repeated])
            X1_mix, Y1_mix = mixmatch(X1, Y1)

            logits1 = model1(X1_mix)
            
            # Séparer supervised et unsupervised loss
            batch_size = x_strong.size(0)
            Lx1 = -(Y1_mix[:batch_size] * F.log_softmax(logits1[:batch_size], 1)).sum(1).mean()
            Lu1 = -(Y1_mix[batch_size:] * F.log_softmax(logits1[batch_size:], 1)).sum(1).mean()
            
            loss1 = Lx1 + lambda_u * Lu1

            # ---- MixMatch pour model2 (utilise STRONG augmentation)
            X2 = torch.cat([x_strong, u_strong_repeated])
            Y2 = torch.cat([y_hat2, q_u_repeated])
            X2_mix, Y2_mix = mixmatch(X2, Y2)

            logits2 = model2(X2_mix)
            
            Lx2 = -(Y2_mix[:batch_size] * F.log_softmax(logits2[:batch_size], 1)).sum(1).mean()
            Lu2 = -(Y2_mix[batch_size:] * F.log_softmax(logits2[batch_size:], 1)).sum(1).mean()
            
            loss2 = Lx2 + lambda_u * Lu2

        # Optimisation
        opt1.zero_grad()
        scaler.scale(loss1).backward()
        torch.nn.utils.clip_grad_norm_(model1.parameters(), CLIP_GRAD)
        scaler.step(opt1)

        opt2.zero_grad()
        scaler.scale(loss2).backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), CLIP_GRAD)
        scaler.step(opt2)

        scaler.update()

    return clean_ratio


def main():
    loss_ema_dict = {'ema1': None, 'ema2': None}

    # Dataset en mode 'train' (weak + strong transforms)
    full_ds = DivideMixDataset("task2/train_data/annotations.csv", mode='train')

    val_size = int(len(full_ds) * VAL_SPLIT)
    train_size = len(full_ds) - val_size
    train_ds, val_ds_indices = random_split(full_ds, [train_size, val_size])

    # Pour validation, on veut mode='validation' (eval_tf sans augmentation)
    val_base = DivideMixDataset("task2/train_data/annotations.csv", mode='validation')
    val_ds = Subset(val_base, val_ds_indices.indices)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE)

    model1 = get_model().to(DEVICE)
    model2 = get_model().to(DEVICE)

    opt1 = torch.optim.Adam(model1.parameters(), LR, weight_decay=5e-4)
    opt2 = torch.optim.Adam(model2.parameters(), LR, weight_decay=5e-4)

    history_clean, history_val1, history_val2 = [], [], []

    print("\n" + "="*50)
    print("WARMUP PHASE")
    print("="*50)
    for e in range(WARMUP_EPOCHS):
        acc1 = warmup_epoch(train_loader, model1, opt1)
        val_acc1 = evaluate(val_loader, model1)
        print(f"Epoch {e} | Train: {acc1:.3%} | Val: {val_acc1:.3%}")

    # Copier model1 vers model2
    model2.load_state_dict(model1.state_dict())

    print("\n" + "="*50)
    print("DIVIDEMIX PHASE")
    print("="*50)
    for e in range(WARMUP_EPOCHS, EPOCHS):
        print(f"\n--- Epoch {e}/{EPOCHS} ---")
        
        clean_ratio = train_dividemix_epoch(
            e, model1, model2, opt1, opt2, train_ds, loss_ema_dict
        )
        
        val_acc1 = evaluate(val_loader, model1)
        val_acc2 = evaluate(val_loader, model2)
        
        print(f"Val → Model1: {val_acc1:.3%} | Model2: {val_acc2:.3%}")

        history_clean.append(clean_ratio)
        history_val1.append(val_acc1)
        history_val2.append(val_acc2)

    # Visualisation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(WARMUP_EPOCHS, EPOCHS)
    ax1.plot(epochs, history_val1, label="Model1", marker='o', linewidth=2)
    ax1.plot(epochs, history_val2, label="Model2", marker='s', linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Accuracy")
    ax1.set_title("Validation Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history_clean, label="Clean Ratio", color='green', marker='^', linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Clean Ratio")
    ax2.set_title("Clean Data Detection")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dividemix_training.png', dpi=150)
    print("\n✅ Graphique: dividemix_training.png")
    plt.show()

    # Sauvegarder meilleur modèle
    best_model = model1 if val_acc1 >= val_acc2 else model2
    best_acc = max(val_acc1, val_acc2)
    torch.save(best_model.state_dict(), 'best_dividemix_model.pth')
    print(f"✅ Modèle sauvegardé (Val Acc: {best_acc:.3%})")

if __name__ == "__main__":
    main()