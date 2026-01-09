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

import argparse

# Importations de tes modules locaux
from src.dataset.dataset import (
    LabeledImageDataset,
    UnlabeledImageDataset,
    load_annotations
)
from src.model.model import get_model_ResNet_34, get_model_ResNet_50

import torch.multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

eval_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_pseudo_labels(model, dataloader, threshold, device):
    model.eval()
    pseudo_samples, pseudo_labels = [], []
    stats = defaultdict(int)

    for images, paths in tqdm(dataloader, desc="Pseudo-labeling (TTA Flip)"):
        images = images.to(device)

        # Original
        logits1 = model(images)
        probs1 = F.softmax(logits1, dim=1)

        # Flip horizontal
        images_flip = torch.flip(images, dims=[3])
        logits2 = model(images_flip)
        probs2 = F.softmax(logits2, dim=1)

        # Moyenne TTA
        probs = (probs1 + probs2) / 2

        max_probs, preds = probs.max(dim=1)
        mask = max_probs > threshold

        for i in range(len(mask)):
            if mask[i]:
                pseudo_samples.append(paths[i])
                pseudo_labels.append(preds[i].item())
                stats[preds[i].item()] += 1

    return pseudo_samples, pseudo_labels, stats

def generate_PL_csv(pseudo_samples, pseudo_labels, output_csv, old_csv):
    with open(old_csv, "r") as f:
        existing_lines = f.readlines()[1:]  # Skip header
    existing_data = set(line.strip() for line in existing_lines)
    with open(output_csv, "w") as f:
        # Add existing data first
        f.write("sample,label\n")
        for line in existing_data:
            f.write(f"{line}\n")
        # Add pseudo-labeled data
        for sample, label in zip(pseudo_samples, pseudo_labels):
            f.write(f"{sample},{label}\n")

def main():
    parser = argparse.ArgumentParser(description="AAIT Task 1 - Pseudo Labels")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--path_csv", type=str, default="pseudo_labels.csv")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument('--model_type', type=str, help='ResNet50 or ResNet34', default='ResNet50')

    args = parser.parse_args()
    
    best_model_path = args.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == 'ResNet50':
        model = get_model_ResNet_50(num_classes=100)
    else:
        model = get_model_ResNet_34(num_classes=100)
        
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model = model.to(device)

    image_dir = os.path.join(args.dataset, 'train_data/images/unlabeled')

    unlabeled_dataset = UnlabeledImageDataset(
        image_dir=image_dir,
        transform=eval_transform
    )

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    threshold = 0.9
    pseudo_samples, pseudo_labels, stats = generate_pseudo_labels(
        model, unlabeled_loader, threshold, device
    )

    print(f"Generated {len(pseudo_samples)} pseudo-labels with threshold {threshold}")

    generate_PL_csv(pseudo_samples, pseudo_labels, args.path_csv, os.path.join(args.dataset,"train_data/annotations.csv"))

if __name__ == "__main__":
    main()