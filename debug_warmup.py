import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms
from src.dataset.dataset import LabeledImageDataset, load_annotations
from src.model.model import get_model

# CONFIG
NUM_CLASSES = 100
BATCH_SIZE = 4  # Petit batch pour debug
LR = 1e-4
WARMUP_EPOCHS = 2  # Juste 2 epochs pour voir

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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

# LOAD DATA
LABELED_CSV = "task1/train_data/annotations.csv"
samples, labels = load_annotations(LABELED_CSV)

print(f"\n📊 Dataset stats:")
print(f"  - Samples: {len(samples)}")
print(f"  - Labels: {len(labels)}")
print(f"  - Unique labels: {len(set(labels))}")
print(f"  - First sample path: {samples[0]}")

# Try loading first image
try:
    from PIL import Image
    img = Image.open(samples[0]).convert("RGB")
    print(f"  - First image loaded: ✅ {img.size}")
except Exception as e:
    print(f"  - First image load: ❌ {e}")

# CREATE DATASET & LOADER
labeled_dataset = LabeledImageDataset(samples, labels, transform=train_transform)
labeled_loader = DataLoader(
    labeled_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0  # Set to 0 for debugging
)

# TEST BATCH LOADING
print(f"\n📦 Testing batch loading...")
try:
    batch_images, batch_labels = next(iter(labeled_loader))
    print(f"  - Batch images shape: {batch_images.shape}")
    print(f"  - Batch labels shape: {batch_labels.shape}")
    print(f"  - Batch labels: {batch_labels}")
    print(f"  - Image values range: [{batch_images.min():.4f}, {batch_images.max():.4f}]")
except Exception as e:
    print(f"  - Batch loading failed: ❌ {e}")
    exit()

# MODEL
model = get_model(num_classes=NUM_CLASSES).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

print(f"\n🤖 Model created")

# TRAIN ONE EPOCH
def train_epoch_debug(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        # DEBUG: Check logits & probs
        if batch_idx == 0:
            probs = F.softmax(logits, dim=1)
            max_probs = probs.max(dim=1)[0]
            preds = logits.argmax(dim=1)
            
            print(f"\n🔍 First batch debug:")
            print(f"  - Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
            print(f"  - Logits mean: {logits.mean().item():.4f}")
            print(f"  - Probs max: {probs.max().item():.4f}")
            print(f"  - Max prob stats - Min: {max_probs.min():.4f}, Max: {max_probs.max():.4f}, Mean: {max_probs.mean():.4f}")
            print(f"  - Predictions: {preds}")
            print(f"  - True labels: {labels}")
            print(f"  - Accuracy: {(preds == labels).sum().item() / len(labels):.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += len(labels)

        if batch_idx % 100 == 0:
            accuracy = total_correct / total_samples
            print(f"  Batch {batch_idx}: Loss {loss.item():.4f}, Accuracy {accuracy:.4f}")

    return total_loss / len(loader), total_correct / total_samples

# RUN WARMUP WITH DEBUG
print(f"\n🔥 Starting warm-up with debug info...")
for epoch in range(WARMUP_EPOCHS):
    loss, acc = train_epoch_debug(model, labeled_loader, optimizer, criterion, device)
    print(f"Epoch [{epoch+1}/{WARMUP_EPOCHS}] - Loss: {loss:.4f}, Accuracy: {acc:.4f}\n")

print("✅ Debug complete!")
