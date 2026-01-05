import os
import csv
import torch
from torchvision import transforms
from PIL import Image

from src.model.model import get_model

# =========================
# CONFIG
# =========================
NUM_CLASSES = 100
BATCH_SIZE = 32

VAL_DIR = "task2/val_data"
MODEL_PATH = "checkpoints/best_model.pth"
OUTPUT_CSV = "submission.csv"

# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================
# TRANSFORMS (NO AUGMENT)
# =========================
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# LOAD MODEL
# =========================
model = get_model(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# =========================
# INFERENCE
# =========================
results = []

image_files = sorted([
    f for f in os.listdir(VAL_DIR)
    if f.endswith(".jpeg") or f.endswith(".jpg") or f.endswith(".png")
])

with torch.no_grad():
    for img_name in image_files:
        img_path = os.path.join(VAL_DIR, img_name)

        image = Image.open(img_path).convert("RGB")
        image = val_transform(image).unsqueeze(0).to(device)

        logits = model(image)
        pred = torch.argmax(logits, dim=1).item()

        results.append((
            f"{VAL_DIR}/{img_name}",
            pred
        ))

# =========================
# WRITE CSV
# =========================
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "label"])
    writer.writerows(results)

print(f"Submission file saved to: {OUTPUT_CSV}")
