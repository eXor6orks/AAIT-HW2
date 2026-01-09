import os
import csv
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from src.model.model import get_model


# =========================
# CONFIG
# =========================
NUM_CLASSES = 100
BATCH_SIZE = 32
CHECKPOINT_PATH = "checkpoints/retrained_student.pth"
VAL_DIR = "task1/val_data"
OUTPUT_CSV = "submission.csv"


# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# =========================
# MODEL
# =========================
model = get_model(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model = model.to(device)
model.eval()


# =========================
# TRANSFORMS
# =========================
eval_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# DATASET
# =========================
class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.images = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, img_name


eval_dataset = EvalDataset(VAL_DIR, transform=eval_transform)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0  # important sous Windows
)


# =========================
# INFERENCE
# =========================
predictions = []

with torch.no_grad():
    for images, names in eval_loader:
        images = images.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        for name, label in zip(names, preds):
            predictions.append((name, label.item()))


# =========================
# WRITE CSV
# =========================
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "label"])

    for name, label in predictions:
        writer.writerow([f"task1/val_data/{name}", label])

print(f"\nSubmission file saved to: {OUTPUT_CSV}")
print(f"Total samples: {len(predictions)}")
