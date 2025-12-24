import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import os

def load_annotations(csv_path):
    df = pd.read_csv(csv_path)
    return df["sample"].values, df["label"].values

class LabeledImageDataset(Dataset):
    def __init__(self, samples, labels, transform=None):
        self.samples = samples
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = int(self.labels[idx])

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

class UnlabeledImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith(".jpeg") or f.endswith(".png")
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

from PIL import Image
import torch

class SoftLabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, samples, soft_labels, transform=None):
        self.samples = samples
        self.soft_labels = soft_labels
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.soft_labels[idx]  # Tensor [num_classes]
        return image, label