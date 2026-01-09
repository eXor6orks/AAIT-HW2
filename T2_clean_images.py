import argparse
import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

from src.model.model import get_model_ResNet_50, get_model_ResNeXt_50

NUM_WORKERS = 4
 
class CleaningDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['sample']
        full_path = os.path.join(self.root_dir, path)
        if not os.path.exists(full_path):
             if 'task' in path:
                 parts = path.split('/')
                 try:
                     task_part = next(p for p in parts if 'task' in p)
                     idx_task = parts.index(task_part)
                     full_path = os.path.join(self.root_dir, *parts[idx_task:])
                 except:
                     pass
 
        try:
            image = Image.open(full_path).convert("RGB")
        except:
            image = Image.new('RGB', (224, 224))
        label = int(row['label'])
        if self.transform:
            image = self.transform(image)
        return image, label, path
 
def clean_labels(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Nettoyage avec taux de rejet: {args.noise_rate*100}%")

    csv_path = os.path.join(args.dataset, 'train_data/annotations.csv')
    if not os.path.exists(csv_path):
        print(f"[ERREUR] Fichier introuvable : {csv_path}")
        return
 
    df = pd.read_csv(csv_path)
    # Correction colonnes
    df.columns = df.columns.str.strip()
    rename_map = {
        'renamed_path': 'sample',
        'label_idx': 'label',
        'Label': 'label',
        'Sample': 'sample',
        'path': 'sample',
        'class': 'label'
    }
    df.rename(columns=rename_map, inplace=True)

    if 'sample' not in df.columns or 'label' not in df.columns:
        print(f"[ERREUR] Colonnes manquantes. Colonnes actuelles: {df.columns.tolist()}")
        return
 
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ds = CleaningDataset(df, args.dataset, transform=transform)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)

    # --- MISE A JOUR ICI : Support ResNeXt ---
    
    if args.model_type == 'ResNeXt50':
        model = get_model_ResNeXt_50(num_classes=100)
    else:
        model = get_model_ResNet_50(num_classes=100)
        
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss(reduction='none') 
    all_losses = []
    all_paths = []
    all_labels = []
    print(">>> Calcul des erreurs par image...")
    with torch.no_grad():
        for images, labels, paths in tqdm(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            losses = criterion(outputs, labels)
            all_losses.extend(losses.cpu().numpy())
            all_paths.extend(paths)
            all_labels.extend(labels.cpu().numpy())
    res_df = pd.DataFrame({'sample': all_paths, 'label': all_labels, 'loss': all_losses})
    threshold = res_df['loss'].quantile(1 - args.noise_rate)
    clean_df = res_df[res_df['loss'] < threshold].copy()
    print(f"[RESULT] Seuil de Loss calculé: {threshold:.4f}")
    print(f"[RESULT] Taille Dataset: {len(res_df)} -> {len(clean_df)} (bruit retiré)")
    output_csv = os.path.join(args.dataset, 'task2_clean.csv')
    clean_df[['sample', 'label']].to_csv(output_csv, index=False)
    print(f"[SUCCESS] Fichier généré : {output_csv}")
 
if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--noise_rate', type=float, default=0.25)
    parser.add_argument('--model_type', type=str, help='ResNet50 or ResNeXt50', default='ResNet50')
    args = parser.parse_args()
    clean_labels(args)