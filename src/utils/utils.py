import random
import numpy as np
import torch

def set_seed(seed=42):
    """Assure la reproductibilité totale des expériences."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed fixée à : {seed}")