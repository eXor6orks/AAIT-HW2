import torch
import torch.nn as nn
import torch.nn.functional as F

class SymmetricCrossEntropy(nn.Module):
    """
    Symmetric Cross Entropy pour l'apprentissage avec labels bruités.
    Reference: Wang et al. ICCV 2019
    """
    def __init__(self, alpha=0.1, beta=1.0, num_classes=10):
        super(SymmetricCrossEntropy, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CE (Robustesse à la convergence)
        ce = self.cross_entropy(pred, labels)

        # RCE (Robustesse au bruit)
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)

        loss = self.alpha * ce + self.beta * rce.mean()
        return loss
