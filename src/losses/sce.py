import torch
import torch.nn as nn
import torch.nn.functional as F

class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, num_classes=100):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets)

        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, min=1e-7, max=1.0)
        one_hot = F.one_hot(targets, self.num_classes).float()
        one_hot = torch.clamp(one_hot, min=1e-4, max=1.0)

        rce = -torch.mean(torch.sum(probs * torch.log(one_hot), dim=1))
        return self.alpha * ce + self.beta * rce
