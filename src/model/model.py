import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn as nn

def get_model(num_classes):
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model