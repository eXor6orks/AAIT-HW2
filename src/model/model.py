import torchvision.models as models
from torchvision.models import ResNet34_Weights
import torch.nn as nn

def get_model(num_classes):
    weights = ResNet34_Weights.DEFAULT
    model = models.resnet34(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model