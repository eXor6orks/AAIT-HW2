import torchvision.models as models
from torchvision.models import ResNet34_Weights
import torch.nn as nn

def get_model(num_classes):
    model = models.resnet34(weights=ResNet34_Weights.DEFAULT)

    # 🔥 Adaptation 64x64
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model