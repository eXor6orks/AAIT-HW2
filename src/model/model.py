import torchvision.models as models
from torchvision.models import ResNet34_Weights, ResNet50_Weights, ResNeXt50_32X4D_Weights
import torch.nn as nn

def get_model_ResNet_34(num_classes):
    model = models.resnet34(weights=ResNet34_Weights.DEFAULT)

    # 🔥 Adaptation 64x64
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_model_ResNet_50(num_classes):
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, num_classes)
    )
    return model

def get_model_ResNeXt_50(num_classes):
    model = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, num_classes)
    )
    return model