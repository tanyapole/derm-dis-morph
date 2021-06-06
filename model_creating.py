from torchvision.models import resnet34
import torch, torch.nn as nn


def create_model(num_classes):
    model = resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes, bias=(model.fc.bias is None))
    return model
