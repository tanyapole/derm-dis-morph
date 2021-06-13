from torchvision.models import resnet34
import torch, torch.nn as nn
from pathlib import Path


def create_model(num_classes, dropout=False):
    model = resnet34(pretrained=True)
    if dropout:
        model.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(model.fc.in_features, num_classes, bias=(model.fc.bias is None))
        )
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes, bias=(model.fc.bias is None))
    return model

def load_model(fldr, num_classes, device=None, dropout=False):
    m = create_model(num_classes, dropout)
    src_fldr = Path('.') #'../derm-dis-morph')
    pt = list((src_fldr/fldr).iterdir())[0]
    m.load_state_dict(torch.load(pt))
    m = m.eval()
    if device is not None:
        m = m.to(device)
    return m