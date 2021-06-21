from enum import Enum
import sklearn.metrics
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import math

from torchvision.models import resnet34
import torch, torch.nn as nn
from pathlib import Path

import metrics
from data_specific import *

def compute_metrics(phase, res):
    d_losses, m_losses, d_preds, d_targs, m_preds, m_targs = res
    D = metrics.compute_metrics(get_class_mode(Data.Diseases), f'dis/{phase}', d_losses, d_preds, d_targs)
    M = metrics.compute_metrics(get_class_mode(Data.Primary), f'morph/{phase}', m_losses, m_preds, m_targs)
    return metrics.append_dict(D, M)

class TwoHeadedModel(nn.Module):
    def __init__(self, num1, num2):
        super(TwoHeadedModel, self).__init__()
        model = resnet34(pretrained=True)
        self.base_model = nn.Sequential(*list(model.children())[:-1])
        self.head1 = nn.Linear(model.fc.in_features, num1, bias=False)
        self.head2 = nn.Linear(model.fc.in_features, num2, bias=False)
    def forward(self, x):
        x = self.base_model(x)
        x = torch.flatten(x, 1)
        x1 = self.head1(x)
        x2 = self.head2(x)
        return x1, x2
    
def create_2_headed_model(num_classes_1, num_classes_2):
    return TwoHeadedModel(num_classes_1, num_classes_2)


class Mode(Enum):
    Train = 'train'
    Eval = 'eval'

    
def _np(t): return t.detach().cpu().numpy()
def _it(t): return _np(t).item()


def step(dl, mode, phase, model, optimizer, d_loss_fn, m_loss_fn, device):
    if mode == Mode.Train: model.train()
    else: model.eval()
    
    d_losses, d_preds, d_targs = [], [], []
    m_losses, m_preds, m_targs = [], [], []
    
    for b in tqdm(dl, leave=False, desc=phase):
        _b = [_.to(device) for _ in b]
        imgs, d_lbls, m_lbls = _b[0], _b[1], _b[2]

        if mode == Mode.Train: 
            optimizer.zero_grad()
        with torch.set_grad_enabled(mode == Mode.Train):
            d_pred, m_pred = model(imgs)
            # diseases
            d_loss = d_loss_fn(d_pred, d_lbls)
            d_losses.append(_it(d_loss))
            d_targs.append(_np(d_lbls))
            d_preds.append(_np(d_pred))
            loss = d_loss
            # morphology
            idxs = m_lbls.sum(dim=-1) > 0
            if idxs.any().item():
                m_pred, m_lbls = m_pred[idxs], m_lbls[idxs]
                m_loss = m_loss_fn(m_pred, m_lbls)
                m_losses.append(_it(m_loss))
                m_targs.append(_np(m_lbls))
                m_preds.append(_np(m_pred))
                loss += m_loss
        if mode == Mode.Train:
            loss.backward()
            optimizer.step()
    
    def _c(L): return np.concatenate(L, axis=0)
    d_preds, d_targs, m_preds, m_targs = _c(d_preds), _c(d_targs), _c(m_preds), _c(m_targs)
    return d_losses, m_losses, d_preds, d_targs, m_preds, m_targs

def _counts(ds, num_classes):
    weights = torch.zeros((num_classes,))
    for img,lbl,_ in ds:
        weights[lbl] += 1
    return weights
  
def create_weights(ds, num_classes):
    weights = _counts(ds, num_classes)
    return (len(ds) - weights) / weights