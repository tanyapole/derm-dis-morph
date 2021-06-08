from enum import Enum
import sklearn.metrics
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import math


class Mode(Enum):
    Train = 'train'
    Eval = 'eval'

    
def _np(t): return t.detach().cpu().numpy()
def _it(t): return _np(t).item()


def step(dl, mode, phase, model, optimizer, loss_fn, device):
    if mode == Mode.Train: model.train()
    else: model.eval()
    
    losses, preds, targs = [], [], []
    for b in tqdm(dl, leave=False, desc=phase):
        imgs, lbls = b[0].to(device), b[1].to(device)
        if mode == Mode.Train:
            optimizer.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, lbls)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                out = model(imgs)
                loss = loss_fn(out, lbls)
        losses.append(_it(loss))
        preds.append(_np(out))
        targs.append(_np(lbls))
    preds = np.concatenate(preds, axis=0)
    targs = np.concatenate(targs, axis=0)
    return losses, preds, targs

