from enum import Enum
import sklearn.metrics
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import math


def _mean(L:list): return sum(L)/len(L)

class Mode(Enum):
    Train = 'train'
    Eval = 'eval'
    
class ClassificationMode(Enum):
    Multiclass = 'multiclass'
    Multilabel = 'multilabel'
    
def create_loss_fn(class_mode):
    if class_mode == ClassificationMode.Multiclass:
        return nn.CrossEntropyLoss()
    else:
        return nn.BCEWithLogitsLoss()
    
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

def compute_metrics(class_mode, phase, losses, preds, targs):
    if class_mode == ClassificationMode.Multiclass:
        preds = np.argmax(preds, axis=1)
        return {f'{phase}/loss': _mean(losses),
             f'{phase}/acc': sklearn.metrics.accuracy_score(targs, preds),
             f'{phase}/f1': sklearn.metrics.f1_score(targs, preds, average='macro')}
    else:
        preds = (preds > 0) * 1
        return {f'{phase}/loss': _mean(losses),
             f'{phase}/f1': sklearn.metrics.f1_score(targs, preds, average='macro')}

def append_dict(D1, D2):
    if len(set(D1.keys()).intersection(set(D2.keys()))) > 0:
        raise Exception("common keys")
    D = {}
    for k,v in D1.items(): D[k] = v
    for k,v in D2.items(): D[k] = v
    return D

def _min(v1, v2):
    if v1 is None or math.isnan(v1): return v2
    return min(v1, v2)

def _max(v1, v2):
    if v1 is None or math.isnan(v1): return v2
    return max(v1, v2)

def update_dict(best_D, cur_D):
    new_best_D = {}
    for k, v in cur_D.items():
        if not k in best_D: new_best_D[k] = v
        else:
            if 'loss' in k:
                new_best_D[k] = _min(best_D[k], v)
            else:
                new_best_D[k] = _max(best_D[k], v)
    return new_best_D