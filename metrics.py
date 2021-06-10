from enum import Enum
import sklearn.metrics
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import math


class ClassificationMode(Enum):
    Multiclass = 'multiclass'
    Multilabel = 'multilabel'
    
def get_post_tfm(class_mode:ClassificationMode):
    if class_mode == ClassificationMode.Multiclass:
        return nn.Softmax(dim=1)
    else:
        return nn.Sigmoid()
    
def create_loss_fn(class_mode):
    if class_mode == ClassificationMode.Multiclass:
        return nn.CrossEntropyLoss()
    else:
        return nn.BCEWithLogitsLoss()
    
def _mean(L:list): return sum(L)/len(L)

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

