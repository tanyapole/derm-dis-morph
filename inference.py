import torch.nn as nn
import data_handling
import torch
from torch.utils.data import DataLoader, Dataset
import augmenting as A
import metrics
from tqdm.auto import tqdm
import numpy as np
import uncertainty as U


class OneImageDs(Dataset):
    def __init__(self, img):
        self.img = img
    def __len__(self): return 1
    def __getitem__(self, i):
        return data_handling._prepare_img(self.img, None)
    
class TTA_OneImageDs(Dataset):
    def __init__(self, img, augm, N):
        self.img = img
        self.augm = augm
        self.N = N
    def __len__(self): return self.N
    def __getitem__(self, i):
        return data_handling._prepare_img(self.img, self.augm)
    
def _np(t): return t.detach().cpu().numpy()

def predict(device, dl, model, class_mode):
    model.eval()
    post_tfm = metrics.get_post_tfm(class_mode)
    all_probs, all_lbls = [], []
    for b in tqdm(dl, leave=False, desc='Inference'):
        imgs, lbls = b[0].to(device), b[1].to(device)
        with torch.no_grad():
            probs = post_tfm(model(imgs))
        all_probs.append(_np(probs))
        all_lbls.append(_np(lbls))
    all_probs = np.concatenate(all_probs, axis=0)
    all_lbls = np.concatenate(all_lbls, axis=0)
    return all_probs, all_lbls

def _check_lbls(all_lbls):
    lbl0 = all_lbls[0]
    for lbl in all_lbls:
        assert (lbl == lbl0).all().item(), "labels don't match"
        
def predict_TTA(device, dl, model, class_mode, num_augm):
    all_probs, all_lbls = [], []
    for t in tqdm(list(range(num_augm)), leave=False, desc='Augm'):
        probs, lbls = predict(device, dl, model, class_mode)
        all_probs.append(probs)
        all_lbls.append(lbls)
    _check_lbls(all_lbls)
    return all_probs, all_lbls

def predict_ensemble(device, dl, models, class_mode):
    all_probs, all_lbls = [], []
    for m in tqdm(models, leave=False, desc='Model'):
        probs, lbls = predict(device, dl, m, class_mode)
        all_probs.append(probs)
        all_lbls.append(lbls)
    _check_lbls(all_lbls)
    return all_probs, all_lbls

def predict_ensemble_TTA(device, dl, models, class_mode, num_augm):
    all_probs, all_lbls = [], []
    for m in tqdm(models, leave=False, desc='Model'):
        probs, lbls = predict_TTA(device, dl, m, class_mode, num_augm)
        all_probs.append(U.get_expected(probs))
        all_lbls.append(lbls[0])
    _check_lbls(all_lbls)
    return all_probs, all_lbls

def _predict(device, dl, model, class_mode):
    model.eval()
    with torch.no_grad():
        probs = model(next(iter(dl)).to(device))
        probs = metrics.get_post_tfm(class_mode)(probs)
    return probs

def one_vanilla_predict(device, img, model, class_mode):
    dl = DataLoader(OneImageDs(img), batch_size=1, shuffle=False)
    return _np(_predict(device, dl, model, class_mode))

def one_TTA_predict(device, img, model, class_mode, augm, num_samples):
    augm = A.get_augm(augm)
    ds = TTA_OneImageDs(img, augm, num_samples)
    dl = DataLoader(ds, batch_size=num_samples, shuffle=False)
    return _np(_predict(device, dl, model, class_mode))

def one_ensemble_predict(device, img, models, class_mode):
    dl = DataLoader(OneImageDs(img), batch_size=1, shuffle=False)
    probs = [_predict(device, dl, model, class_mode) for model in models]
    return _np(torch.cat(probs, dim=0))