import torch.nn as nn
import data_handling
import torch
from torch.utils.data import DataLoader, Dataset
import augmenting as A
import metrics


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

def _predict(device, dl, model, class_mode):
    with torch.no_grad():
        probs = model(next(iter(dl)).to(device))
        probs = metrics.get_post_tfm(class_mode)(probs)
    return probs

def vanilla_predict(device, img, model, class_mode):
    dl = DataLoader(OneImageDs(img), batch_size=1, shuffle=False)
    return _np(_predict(device, dl, model, class_mode))

def TTA_predict(device, img, model, class_mode, augm, num_samples):
    augm = A.get_augm(augm)
    ds = TTA_OneImageDs(img, augm, num_samples)
    dl = DataLoader(ds, batch_size=num_samples, shuffle=False)
    return _np(_predict(device, dl, model, class_mode))

def ensemble_predict(device, img, models, class_mode):
    dl = DataLoader(OneImageDs(img), batch_size=1, shuffle=False)
    probs = [_predict(device, dl, model, class_mode) for model in models]
    return _np(torch.cat(probs, dim=0))