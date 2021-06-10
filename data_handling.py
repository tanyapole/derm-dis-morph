import pickle
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from tqdm.auto import tqdm
import torchvision.transforms as TF
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
import augmenting as A

_lbl_folder = Path('_data_lbls')

def _read_pickle(fname):
    with open(_lbl_folder/f'{fname}.pkl', 'rb') as f: 
        _ = pickle.load(f)
    return _

def _load_img(pt, size=224): 
    img = np.array(Image.open(str(pt)))
    img = cv2.resize(img, (size,size))
    return img

NUM_DISEASES = len(_read_pickle('mapping_diseases'))
NUM_MORPH = len(_read_pickle('mapping_morph'))
NUM_PRIMARY = 8

def get_ds_names(allowed, is_demo):
    if allowed:
        if is_demo: return ['gsa']
        else: return ['gsa', 'ulb', 'atlas_derm', 'hellenic', 'dermnetnz']
    else:
        if is_demo: return ['dermis']
        else: return ['chicago', 'dermnet', 'dermis', 'iowa']

def _load_ds_metadata(ds_name):
    return _read_pickle(ds_name)

def load_metadata(ds_names):
    return {name: _load_ds_metadata(name) for name in ds_names}

def _to_primary(morph):
    primary = {}
    for k, v in morph.items():
        primary[k] = [_ for _ in v if _ < NUM_PRIMARY]
    return primary

def _get_idxs(ds_names, metadata):
    return [(ds_name, i) for ds_name in ds_names for i in range(len(metadata[ds_name]))]
def _get_diseases(idxs, metadata):
    return {(ds_name, i): metadata[ds_name][i][2] for (ds_name, i) in idxs}
def _get_morph(idxs, metadata):
    return {(ds_name, i): metadata[ds_name][i][3] for (ds_name, i) in idxs}
def _get_img_paths(idxs, metadata):
    return {(ds_name, i): metadata[ds_name][i][1] for (ds_name, i) in idxs}
def _load_imgs(idxs, metadata, size=224):
    img_paths = _get_img_paths(idxs, metadata)
    return {idx: _load_img(img_paths[idx], size) for idx in tqdm(idxs)}  

def _get_primary_morph(idxs, metadata):
    return _to_primary(_get_morph(idxs, metadata))

def get_no_split(ds_names, metadata):
    return _get_idxs(ds_names, metadata)

def get_split(ds_names, metadata, fold):
    idxs = _get_idxs(ds_names, metadata)
    diseases = _get_diseases(idxs, metadata)
    diseases = [diseases[idx] for idx in idxs]
    
    if fold is None:
        return train_test_split(idxs, train_size=0.8, random_state=0, stratify=diseases)
    
    skf = StratifiedKFold(random_state=0, shuffle=True)
    splits =list(skf.split(idxs, diseases))    
    trn, val = splits[fold]
    return [idxs[i] for i in trn], [idxs[i] for i in val]

_normalize = TF.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
_to_tensor = TF.ToTensor()

def _prepare_img(img, augm):
    if augm is not None:
        img = augm(images=[img])[0]
    img = _to_tensor(img)
    img = _normalize(img)
    return img

def primary_to_ohe(primary):
    ohe = torch.zeros((NUM_PRIMARY,))
    for i in primary: ohe[i] = 1
    return ohe

class DiseaseDataset(Dataset):
    def __init__(self, idxs, metadata, img_size, augm=None):
        self.idxs = idxs
        self.diseases = _get_diseases(idxs, metadata)
        self.imgs = _load_imgs(idxs, metadata, img_size)
        self.augm = A.get_augm(augm)
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        idx = self.idxs[i]
        img, lbl = self.imgs[idx], self.diseases[idx]
        img = _prepare_img(img, self.augm)
        return img, lbl
    
class PrimaryMorphDataset(Dataset):
    def __init__(self, idxs, metadata, img_size, augm=None):
        primary = _get_primary_morph(idxs, metadata)
        idxs = [idx for idx in idxs if len(primary[idx]) > 0]
        self.idxs = idxs
        self.primary = {k: primary_to_ohe(v) for k,v in primary.items() if k in self.idxs}
        self.imgs = _load_imgs(idxs, metadata, img_size)
        self.augm = A.get_augm(augm)
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        idx = self.idxs[i]
        img, lbl = self.imgs[idx], self.primary[idx]
        img = _prepare_img(img, self.augm)
        return img, lbl

def create_trn_val_ds(ds_names, ds_class, img_size, fold, trn_augm, val_augm):
    metadata = load_metadata(ds_names)
    trn, val = get_split(ds_names, metadata, fold=fold)
    trn_ds = ds_class(trn, metadata, img_size, trn_augm)
    val_ds = ds_class(val, metadata, img_size, val_augm)
    return trn_ds, val_ds
def create_val_ds(ds_names, ds_class, img_size, fold, augm):
    metadata = load_metadata(ds_names)
    trn, val = get_split(ds_names, metadata, fold=fold)
    val_ds = ds_class(val, metadata, img_size, augm)
    return val_ds
def create_total_ds(ds_names, ds_class, img_size, augm):
    metadata = load_metadata(ds_names)
    total = get_no_split(ds_names, metadata)
    return ds_class(total, metadata, img_size, augm)

def show_tensor(img):
    img = img * torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)
    img = img.detach().cpu().numpy()
    img = np.moveaxis(img, 0, -1)
    img = np.uint8(img * 255)
    return Image.fromarray(img)