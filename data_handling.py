import pickle
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from tqdm.auto import tqdm
import torchvision.transforms as TF
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split


_lbl_folder = Path('_data_lbls')

def _read_pickle(fname):
    with open(_lbl_folder/f'{fname}.pkl', 'rb') as f: 
        _ = pickle.load(f)
    return _

def _load_img(pt): 
    img = np.array(Image.open(str(pt)))
    img = cv2.resize(img, (224,224))
    return img


NUM_DISEASES = len(_read_pickle('mapping_diseases'))
NUM_MORPH = len(_read_pickle('mapping_morph'))
NUM_PRIMARY = 8

def to_primary(morph):
    primary = {}
    for k, v in morph.items():
        primary[k] = [_ for _ in v if _ < NUM_PRIMARY]
    return primary

def get_ds_names(is_demo):
    if is_demo: return ['gsa']
    else: return ['gsa', 'ulb', 'atlas_derm', 'hellenic', 'dermnetnz']
    
def _load_ds_metadata(ds_name):
    return _read_pickle(ds_name)

def _load_ds_imgs(ds_name, ds_metadata):
    return [_load_img(el[1]) for el in tqdm(ds_metadata, desc=ds_name)]

def load_data(ds_names):
    metadata = {ds_name: _load_ds_metadata(ds_name) for ds_name in ds_names}
    idxs = [(ds_name, i) for ds_name in ds_names for i in range(len(metadata[ds_name]))]
    print(f'Hadling {len(idxs)} data instances')
    diseases = {(ds_name, i): metadata[ds_name][i][2] for (ds_name, i) in idxs}
    morph = {(ds_name, i): metadata[ds_name][i][3] for (ds_name, i) in idxs}
    imgs = {ds_name: _load_ds_imgs(ds_name, metadata[ds_name]) for ds_name in ds_names}
    imgs = {(ds, i): imgs[ds][i] for (ds, i) in idxs}
    return idxs, imgs, diseases, morph


_normalize = TF.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
_to_tensor = TF.ToTensor()

def _prepare_img(img, augm):
    if augm is not None:
        img = augm(images=[img])[0]
    img = _to_tensor(img)
    img = _normalize(img)
    return img

class DiseaseDataset(Dataset):
    def __init__(self, idxs, imgs, diseases, augm=None):
        self.idxs = idxs
        self.diseases = diseases
        self.imgs = imgs
        self.augm = augm
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        idx = self.idxs[i]
        img, lbl = self.imgs[idx], self.diseases[idx]
        img = _prepare_img(img, self.augm)
        return img, lbl

class PrimaryMorphDataset(Dataset):
    def __init__(self, idxs, imgs, primary, augm=None):
        self.idxs = idxs
        self.primary = primary
        self.imgs = imgs
        self.augm = augm
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        idx = self.idxs[i]
        img = _prepare_img(self.imgs[idx], self.augm)
        lbl = torch.zeros((NUM_PRIMARY,))
        for i in self.primary[idx]: lbl[i] = 1
        return img, lbl
    
def create_disease_datasets(trn_augm, demo:bool):
    IS_DEMO = demo
    datasets = get_ds_names(IS_DEMO)
    print('Using datasets: ', ', '.join(datasets))

    idxs, imgs, diseases, _ = load_data(datasets)

    trn_idxs, val_idxs = train_test_split(idxs, train_size=0.8, random_state=0, stratify=[diseases[idx] for idx in idxs])

    trn_ds = DiseaseDataset(trn_idxs, imgs, diseases, trn_augm)
    val_ds = DiseaseDataset(val_idxs, imgs, diseases)
    return trn_ds, val_ds

def create_primary_datasets(trn_augm, demo:bool):
    IS_DEMO = demo
    datasets = get_ds_names(IS_DEMO)
    print('Using datasets: ', ', '.join(datasets))

    idxs, imgs, diseases, morph = load_data(datasets)
    primary = to_primary(morph)

    trn_idxs, val_idxs = train_test_split(idxs, train_size=0.8, random_state=0, stratify=[diseases[idx] for idx in idxs])
    trn_idxs = [idx for idx in trn_idxs if len(primary[idx]) > 0]
    val_idxs = [idx for idx in val_idxs if len(primary[idx]) > 0]
    
    trn_ds = PrimaryMorphDataset(trn_idxs, imgs, primary, trn_augm)
    val_ds = PrimaryMorphDataset(val_idxs, imgs, primary)
    return trn_ds, val_ds

def show_tensor(img):
    img = img * torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)
    img = img.detach().cpu().numpy()
    img = np.moveaxis(img, 0, -1)
    img = np.uint8(img * 255)
    return Image.fromarray(img)