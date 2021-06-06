import pickle
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from tqdm.auto import tqdm
import torchvision.transforms as TF
from torch.utils.data import Dataset, DataLoader
import torch


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
    imgs = {ds_name: _load_ds_imgs(ds_name, metadata[ds_name]) for ds_name in ds_names}
    imgs = {(ds, i): imgs[ds][i] for (ds, i) in idxs}
    return idxs, imgs, diseases


_normalize = TF.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
_to_tensor = TF.ToTensor()

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
        if self.augm is not None: img = self.augm(image=img)['image']
        img = _to_tensor(img)
        img = _normalize(img)
        return img, lbl

def show_tensor(img):
    img = img * torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)
    img = img.detach().cpu().numpy()
    img = np.moveaxis(img, 0, -1)
    img = np.uint8(img * 255)
    return Image.fromarray(img)