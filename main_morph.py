import argparse
from data_specific import *

# data
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import data_handling
import numpy as np

# training
import model_saving, training, model_creating, metrics
import wandb
import torch, torch.nn as nn
from tqdm.auto import tqdm


def _base_run_name(config):
    def add(s, s1):
        if s is None: return s1
        return f'{s} {s1}'
    s = None
    if config.plateau: s = add(s, 'plateau')
    if config.augm is not None: s = add(s, config.augm)
    if s is None: s = 'baseline'
    return s

def _run_name(config):
    name = _base_run_name(config)
    name += f' bs={config.batch_size}'
    name += f' lr={config.lr}'
    return name

use_primary = [0,2,3,5,6,7]
def _cut_primary(L):
    return [l for l in L if l in use_primary]
def _map_primary(L):
    mapping = {idx: i for (i,idx) in enumerate(use_primary)}
    return [mapping[l] for l in L]
def _fix_primary(L):
    return _map_primary(_cut_primary(L))
def prepare_primary(L):
    L = [(l[0], l[1], l[2], _fix_primary(l[3])) for l in L]
    L = filter(lambda l: len(l[-1]) > 0, L)
    return list(L)
def primary_to_ohe(L):
    r = np.zeros((len(use_primary),))
    for l in L:
        r[l] = 1
    return r

class PrimaryMorphDataset(Dataset):
    def __init__(self, idxs, metadata, img_size=448):
        primary = data_handling._get_morph(idxs, metadata)
        self.idxs = idxs
        self.primary = {k: primary_to_ohe(v) for k,v in primary.items() if k in self.idxs}
        self.imgs = data_handling._load_imgs(idxs, metadata, img_size)
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        idx = self.idxs[i]
        return self.imgs[idx], self.primary[idx]

def main(data, config, tags):
    # wandb
    project_name = get_project_name(data)
    run = wandb.init(project=project_name, config=config, tags=tags)
    run.name = _run_name(run.config)
    
    # config env
    IS_DEMO = run.config.demo
    target_metric = get_target_metric(data)
    device = torch.device(f'cuda:{run.config.device}')
    if run.config.save:
        save_folder = model_saving.get_save_folder()
        run.config.save_folder = save_folder
        model_saver = model_saving.BestModelSaver(save_folder, device, target_metric)
        print(model_saver.save_fldr)
    
    # data loading
    ds_names = ['gsa', 'ulb', 'dermis']
    metadata = data_handling.load_metadata(ds_names)
    meta = {k: prepare_primary(v) for k,v in metadata.items()}
    idxs = data_handling._get_idxs(ds_names, meta)
    kf = KFold(shuffle=True, random_state=0, n_splits=5)
    trn, val = list(kf.split(idxs))[run.config.fold]
    trn, val = [idxs[i] for i in trn], [idxs[i] for i in val]
    trn_ds, val_ds = PrimaryMorphDataset(trn, meta), PrimaryMorphDataset(val, meta)

    bs = run.config.batch_size
    trn_dl = DataLoader(data_handling.AugmDataset(trn_ds, run.config.augm), batch_size=bs, shuffle=run.config.shuffle_train)
    val_dl = DataLoader(data_handling.AugmDataset(val_ds, None), batch_size=bs+30, shuffle=False)

    # Prepare training
    NUM_CLASSES = len(use_primary)
    class_mode = get_class_mode(data)
    if run.config.weighted_loss:
        weights = metrics.create_weights(class_mode, trn_ds, NUM_CLASSES).to(device)
        print('Using loss weights: ', weights)
    else:
        weights = None
    loss_fn = metrics.create_loss_fn(class_mode, weights)
    model = model_creating.create_model(NUM_CLASSES, run.config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=run.config.lr)
    if run.config.plateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, threshold=0.001)

    # Training
    torch.set_num_threads(2)
    best_D = {}
    common_params = {'model': model, 'optimizer': optimizer, 'loss_fn': loss_fn, 'device': device}
    for epoch in tqdm(list(range(run.config.epochs)), desc='Epoch'):
        D = {'epoch': epoch}

        losses, preds, targs = training.step(trn_dl, training.Mode.Train, 'Train', **common_params)
        upD = metrics.compute_metrics(class_mode, 'trn', losses, preds, targs)
        D = metrics.append_dict(D, upD)

        losses, preds, targs = training.step(val_dl, training.Mode.Eval, 'Valid', **common_params)
        upD = metrics.compute_metrics(class_mode, 'val', losses, preds, targs)
        D = metrics.append_dict(D, upD)

        wandb.log(D)
        best_D = metrics.update_dict(best_D, D)
        for k, v in best_D.items(): run.summary[k] = v
        # print('acc=', D[target_metric], 'lr=', optimizer.param_groups[0]['lr'])
            
        if run.config.save: model_saver.update(model, D)
        if run.config.plateau: scheduler.step(D[target_metric])

    run.finish();
    
def _form_parser():
    parser = argparse.ArgumentParser('derm')
    parser.add_argument('--cuda', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--plateau', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--augm', type=str, default=None)
    parser.add_argument('--tags', nargs='*')
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--img_size', type=int, default=448)
    parser.add_argument('--weighted_loss', action='store_true')
    parser.add_argument('--dropout', action='store_true')
    return parser

def _to_config(args):
    return {
        'device': args.cuda,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'epochs': args.epochs,
        'shuffle_train': True,
        'plateau': args.plateau,
        'save': args.save,
        'demo': args.demo,
        'augm': args.augm,
        'fold': args.fold,
        'img_size': args.img_size,
        'weighted_loss': args.weighted_loss,
        'dropout': args.dropout
    }

if __name__ == '__main__':
    parser = _form_parser()
    args = parser.parse_args()
    print(args)
    config = _to_config(args)
    data = Data.Primary
    print(data, config)
    main(data, config, args.tags)
