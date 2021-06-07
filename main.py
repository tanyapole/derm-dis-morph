import argparse
from data_specific import *

# data
from torch.utils.data import DataLoader
import data_handling
import augmenting as A

# training
import model_saving, training, model_creating
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

def main(data, config):
    # wandb
    project_name = get_project_name(data)
    run = wandb.init(project=project_name, config=config)
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
    trn_augm = A.get_augm(run.config.augm)
    ds_create_fn = get_ds_create_fn(data)
    trn_ds, val_ds = ds_create_fn(trn_augm, IS_DEMO)

    bs = run.config.batch_size
    trn_dl = DataLoader(trn_ds, batch_size=bs, shuffle=run.config.shuffle_train)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False)

    # Prepare training
    NUM_CLASSES = get_num_classes(data)
    class_mode = get_class_mode(data)
    loss_fn = training.create_loss_fn(class_mode)
    model = model_creating.create_model(NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=run.config.lr)
    if run.config.plateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, threshold=0.001)

    # Training
    torch.set_num_threads(2)
    common_params = {'model': model, 'optimizer': optimizer, 'loss_fn': loss_fn, 'device': device}
    for epoch in tqdm(list(range(run.config.epochs)), desc='Epoch'):
        D = {'epoch': epoch}

        losses, preds, targs = training.step(trn_dl, training.Mode.Train, 'Train', **common_params)
        metrics = training.compute_metrics(class_mode, 'trn', losses, preds, targs)
        D = training.append_dict(D, metrics)

        losses, preds, targs = training.step(val_dl, training.Mode.Eval, 'Valid', **common_params)
        metrics = training.compute_metrics(class_mode, 'val', losses, preds, targs)
        D = training.append_dict(D, metrics)

        wandb.log(D)
        if run.config.save: model_saver.update(model, D)
        if run.config.plateau: scheduler.step(D[target_metric])
        # print('acc=', D[target_metric], 'lr=', optimizer.param_groups[0]['lr'])

    run.finish();
    
def _form_parser():
    parser = argparse.ArgumentParser('derm')
    parser.add_argument('--data', choices=[e.value for e in Data], required=True)
    parser.add_argument('--cuda', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--shuffle_train', action='store_true')
    parser.add_argument('--plateau', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--augm', type=str, default=None)
    return parser

def _to_config(args):
    return {
        'device': args.cuda,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'epochs': args.epochs,
        'shuffle_train': args.shuffle_train,
        'plateau': args.plateau,
        'save': args.save,
        'demo': args.demo,
        'augm': args.augm
    }

def _to_data(args):
    for e in Data:
        if e.value == args.data:
            return e
    raise Exception(f'Unsupported data type {data}')
    
if __name__ == '__main__':
    parser = _form_parser()
    args = parser.parse_args()
    config = _to_config(args)
    data = _to_data(args)
    print(data, config)
    main(data, config)
