import argparse
from data_specific import *

# data
from torch.utils.data import DataLoader
import data_handling

# training
import model_saving, training, model_creating, metrics
import wandb
import torch, torch.nn as nn
from tqdm.auto import tqdm

def get_remaining_datasets(trn_datasets):
    ds = data_handling.get_ds_names(True,False) + data_handling.get_ds_names(False,False)
    ds = set(ds).difference(set(trn_datasets))
    return sorted(ds)

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
    trn_ds_names = run.config.datasets.split(",")
    val_ds_names = get_remaining_datasets(trn_ds_names)
    if run.config.demo:
        trn_ds_names = trn_ds_names[0:1]
        val_ds_names = val_ds_names[0:1]
    print('Using for training: ', ','.join(trn_ds_names))
    print('Using for validation: ', ','.join(val_ds_names))
    trn_ds = data_handling.create_total_ds(trn_ds_names, get_ds_class(data), run.config.img_size)
    val_ds = data_handling.create_total_ds(val_ds_names, get_ds_class(data), run.config.img_size)        

    bs = run.config.batch_size
    trn_dl = DataLoader(data_handling.AugmDataset(trn_ds, run.config.augm), batch_size=bs, shuffle=run.config.shuffle_train)
    val_dl = DataLoader(data_handling.AugmDataset(val_ds, None), batch_size=bs+30, shuffle=False)

    # Prepare training
    NUM_CLASSES = get_num_classes(data)
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
    parser.add_argument('--data', choices=[e.value for e in Data], required=True)
    parser.add_argument('--cuda', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--plateau', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--augm', type=str, default=None)
    parser.add_argument('--tags', nargs='*')
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--img_size', type=int, default=448)
    parser.add_argument('--datasets', nargs='+', required=True)
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
        'datasets': ",".join(args.datasets),
        'weighted_loss': args.weighted_loss,
        'dropout': args.dropout
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
    main(data, config, args.tags)
