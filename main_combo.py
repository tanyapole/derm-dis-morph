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

# combo
import combo

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

def main(config, tags):
    # wandb
    project_name = "derm-combined"
    run = wandb.init(project=project_name, config=config, tags=tags)
    run.name = _run_name(run.config)

    # config env
    IS_DEMO = run.config.demo
    target_metric = 'dis/' + get_target_metric(Data.Diseases)
    device = torch.device(f'cuda:{run.config.device}')
    if run.config.save:
        save_folder = model_saving.get_save_folder()
        run.config.save_folder = save_folder
        model_saver = model_saving.BestModelSaver(save_folder, device, target_metric)
        print(model_saver.save_fldr)

    # data loading
    ds_class = data_handling.CombinedDataset
    if run.config.datasets is None:
        ds_names = data_handling.get_ds_names(allowed=True, is_demo=IS_DEMO)
    else:
        ds_names = run.config.datasets.split(",")
    trn_ds, val_ds = data_handling.create_trn_val_ds(
        ds_names, 
        ds_class,
        run.config.img_size,
        run.config.fold,
    )

    bs = run.config.batch_size
    trn_dl = DataLoader(data_handling.AugmDataset(trn_ds, run.config.augm), batch_size=bs, shuffle=run.config.shuffle_train)
    val_dl = DataLoader(data_handling.AugmDataset(val_ds, None), batch_size=bs, shuffle=False)



    model = combo.create_2_headed_model(get_num_classes(Data.Diseases), get_num_classes(Data.Primary)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=run.config.lr)

    if run.config.weighted_loss:
        weights = combo.create_weights(trn_ds, get_num_classes(Data.Diseases)).to(device)
        print('Using loss weights: ', weights)
    else:
        weights = None
    dis_loss_fn = metrics.create_loss_fn(get_class_mode(Data.Diseases), weights)
    morph_loss_fn = metrics.create_loss_fn(get_class_mode(Data.Primary))

    # Training
    torch.set_num_threads(2)
    best_D = {}
    common_params = {'model': model, 'optimizer': optimizer, 'd_loss_fn': dis_loss_fn, 'm_loss_fn': morph_loss_fn, 'device': device}

    for epoch in tqdm(list(range(run.config.epochs)), desc='Epoch'):
        D = {'epoch': epoch}

        res = combo.step(trn_dl, training.Mode.Train, 'Train', **common_params)
        upD = combo.compute_metrics('trn', res)
        D = metrics.append_dict(D, upD)

        res = combo.step(val_dl, training.Mode.Eval, 'Valid', **common_params)
        upD = combo.compute_metrics('val', res)
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
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--img_size', type=int, default=448)
    parser.add_argument('--datasets', nargs='*', required=False)
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
        'datasets': None if args.datasets is None else ",".join(args.datasets),
        'weighted_loss': args.weighted_loss,
        'dropout': args.dropout
    }


if __name__ == '__main__':
    parser = _form_parser()
    args = parser.parse_args()
    config = _to_config(args)
    print(config)
    main(config, args.tags)
