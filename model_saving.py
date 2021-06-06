from datetime import datetime as dt
from pathlib import Path
import torch
import os


def get_save_folder():
    now = dt.now().strftime('%Y%m%d_%H%M%S')
    fldr = Path(f'./logs/weights/run_{now}')
    fldr.mkdir(parents=True, exist_ok=False)
    return fldr

def _save_model(model, pt, main_device):
    model.to(torch.device('cpu'))
    torch.save(model.state_dict(), pt)
    model.to(main_device)

class ModelSaver:
    def __init__(self, save_fldr, device):
        self.save_fldr = save_fldr
        self.device = device
    def update(self, model, metrics):
        raise NotImplementedError
        
class NoSaver(ModelSaver):
    def __init__(self, save_fldr, device):
        super().__init__(save_fldr, device)
    def update(self, model, metrics):
        pass

def _form_path(fldr, acc, epoch):
    return fldr / f'model_{epoch}_acc{acc:.4f}.pth'

class ImprovementModelSaver(ModelSaver):
    def __init__(self, save_fldr, device):
        super().__init__(save_fldr, device)
        self.best_acc = -1
    def update(self, model, metrics):
        cur_acc = metrics['val/acc']
        if curr_acc > self.best_acc:
            self.best_acc = cur_acc
            epoch = metrics['epoch']
            pt = _form_path(self.save_fldr, cur_acc, epoch)
            _save_model(model, pt, self.device)
            
class BestModelSaver(ModelSaver):
    def __init__(self, save_fldr, device):
        super().__init__(save_fldr, device)
        self.best_acc = -1
        self.best_epoch = None
    def update(self, model, metrics):
        cur_acc = metrics['val/acc']
        cur_epoch = metrics['epoch']
        if cur_acc > self.best_acc:
            if self.best_epoch is not None:
                os.remove(self._form_best_path())
            self.best_acc, self.best_epoch = cur_acc, cur_epoch
            _save_model(model, self._form_best_path(), self.device)
    def _form_best_path(self):
        return _form_path(self.save_fldr, self.best_acc, self.best_epoch)