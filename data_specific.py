from enum import Enum
import training, data_handling


class Data(Enum):
    Diseases = 'diseases'
    Primary = 'primary'
    
def get_project_name(data:Data):
    if data == Data.Diseases: return 'derm-dis-morph'
    else: return 'derm-primary'
    
def get_class_mode(data:Data):
    if data == Data.Diseases:
        return training.ClassificationMode.Multiclass
    else:
        return training.ClassificationMode.Multilabel
    
def get_ds_create_fn(data:Data):
    if data == Data.Diseases:
        return data_handling.create_disease_datasets
    else:
        return data_handling.create_primary_datasets
    
def get_num_classes(data:Data):
    if data == Data.Diseases:
        return data_handling.NUM_DISEASES
    else:
        return data_handling.NUM_PRIMARY
        
def get_target_metric(data:Data):
    if data == Data.Diseases:
        return 'val/acc'
    else:
        return 'val/f1'