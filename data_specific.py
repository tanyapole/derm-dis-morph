from enum import Enum
import data_handling, metrics


class Data(Enum):
    Diseases = 'diseases'
    Primary = 'primary'
    
def get_project_name(data:Data):
    if data == Data.Diseases: return 'derm-dis-morph'
    else: return 'derm-primary'
    
def get_class_mode(data:Data):
    if data == Data.Diseases:
        return metrics.ClassificationMode.Multiclass
    else:
        return metrics.ClassificationMode.Multilabel
    
def get_ds_class(data:Data):
    if data == Data.Diseases:
        return data_handling.DiseaseDataset
    else:
        return data_handling.PrimaryMorphDataset
    
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