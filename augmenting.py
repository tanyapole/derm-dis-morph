import albumentations as A


def _get_augm_by_name(name):
    if name == 'hflip': return A.HorizontalFlip()
    raise Exception(f'Unsupported augmentation: {name}')
    
def get_augm(augm_str):
    if augm_str is None: return None
    augm_names = augm_str.split(',')
    return A.Compose([_get_augm_by_name(name) for name in augm_names])
    