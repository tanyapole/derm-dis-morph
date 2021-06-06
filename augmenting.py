import albumentations as A
import imgaug as ia
import imgaug.augmenters as iaa



def _get_augm_by_name(name):
    if name == 'hflip': return iaa.Fliplr(p=0.5)#A.HorizontalFlip()
    elif name == 'salt_pepper': return iaa.SaltAndPepper(0.1, fill_per_channel=True)
    elif name == 'cutout': return iaa.Cutout(fill_mode="gaussian", fill_per_channel=True)
    elif name == 'gauss_noise': return iaa.AdditiveGaussianNoise(scale=(0, 0.2*255))
    elif name == 'gaussblur': return iaa.GaussianBlur(sigma=(0.0, 3.0))
    elif name == 'saturation': return iaa.MultiplySaturation((0.5, 1.5))
    elif name == 'brightness': return iaa.MultiplyBrightness((0.5, 1.5))
    elif name == 'perspective': return iaa.PerspectiveTransform(scale=(0.01, 0.15))
    raise Exception(f'Unsupported augmentation: {name}')
    
def get_augm(augm_str):
    if augm_str is None: return None
    augm_names = augm_str.split(',')
    return iaa.Sequential([_get_augm_by_name(name) for name in augm_names])



    