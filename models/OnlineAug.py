import torchvision.transforms.functional as TF
import torch

def augmentation(X, mask):
    randomNumbers_arr = torch.rand((7,7)).numpy()
    X_aug=apply_randomAug(X,randomNumbers_arr)
    mask_aug=apply_randomAug(mask,randomNumbers_arr, is_mask=True)
    return X_aug, mask_aug

def apply_randomAug(X, randomNumbers_arr, is_mask=False):
    X_aug = None
    # Random variables
    angle = 260 * (randomNumbers_arr[0, 0] - 0.5)  # -180 to 180
    shear = 260 * (randomNumbers_arr[1, 0] - 0.5)  # -180 to 180
    hflip = randomNumbers_arr[2, 0]  # 0 to 1
    vflip = randomNumbers_arr[3, 0]  # 0 to 1
    brightness = randomNumbers_arr[4, 0] + 0.5  # 0.5 to 1.5
    hue = randomNumbers_arr[5, 0] / 2 - 0.25  # -0.25 to 0.25

    # Transformations
    X_aug = TF.affine(X, angle=angle, shear=shear, translate=[0,0],  scale=1)
    if hflip >= 0.5:
        X_aug = TF.hflip(X_aug)
    if vflip >= 0.5:
        X_aug = TF.vflip(X_aug)

    if not is_mask:
        X_aug = TF.adjust_brightness(X_aug, brightness_factor=brightness)
        X_aug = TF.adjust_hue(X_aug, hue_factor=hue)
    return X_aug