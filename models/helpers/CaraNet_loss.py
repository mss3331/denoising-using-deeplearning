import torch
import torch.nn.functional as F

def structure_loss(pred, mask):

    weit = 1 + 5* torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

def CaraNet_structure_loss (special_outputs, gts):
    lateral_map_5, lateral_map_3, lateral_map_2, lateral_map_1 = special_outputs
    # ---- loss function ----
    loss5 = structure_loss(lateral_map_5, gts)
    loss3 = structure_loss(lateral_map_3, gts)
    loss2 = structure_loss(lateral_map_2, gts)
    loss1 = structure_loss(lateral_map_1, gts)

    loss = loss5 + loss3 + loss2 + loss1
    return loss