import torch
import torch.nn.functional as F


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def pranet_structure_loss(special_outputs, gts):
    loss5 = structure_loss(special_outputs[0], gts)
    loss4 = structure_loss(special_outputs[1], gts)
    loss3 = structure_loss(special_outputs[2], gts)
    loss2 = structure_loss(special_outputs[3], gts)
    loss = loss2 + loss3 + loss4 + loss5
    return loss