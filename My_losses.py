import torch
import numpy as np
from torch import nn

def gradMaskLoss_Eq1(images,mask,loss_fn):
    r'''‖∇g∙mask(g)‖^2 == MSELoss(∇g∙mask(g),torch.zeros(∇g.shape)'''
    images_grad=image_gradient(images, reduction=None)
    masked_grad = torch.mul(images_grad,mask)
    return loss_fn(masked_grad,torch.zeros(masked_grad.shape))


def image_gradient(images,reduction='mean'):
    device = torch.device('cuda:0')
    a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight = nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
    # print(conv1.weight)
    conv1.to(device)
    # -----------------------------------------------------------
    b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight = nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
    conv2.to(device)
    # -----------------------------------------
    # images.shape = [batch, C, H, W]
    images_shape = images.shape
    # images.reshape = [batch*C, 1, H, W]
    images = images.view(-1, 1, *images_shape[-2:])

    G_x = conv1(images)
    G_y = conv2(images)
    G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2)+0.000000000000001)
    if reduction=='mean':
        grad_loss = torch.sum(G) / (images_shape[0] * images_shape[1] * images_shape[2] * images_shape[3])
    elif reduction=='sum':
        grad_loss = torch.sum(G)
    else:
        grad_loss=G.view(*images_shape)
    return grad_loss







def denoising_loss(created_images, original_images):
    alpha = torch.sum(torch.pow(created_images - original_images, 2)) / (
            original_images.shape[-1] * original_images.shape[-2])
    beta = 0.1 * image_gradient(created_images)
    if alpha > 1000 or beta > 100:
        print((original_images.shape[-1] * original_images.shape[-2]))

    total = beta  # alpha + beta
    print(alpha, beta)
    return total
