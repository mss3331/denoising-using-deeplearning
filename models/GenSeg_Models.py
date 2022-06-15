import torch
from models.DeepLabModels import Deeplabv3
from torch import nn
from models import MyModelV1, FCNModels, DeepLabModels, unet

'''This model should handle the following:
1- Load pretrained models
2- Flexibility at selecting the Generator and the Segmentor architectures
3- Flexibility at handling the original images and generated images with respect to the final mask
(i.e., type of voting mechanism for generating masks for train/val/test phases).
 The signature of forward: forward(X, phase)-> (generated_images, mask)'''
def getModel(model_name='unet', in_channels=3, out_channels=2):
    if model_name=='deeplab':
        model = Deeplabv3(num_classes=out_channels)
    elif model_name == 'unet':
        model = unet.UNet(in_channels=in_channels,
              out_channels=out_channels,
              n_blocks=4,
              activation='relu',
              normalization='batch',
              conv_mode='same',
              dim=2)
    else:
        print('unknnown model for the Gen Seg models')
        exit(-1)

    return model

def catOrSplit(tensor_s):
    if isinstance(tensor_s,list):#if list, means we need to concat
        return torch.cat(tensor_s,dim=0)
    else: # or split
        return tensor_s.chunk(chunks=2)

class GenSeg_IncludeX(nn.Module):
    '''
    This is the base class for GenSeg_IncludeX class. Now we will create subclasses
    '''

    def __init__(self, Gen_Seg_arch=('unet','unet')):
        super().__init__()
        base = getModel(Gen_Seg_arch[0],out_channels=3)
        self.Generator = nn.Sequential(base, nn.Sigmoid())
        self.Segmentor = getModel(Gen_Seg_arch[1])

    def forward(self,X):
        generated_images = self.Generator(X)
        generated_images_clone = generated_images.clone().detach()
        X_and_generated_images = catOrSplit([X, generated_images_clone])
        masks = self.Segmentor(X_and_generated_images)
        return generated_images, masks

class GenSeg_IncludeX_max(nn.Module):
    def __init__(self, Gen_Seg_arch=('unet','unet')):
        super().__init__()
        self.baseGenSeg_model = GenSeg_IncludeX(Gen_Seg_arch)

    def forward(self,X, phase, truth_masks):
        generated_images, predicted_masks = self.baseGenSeg_model(X)
        #predicted_masks = (2*N,2,H,W) i.e., original images masks and generated images masks
        if phase !='train':
            predicted_masks_X, predicted_masks_gen = catOrSplit(predicted_masks)
            #the results would be (N,C,[orig gen],H,W)
            generated_X_masks_stacked = torch.stack((predicted_masks_gen, predicted_masks_X), dim=2)
            predicted_masks, _ = generated_X_masks_stacked.max(dim=2)
        else:#if it is train, double the number of labels to be (2*N,2,H,W)
            truth_masks = catOrSplit([truth_masks, truth_masks])

        return generated_images, predicted_masks, truth_masks



def unet_proposed():
    generator = unet.UNet(in_channels=3,
                          out_channels=3,
                          n_blocks=4,
                          activation='relu',
                          normalization='batch',
                          conv_mode='same',
                          dim=2)
    generator = nn.Sequential(generator, nn.Sigmoid())
    segmentor = unet.UNet(in_channels=3,
                          out_channels=2,
                          n_blocks=4,
                          activation='relu',
                          normalization='batch',
                          conv_mode='same',
                          dim=2)
    return nn.ModuleList([generator, segmentor])
