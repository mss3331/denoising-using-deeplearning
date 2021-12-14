import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self,num_classes,backbone="resnet50"):
        super(FCN, self).__init__()

        if backbone.find("resnet50") >= 0:
            self.dl = models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=num_classes)
        elif backbone.find("resnet101") >= 0:
            self.dl = models.segmentation.fcn_resnet101(pretrained=False, progress=True, num_classes=num_classes)
        else:
            print("backbone for FCN not recognized ...\n")
            exit(-1)

    def forward(self, x):
        x = self.dl(x)['out']
        #x_softmax = F.softmax(x, dim=1)
        return x#, x_softmax
