import torch.nn as nn
from DL_TOV_threeStages_main import getModel as getModel
def numOfParam(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

if __name__=='__main__':
    #GenSeg_Vanilla_none_CaraNet
    caranet = 'GenSeg_Vanilla_none_CaraNet'
    total_params_caranet = numOfParam(getModel(caranet))
    print('{} has {} parameters'.format(caranet,total_params_caranet ))
    lraspp = 'GenSeg_Vanilla_none_MSNet'
    total_params_lraspp = numOfParam(getModel(lraspp))
    print('{} has {} parameters'.format(caranet, total_params_lraspp))
    print('The ratio lraspp/cara=',total_params_lraspp/total_params_caranet)






