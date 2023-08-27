#9064466c4a4f16db52c1672e03ee3c52060a24e4 token
import torch.optim as optim
from requests import get
# import matplotlib.pyplot as plt
import wandb
import time
from Metrics import *
from models import unet
import torch
from MyDataloaders_denoising import getDataloadersDic
from torch import nn

from Training_GenSeg import *

wandb.login(key="38818beaffe50c5403d3f87b288d36d1b38372f8")
# from prettytable import PrettyTable
def initializWandb():
    wandbproject_name = "denoising"
    wandb.init(
        project=wandbproject_name,
        entity="mss3331",
        name=experiment_name,
        # Track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "architecture": model_name,
            "batch_size": batch_size,
            "lamda": lamda,
            "num_epochs": num_epochs,
            "dataset": root_dir.split("/")[-1], })

def repreducibility():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def getModel(train_Seg_or_Gen):
    generator = unet.UNet(in_channels=input_channels,
                          out_channels=number_classes,
                          n_blocks=4,
                          activation='relu',
                          normalization='batch',
                          conv_mode='same',
                          dim=2)
    generator = nn.Sequential(generator, nn.Sigmoid())
    segmentor = unet.UNet(in_channels=input_channels,
                          out_channels=2,
                          n_blocks=4,
                          activation='relu',
                          normalization='batch',
                          conv_mode='same',
                          dim=2)
    if train_Seg_or_Gen=='Gen':
        model = generator
    elif train_Seg_or_Gen=='Seg':
        model = nn.ModuleList([generator, segmentor])
    return model


def printCheckpoint(checkpoint):
    ''''epoch': epoch + 1,
        'description': "add your description",
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'Validation Loss': val_loss,
        'Test Loss': test_loss,
        'MeanIOU test': test_mIOU,
        'MeanIOU val': val_mIOU'''
    for key,value in checkpoint.items():
        if key.find('state')>=0:
            continue
        print(key,":",value)
    return checkpoint['state_dict']

def load_pretrained_model(model, checkpoint,train_Seg_or_Gen, inference):
    if checkpoint:
        state_dict = printCheckpoint(checkpoint)
        if inference:
            model.load_state_dict(state_dict)
        elif train_Seg_or_Gen=='Seg':
            model.load_state_dict(state_dict) #load weights for generator only

    if train_Seg_or_Gen == 'Seg' and not inference:
        for param in model.parameters():
            param.requires_grad = False
        #in any case, the segmentor should be re-initialize.
        # Except for inference, though, inference ha entirely different code
        seg = unet.UNet(in_channels=input_channels,
                              out_channels=2,
                              n_blocks=4,
                              activation='relu',
                              normalization='batch',
                              conv_mode='same',
                              dim=2)
        model = nn.ModuleList([model, seg])

    return model


if __name__ == '__main__':
    '''This main is created to do side experiments'''
    repreducibility()
    #either Seg or Gen
    train_Seg_or_Gen = "Seg"
    inference=False
    experiment_name=get('http://172.28.0.2:9000/api/sessions').json()[0]['name'].split('.')[0]
    learning_rate = 0.01
    input_channels = 3
    number_classes = 3  # output channels should be one mask for binary class
    if train_Seg_or_Gen=='Gen':
        if inference:
          switch_epoch = [-1,1500000] # ignore stage1 for inference
        else:
          switch_epoch = [50,1500000] # when to switch to the next training stage?
    elif train_Seg_or_Gen=='Seg':
        switch_epoch = [-1,-1]  # when to switch to the next training stage?
    run_in_colab = True
    root_dir = r"E:\Databases\dummyDataset\train"
    child_dir = "data_C1"
    imageDir = 'images_C1'
    maskDir = 'mask_C1'
    colab_dir = ".."
    if run_in_colab:
        root_dir = "/content/CVC-ClinicDB"
        colab_dir = "/content/denoising-using-deeplearning"
    num_epochs = 300
    batch_size = 7
    shuffle = False
    if train_Seg_or_Gen=='Gen':
        lamda = {"l2":1,"grad":10} #L2 and Grad
    else: lamda = {"l2":1,"grad":1} #L2 and Grad
    if inference:
        lamda = {"l2":1,"grad":1} #L2 and Grad

    # ************** modify for full experiment *************
    # load_to_RAM = True

    resize_factor = 0.75
    target_img_size = (int(288*resize_factor), int(384*resize_factor))
    train_val_ratio = 0.5

    print("resize_factor={} and image size={}".format(resize_factor, target_img_size))
    # ************** modify for full experiment *************
    # [SegNet, SegNetGRU, SegNetGRU_Symmetric, SegNetGRU_Symmetric_columns,
    # SegNetGRU_Symmetric_columns_shared_EncDec, SegNetGRU_Symmetric_columns_UltimateShare,
    # SegNetGRU_Symmetric_columns_last2stages, SegNetGRU_Symmetric_columns_last2stages_Notshared_EncDec
    # SegNetGRU_Symmetric_columns_last2stages_Notshared_EncDec_smallerH, SegNetGRU_5thStage_only_not_shared,
    # SegNetGRU_4thStage_only_not_shared, SegNetGRU_Symmetric_last2stages_FromEncToDec]
    ########################### Deeplab versions ###################################
    # [Deeplap_resnet50, Deeplap_resnet101, FCN_resnet50, FCN_resnet101, Deeplabv3_GRU_ASPP_resnet50,
    # Deeplabv3_GRU_CombineChannels_resnet50, Deeplabv3_GRU_ASPP_CombineChannels_resnet50, Deeplabv3_LSTM_resnet50]
    ########################### unet model #####################################################
    # [unit.UNET]
    model_name = "unet-proposed-"+train_Seg_or_Gen
    model = getModel('Gen')
    # Start WandB recording
    initializWandb()
    print('#'*50,'training which part?:',train_Seg_or_Gen,'#'*50)
    print("Experiment name:",experiment_name)
    print("epochs {} batch size {}".format(num_epochs, batch_size))

    dataset_info = [(root_dir, child_dir, imageDir, maskDir, target_img_size)]#,
                    #("/content/trainData_EndoCV2021_5_Feb2021","data_C2","images_C2","mask_C2",target_img_size)]
    dataloder_info = (train_val_ratio,batch_size, shuffle)
    Dataloaders_dic = getDataloadersDic(dataset_info, dataloder_info)

    dataset_info = ("/content/trainData_EndoCV2021_5_Feb2021", child_dir, imageDir, maskDir, target_img_size)
    dataloder_info = (0.01,batch_size, shuffle) # from 0:(0.01*datasize) will be for val the rest for test
    Dataloaders_test_dic = getDataloadersDic(dataset_info, dataloder_info)
    Dataloaders_dic['test']=Dataloaders_test_dic['val']

    print("training images:", len(Dataloaders_dic['train'].dataset))
    print("val images:", len(Dataloaders_dic['val'].dataset))
    print("test images:", len(Dataloaders_dic['test'].dataset))
    print("model name:", model_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    print("Training will be on:", device)

    if train_Seg_or_Gen == 'Seg' or inference:
        checkpoint = torch.load('./denoising-using-deeplearning/checkpoints/highest_IOU_unet-proposed-Gen.pt')
    else:
        checkpoint = None
    model = load_pretrained_model(model, checkpoint, train_Seg_or_Gen, inference)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_dic = {'generator':nn.MSELoss(reduction='sum'), # this is generator loss,
               'segmentor':nn.BCEWithLogitsLoss()}

    # call the training loop,
    # make sure to pass correct checkpoint path, or none if starting with the training
    start = time.time()

    if inference:
        num_epochs=0
        Dataloaders_dic.pop('train')
    Dl_TOV_GenSeg_loop(num_epochs, optimizer, lamda, model, loss_dic,
                       Dataloaders_dic, device, switch_epoch,colab_dir,
                       model_name,train_Seg_or_Gen, inference)

    wandb.save(colab_dir + '/*.py')
    wandb.save(colab_dir + '/results/*')
    wandb.save(colab_dir + '/models/*')
    wandb.finish()

    total_time = time.time() - start
    print('-' * 50, '\nThe entire experiments completed in {:.0f}h {:.0f}m'.format(total_time // 60 ** 2,
                                                                                   (total_time % 60 ** 2) // 60))