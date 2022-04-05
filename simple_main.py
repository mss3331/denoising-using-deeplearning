import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import os
import numpy as np
from torchvision import transforms
from Plotting import plot, plot_test
from torch.nn import functional as F
from MyDataloaders import *
from Metrics import *
from models import MyModelV1, FCNModels, DeepLabModels, unet
import torch
from MyDataloaders_denoising import getDataloadersDic
from torch import nn
from torchvision import datasets
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

wandb.login(key="38818beaffe50c5403d3f87b288d36d1b38372f8")
from prettytable import PrettyTable


def show(torch_img, original_imgs,phase, index, save):
    # if not isinstance(torch_img,list):
    #     torch_img = [torch_img]
    toPIL = transforms.ToPILImage()
    for i, img in enumerate(torch_img):
        if (i == 5): return
        generated_img = img.clone().detach().cpu()
        original_img = original_imgs[i].clone().detach().cpu()
        img = torch.cat((original_img, generated_img), 2)
        img = toPIL(img)  # .numpy().transpose((1, 2, 0))
        img.save('./generatedImages_'+phase+'/' + str(index) + '_' + str(i) + 'generated.png')
        # plt.imshow(img)
        # if save:
        #     plt.savefig('./generatedImages/'+str(index)+'_'+str(i)+'generated.jpg')
        #     plt.clf()
        # else:
        #     plt.show()
        #     plt.clf()
        # print(img)


def image_gradient(images):

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
    grad_loss = torch.sum(G) / (images_shape[0] * images_shape[1] * images_shape[2] * images_shape[3])
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


def training_loop(n_epochs, optimizer, lr_scheduler, model, loss_fn, train_loader, device):
    best_val_loss = 100
    for epoch in range(0, n_epochs + 1):

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            flag = True
            total_train_images = 0
            #TODO: the Loss here are normalized using np.mean() to get the average loss across all images
            loss = []
            loss_l2 = []
            loss_grad = []
            original_images_grad = []



            pbar = tqdm(train_loader, total=len(train_loader))
            for X, y in pbar:
                batch_size = len(X)
                total_train_images += batch_size

                # torch.cuda.empty_cache()
                model.train()
                X = X.to(device).float()
                y = y.to(device).float()
                ypred = model(X)

                optimizer.zero_grad()
                # show(X)
                # image_gradient(X)
                # Calculating the loss starts here
                with torch.set_grad_enabled(phase == 'train'):
                    loss_l2 = loss_fn(ypred, X)*lamda["l2"]
                    loss_grad = image_gradient(ypred)*lamda["grad"]

                    loss = loss_l2 + loss_grad

                    # if (loss.item() <= 0.01):
                    #     scaler += 10
                    #     print("scaler is used to increase the loss=", scaler)

                    loss.append(loss.clone().detach().cpu().numpy())
                    loss_l2.append(loss_l2.clone().detach().cpu().numpy())
                    loss_grad.append(loss_grad.clone().detach().cpu().numpy())
                    original_images_grad.append(image_gradient(X).clone().detach().cpu().numpy())

                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                    show(ypred, X,phase, index=100 + epoch, save=True)

                # ************ store sub-batch results ********************
                # loss.append(loss.item()*batch_size)
                # ioutrain += IOU_class01(y, ypred) # appending list of images' IOU
                # dice_train += dic(y, ypred)
                # pixelacctrain += pixelAcc(y, ypred) # appending list of images' pixel accuracy

                # temp_epoch_loss += loss.item()
                # temp_epoch_iou += IOU_class01(y, ypred)
                # temp_epoch_pixelAcc +=pixelAcc(y, ypred)
                # ******************* finish storing sub-batch result *********

                # update the progress bar
                pbar.set_postfix({phase+' Epoch': str(epoch)+"/"+str(num_epochs-1),
                                  'Loss': np.mean(loss),
                                  'L2': np.mean(loss_l2),
                                  'grad': np.mean(loss_grad),
                                  'original_images_grad': np.mean(original_images_grad)
                                  })
            if phase=='val' and np.mean(loss) < best_val_loss:
                print('best loss={} so far ...'.format(np.mean(loss)))
                wandb.run.summary["best_epoch"] = epoch
                wandb.run.summary["val_loss"] = np.mean(loss)
                best_val_loss = np.mean(loss)

                print('saving a checkpoint')


            wandb.log({phase+"_loss": np.mean(loss),
                       phase+"_L2": np.mean(loss_l2), phase+"_grad": np.mean(loss_grad),
                       phase+'_original_images_grad': np.mean(original_images_grad), phase+"_epoch": epoch},
                      step=epoch)


if __name__ == '__main__':

    learning_rate = 0.01
    input_channels = 3
    number_classes = 3  # output channels should be one mask for binary class

    run_in_colab = True
    root_dir = r"E:\Databases\dummyDataset\train"
    child_dir = "CVC-ClinicDB"
    imageDir = 'images_C1'
    maskDir = 'mask_C1'
    colab_dir = "."
    if run_in_colab:
        root_dir = "/content/cvc_samples_denosing"
        colab_dir = "/content/denoising-using-deeplearning"
    num_epochs = 300
    batch_size = 30
    shuffle = True
    lamda = {"l2":1,"grad":1} #L2 and Grad
    print("epochs {} batch size {}".format(num_epochs, batch_size))
    # ************** modify for full experiment *************
    # load_to_RAM = True

    resize_factor = None
    target_img_size = (288, 384)
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
    model_name = "unet"
    model = unet.UNet(in_channels=input_channels,
                      out_channels=number_classes,
                      n_blocks=4,
                      activation='relu',
                      normalization='batch',
                      conv_mode='same',
                      dim=2)

    # image_transform = transforms.Compose([
    #     transforms.Resize(target_img_size),  # Resizing the image as the VGG only take 224 x 244 as input size
    #     transforms.ToTensor()])
    # train_dataset = datasets.ImageFolder(root_dir, transform=image_transform)
    # trainLoader = DataLoader(train_dataset, batch_size=batch_size)
    dataset_info = (root_dir, child_dir, imageDir, maskDir, target_img_size)
    dataloder_info = (train_val_ratio,batch_size, shuffle)
    Dataloaders_dic = getDataloadersDic(dataset_info, dataloder_info)
    # print(trainDataset[1])
    # exit(0)
    # trainLoader = DataLoader(trainDataset, batch_size = batchSize, shuffle=False, drop_last=False,worker_init_fn=seed_worker)

    print("training images:", len(Dataloaders_dic['train'].dataset))

    # model = getModel(model_name,input_channels, number_classes)

    # count_parameters(model)
    print("model name:", model_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    print("Training will be on:", device)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # loss_fn = nn.BCELoss()
    # weight = torch.tensor([0.2, 0.8]).to(device)
    # loss_fn = nn.CrossEntropyLoss(weight) this is the loss of the accepted paper
    loss_fn = nn.MSELoss()  # this is the handseg loss
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.8)
    lr_scheduler = None
    # call the training loop,
    # make sure to pass correct checkpoint path, or none if starting with the training
    start = time.time()
    wandbproject_name = "denoising"
    wandb.init(
        project=wandbproject_name,
        entity="mss3331",
        name="Denoising_Exp1_Unet_Adam_30Images",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "architecture": model_name,
            "batch_size": batch_size,
            "lamda":lamda,
            "num_epochs": num_epochs,
            "dataset": root_dir.split("/")[-1], })
    training_loop(num_epochs, optimizer, lr_scheduler, model, loss_fn,
                  Dataloaders_dic,
                  device)
    wandb.save(colab_dir + '/*.py')
    wandb.save(colab_dir + '/results/*')
    wandb.save(colab_dir + '/models/*')
    wandb.finish()

    total_time = time.time() - start
    print('-' * 50, '\nThe entire experiments completed in {:.0f}h {:.0f}m'.format(total_time // 60 ** 2,
                                                                                   (total_time % 60 ** 2) // 60))