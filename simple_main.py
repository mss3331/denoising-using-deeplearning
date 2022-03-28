import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import os
from torchvision import transforms
from Plotting import plot, plot_test
from torch.nn import functional as F
from MyDataloaders import *
from Metrics import *
from models import MyModelV1, FCNModels, DeepLabModels, unet
from torch import nn
from torchvision import datasets
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from prettytable import PrettyTable


def show(torch_img, index, save):
    # if not isinstance(torch_img,list):
    #     torch_img = [torch_img]
    toPIL = transforms.ToPILImage()
    for i, img in enumerate(torch_img):
        if (i == 10): return
        img = toPIL(img.clone().detach().cpu())  # .numpy().transpose((1, 2, 0))
        plt.imshow(img)
        if save:
            plt.savefig('./generatedImages/' + str(index) + '_' + str(i) + 'generated.jpg')
            plt.clf()
        else:
            plt.show()
            plt.clf()
        # print(img)


def image_gradient(images):
    a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight = nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
    print(conv1.weight)
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
    G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
    return torch.sum(G) / (images_shape[-1] * images_shape[-2])


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
    for epoch in range(0, n_epochs + 1):
        flag = True
        total_train_images = 0
        tr_loss_arr = []

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
            loss = loss_fn(ypred, X)

            tr_loss_arr.append(loss.clone().detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            show(ypred, index=100 + epoch, save=True)
            # ************ store sub-batch results ********************
            # tr_loss_arr.append(loss.item()*batch_size)
            # ioutrain += IOU_class01(y, ypred) # appending list of images' IOU
            # dice_train += dic(y, ypred)
            # pixelacctrain += pixelAcc(y, ypred) # appending list of images' pixel accuracy

            # temp_epoch_loss += loss.item()
            # temp_epoch_iou += IOU_class01(y, ypred)
            # temp_epoch_pixelAcc +=pixelAcc(y, ypred)
            # ******************* finish storing sub-batch result *********

            # update the progress bar
            pbar.set_postfix({'Epoch': epoch + 1,
                              'Training Loss': np.average(tr_loss_arr) / total_train_images,
                              })
        # average epoch results for training
        temp_epoch_loss = np.average(tr_loss_arr) / total_train_images

    return epoch_based_result, test_results


if __name__ == '__main__':

    learning_rate = 0.01
    input_channels = 3
    number_classes = 3  # output channels should be one mask for binary class

    run_in_colab = True
    root_dir = r"E:\Databases\dummyDataset\train"
    colab_dir = "."
    if run_in_colab:
        root_dir = "/content/cvc_samples_denosing/"
        colab_dir = "/content/denoising-using-deeplearning/"
    epochs = 100
    batchSize = 30

    print("epochs {} batch size {}".format(epochs, batchSize))
    # ************** modify for full experiment *************
    # load_to_RAM = True

    resize_factor = None
    target_img_size = (255, 255)
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

    image_transform = transforms.Compose([
        transforms.Resize(target_img_size),  # Resizing the image as the VGG only take 224 x 244 as input size
        transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(root_dir, transform=image_transform)
    trainLoader = DataLoader(train_dataset, batch_size=batchSize)
    # print(trainDataset[1])
    # exit(0)
    # trainLoader = DataLoader(trainDataset, batch_size = batchSize, shuffle=False, drop_last=False,worker_init_fn=seed_worker)

    print("training images:", len(trainLoader))

    # model = getModel(model_name,input_channels, number_classes)

    # count_parameters(model)
    print("model name:", model_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    print("Training will be on:", device)

    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # loss_fn = nn.BCELoss()
    # weight = torch.tensor([0.2, 0.8]).to(device)
    # loss_fn = nn.CrossEntropyLoss(weight) this is the loss of the accepted paper
    loss_fn = nn.MSELoss()  # this is the handseg loss
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.8)
    lr_scheduler = None
    # call the training loop,
    # make sure to pass correct checkpoint path, or none if starting with the training
    start = time.time()

    all_results = training_loop(epochs, optimizer, lr_scheduler, model, loss_fn,
                                trainLoader,
                                device)
    # batch_based_result, epoch_based_result, test_results = all_results
    epoch_based_result, test_results = all_results
    total_time = time.time() - start
    print('-' * 50, '\nThe entire experiments completed in {:.0f}h {:.0f}m'.format(total_time // 60 ** 2,
                                                                                   (total_time % 60 ** 2) // 60))