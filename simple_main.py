import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import os
from torchvision import transforms
from Plotting import plot, plot_test
from MyDataloaders import *
from Metrics import *
from models import MyModelV1, FCNModels, DeepLabModels
from torch import nn
from torchvision import datasets
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from prettytable import PrettyTable

def show(torch_img, index):
    # if not isinstance(torch_img,list):
    #     torch_img = [torch_img]
    toPIL = transforms.ToPILImage()
    for i, img in enumerate(torch_img):
        img = toPIL(img.clone().detach().cpu())#.numpy().transpose((1, 2, 0))
        plt.imshow(img)
        plt.savefig('./generatedImages/'+str(index+i)+'generated.jpg')
        plt.clf()
        # print(img)

def image_gradient(images):
    a=np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight = nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0),requires_grad=False)
    conv1.to(device)
    #-----------------------------------------------------------
    b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight = nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0),requires_grad=False)
    conv2.to(device)
    #-----------------------------------------
    #images.shape = [batch, C, H, W]
    images_shape = images.shape
    #images.reshape = [batch*C, 1, H, W]
    images = images.view(-1,1,*images_shape[-2:])

    G_x = conv1(images)
    G_y = conv2(images)
    G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
    return torch.sum(G)/(images_shape[-1]*images_shape[-2])

def denoising_loss(created_images, original_images, lr=1):
    alpha = torch.sum(torch.pow(created_images-original_images,2))/(original_images.shape[-1]*original_images.shape[-2])
    beta = lr * image_gradient(created_images)
    total = alpha + beta
    return total

def training_loop(n_epochs, optimizer, lr_scheduler, model, loss_fn, train_loader,device):


    for epoch in range(0, n_epochs+1):
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
            loss = denoising_loss(ypred,X)
            tr_loss_arr.append(loss.clone().detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            if epoch%10 == 0 and flag:
                show(ypred,index = total_train_images)
            #************ store sub-batch results ********************
            # tr_loss_arr.append(loss.item()*batch_size)
            # ioutrain += IOU_class01(y, ypred) # appending list of images' IOU
            # dice_train += dic(y, ypred)
            # pixelacctrain += pixelAcc(y, ypred) # appending list of images' pixel accuracy

            # temp_epoch_loss += loss.item()
            # temp_epoch_iou += IOU_class01(y, ypred)
            # temp_epoch_pixelAcc +=pixelAcc(y, ypred)
            # ******************* finish storing sub-batch result *********

            #update the progress bar
            pbar.set_postfix({'Epoch': epoch + 1,
                              'Training Loss': np.average(tr_loss_arr)/total_train_images,
                              })
        #average epoch results for training
        temp_epoch_loss = np.average(tr_loss_arr)/total_train_images
    #     temp_epoch_iou = np.sum(ioutrain)/total_train_images # to make sure that is equal to np.mean(iotrain)
    #     temp_epoch_dice = np.mean(dice_train)
    #     temp_epoch_pixelAcc = np.mean(pixelacctrain)
    #     # print("Train: Epoch{} results:loss:{:.4f} iou:{:.4f} pixelAcc:{:.4f}".format(epoch+1,temp_epoch_loss,temp_epoch_iou,temp_epoch_pixelAcc))
    #     valtraintest_loss["train"].append(temp_epoch_loss)
    #     valtraintest_IOU["train"].append(temp_epoch_iou)
    #     valtraintest_Dice["train"].append(temp_epoch_dice)
    #     valtraintest_pixelAcc["train"].append(temp_epoch_pixelAcc)
    #
    #     #************************************** Validation starts here ************************************
    #     # temp_epoch_loss = 0
    #     # temp_epoch_iou = 0
    #     # temp_epoch_pixelAcc = 0
    #     total_val_images = 0
    #     with torch.no_grad():
    #
    #         val_loss = 0
    #         pbar = tqdm(val_loader, total=len(val_loader))
    #         for X, y in pbar:
    #             torch.cuda.empty_cache()
    #             X = X.to(device).float()
    #             y = y.to(device).float()
    #             batch_size = len(X)
    #             total_val_images += batch_size
    #             model.eval()
    #             ypred = model(X)
    #             # ************ store sub-batch results ********************
    #             val_loss = loss_fn(ypred,y)*batch_size
    #             val_loss_arr.append(val_loss.item())
    #             pixelaccval += pixelAcc(y,ypred)
    #             iouval += IOU_class01(y, ypred)
    #             dice_val += dic(y, ypred)
    #
    #             # temp_epoch_loss += val_loss.item()
    #             # temp_epoch_iou += IOU_class01(y, ypred)
    #             # temp_epoch_pixelAcc += pixelAcc(y, ypred)
    #             # ************ finish storing sub-batch results ********************
    #
    #             pbar.set_postfix({'Epoch': epoch + 1 + prevEpoch,
    #                               'Validation Loss': np.sum(val_loss_arr)/total_val_images,
    #                               'Mean IOU': np.mean(iouval),
    #                               'Mean Dice': np.mean(dice_val),
    #                               'Pixel Acc': np.mean(pixelaccval)
    #                               })
    #     # average epoch results for training
    #     temp_epoch_loss = np.sum(val_loss_arr)/total_val_images
    #     temp_epoch_iou = np.mean(iouval)
    #     temp_epoch_dice = np.mean(dice_val)
    #     temp_epoch_pixelAcc = np.mean(pixelaccval)
    #     # print("Val: Epoch{} results:loss:{:.4f} iou:{:.4f} pixelAcc:{:.4f}".format(epoch + 1, temp_epoch_loss,
    #     #                                                                  temp_epoch_iou, temp_epoch_pixelAcc))
    #
    #     #store the model if higher iou achieved
    #     if temp_epoch_iou > best_val_iou:#best recordered val iou
    #         best_val_iou = temp_epoch_iou
    #         print("Best Val so far\n")
    #
    #         #run the model on test data
    #         if len(test_loader) != 0:# check if the test set is avialable otherwise don't run a test set
    #             test_epoch_loss,test_epoch_iou, test_epoch_dice, test_epoch_pixelAcc=applytoTest(model, epoch,loss_fn, prevEpoch, test_loader,device)
    #         else: test_epoch_loss,test_epoch_iou, test_epoch_dice, test_epoch_pixelAcc= (0,0,0,0)
    #
    #         valtraintest_loss["test"].append(test_epoch_loss)
    #         valtraintest_IOU["test"].append(test_epoch_iou)
    #         valtraintest_pixelAcc["test"].append(test_epoch_pixelAcc)
    #
    #
    #         checkpoint = {
    #             'epoch': epoch + 1,
    #             'description': "add your description",
    #             'state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'Training Loss': np.sum(tr_loss_arr)/total_train_images,
    #             'Validation Loss': np.sum(val_loss_arr)/total_val_images,
    #             'MeanIOU train': ioutrain,
    #             'PixelAcc train': pixelacctrain,
    #             'MeanIOU test': ioutest,
    #             'PixelAcc test': pixelacctest,
    #             'MeanIOU val': iouval,
    #             'PixelAcc val': pixelaccval
    #         }
    #         torch.save(checkpoint,
    #                    colab_dir+'/checkpoints/highest_IOU_'+model_name+'.pt')
    #         print("finished saving checkpoint")
    #
    #     valtraintest_loss["val"].append(temp_epoch_loss)
    #     valtraintest_IOU["val"].append(temp_epoch_iou)
    #     valtraintest_Dice["val"].append(temp_epoch_dice)
    #     valtraintest_pixelAcc["val"].append(temp_epoch_pixelAcc)
    #
    #     #lr_scheduler.step()
    # # batch_based_result = (tr_loss_arr, val_loss_arr, meanioutrain, pixelacctrain, meanioutest, pixelacctest)
    # epoch_based_result = (valtraintest_loss["train"], valtraintest_loss["val"],
    #                       valtraintest_pixelAcc["val"], valtraintest_pixelAcc["val"],
    #                       valtraintest_IOU["train"], valtraintest_IOU["val"],
    #                       valtraintest_Dice["train"], valtraintest_Dice["val"] )
    #
    # test_results = (valtraintest_loss["test"],valtraintest_IOU["test"],valtraintest_Dice['test'], valtraintest_pixelAcc["test"])
    # # return  batch_based_result, epoch_based_result,test_results
    return  epoch_based_result,test_results



if __name__=='__main__':



    learning_rate = 0.1
    input_channels = 3
    number_classes = 3 # output channels should be one mask for binary class

    root_dir = r"E:\Databases\dummyDataset\train"

    epochs= 2
    batchSize = 2


    print("epochs {} batch size {}".format(epochs, batchSize))
    #************** modify for full experiment *************
    # load_to_RAM = True

    resize_factor = 10
    target_img_size = (500, 570)
    train_val_ratio = 0.8

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
    model_name = "Deeplap_resnet50"
    model = DeepLabModels.Deeplabv3(num_classes=number_classes, backbone=model_name)

    image_transform = transforms.Compose([
            transforms.Resize(target_img_size),  # Resizing the image as the VGG only take 224 x 244 as input size
            transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(root_dir,transform=image_transform)
    trainLoader = DataLoader(train_dataset,batch_size=batchSize)
    # print(trainDataset[1])
    # exit(0)
    # trainLoader = DataLoader(trainDataset, batch_size = batchSize, shuffle=False, drop_last=False,worker_init_fn=seed_worker)

    print("training images:",len(trainLoader))

    # model = getModel(model_name,input_channels, number_classes)

    # count_parameters(model)
    print("model name:", model_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")


    print("Training will be on:",device)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # loss_fn = nn.BCELoss()
    # weight = torch.tensor([0.2, 0.8]).to(device)
    # loss_fn = nn.CrossEntropyLoss(weight) this is the loss of the accepted paper
    loss_fn = nn.BCEWithLogitsLoss() # this is the handseg loss
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.8)
    lr_scheduler = None
    #call the training loop,
    #make sure to pass correct checkpoint path, or none if starting with the training
    start = time.time()

    all_results = training_loop(epochs, optimizer, lr_scheduler, model,loss_fn,
                                                              trainLoader,
                                                              device)
        # batch_based_result, epoch_based_result, test_results = all_results
    epoch_based_result, test_results = all_results
    total_time = time.time() - start
    print('-' * 50, '\nThe entire experiments completed in {:.0f}h {:.0f}m'.format(total_time // 60 ** 2,
                                                                                   (total_time % 60 ** 2) // 60))