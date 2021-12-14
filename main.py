'''
check the train, inference, and hyperparameter code in the original github implementation of SegNet:https://github.com/say4n/pytorch-segnet/tree/f7738c6bce384b54fcbb3fe8aff02736d6ec2285/src

'''
import torch.optim as optim
import pandas as pd
import random
import time
import os
from torchvision import transforms as T
from Plotting import plot, plot_test
from MyDataloaders import *
from Metrics import *
from models import MyModelV1, FCNModels, DeepLabModels
from torch import nn
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
def my_loss(truth,outputs):
    N,C,H,W = truth.shape
    diff = ((outputs-truth)**2).sum() /(N*C*H*W)
    # nig_diff = 1 - diff
    # outputs = (outputs >= 0.5).type(torch.int)
    # labels = truth.type(torch.int)
    # intersection = torch.sum(outputs & labels)
    # union = torch.sum(outputs | labels)
    # iou = intersection.type(torch.float) / union.type(torch.float)
    # ratio = torch.sum(diff)/torch.sum(truth)
    return -torch.log(1-diff)


def applytoTest(model, epoch,loss_fn, prevEpoch, test_loader,device):
    # temp_epoch_loss = 0
    # temp_epoch_iou = 0
    # temp_epoch_dice = 0
    # temp_epoch_pixelAcc = 0
    test_loss_arr=[]
    pixelacctest=[]
    meanioutest=[]
    meandicetest=[]
    total_images = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, total=len(test_loader))
        for X, y in pbar:
            batchSize = len(X)
            total_images += batchSize
            torch.cuda.empty_cache()
            X = X.to(device).float()
            y = y.to(device).float()
            model.eval()
            ypred = model(X)
            # ************ store sub-batch results ********************
            test_loss = loss_fn(ypred, y) * batchSize
            test_loss_arr.append(test_loss.item())
            pixelacctest += pixelAcc(y, ypred)
            meanioutest += IOU_class01(y, ypred)
            meandicetest += dic(y, ypred)

            # temp_epoch_loss += test_loss.item()
            # temp_epoch_iou += IOU_class01(y, ypred)
            # temp_epoch_dice += meanDic(y, ypred)
            # temp_epoch_pixelAcc += pixelAcc(y, ypred)
            # ************ finish storing sub-batch results ********************

            pbar.set_postfix({'Epoch': epoch + 1 + prevEpoch,
                              'Test Loss': np.sum(test_loss_arr)/total_images,
                              'Mean IOU': np.mean(meanioutest),
                              'Mean Dice': np.mean(meandicetest),
                              'Pixel Acc': np.mean(pixelacctest)
                              })
    # average epoch results for training
    temp_epoch_loss = np.sum(test_loss_arr)/total_images
    temp_epoch_iou = np.mean(meanioutest)
    temp_epoch_dice = np.mean(meandicetest)
    temp_epoch_pixelAcc = np.mean(pixelacctest)
    print("Test: Epoch{} results:loss:{:.4f} iou:{:.5f} dice:{:.5f} pixelAcc:{:.4f}".format(epoch + 1+ prevEpoch, temp_epoch_loss,
                                                                               temp_epoch_iou,temp_epoch_dice,
                                                                                temp_epoch_pixelAcc))
    return (temp_epoch_loss,temp_epoch_iou, temp_epoch_dice, temp_epoch_pixelAcc)


def training_loop(n_epochs, optimizer, lr_scheduler, model, loss_fn, train_loader, val_loader,test_loader, lastCkptPath,device):

    prevEpoch = 0
    check_point_threshold = 0.0
    #************ Best Epoch variables *******************
    best_val_epoch = 0
    valtraintest_loss = {"val":[],"train":[],"test":[]}
    valtraintest_IOU  = {"val":[],"train":[],"test":[]}
    valtraintest_Dice  = {"val":[],"train":[],"test":[]}
    valtraintest_pixelAcc  = {"val":[],"train":[],"test":[]}
    #*****************************************************
    if lastCkptPath != None:
        checkpoint = torch.load(lastCkptPath)
        prevEpoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.to(device)
        tr_loss_arr = checkpoint['Training Loss']
        val_loss_arr = checkpoint['Validation Loss']
        meanioutrain = checkpoint['MeanIOU train']
        pixelacctrain = checkpoint['PixelAcc train']
        meanioutest = checkpoint['MeanIOU test']
        pixelacctest = checkpoint['PixelAcc test']
        print("loaded model, ", checkpoint['description'], "at epoch", prevEpoch)
        model.to(device)
    count=0
    best_val_iou = 0
    for epoch in range(0, n_epochs-prevEpoch+1):
        tr_loss_arr = []
        val_loss_arr = []
        ioutrain = []
        iouval = []
        dice_train = []
        dice_val = []
        pixelacctrain = []
        pixelaccval = []
        ioutest = []
        pixelacctest = []
        total_train_images = 0

        pbar = tqdm(train_loader, total=len(train_loader))
        for X, y in pbar:
            batch_size = len(X)
            total_train_images += batch_size

            count +=2
            # torch.cuda.empty_cache()
            model.train()
            X = X.to(device).float()
            y = y.to(device).float()
            ypred = model(X)

            optimizer.zero_grad()
            loss = loss_fn(ypred,y)
            loss.backward()
            optimizer.step()
            #************ store sub-batch results ********************
            tr_loss_arr.append(loss.item()*batch_size)
            ioutrain += IOU_class01(y, ypred) # appending list of images' IOU
            dice_train += dic(y, ypred)
            pixelacctrain += pixelAcc(y, ypred) # appending list of images' pixel accuracy

            # temp_epoch_loss += loss.item()
            # temp_epoch_iou += IOU_class01(y, ypred)
            # temp_epoch_pixelAcc +=pixelAcc(y, ypred)
            # ******************* finish storing sub-batch result *********

            #update the progress bar
            pbar.set_postfix({'Epoch': epoch + 1 + prevEpoch,
                              'Training Loss': np.sum(tr_loss_arr)/total_train_images,
                              'Mean IOU': np.mean(ioutrain),
                              'Mean Dice': np.mean(dice_train),
                              'Pixel Acc': np.mean(pixelacctrain)})
        #average epoch results for training
        temp_epoch_loss = np.sum(tr_loss_arr)/total_train_images
        temp_epoch_iou = np.sum(ioutrain)/total_train_images # to make sure that is equal to np.mean(iotrain)
        temp_epoch_dice = np.mean(dice_train)
        temp_epoch_pixelAcc = np.mean(pixelacctrain)
        # print("Train: Epoch{} results:loss:{:.4f} iou:{:.4f} pixelAcc:{:.4f}".format(epoch+1,temp_epoch_loss,temp_epoch_iou,temp_epoch_pixelAcc))
        valtraintest_loss["train"].append(temp_epoch_loss)
        valtraintest_IOU["train"].append(temp_epoch_iou)
        valtraintest_Dice["train"].append(temp_epoch_dice)
        valtraintest_pixelAcc["train"].append(temp_epoch_pixelAcc)

        #************************************** Validation starts here ************************************
        # temp_epoch_loss = 0
        # temp_epoch_iou = 0
        # temp_epoch_pixelAcc = 0
        total_val_images = 0
        with torch.no_grad():

            val_loss = 0
            pbar = tqdm(val_loader, total=len(val_loader))
            for X, y in pbar:
                torch.cuda.empty_cache()
                X = X.to(device).float()
                y = y.to(device).float()
                batch_size = len(X)
                total_val_images += batch_size
                model.eval()
                ypred = model(X)
                # ************ store sub-batch results ********************
                val_loss = loss_fn(ypred,y)*batch_size
                val_loss_arr.append(val_loss.item())
                pixelaccval += pixelAcc(y,ypred)
                iouval += IOU_class01(y, ypred)
                dice_val += dic(y, ypred)

                # temp_epoch_loss += val_loss.item()
                # temp_epoch_iou += IOU_class01(y, ypred)
                # temp_epoch_pixelAcc += pixelAcc(y, ypred)
                # ************ finish storing sub-batch results ********************

                pbar.set_postfix({'Epoch': epoch + 1 + prevEpoch,
                                  'Validation Loss': np.sum(val_loss_arr)/total_val_images,
                                  'Mean IOU': np.mean(iouval),
                                  'Mean Dice': np.mean(dice_val),
                                  'Pixel Acc': np.mean(pixelaccval)
                                  })
        # average epoch results for training
        temp_epoch_loss = np.sum(val_loss_arr)/total_val_images
        temp_epoch_iou = np.mean(iouval)
        temp_epoch_dice = np.mean(dice_val)
        temp_epoch_pixelAcc = np.mean(pixelaccval)
        # print("Val: Epoch{} results:loss:{:.4f} iou:{:.4f} pixelAcc:{:.4f}".format(epoch + 1, temp_epoch_loss,
        #                                                                  temp_epoch_iou, temp_epoch_pixelAcc))

        #store the model if higher iou achieved
        if temp_epoch_iou > best_val_iou:#best recordered val iou
            best_val_iou = temp_epoch_iou
            print("Best Val so far\n")

            #run the model on test data
            if len(test_loader) != 0:# check if the test set is avialable otherwise don't run a test set
                test_epoch_loss,test_epoch_iou, test_epoch_dice, test_epoch_pixelAcc=applytoTest(model, epoch,loss_fn, prevEpoch, test_loader,device)
            else: test_epoch_loss,test_epoch_iou, test_epoch_dice, test_epoch_pixelAcc= (0,0,0,0)

            valtraintest_loss["test"].append(test_epoch_loss)
            valtraintest_IOU["test"].append(test_epoch_iou)
            valtraintest_pixelAcc["test"].append(test_epoch_pixelAcc)


            checkpoint = {
                'epoch': epoch + 1,
                'description': "add your description",
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'Training Loss': np.sum(tr_loss_arr)/total_train_images,
                'Validation Loss': np.sum(val_loss_arr)/total_val_images,
                'MeanIOU train': ioutrain,
                'PixelAcc train': pixelacctrain,
                'MeanIOU test': ioutest,
                'PixelAcc test': pixelacctest,
                'MeanIOU val': iouval,
                'PixelAcc val': pixelaccval
            }
            torch.save(checkpoint,
                       colab_dir+'/checkpoints/highest_IOU_'+model_name+'.pt')
            print("finished saving checkpoint")

        valtraintest_loss["val"].append(temp_epoch_loss)
        valtraintest_IOU["val"].append(temp_epoch_iou)
        valtraintest_Dice["val"].append(temp_epoch_dice)
        valtraintest_pixelAcc["val"].append(temp_epoch_pixelAcc)

        #lr_scheduler.step()
    # batch_based_result = (tr_loss_arr, val_loss_arr, meanioutrain, pixelacctrain, meanioutest, pixelacctest)
    epoch_based_result = (valtraintest_loss["train"], valtraintest_loss["val"],
                          valtraintest_pixelAcc["val"], valtraintest_pixelAcc["val"],
                          valtraintest_IOU["train"], valtraintest_IOU["val"],
                          valtraintest_Dice["train"], valtraintest_Dice["val"] )

    test_results = (valtraintest_loss["test"],valtraintest_IOU["test"],valtraintest_Dice['test'], valtraintest_pixelAcc["test"])
    # return  batch_based_result, epoch_based_result,test_results
    return  epoch_based_result,test_results
def seed_worker(worker_id):
    worker_seed = 0
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def repreducibility(repreducable):
    if repreducable:
        print("repreducibility=",repreducable)
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # if torch.__version__.find('1.8')>=0:
        #     torch.use_deterministic_algorithms(True)


def saveToCSV(data, columns,path):
    frame = pd.DataFrame(data).T
    frame.columns = columns
    frame.to_csv(path)


def getModel(model_name, input_channels, number_classes):
    if model_name == "SegNet":
        model = MyModelV1.SegNet(input_channels=input_channels, num_classes=number_classes)
    elif model_name == "SegNetGRU":
        model = MyModelV1.SegNetGRU(input_channels=input_channels, num_classes=number_classes)
    elif model_name == "SegNetGRU_Symmetric":
        model = MyModelV1.SegNetGRU_Symmetric(input_channels=input_channels, num_classes=number_classes)
    elif model_name == "SegNetGRU_Symmetric_columns":
        model = MyModelV1.SegNetGRU_Symmetric_columns(input_channels=input_channels, num_classes=number_classes)
    elif model_name == "SegNetGRU_Symmetric_columns_shared_EncDec":
        model = MyModelV1.SegNetGRU_Symmetric_columns_shared_EncDec(input_channels=input_channels,
                                                                    num_classes=number_classes)
    elif model_name == "SegNetGRU_Symmetric_columns_UltimateShare":
        model = MyModelV1.SegNetGRU_Symmetric_columns_UltimateShare(input_channels=input_channels,
                                                                    num_classes=number_classes)
    elif model_name == "SegNetGRU_Symmetric_columns_last2stages":
        model = MyModelV1.SegNetGRU_Symmetric_columns_last2stages(input_channels=input_channels,
                                                                    num_classes=number_classes)
    elif model_name == "SegNetGRU_Symmetric_columns_last2stages_Notshared_EncDec":
        model = MyModelV1.SegNetGRU_Symmetric_columns_last2stages_Notshared_EncDec(input_channels=input_channels,
                                                                    num_classes=number_classes)
    elif model_name == "SegNetGRU_Symmetric_columns_last2stages_Notshared_EncDec_smallerH":
        model = MyModelV1.SegNetGRU_Symmetric_columns_last2stages_Notshared_EncDec_smallerH(input_channels=input_channels,
                                                                                   num_classes=number_classes)
    elif model_name == "SegNetGRU_5thStage_only_not_shared":
        model = MyModelV1.SegNetGRU_5thStage_only_not_shared(input_channels=input_channels, num_classes=number_classes)
    elif model_name == "SegNetGRU_4thStage_only_not_shared":
        model = MyModelV1.SegNetGRU_4thStage_only_not_shared(input_channels=input_channels, num_classes=number_classes)
    elif model_name == "SegNetGRU_Symmetric_last2stages_FromEncToDec":
        model = MyModelV1.SegNetGRU_Symmetric_last2stages_FromEncToDec(input_channels=input_channels, num_classes=number_classes)

    ################# Deeplab and FCN variations ###################################
    elif model_name.find("Deeplap_resnet") == 0:
        model = DeepLabModels.Deeplabv3(num_classes=number_classes,backbone=model_name)
    elif model_name.find("Deeplabv3_GRU_ASPP") >=0:
        model = DeepLabModels.Deeplabv3_GRU_ASPP(num_classes=number_classes,backbone=model_name)
    elif model_name.find("Deeplabv3_GRU_CombineChannels") >=0:
        model = DeepLabModels.Deeplabv3_GRU_CombineChannels(num_classes=number_classes,backbone=model_name)
    elif model_name.find("Deeplabv3_GRU_ASPP_CombineChannels") >=0:
        model = DeepLabModels.Deeplabv3_GRU_ASPP_CombineChannels(num_classes=number_classes,backbone=model_name)
    elif model_name.find("Deeplabv3_LSTM_resnet50") >=0:
        model = DeepLabModels.Deeplabv3_LSTM(num_classes=number_classes,backbone=model_name)
    elif model_name.find("FCN_resnet") >=0:
        model = FCNModels.FCN(num_classes=number_classes,backbone=model_name)
    else:
        print("Unrecogenized model name...\n")
        exit(-1)
    return model


def testModel(model,optimizer,loss_fn, data_loaders, device,lastCkptPath,epoch=0):

    checkpoint = torch.load(lastCkptPath)
    prevEpoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('val loss {}, val iou {}, val pixel acc {}'.format(checkpoint['Validation Loss'],
          np.mean(checkpoint['MeanIOU val']),np.mean(checkpoint['PixelAcc val'])))
    # THOSE STORED RESULTS IS BACH-BASED NOT EPOCH BASED you need to apply mean
    # val_loss_arr = checkpoint['Validation Loss']
    # meanioutrain = checkpoint['MeanIOU train']
    # pixelacctrain = checkpoint['PixelAcc train']
    # meanioutest = checkpoint['MeanIOU test']
    # pixelacctest = checkpoint['PixelAcc test']
    # print("val_loss_arr{}, meanioutrain{}, pixelacctrain{}, meaniouval{}, pixelaccval{}".format(np.mean(val_loss_arr),
    #                                                                                             np.mean(meanioutrain),
    #                                                                                             np.mean(pixelacctrain),
    #                                                                                             np.mean(meanioutest),
    #                                                                                             np.mean(pixelacctest)))
    print("Validation")
    for key in data_loaders.keys():
        print(key)
        result = applytoTest(model, epoch,loss_fn, prevEpoch, data_loaders[key], device)
        print('*' * 20)
    # checkpoints_pth = os.scandir("./models/ready to use checkpoints")
    # for dirEntry in checkpoints_pth: #test all models checkpoints in checkpoints_pt
    #     print(dirEntry.path)
    #     checkpoint = torch.load(dirEntry.path)
    #     prevEpoch = checkpoint['epoch']
    #     model.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     # THOSE STORED RESULTS IS BACH-BASED NOT EPOCH BASED
    #     # val_loss_arr = checkpoint['Validation Loss']
    #     # meanioutrain = checkpoint['MeanIOU train']
    #     # pixelacctrain = checkpoint['PixelAcc train']
    #     # meanioutest = checkpoint['MeanIOU test']
    #     # pixelacctest = checkpoint['PixelAcc test']
    #     # print("val_loss_arr{}, meanioutrain{}, pixelacctrain{}, meaniouval{}, pixelaccval{}".format(np.mean(val_loss_arr),
    #     #                                                                                             np.mean(meanioutrain),
    #     #                                                                                             np.mean(pixelacctrain),
    #     #                                                                                             np.mean(meanioutest),
    #     #                                                                                             np.mean(pixelacctest)))
    #     print("Validation")
    #     for key in data_loaders.keys():
    #         print(key)
    #         result = applytoTest(model, epoch, prevEpoch, data_loaders[key],device)
    #         print('*'*20)
def createAugmentedSegDataset(augmentation):
    c1 = SegDataset(root_dir, 'C1', 'images', 'mask', target_img_size,augmentation, load_to_RAM=load_to_RAM)  # SD images
    c2 = SegDataset(root_dir, 'C2', 'images', 'mask', target_img_size,augmentation, load_to_RAM=load_to_RAM)  # HD images
    c3 = SegDataset(root_dir, 'C3', 'images', 'mask', target_img_size,augmentation, load_to_RAM=load_to_RAM)
    c4 = SegDataset(root_dir, 'C4', 'images', 'mask', target_img_size,augmentation, load_to_RAM=load_to_RAM)  # test 1
    c5 = SegDataset(root_dir, 'C5', 'images', 'mask', target_img_size,augmentation, load_to_RAM=load_to_RAM)  # test 2

    return (c1,c2,c3,c4,c5)


if __name__=='__main__':
    repreducibility(repreducable=True)
    Test = False
    load_last_checkpoint = False
    run_in_colab = False
    learning_rate = 0.00005
    input_channels = 3
    number_classes = 2 # output channels should be one mask for binary class

    root_dir = r"E:\Databases\EndoCV21\trainData_EndoCV2021_5_Feb2021"
    colab_dir = "."
    if run_in_colab:
        root_dir = r"/content/trainData_EndoCV2021_5_Feb2021"
        colab_dir = "/content/GIANA21/"

    # root_dir = "/home/bb79e4/Mahmood_databases/EndoCV21"


    epochs= 50
    batchSize = 4
    moving_avg_window = 100

    print("epochs {} batch size {}".format(epochs, batchSize))
    #************** modify for full experiment *************
    # load_to_RAM = True
    load_to_RAM = False
    resize_factor = 10
    target_img_size = (int(1080 / resize_factor), int(1350 / resize_factor))
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
    model_name = "SegNetGRU_Symmetric_columns_last2stages_Notshared_EncDec"

    if load_last_checkpoint or Test:
        lastCkptPath = colab_dir + "/checkpoints/highest_IOU_" + model_name + ".pt"
        print("loaded a checkpoint:", lastCkptPath.split("/")[-1])
    else: lastCkptPath = None

    augmentation = T.ColorJitter(contrast=0.5, saturation=0.5)
    no_augmentation = T.ColorJitter(contrast=0, saturation=0)
    c_augmented = createAugmentedSegDataset(augmentation) # C1, C2, C3 ... Augmented datasets
    c_no_augmented = createAugmentedSegDataset(no_augmentation)

    dummy_dataset = SegDataset(root_dir, 'dummy', 'images', 'mask', target_img_size,augmentation, load_to_RAM=load_to_RAM)
    megaDataset_augmented = ConcatDataset([*c_augmented])#c2, c3])#train validate
    megaDataset_no_augmented = ConcatDataset([*c_no_augmented])#c2, c3])#train validate
    megaDataset_test = ConcatDataset([dummy_dataset])
    #************** comment for full experiment *************
    # megaDataset = ConcatDataset([c1, c2])  # comment this line if you want to do full experiment
    # moving_avg_window = 10  # this threshold used for plotting moving average
    # epochs = 1
    # batchSize = 4
    #********* for full experiment


    trainDataset, valDataset = trainTestSplit((megaDataset_augmented,megaDataset_no_augmented), train_val_ratio)
    # print(trainDataset[1])
    # exit(0)
    trainLoader = DataLoader(trainDataset, batch_size = batchSize, shuffle=False, drop_last=False,worker_init_fn=seed_worker)
    valLoader = DataLoader(valDataset, batch_size = batchSize, shuffle=False, drop_last=False,worker_init_fn=seed_worker)
    testLoader = DataLoader(megaDataset_test, batch_size = batchSize, shuffle=False, drop_last=False,worker_init_fn=seed_worker)
    print("training images:",len(trainDataset))
    print("val images:",len(valDataset))
    print("test images:",len(megaDataset_test))

    model = getModel(model_name,input_channels, number_classes)

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
    if Test:# True if you want to test a model
        testModel(model,optimizer,loss_fn, {"Validations":valLoader,"Test":trainLoader}, device,lastCkptPath,epoch=0)
        exit(0)
    else:
        all_results = training_loop(epochs, optimizer, lr_scheduler, model,loss_fn,
                                                              trainLoader,
                                                              valLoader,
                                                              testLoader,
                                                              lastCkptPath, device)
        # batch_based_result, epoch_based_result, test_results = all_results
        epoch_based_result, test_results = all_results
        total_time = time.time() - start
        print('-' * 50, '\nThe entire experiments completed in {:.0f}h {:.0f}m'.format(total_time // 60 ** 2,
                                                                                       (total_time % 60 ** 2) // 60))
        # saveToCSV(batch_based_result,
        #           columns=['tr_loss_arr, val_loss_arr, meanioutrain, pixelacctrain, meanioutest, pixelacctest'.split(', ')],
        #           path=colab_dir+"/results/batch_based_result.csv")
        saveToCSV(epoch_based_result,
                  columns=['train loss','val loss',
                           'train PixelAcc','val PixelAcc',
                           'train IOU', 'val IOU',
                           'train Dice', 'val Dice'],
                  path=colab_dir+"/results/epoch_based_result.csv")
        saveToCSV(test_results,
                  columns=['test loss', 'test IOU', 'test Dice','test pixelAcc'],
                  path=colab_dir+"/results/epoch_based_teset_result.csv")

        # plot(batch_based_result, moving_avg_window,running_m=True,path=colab_dir+"/results/batch_based_movingAvg.png")
        plot(epoch_based_result, moving_avg_window,running_m=False,path=colab_dir+"/results/epoch_based.png")
        plot_test(test_results,path=colab_dir+"/results/epoch_based_test.png")

    # if run_in_colab:
    #     %cp - r "/content/SegNet" "/content/drive/MyDrive/Experiments/Colab_Ex1_EndoCVJointPaper"
    #     pass
