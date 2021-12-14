import numpy as np
import matplotlib.pyplot as pl
from torch import sigmoid
import torch

def  iou_my(outputs,labels):
    outputs = outputs.detach().clone().cpu()
    labels = labels.detach().clone().cpu().argmax(1)
    outputs = (outputs>=0.5).type(torch.int)
    labels = labels.type(torch.int)
    intersection = torch.sum(outputs & labels)
    union = torch.sum(outputs | labels)
    iou = intersection.type(torch.float)/union.type(torch.float)
    return iou.item()

SMOOTH = 1e-6
def iou_pytorch(outputs, labels):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.detach().cpu()
    labels = labels.detach().cpu()
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded  # Or thresholded.mean() if you are interested in average across the batch
def show_image(target_images,predicted_images):
    counter=1
    fig = pl.figure(figsize=(8, 8))
    rows=2
    columns=2
    for index in range(2):
        # temp = image.detach().cpu().numpy()
        temp_tar = target_images[index]
        # print(temp.squeeze().astype(np.int))
        fig.add_subplot(rows,columns,counter)
        pl.imshow(temp_tar)
        counter+=1
        fig.add_subplot(rows, columns, counter)
        pl.imshow((predicted_images[index].argmax(0).astype(int)))
        counter += 1
    pl.show()

def dic(target, predicted,count=0):
    # target = target.detach().cpu()
    # predicted = predicted.detach().cpu()  # simgoid is implecitly applied with calculating the loss
    #
    # dice_sum = 0
    # for i in range(target.shape[0]):
    #     target_arr = target[i, :,:, :].clone().detach().cpu().numpy().argmax(0)
    #     predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
    #
    #     intersection = np.logical_and(target_arr, predicted_arr).sum()
    #     union = np.logical_or(target_arr, predicted_arr).sum()
    #     if union == 0:
    #         dice_score = 0
    #     else:
    #         dice_score = 2*intersection / (union+intersection)
    #     dice_sum += dice_score
    #
    # miou = dice_sum / target.shape[0]
    target = target.detach().cpu()
    predicted = predicted.detach().cpu()  # simgoid is implecitly applied with calculating the loss

    dicsum = []
    # for each image in the batch
    for i in range(target.shape[0]):
        target_arr = target[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
        predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
        iou_score_1 = _iou(target_arr, predicted_arr)
        iou_score_0 = _iou(1 - target_arr, 1 - predicted_arr)
        dice_1 = 2*iou_score_1/(1+iou_score_1)
        dice_0 = 2*iou_score_0/(1+iou_score_0)
        dicsum.append((dice_0 + dice_1) / 2)


    return dicsum

def _iou(target_arr,predicted_arr):
    intersection = np.logical_and(target_arr, predicted_arr).sum()
    union = np.logical_or(target_arr, predicted_arr).sum()
    if union == 0:
        if np.sum(target_arr) == 0: #if the image doesn't have a polyp at all, then, nothing to calculate here
            iou_score = 1
        else:
            iou_score = 0
    else:
        iou_score = intersection / union
    return iou_score

def IOU_class01(target, predicted,count=0):

    # if target.shape != predicted.shape:
    #     print("target has dimension", target.shape, ", predicted values have shape", predicted.shape)
    #     return

    # if target.dim() != 3:
    #     print("target has dim", target.dim(), ", Must be 4.")
    #     return
    target = target.detach().cpu()
    predicted = predicted.detach().cpu()  # simgoid is implecitly applied with calculating the loss

    # if count%80==0:
    #     show_image(target.numpy(),predicted.numpy())





    iou_list = []
    # for each image in the batch
    for i in range(target.shape[0]):
        target_arr = target[i,:, :, :].clone().detach().cpu().numpy().argmax(0)
        predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
        iou_score_1 =  _iou(target_arr,predicted_arr)
        iou_score_0 = _iou(1-target_arr, 1-predicted_arr)

        iou_list.append((iou_score_0+iou_score_1)/2)

    # miou = iousum / target.shape[0]
    return iou_list
def meanIOU(target, predicted,count):


    target = target.detach().cpu()
    predicted = predicted.detach().cpu()  # simgoid is implecitly applied with calculating the loss

    iousum = 0
    # for each image in the batch
    for i in range(target.shape[0]):
        target_arr = target[i,:, :, :].clone().detach().cpu().numpy().argmax(0)
        predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
        iou_score_1 =  _iou(target_arr,predicted_arr)

        iousum += iou_score_1

    miou = iousum / target.shape[0]
    return miou


def pixelAcc(target, predicted):
    # if target.shape != predicted.shape:
    #     print("target has dimension", target.shape, ", predicted values have shape", predicted.shape)
    #     return
    #
    # if target.dim() != 4:
    #     print("target has dim", target.dim(), ", Must be 4.")
    #     return

    accsum = []
    for i in range(target.shape[0]):
        target_arr = target[i,:, :, :].clone().detach().cpu().numpy().argmax(0)
        predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy().argmax(0)

        same = (target_arr == predicted_arr).sum()
        a, b = target_arr.shape
        total = a * b
        accsum.append(same / total)


    return accsum