import torch
import wandb
from Training import *
import numpy as np
from tqdm import tqdm
from torch import nn
from pprint import pprint
from torchvision import transforms
from MedAI_code_segmentation_evaluation import IOU_class01
from My_losses import *
#TODO: delete the unecessarly model.train after the phase loop
# TODO: Implement saving a checkpoint


def Dl_TOV_GenSeg_loop(num_epochs, optimizer, lamda, model, loss_dic,
                       data_loader_dic, device, switch_epoch,colab_dir,
                       model_name,train_Seg_or_Gen, inference):
    best_loss = {k: 1000 for k in data_loader_dic.keys()}
    best_iou = {k: 0 for k in data_loader_dic.keys()}
    best_iou_epoch = -1
    loss_fn_sum = loss_dic['generator']
    bce = loss_dic['segmentor']


    for epoch in range(0, num_epochs + 1):

        for phase in data_loader_dic.keys():
            if phase == 'test' and best_iou_epoch!=epoch: #skip testing if no better iou val achieved
                if not inference:# skip if we are not doing inference
                    continue
            if phase == 'train':
                if train_Seg_or_Gen=='Gen':
                    #The model is actually Sequential(generator,Sigmoid)
                    model.train()  # Set model to training mode.
                if train_Seg_or_Gen=='Seg':
                    model[0].eval()  # Set model to training mode
                    model[1].train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            flag = True #flag for showing first batch
            total_train_images = 0
            # TODO: the Loss here are normalized using np.mean() to get the average loss across all images. However,
            # TODO: last epoch may have less images hence, mean is not accurate
            loss_batches = []
            loss_l2_batches = []
            loss_grad_batches = []
            loss_mask_batches = []
            iou_batches = []
            original_images_grad = []

            pbar = tqdm(data_loader_dic[phase], total=len(data_loader_dic[phase]))
            for X, intermediate, original_masks in pbar:
                batch_size = len(X)
                total_train_images += batch_size

                X = X.to(device).float()
                intermediate = intermediate.to(device).float()  # intermediate is the mask with type of float
                original_masks = original_masks.to(device)#this is 2 channels mask
                if train_Seg_or_Gen=='Gen':
                    generated_images = model(X)
                else:
                    generated_images = model[0](X)

                generated_X = generated_images.clone().detach()
                if epoch >= switch_epoch[1]:
                    generated_masks = model[1](generated_X)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    loss_mask = torch.zeros((1)).int()
                    iou=0
                    #update loss threshold for stage 2 and 3
                    if epoch == switch_epoch[0]:  # update the best_val_loss threshold
                        best_loss['val'] = 1000
                    if epoch == switch_epoch[1]:  # update the best_val_loss threshold for the segmentor
                        best_loss['val'] = 1000

                    if epoch < switch_epoch[0]:  # focus on minimizing ‖f-g‖^2
                        loss_l2 = loss_fn_sum(generated_images, X) / X.numel()
                        loss_grad = color_gradient(generated_images)
                        loss = loss_l2
                    else:  # epoch >= switch_epoch and epoch <switch_epoch*2:  # move to stage 2 loss: ‖f-g * mask(polyp)‖^2 + ‖∇g *mask(1-polyp)‖^2
                        loss_l2 = torch.sum(torch.pow(torch.mul(generated_images - X, intermediate), 2)) / torch.sum(intermediate)
                        gradients = color_gradient(generated_images, 'No reduction')
                        gradients_masked = torch.mul(gradients, 1 - intermediate)  # consider only background
                        loss_grad = torch.sum(torch.pow(gradients_masked, 2)) / torch.sum(1 - intermediate)
                        # if epoch >= switch_epoch[1]:#increse the polyp reconstruction loss to balance it with seg loss
                        #     loss_l2 = loss_l2 * lamda['l2']
                        if epoch < switch_epoch[1]:  # if we are in stage 2 calculate don't include lamda['l2']
                            loss = loss_grad * lamda['grad'] + loss_l2 * lamda['l2']
                        if epoch >= switch_epoch[1]:  # move to stage 3 loss: ‖f-g * mask(polyp)‖^2 + ‖∇g *mask(1-polyp)‖^2 + BCEWithLoggits
                            loss_mask = bce(generated_masks, original_masks)
                            #the only loss we care about is the BCE of the mask
                            loss = loss_mask
                            iou = IOU_class01(original_masks, generated_masks)

                    loss_batches.append(loss.clone().detach().cpu().numpy())
                    loss_l2_batches.append(loss_l2.clone().detach().cpu().numpy())
                    loss_grad_batches.append(loss_grad.clone().detach().cpu().numpy())
                    loss_mask_batches.append(loss_mask.clone().detach().cpu().numpy())
                    iou_batches.append(iou)
                    original_images_grad.append(color_gradient(X).clone().detach().cpu().numpy())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                if flag:  # this flag
                    flag = False
                    if inference:# show all batches if we are in inference phase
                        flag=True
                    true_mask = intermediate
                    if epoch >= switch_epoch[1]:#stage 3
                        max, generated_mask = generated_masks.max(dim=1)
                        generated_mask = generated_mask.unsqueeze(dim=1)
                        show2(generated_images, X, generated_mask,true_mask, phase,
                              index=100 + epoch, save=True, limit=limit,save_all=(inference,total_train_images))
                    else: #stage 1 and 2
                        generated_mask = torch.zeros(generated_images.shape)
                        show2(generated_images, X, generated_mask,true_mask, phase,
                              index=100 + epoch, save=True,limit=limit,save_all=(inference,total_train_images))

                # update the progress bar
                pbar.set_postfix({phase + ' Epoch': str(epoch) + "/" + str(num_epochs - 1),
                                  'Loss': np.mean(loss_batches),
                                  'L2': np.mean(loss_l2_batches),
                                  'grad': np.mean(loss_grad_batches),
                                  'BCE_loss': np.mean(loss_mask_batches),
                                  'iou': np.mean(iou_batches),
                                  'original_images_grad': np.mean(original_images_grad),
                                  'best_val_loss': best_loss['val'],
                                  'best_val_iou': best_iou['val'],
                                  'best_test_loss': best_loss['test'],
                                  'best_test_iou': best_iou['test'],
                                  })
            if phase != 'train':
                if np.mean(loss_batches) < best_loss[phase]:
                    print('best {} loss={} so far ...'.format(phase,np.mean(loss_batches)))
                    wandb.run.summary["best_{}_loss_epoch".format(phase)] = epoch
                    wandb.run.summary["best_{}_loss".format(phase)] = np.mean(loss_batches)
                    best_loss[phase] = np.mean(loss_batches)
                    if phase=='val':
                        print('better validation loss')
                        if train_Seg_or_Gen=='Gen':
                            print('Saving a Checkpoint for the Generator')
                            saving_checkpoint(epoch, model, optimizer,
                                              best_loss['val'], best_loss['test'],
                                              best_iou['val'], best_iou['test'],
                                              colab_dir, model_name)

                if np.mean(iou_batches) > best_iou[phase]:
                    wandb.run.summary["best_{}_iou".format(phase)] = np.mean(iou_batches)
                    wandb.run.summary["best_{}_iou_epoch".format(phase)] = epoch
                    best_iou[phase] = np.mean(iou_batches)
                    best_loss[phase] = np.mean(loss_batches)
                    if phase=='val':
                        best_iou_epoch = epoch
                        print('best val_iou')
                        print('testing on a test set....\n')
                if phase=='test':#if we reach inside this, it means we achieved a better val iou
                    print('saving a checkpoint')
                    saving_checkpoint(epoch, model, optimizer,
                                      best_loss['val'], best_loss['test'],
                                      best_iou['val'], best_iou['test'],
                                      colab_dir, model_name)

            wandb.log({phase + "_loss": np.mean(loss_batches),
                       phase + "_L2": np.mean(loss_l2_batches), phase + "_grad": np.mean(loss_grad_batches),
                       phase + '_BCE_loss': np.mean(loss_mask_batches), phase + '_iou': np.mean(iou_batches),
                       phase + '_original_images_grad': np.mean(original_images_grad), "best_val_loss": best_loss['val'],
                       'best_val_iou': best_iou['val'], phase + "_epoch": epoch},
                      step=epoch)
