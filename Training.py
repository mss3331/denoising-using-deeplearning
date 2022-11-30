import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch import nn
from pprint import pprint
import pandas
from torchvision import transforms
from MedAI_code_segmentation_evaluation import IOU_class01, calculate_metrics_torch
from My_losses import *
#TODO: delete the unecessarly model.train after the phase loop
# TODO: Implement saving a checkpoint
def saving_checkpoint(epoch,model,optimizer,val_loss,test_loss,
                      val_mIOU,test_mIOU, colab_dir, model_name, save_generator_checkpoints = False):

    checkpoint = {
        'epoch': epoch + 1,
        'description': "add your description",
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'Validation Loss': val_loss,
        'Test Loss': test_loss,
        'IOU Polyp test': test_mIOU,
        'IOU Polyp val': val_mIOU
    }
    if save_generator_checkpoints:
        torch.save(checkpoint,
                   colab_dir + '/checkpoints/gen_highest_loss_' + model_name + '.pt')
    else:
        torch.save(checkpoint,
               colab_dir + '/checkpoints/highest_IOU_' + model_name + '.pt')

    print("finished saving checkpoint")

def training_loop(num_epochs, optimizer, lamda, model, loss_fn, data_loader_dic, device):
    best_val_loss = 100
    for epoch in range(0, num_epochs + 1):

        for phase in data_loader_dic.keys():
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            flag = True
            total_train_images = 0
            #TODO: the Loss here are normalized using np.mean() to get the average loss across all images. However,
            #TODO: last epoch may have less images hence, mean is not accurate
            loss_batches = []
            loss_l2_batches = []
            loss_grad_batches = []
            original_images_grad = []



            pbar = tqdm(data_loader_dic[phase], total=len(data_loader_dic[phase]))
            for X,intermediate, y in pbar:
                batch_size = len(X)
                total_train_images += batch_size


                model.train()
                X = X.to(device).float()
                intermediate = intermediate.to(device).float() #intermediate is the mask with type of float
                y = y.to(device) #this is the mask as Boolean
                ypred = model(X)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    loss_l2 = loss_fn(ypred, X)*lamda["l2"]
                    loss_grad = gradMaskLoss_Eq1(ypred,intermediate,loss_fn)*lamda["grad"]

                    loss = loss_grad+loss_l2

                    loss_batches.append(loss.clone().detach().cpu().numpy())
                    loss_l2_batches.append(loss_l2.clone().detach().cpu().numpy())
                    loss_grad_batches.append(loss_grad.clone().detach().cpu().numpy())
                    original_images_grad.append(image_gradient(X).clone().detach().cpu().numpy())

                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                if flag:
                    show(ypred, X, intermediate, phase, index=100 + epoch, save=True)
                    flag=False

                # update the progress bar
                pbar.set_postfix({phase+' Epoch': str(epoch)+"/"+str(num_epochs-1),
                                  'Loss': np.mean(loss_batches),
                                  'L2': np.mean(loss_l2_batches),
                                  'grad': np.mean(loss_grad_batches),
                                  'original_images_grad': np.mean(original_images_grad)
                                  })
            if phase=='val' and np.mean(loss_batches) < best_val_loss:
                print('best loss={} so far ...'.format(np.mean(loss_batches)))
                wandb.run.summary["best_epoch"] = epoch
                wandb.run.summary["val_loss"] = np.mean(loss_batches)
                best_val_loss = np.mean(loss_batches)

                print('saving a checkpoint')


            wandb.log({phase+"_loss": np.mean(loss_batches),
                       phase+"_L2": np.mean(loss_l2_batches), phase+"_grad": np.mean(loss_grad_batches),
                       phase+'_original_images_grad': np.mean(original_images_grad),"best_val_loss":best_val_loss, phase+"_epoch": epoch},
                      step=epoch)

def show2(generated_images, X, generated_mask,true_mask, phase, index, save,save_all=(False,-1),limit=5):
    if phase[-1].isnumeric():
        if int(phase[-1]) != 1:
            return

    original_imgs = X
    if not generated_mask.shape == original_imgs.shape:
        generated_mask = generated_mask.repeat(1, 3, 1, 1)
    if not true_mask.shape == original_imgs.shape:
        true_mask = true_mask.repeat(1, 3, 1, 1)
    if not generated_images.shape == original_imgs.shape:
        generated_images.unsqueeze_(1)  # (N,H,W) ==> (N,1,H,W)
        generated_images = generated_images.repeat(1, 3, 1, 1)  # (N,1,H,W) ==> (N,3,H,W)

    # inference mode configuration
    inference_mode = False
    total_images_so_far = save_all[1]
    if save_all[0]:  # True if we are in inference phase
        inference_mode = True
        limit=-1 #no limit
    # End inference mode configuration

    toPIL = transforms.ToPILImage()
    for i, img in enumerate(generated_images):
        if (i == limit): return
        generated_img = img.clone().detach().cpu()
        original_img = original_imgs[i].clone().detach().cpu()
        mask_img = generated_mask[i].clone().detach().cpu()
        true_mask_img = true_mask[i].clone().detach().cpu()
        imgs_cat = torch.cat((original_img, generated_img), 2)
        masks_cat = torch.cat((true_mask_img, mask_img), 2)
        img = torch.cat((imgs_cat, masks_cat), 1)
        img = toPIL(img)  # .numpy().transpose((1, 2, 0))
        if inference_mode:
            image_numbering = str(index+i+ (total_images_so_far-len(generated_images)))
        else:
            image_numbering = str(index) + '_' + str(i)
        img.save('./generatedImages_' + phase + '/' + image_numbering + 'generated.png')

def show(generated_imgs, original_imgs,masks, phase, index, save):
    # if not isinstance(torch_img,list):
    #     torch_img = [torch_img]
    if not masks.shape == original_imgs.shape:
        masks = masks.repeat(1,3,1,1)
    if not generated_imgs.shape == original_imgs.shape:
        generated_imgs.unsqueeze_(1) # (N,H,W) ==> (N,1,H,W)
        generated_imgs=generated_imgs.repeat(1,3,1,1) # (N,1,H,W) ==> (N,3,H,W)

    toPIL = transforms.ToPILImage()
    for i, img in enumerate(generated_imgs):
        if (i == 5): return
        generated_img = img.clone().detach().cpu()
        original_img = original_imgs[i].clone().detach().cpu()
        mask_img = masks[i].clone().detach().cpu()
        img = torch.cat((original_img, generated_img,mask_img), 2)
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

def blure_background_training_loop(n_epochs, optimizer, lamda, model, loss_fn, data_loader_dic, device,num_epochs):
    best_val_loss = 1000
    for epoch in range(0, n_epochs + 1):

        for phase in data_loader_dic.keys():
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            flag = True
            total_train_images = 0
            # TODO: the Loss here are normalized using np.mean() to get the average loss across all images. However,
            # TODO: last epoch may have less images hence, mean is not accurate
            loss_batches = []
            loss_l2_batches = []
            loss_grad_batches = []
            original_images_grad = []

            pbar = tqdm(data_loader_dic[phase], total=len(data_loader_dic[phase]))
            for X, intermediate, y in pbar:
                batch_size = len(X)
                total_train_images += batch_size

                model.train()
                X = X.to(device).float()
                intermediate = intermediate.to(device).float()  # intermediate is the mask with type of float
                y = y.to(device)  # this is the mask as Boolean
                ypred = model(X)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    loss_l2 = loss_fn(ypred, X) * lamda["l2"]
                    loss_grad = image_gradient(ypred) * lamda["grad"]

                    loss = loss_l2

                    loss_batches.append(loss.clone().detach().cpu().numpy())
                    loss_l2_batches.append(loss_l2.clone().detach().cpu().numpy())
                    loss_grad_batches.append(loss_grad.clone().detach().cpu().numpy())
                    original_images_grad.append(image_gradient(X).clone().detach().cpu().numpy())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                if flag:
                    show(ypred, X, intermediate, phase, index=100 + epoch, save=True)
                    flag = False

                # update the progress bar
                pbar.set_postfix({phase + ' Epoch': str(epoch) + "/" + str(num_epochs - 1),
                                  'Loss': np.mean(loss_batches),
                                  'L2': np.mean(loss_l2_batches),
                                  'grad': np.mean(loss_grad_batches),
                                  'original_images_grad': np.mean(original_images_grad)
                                  })
            if phase == 'val' and np.mean(loss_batches) < best_val_loss:
                print('best loss={} so far ...'.format(np.mean(loss_batches)))
                wandb.run.summary["best_epoch"] = epoch
                wandb.run.summary["val_loss"] = np.mean(loss_batches)
                best_val_loss = np.mean(loss_batches)

                print('saving a checkpoint')

            wandb.log({phase + "_loss": np.mean(loss_batches),
                       phase + "_L2": np.mean(loss_l2_batches), phase + "_grad": np.mean(loss_grad_batches),
                       phase + '_original_images_grad': np.mean(original_images_grad), "best_val_loss": best_val_loss,
                       phase + "_epoch": epoch},
                      step=epoch)

def blure_background_trainingMechanism_training_loop(n_epochs, optimizer, lamda, model, loss_fn,
                                                     data_loader_dic, device, num_epochs,switch_epoch):
    best_val_loss = 1000

    for epoch in range(0, n_epochs + 1):

        for phase in data_loader_dic.keys():
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            flag = True
            total_train_images = 0
            # TODO: the Loss here are normalized using np.mean() to get the average loss across all images. However,
            # TODO: last epoch may have less images hence, mean is not accurate
            loss_batches = []
            loss_l2_batches = []
            loss_grad_batches = []
            original_images_grad = []

            pbar = tqdm(data_loader_dic[phase], total=len(data_loader_dic[phase]))
            for X, intermediate, y in pbar:
                batch_size = len(X)
                total_train_images += batch_size

                model.train()
                X = X.to(device).float()
                intermediate = intermediate.to(device).float()  # intermediate is the mask with type of float
                y = y.to(device)  # this is the mask as Boolean
                ypred = model(X)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    loss_l2 = loss_fn(ypred, X) * lamda["l2"]
                    loss_grad = image_gradient(ypred) * lamda["grad"]

                    if epoch <= switch_epoch:
                        loss = loss_l2
                    else:
                        loss = loss_fn(torch.mul(ypred,intermediate), torch.mul(X,intermediate))*torch.sum(intermediate)


                    loss_batches.append(loss.clone().detach().cpu().numpy())
                    loss_l2_batches.append(loss_l2.clone().detach().cpu().numpy())
                    loss_grad_batches.append(loss_grad.clone().detach().cpu().numpy())
                    original_images_grad.append(image_gradient(X).clone().detach().cpu().numpy())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                if flag:
                    show(ypred, X, intermediate, phase, index=100 + epoch, save=True)
                    flag = False

                # update the progress bar
                pbar.set_postfix({phase + ' Epoch': str(epoch) + "/" + str(num_epochs - 1),
                                  'Loss': np.mean(loss_batches),
                                  'L2': np.mean(loss_l2_batches),
                                  'grad': np.mean(loss_grad_batches),
                                  'original_images_grad': np.mean(original_images_grad)
                                  })
            if epoch==switch_epoch:
                best_val_loss = np.mean(loss_batches)
            if phase == 'val' and np.mean(loss_batches) < best_val_loss:
                print('best loss={} so far ...'.format(np.mean(loss_batches)))
                wandb.run.summary["best_epoch"] = epoch
                wandb.run.summary["val_loss"] = np.mean(loss_batches)
                best_val_loss = np.mean(loss_batches)

                print('saving a checkpoint')

            wandb.log({phase + "_loss": np.mean(loss_batches),
                       phase + "_L2": np.mean(loss_l2_batches), phase + "_grad": np.mean(loss_grad_batches),
                       phase + '_original_images_grad': np.mean(original_images_grad),
                       "best_val_loss": best_val_loss,
                       phase + "_epoch": epoch},
                      step=epoch)
def two_stages_training_loop(num_epochs, optimizer, lamda, model, loss_fn_sum, data_loader_dic, device,switch_epoch):
    best_val_loss = 1000
    for epoch in range(0, num_epochs + 1):

        for phase in data_loader_dic.keys():
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            flag = True
            total_train_images = 0
            #TODO: the Loss here are normalized using np.mean() to get the average loss across all images. However,
            #TODO: last epoch may have less images hence, mean is not accurate
            loss_batches = []
            loss_l2_batches = []
            loss_grad_batches = []
            original_images_grad = []



            pbar = tqdm(data_loader_dic[phase], total=len(data_loader_dic[phase]))
            for X,intermediate, y in pbar:
                batch_size = len(X)
                total_train_images += batch_size


                model.train()
                X = X.to(device).float()
                intermediate = intermediate.to(device).float() #intermediate is the mask with type of float
                y = y.to(device) #this is the mask as Boolean
                ypred = model(X)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    if epoch==switch_epoch: #update the best_val_loss threshold
                        best_val_loss = 1000
                    if epoch < switch_epoch: # focus on minimizing ‖f-g‖^2
                        loss_l2 = loss_fn_sum(ypred,X)/X.numel()
                        loss_grad = color_gradient(ypred)
                        loss = loss_l2
                    else:  # move to stage 2 loss: ‖f-g * mask(polyp)‖^2 + ‖∇g *mask(1-polyp)‖^2
                        loss_l2 = torch.sum(torch.pow(torch.mul(ypred-X,intermediate),2))/torch.sum(intermediate)
                        gradients = color_gradient(ypred,'No reduction')
                        gradients_masked = torch.mul(gradients, 1-intermediate) #consider only background
                        loss_grad = torch.sum(torch.pow(gradients_masked,2))/torch.sum(1-intermediate)
                        loss = loss_grad*lamda['grad'] + loss_l2*lamda['l2']

                    loss_batches.append(loss.clone().detach().cpu().numpy())
                    loss_l2_batches.append(loss_l2.clone().detach().cpu().numpy())
                    loss_grad_batches.append(loss_grad.clone().detach().cpu().numpy())
                    original_images_grad.append(color_gradient(X).clone().detach().cpu().numpy())

                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                if flag:
                    show(ypred, X, intermediate, phase, index=100 + epoch, save=True)
                    flag=False

                # update the progress bar
                pbar.set_postfix({phase+' Epoch': str(epoch)+"/"+str(num_epochs-1),
                                  'Loss': np.mean(loss_batches),
                                  'L2': np.mean(loss_l2_batches),
                                  'grad': np.mean(loss_grad_batches),
                                  'original_images_grad': np.mean(original_images_grad)
                                  })
            if phase=='val' and np.mean(loss_batches) < best_val_loss:
                print('best loss={} so far ...'.format(np.mean(loss_batches)))
                wandb.run.summary["best_epoch"] = epoch
                wandb.run.summary["val_loss"] = np.mean(loss_batches)
                best_val_loss = np.mean(loss_batches)

                print('saving a checkpoint')


            wandb.log({phase+"_loss": np.mean(loss_batches),
                       phase+"_L2": np.mean(loss_l2_batches), phase+"_grad": np.mean(loss_grad_batches),
                       phase+'_original_images_grad': np.mean(original_images_grad),"best_val_loss":best_val_loss, phase+"_epoch": epoch},
                      step=epoch)
def three_stages_training_loop(num_epochs, optimizer, lamda, model, loss_dic, data_loader_dic, device,switch_epoch):
    best_val_loss = 1000
    best_val_iou = 0
    loss_fn_sum = loss_dic['generator']
    for epoch in range(0, num_epochs + 1):

        for phase in data_loader_dic.keys():
            iou = []

            if phase == 'train':
                model[0].train()  # Set model to training mode
                model[1].train()  # Set model to training mode
            else:
                model[0].eval()   # Set model to evaluate mode
                model[1].eval()  # Set model to evaluate mode

            flag = True
            total_train_images = 0
            #TODO: the Loss here are normalized using np.mean() to get the average loss across all images. However,
            #TODO: last epoch may have less images hence, mean is not accurate
            loss_batches = []
            loss_l2_batches = []
            loss_grad_batches = []
            loss_mask_batches = []
            iou_batches = []
            original_images_grad = []




            pbar = tqdm(data_loader_dic[phase], total=len(data_loader_dic[phase]))
            for X,intermediate, y in pbar:
                batch_size = len(X)
                total_train_images += batch_size



                X = X.to(device).float()
                intermediate = intermediate.to(device).float() #intermediate is the mask with type of float
                y = y.to(device)

                ypred = model[0](X)
                if epoch>=switch_epoch*2:
                    ymask = model[1](ypred)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    loss_mask = torch.zeros((1)).int()
                    if epoch==switch_epoch: #update the best_val_loss threshold
                        best_val_loss = 1000
                    if epoch==switch_epoch*2: #update the best_val_loss threshold for the segmentor
                        best_val_loss = 1000
                    if epoch < switch_epoch: # focus on minimizing ‖f-g‖^2
                        loss_l2 = loss_fn_sum(ypred,X)/X.numel()
                        loss_grad = color_gradient(ypred)
                        loss = loss_l2
                    else: #epoch >= switch_epoch and epoch <switch_epoch*2:  # move to stage 2 loss: ‖f-g * mask(polyp)‖^2 + ‖∇g *mask(1-polyp)‖^2
                        loss_l2 = torch.sum(torch.pow(torch.mul(ypred-X,intermediate),2))/torch.sum(intermediate)
                        gradients = color_gradient(ypred,'No reduction')
                        gradients_masked = torch.mul(gradients, 1-intermediate) #consider only background
                        loss_grad = torch.sum(torch.pow(gradients_masked,2))/torch.sum(1-intermediate)
                        loss = loss_grad*lamda['grad'] + loss_l2*lamda['l2']
                        if epoch>=switch_epoch*2:# move to stage 3 loss: ‖f-g * mask(polyp)‖^2 + ‖∇g *mask(1-polyp)‖^2 + BCEWithLoggits
                            bce = loss_dic['segmentor']
                            loss_mask = bce(ymask,y)
                            loss = loss + loss_mask
                            iou += IOU_class01(y, ymask)




                    loss_batches.append(loss.clone().detach().cpu().numpy())
                    loss_l2_batches.append(loss_l2.clone().detach().cpu().numpy())
                    loss_grad_batches.append(loss_grad.clone().detach().cpu().numpy())
                    loss_mask_batches.append(loss_mask.clone().detach().cpu().numpy())
                    iou_batches.append(iou)
                    original_images_grad.append(color_gradient(X).clone().detach().cpu().numpy())

                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                if flag: #this flag
                    flag = False
                    if epoch>=switch_epoch*2:
                        max, generated_mask = ymask.max(dim=1)
                        generated_mask = generated_mask.unsqueeze(dim=1)
                        show(ypred, X, generated_mask, phase, index=100 + epoch, save=True)
                    else:
                        show(ypred, X, intermediate, phase, index=100 + epoch, save=True)


                # update the progress bar
                pbar.set_postfix({phase+' Epoch': str(epoch)+"/"+str(num_epochs-1),
                                  'Loss': np.mean(loss_batches),
                                  'L2': np.mean(loss_l2_batches),
                                  'grad': np.mean(loss_grad_batches),
                                  'BCE_loss': np.mean(loss_mask_batches),
                                  'iou':np.mean(iou_batches),
                                  'original_images_grad': np.mean(original_images_grad),
                                  'best_val_loss':best_val_loss,
                                  'best_val_iou':best_val_iou
                                  })
            if phase=='val':
                if np.mean(loss_batches) < best_val_loss:
                    print('best loss={} so far ...'.format(np.mean(loss_batches)))
                    wandb.run.summary["best_epoch"] = epoch
                    wandb.run.summary["best_val_loss"] = np.mean(loss_batches)
                    best_val_loss = np.mean(loss_batches)

                    print('saving a checkpoint')
                if np.mean(iou_batches) > best_val_iou:
                    wandb.run.summary["best_val_iou"] = np.mean(iou_batches)
                    best_val_iou = np.mean(iou_batches)
                    print('best val_iou')




            wandb.log({phase+"_loss": np.mean(loss_batches),
                       phase+"_L2": np.mean(loss_l2_batches), phase+"_grad": np.mean(loss_grad_batches),
                       phase+'_BCE_loss':np.mean(loss_mask_batches),phase+'_iou':np.mean(iou_batches),
                       phase+'_original_images_grad': np.mean(original_images_grad),"best_val_loss":best_val_loss,
                       'best_val_iou':best_val_iou ,phase+"_epoch": epoch},
                      step=epoch)

def Dl_TOV_training_loop(num_epochs, optimizer, lamda, model, loss_dic, data_loader_dic,
                         device,switch_epoch,colab_dir, model_name, save_generator_checkpoints):
    best_loss = {k: 1000 for k in data_loader_dic.keys()}
    best_iou = {k: 0 for k in data_loader_dic.keys()}
    best_iou_epoch = -1
    loss_fn_sum = loss_dic['generator']
    #this variable to track the performance of the generator
    # this number is created according to the best gen loss at
    # Denoising_trainCVC_testKvasir_Exp4_IncludeAugX_hue_avgV2_unet_Lraspp
    # best_val_generator_loss=0.005
    best_val_generator_loss=1000
    for epoch in range(0, num_epochs + 1):

        for phase in data_loader_dic.keys():
            if phase.find('test')>=0 and best_iou_epoch != epoch: #skip testing if no better iou val achieved
                continue
            if phase == 'train':
                model.train()
                # model[0].train()  # Set model to training mode
                # model[1].train()  # Set model to training mode
            else:
                model.eval()
                # model[0].eval()  # Set model to evaluate mode

            flag = True #flag for showing first batch
            total_train_images = 0
            # TODO: the Loss here are normalized using np.mean() to get the average loss across all images. However,
            # TODO: last epoch may have less images hence, mean is not accurate
            loss_batches = []
            loss_l2_batches = []
            loss_grad_batches = []
            loss_mask_batches = []
            iou_batches = np.array([])
            iou_background_batches = np.array([])
            # metrics_polyp = []
            # metrics_background = []
            original_images_grad = []
            generated_images_grad = []

            all_true_maskes_torch = []
            all_pred_maskes_torch = []

            pbar = tqdm(data_loader_dic[phase], total=len(data_loader_dic[phase]))
            for X, intermediate, original_masks in pbar:
                batch_size = len(X)
                total_train_images += batch_size

                X = X.to(device).float()
                intermediate = intermediate.to(device).float()  # intermediate is the mask with type of float
                original_masks = original_masks.to(device)#this is 2 channels mask

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                #Generate polyp images and masks
                    if model_name.find('GenSeg') >= 0:
                        results = model(X, phase, original_masks)
                        generated_images, generated_masks, original_masks = results
                    else:  # the old version code i.e., other than GenSeg_IncludeX models
                        generated_images = model[0](X)
                        generated_X = generated_images.clone().detach()
                        if epoch >= switch_epoch[1]:
                            generated_masks = model[1](generated_X)

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
                        gradients = color_gradient(generated_images, 'No reduction', model_name)
                        gradients_masked = torch.mul(gradients, 1 - intermediate)  # consider only background
                        loss_grad = torch.sum(torch.pow(gradients_masked, 2)) / torch.sum(1 - intermediate)
                        # if epoch >= switch_epoch[1]:#increse the polyp reconstruction loss to balance it with seg loss
                        #     loss_l2 = loss_l2 * lamda['l2']
                        if epoch < switch_epoch[1]:  # if we are in stage 2 calculate don't include lamda['l2']
                            loss = loss_grad * lamda['grad'] + loss_l2
                        if epoch >= switch_epoch[1]:  # move to stage 3 loss: ‖f-g * mask(polyp)‖^2 + ‖∇g *mask(1-polyp)‖^2 + BCEWithLoggits
                            bce = loss_dic['segmentor']
                            loss_mask = bce(generated_masks, original_masks)
                            loss = loss_grad * lamda['grad'] + loss_l2 * lamda['l2'] + loss_mask

                            # iou = IOU_class01(original_masks, generated_masks)
                            # iou is numpy array for each image
                            iou = calculate_metrics_torch(true=original_masks,pred=generated_masks,metrics='jaccard')
                            iou_background = calculate_metrics_torch(true=original_masks,pred=generated_masks,
                                                                     metrics='jaccard',ROI='background')
                            #store all the true and pred masks
                            if len(all_true_maskes_torch)==0:
                                all_true_maskes_torch = original_masks.clone().detach()
                                all_pred_maskes_torch = generated_masks.clone().detach()
                            else:
                                all_true_maskes_torch = torch.cat((all_true_maskes_torch,original_masks.clone().detach()))
                                all_pred_maskes_torch = torch.cat((all_pred_maskes_torch,generated_masks.clone().detach()))

                    loss_batches.append(loss.clone().detach().cpu().numpy())
                    loss_l2_batches.append(loss_l2.clone().detach().cpu().numpy())
                    loss_grad_batches.append(loss_grad.clone().detach().cpu().numpy())
                    loss_mask_batches.append(loss_mask.clone().detach().cpu().numpy())
                    iou_batches=np.append(iou_batches,iou)
                    iou_background_batches=np.append(iou_background_batches,iou_background)
                    original_images_grad.append(color_gradient(X).clone().detach().cpu().numpy())
                    generated_images_grad.append(color_gradient(generated_images).clone().detach().cpu().numpy())
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                if flag:  # this flag
                    flag = False
                    true_mask = intermediate
                    if epoch >= switch_epoch[1]:#stage 3
                        max, generated_mask = generated_masks.max(dim=1)
                        generated_mask = generated_mask.unsqueeze(dim=1)
                        show2(generated_images, X, generated_mask,true_mask, phase, index=100 + epoch, save=True)
                    else: #stage 1 and 2
                        generated_mask = torch.zeros(generated_images.shape)
                        show2(generated_images, X, generated_mask,true_mask, phase, index=100 + epoch, save=True)

                # update the progress bar
                pbar.set_postfix({phase + ' Epoch': str(epoch) + "/" + str(num_epochs - 1),
                                  'polypIOU': iou_batches.mean(),
                                  'best_val_iou': best_iou['val'],
                                  'best_test_iou': best_iou['test1'],
                                  'mIOU': np.mean((iou_batches+iou_background_batches)/2),
                                  'Loss': np.mean(loss_batches),
                                  'L2': np.mean(loss_l2_batches),
                                  'grad': np.mean(loss_grad_batches),
                                  'BCE_loss': np.mean(loss_mask_batches),
                                  'original_images_grad': np.mean(original_images_grad),
                                  })

            # !!!! calculate metrics for all images. The results are dictionary
            mean_metrics_polyp = calculate_metrics_torch(all_true_maskes_torch, all_pred_maskes_torch,
                                                         reduction='mean',cloned_detached=True)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            if phase != 'train':
                if np.mean(loss_batches) < best_loss[phase]:
                    print('best {} loss={} so far ...'.format(phase,np.mean(loss_batches)))
                    wandb.run.summary["best_{}_loss_epoch".format(phase)] = epoch
                    wandb.run.summary["best_{}_loss".format(phase)] = np.mean(loss_batches)
                    best_loss[phase] = np.mean(loss_batches)
                    if phase=='val':
                        print('better validation loss')
                if phase=='val' :
                    # if the generator is getting better save a checkpoint for the generator
                    generator_loss = np.mean(loss_l2_batches) + np.mean(loss_grad_batches)

                    if best_val_generator_loss > generator_loss:
                        print('saving a checkpoint for the best generator '
                              '\n previously={} and now={}'.format(best_val_generator_loss,generator_loss))
                        saving_checkpoint(epoch, model, optimizer,
                                          generator_loss, generator_loss,
                                          generator_loss, generator_loss,
                                          colab_dir, model_name, save_generator_checkpoints=True)
                        best_val_generator_loss = generator_loss
                        
                    #if Polyp mean is getting better
                    if mean_metrics_polyp['jaccard'] > best_iou[phase]:
                        wandb.run.summary["best_{}_iou".format(phase)] = np.mean(iou_batches)
                        wandb.run.summary["best_{}_iou_epoch".format(phase)] = epoch
                        best_iou[phase] = mean_metrics_polyp['jaccard'] #Jaccard/IOU of polyp
                        best_loss[phase] = np.mean(loss_batches)
                        best_iou_epoch = epoch
                        print('best val_iou')
                        print('testing on a test set....\n')


                if phase.find('test')>=0:#if we reach inside this, it means we achieved a better val iou
                    print('saving a checkpoint')
                    best_iou[phase] = mean_metrics_polyp['jaccard']
                    best_loss[phase] = np.mean(loss_batches)
                    if phase == 'test1': #I want to save the model weights only once, not for every test set
                        saving_checkpoint(epoch, model, optimizer,
                                      best_loss['val'], best_loss['test1'],
                                      best_iou['val'], best_iou['test1'],
                                      colab_dir, model_name)
                # calculate summary results and store them in results
                if best_iou_epoch == epoch:#calculate metrics for val and test if it is the best epoch
                    metrics_dic_background = calculate_metrics_torch(all_true_maskes_torch, all_pred_maskes_torch,
                                                         reduction='mean',cloned_detached=True,ROI='background')
                    metrics_dic_polyp = mean_metrics_polyp

                    metrics_mMetrics_dic = {metric:(metrics_dic_polyp[metric]+metrics_dic_background[metric])/2
                                            for metric in metrics_dic_background.keys()}

                    print(phase,':',metrics_dic_polyp)
                    wandb.run.summary["dict_{}".format(phase)] = metrics_dic_polyp
                    pandas.DataFrame.from_dict({'Polyp':metrics_dic_polyp,'Background': metrics_dic_background,
                                                'Mean':metrics_mMetrics_dic}).transpose().to_excel(colab_dir + "/results/{}_summary_report.xlsx".format(phase))

            wandb.log({phase + "_loss": np.mean(loss_batches),
                       phase + "_L2": np.mean(loss_l2_batches), phase + "_grad": np.mean(loss_grad_batches),
                       phase + '_BCE_loss': np.mean(loss_mask_batches), phase + '_iou': np.mean(iou_batches),
                       phase + '_original_images_grad': np.mean(original_images_grad), phase + '_generated_images_grad': np.mean(generated_images_grad),
                       "best_val_loss": best_loss['val'],
                       'best_val_iou': best_iou['val'], phase + "_epoch": epoch},
                      step=epoch)




def literature_training_loop(num_epochs, optimizer, lamda, model, BCE, data_loader_dic, device,colab_dir, model_name):
    best_loss = {k: 1000 for k in data_loader_dic.keys()}
    best_iou = {k: 0 for k in data_loader_dic.keys()}
    best_iou_epoch = -1
    for epoch in range(0, num_epochs + 1):

        for phase in data_loader_dic.keys():
            if phase == 'test' and best_iou_epoch!=epoch: #skip testing if no better iou val achieved
                continue
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            flag = True #flag for showing the first batch
            total_train_images = 0
            # TODO: the Loss here are normalized using np.mean() to get the average loss across all images. However,
            # TODO: last epoch may have less images hence, mean is not accurate
            loss_batches = []
            iou_batches = []

            pbar = tqdm(data_loader_dic[phase], total=len(data_loader_dic[phase]))
            for X, intermediate, y in pbar:
                batch_size = len(X)
                total_train_images += batch_size

                X = X.to(device).float()
                intermediate = intermediate.to(device).float()  # intermediate is the mask with type of float
                y = y.to(device)

                # Feed forward
                ymask = model(X)
                # clear any gradient before this batch
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    loss = BCE(ymask, y)
                    iou = IOU_class01(y, ymask)

                    loss_batches.append(loss.clone().detach().cpu().numpy())
                    iou_batches.append(iou)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                if flag:  # this flag
                    flag = False
                    max, generated_mask = ymask.max(dim=1)
                    generated_mask = generated_mask.unsqueeze(dim=1)
                    show(y[:,1,:,:], X, generated_mask, phase, index=100 + epoch, save=True)

                # update the progress bar
                pbar.set_postfix({phase + ' Epoch': str(epoch) + "/" + str(num_epochs - 1),
                                  'Loss': np.mean(loss_batches),
                                  'iou': np.mean(iou_batches),
                                  'best_val_loss': best_loss['val'],
                                  'best_val_iou': best_iou['val'],
                                  'best_test_loss': best_loss['test'],
                                  'best_test_iou': best_iou['test']
                                  })
            if phase != 'train':
                if np.mean(loss_batches) < best_loss[phase]:
                    print('best {} loss={} so far ...'.format(phase, np.mean(loss_batches)))
                    wandb.run.summary["best_{}_loss_epoch".format(phase)] = epoch
                    wandb.run.summary["best_{}_loss".format(phase)] = np.mean(loss_batches)
                    best_loss[phase] = np.mean(loss_batches)
                    if phase == 'val':
                        print('better validation loss')
                if phase == 'val':
                    if np.mean(iou_batches) > best_iou[phase]:
                        wandb.run.summary["best_{}_iou".format(phase)] = np.mean(iou_batches)
                        wandb.run.summary["best_{}_iou_epoch".format(phase)] = epoch
                        best_iou[phase] = np.mean(iou_batches)
                        best_loss[phase] = np.mean(loss_batches)
                        best_iou_epoch = epoch
                        print('best val_iou')
                        print('testing on a test set....\n')
                if phase == 'test':  # if we reach inside this, it means we achieved a better val iou
                    print('saving a checkpoint')
                    best_iou['test'] = np.mean(iou_batches)
                    best_loss['test'] = np.mean(loss_batches)
                    saving_checkpoint(epoch, model, optimizer,
                                      best_loss['val'], best_loss['test'],
                                      best_iou['val'], best_iou['test'],
                                      colab_dir, model_name)

            wandb.log({phase + "_loss": np.mean(loss_batches),
                       phase + '_iou': np.mean(iou_batches),
                       "best_val_loss": best_loss['val'],
                       'best_val_iou': best_iou['val'], phase + "_epoch": epoch},
                      step=epoch)
def pefect_filter_training_loop(num_epochs, optimizer, lamda, model, loss_fn,
                  data_loader_dic, device, switch_epoch):
    best_val_loss = 1000
    for epoch in range(0, num_epochs + 1):

        for phase in data_loader_dic.keys():
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            flag = True
            total_train_images = 0
            # TODO: the Loss here are normalized using np.mean() to get the average loss across all images. However,
            # TODO: last epoch may have less images hence, mean is not accurate
            loss_batches = []
            loss_l2_batches = []
            loss_grad_batches = []
            original_images_grad = []

            pbar = tqdm(data_loader_dic[phase], total=len(data_loader_dic[phase]))
            for X, intermediate, y in pbar:
                batch_size = len(X)
                total_train_images += batch_size

                model.train()
                X = X.to(device).float()
                intermediate = intermediate.to(device).float()  # intermediate is the mask with type of float
                y = y.to(device)  # this is the mask as Boolean
                ypred = model(X)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    #ypred = (N,1,H,W)
                    loss = loss_fn(ypred.squeeze(),intermediate.squeeze())
                    loss_l2 = torch.Tensor((1))
                    loss_grad = torch.Tensor((1))
                    loss_batches.append(loss.clone().detach().cpu().numpy())
                    loss_l2_batches.append(loss_l2.clone().detach().cpu().numpy())
                    loss_grad_batches.append(loss_grad.clone().detach().cpu().numpy())
                    original_images_grad.append(image_gradient(X).clone().detach().cpu().numpy())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                if flag:
                    kernel_81 = model.models[2].weight
                    #resize
                    kernel_81 = nn.functional.interpolate(kernel_81,size=intermediate.shape[2:], mode='bilinear')
                    #ypred is mask (N,1,H,W), original imgs here are the original mask. masks here is the kernel
                    #all of them have the same dimensions except the kernel it has three dim
                    show_filter(generated_masks=ypred, original_masks=intermediate,kernel3D=kernel_81,
                         phase=phase, index=100 + epoch, save=True)
                    flag = False

                # update the progress bar
                pbar.set_postfix({phase + ' Epoch': str(epoch) + "/" + str(num_epochs - 1),
                                  'Loss': np.mean(loss_batches),
                                  'L2': np.mean(loss_l2_batches),
                                  'grad': np.mean(loss_grad_batches),
                                  'original_images_grad': np.mean(original_images_grad)
                                  })
            if phase == 'val' and np.mean(loss_batches) < best_val_loss:
                print('best loss={} so far ...'.format(np.mean(loss_batches)))
                wandb.run.summary["best_epoch"] = epoch
                wandb.run.summary["val_loss"] = np.mean(loss_batches)
                best_val_loss = np.mean(loss_batches)

                print('saving a checkpoint')

            wandb.log({phase + "_loss": np.mean(loss_batches),
                       phase + "_L2": np.mean(loss_l2_batches), phase + "_grad": np.mean(loss_grad_batches),
                       phase + '_original_images_grad': np.mean(original_images_grad), "best_val_loss": best_val_loss,
                       phase + "_epoch": epoch},
                      step=epoch)

def show_filter(generated_masks, original_masks,kernel3D, phase, index, save):
    # if not isinstance(torch_img,list):
    #     torch_img = [torch_img]
    kernel3D = kernel3D.squeeze() # (1,3,H,W) ==> (3,H,W)
    if not original_masks.shape == kernel3D.shape:
        original_masks = original_masks.repeat(1,3,1,1)
    if not generated_masks.shape == kernel3D.shape:
        generated_masks.unsqueeze_(dim=1)
        generated_masks = generated_masks.repeat(1,3,1,1)
    kernel_img = kernel3D.clone().detach()
    # normalize kernel_img
    kernel_img = (kernel_img - torch.min(kernel_img))/(torch.max(kernel_img) - torch.min(kernel_img))
    kernel_img =kernel_img.cpu()
    toPIL = transforms.ToPILImage()
    for i, img in enumerate(generated_masks):
        if (i == 5): return
        generated_mask = img.clone().detach().cpu()
        original_mask = original_masks[i].clone().detach().cpu()

        img = torch.cat((original_mask, generated_mask,kernel_img), 2)
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

def Dl_TOV_inference_loop(num_epochs, optimizer, lamda, model, loss_dic, data_loader_dic, device,checkpoint):
    best_loss = {k: 1000 for k in data_loader_dic.keys()}
    best_iou = {k: 0 for k in data_loader_dic.keys()}
    best_iou_epoch = -1
    loss_fn_sum = loss_dic['generator']

    for epoch in range(0, num_epochs + 1):

        for phase in data_loader_dic.keys():
            if phase == 'test' and best_iou_epoch != epoch:  # skip testing if no better iou val achieved
                continue
            if phase == 'train':
                model[0].train()  # Set model to training mode
                model[1].train()  # Set model to training mode
            else:
                model[0].eval()  # Set model to evaluate mode
                model[1].eval()  # Set model to evaluate mode

            flag = True  # flag for showing first batch
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
                original_masks = original_masks.to(device)  # this is 2 channels mask

                generated_images = model[0](X)
                generated_masks = model[1](generated_images)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    loss_mask = torch.zeros((1)).int()
                    iou = 0
                         # move to stage 2 loss: ‖f-g * mask(polyp)‖^2 + ‖∇g *mask(1-polyp)‖^2
                    loss_l2 = torch.sum(torch.pow(torch.mul(generated_images - X, intermediate), 2)) / torch.sum(
                        intermediate)
                    gradients = color_gradient(generated_images, 'No reduction')
                    gradients_masked = torch.mul(gradients, 1 - intermediate)  # consider only background
                    loss_grad = torch.sum(torch.pow(gradients_masked, 2)) / torch.sum(1 - intermediate)
                    #stage 3 loss: ‖f-g * mask(polyp)‖^2 + ‖∇g *mask(1-polyp)‖^2 + BCEWithLoggits
                    bce = loss_dic['segmentor']
                    loss_mask = bce(generated_masks, original_masks)
                    loss = loss_grad * lamda['grad'] + loss_l2 * lamda['l2'] + loss_mask
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
                # if flag:  # this flag
                #     # flag = False
                true_mask = intermediate
                max, generated_mask = generated_masks.max(dim=1)
                generated_mask = generated_mask.unsqueeze(dim=1)
                show2(generated_images, X, generated_mask, true_mask, phase, index=100 + total_train_images-batch_size, save=True,limit=-1)


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
                    print('best {} loss={} so far ...'.format(phase, np.mean(loss_batches)))
                    wandb.run.summary["best_{}_loss_epoch".format(phase)] = epoch
                    wandb.run.summary["best_{}_loss".format(phase)] = np.mean(loss_batches)
                    best_loss[phase] = np.mean(loss_batches)
                    if phase == 'val':
                        print('better validation loss')
                if np.mean(iou_batches) > best_iou[phase]:
                    wandb.run.summary["best_{}_iou".format(phase)] = np.mean(iou_batches)
                    wandb.run.summary["best_{}_iou_epoch".format(phase)] = epoch
                    best_iou[phase] = np.mean(iou_batches)
                    if phase == 'val':
                        best_iou_epoch = epoch
                        print('best val_iou')
                        print('testing on a test set....\n')

            wandb.log({phase + "_loss": np.mean(loss_batches),
                       phase + "_L2": np.mean(loss_l2_batches), phase + "_grad": np.mean(loss_grad_batches),
                       phase + '_BCE_loss': np.mean(loss_mask_batches), phase + '_iou': np.mean(iou_batches),
                       phase + '_original_images_grad': np.mean(original_images_grad),
                       "best_val_loss": best_loss['val'],
                       'best_val_iou': best_iou['val'], phase + "_epoch": epoch},
                      step=epoch)

def Dl_TOV_IncludeX_loop(num_epochs, optimizer, lamda, model, loss_dic, data_loader_dic, device,switch_epoch,colab_dir, model_name):
    best_loss = {k: 1000 for k in data_loader_dic.keys()}
    best_iou = {k: 0 for k in data_loader_dic.keys()}
    best_iou_epoch = -1
    loss_fn_sum = loss_dic['generator']
    for epoch in range(0, num_epochs + 1):

        for phase in data_loader_dic.keys():
            if phase == 'test' and best_iou_epoch!=epoch: #skip testing if no better iou val achieved
                continue
            if phase == 'train':
                model[0].train()  # Set model to training mode
                model[1].train()  # Set model to training mode
            else:
                model[0].eval()  # Set model to evaluate mode
                model[1].eval()  # Set model to evaluate mode

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

                generated_images = model[0](X)
                generated_X = generated_images.clone().detach()
                if epoch >= switch_epoch[1]:
                    # concatenate original images as well as generated images
                    input = cat_split([generated_X, X])
                    output_masks = model[1](input)
                    [g_masks, x_masks] = cat_split(output_masks)
                    generated_masks = g_masks*0.6 + x_masks*0.4

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
                            loss = loss_grad * lamda['grad'] + loss_l2
                        if epoch >= switch_epoch[1]:  # move to stage 3 loss: ‖f-g * mask(polyp)‖^2 + ‖∇g *mask(1-polyp)‖^2 + BCEWithLoggits
                            bce = loss_dic['segmentor']
                            loss_mask = bce(generated_masks, original_masks)
                            loss = loss_grad * lamda['grad'] + loss_l2 * lamda['l2'] + loss_mask
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
                    true_mask = intermediate
                    if epoch >= switch_epoch[1]:#stage 3
                        max, generated_mask = generated_masks.max(dim=1)
                        generated_mask = generated_mask.unsqueeze(dim=1)
                        show2(generated_images, X, generated_mask,true_mask, phase, index=100 + epoch, save=True)
                    else: #stage 1 and 2
                        generated_mask = torch.zeros(generated_images.shape)
                        show2(generated_images, X, generated_mask,true_mask, phase, index=100 + epoch, save=True)

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
                if phase == 'val':
                    if np.mean(iou_batches) > best_iou[phase]:
                        wandb.run.summary["best_{}_iou".format(phase)] = np.mean(iou_batches)
                        wandb.run.summary["best_{}_iou_epoch".format(phase)] = epoch
                        best_iou[phase] = np.mean(iou_batches)
                        best_loss[phase] = np.mean(loss_batches)
                        best_iou_epoch = epoch
                        print('best val_iou')
                        print('testing on a test set....\n')
                if phase == 'test':  # if we reach inside this, it means we achieved a better val iou
                    print('saving a checkpoint')
                    best_iou['test'] = np.mean(iou_batches)
                    best_loss['test'] = np.mean(loss_batches)
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

def cat_split(tensor_s):
    split=True
    if isinstance(tensor_s,list):#if list, means we need to concat
        return torch.cat(tensor_s,dim=0)
    else: # or split
        return tensor_s.chunk(chunks=2)

def Dl_TOV_IncludeXV2_loop(num_epochs, optimizer, lamda, model, loss_dic, data_loader_dic, device,switch_epoch,colab_dir, model_name):
    '''In the training, X and G are different images. Val/Test the highest probability should be selected for each
    pixel'''
    best_loss = {k: 1000 for k in data_loader_dic.keys()}
    best_iou = {k: 0 for k in data_loader_dic.keys()}
    best_iou_epoch = -1
    loss_fn_sum = loss_dic['generator']
    for epoch in range(0, num_epochs + 1):

        for phase in data_loader_dic.keys():
            if phase == 'test' and best_iou_epoch!=epoch: #skip testing if no better iou val achieved
                continue
            if phase == 'train':
                model[0].train()  # Set model to training mode
                model[1].train()  # Set model to training mode
            else:
                model[0].eval()  # Set model to evaluate mode
                model[1].eval()  # Set model to evaluate mode

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

                generated_images = model[0](X)
                generated_X = generated_images.clone().detach()
                if epoch >= switch_epoch[1]:
                    # concatenate original images as well as generated images
                    input = cat_split([generated_X, X])
                    output_masks = model[1](input)
                    [generated_masks, x_masks] = cat_split(output_masks)



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
                            loss = loss_grad * lamda['grad'] + loss_l2
                        if epoch >= switch_epoch[1]:  # move to stage 3 loss: ‖f-g * mask(polyp)‖^2 + ‖∇g *mask(1-polyp)‖^2 + BCEWithLoggits
                            bce = loss_dic['segmentor']
                            if phase=='train':
                                loss_generated_mask = bce(generated_masks, original_masks)
                                loss_x_mask = bce(x_masks, original_masks)
                                loss_mask = loss_generated_mask*0.6 + loss_x_mask*0.4
                                iou = IOU_class01(original_masks, generated_masks)
                            else:
                                generated_X_masks_stacked = torch.stack((generated_masks,x_masks), dim=2)
                                unified_masks, _ = generated_X_masks_stacked.max(dim=2)
                                loss_mask = bce(generated_masks, original_masks)
                                iou = IOU_class01(original_masks, unified_masks)

                            loss = loss_grad * lamda['grad'] + loss_l2 * lamda['l2'] + loss_mask


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
                    true_mask = intermediate
                    if epoch >= switch_epoch[1]:#stage 3
                        max, generated_mask = generated_masks.max(dim=1)
                        generated_mask = generated_mask.unsqueeze(dim=1)
                        show2(generated_images, X, generated_mask,true_mask, phase, index=100 + epoch, save=True)
                    else: #stage 1 and 2
                        generated_mask = torch.zeros(generated_images.shape)
                        show2(generated_images, X, generated_mask,true_mask, phase, index=100 + epoch, save=True)

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
                if phase == 'val':
                    if np.mean(iou_batches) > best_iou[phase]:
                        wandb.run.summary["best_{}_iou".format(phase)] = np.mean(iou_batches)
                        wandb.run.summary["best_{}_iou_epoch".format(phase)] = epoch
                        best_iou[phase] = np.mean(iou_batches)
                        best_loss[phase] = np.mean(loss_batches)
                        best_iou_epoch = epoch
                        print('best val_iou')
                        print('testing on a test set....\n')
                if phase == 'test':  # if we reach inside this, it means we achieved a better val iou
                    print('saving a checkpoint')
                    best_iou['test'] = np.mean(iou_batches)
                    best_loss['test'] = np.mean(loss_batches)
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
