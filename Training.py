import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch import nn
from torchvision import transforms
from My_losses import *

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


def show(generated_imgs, original_imgs,masks, phase, index, save):
    # if not isinstance(torch_img,list):
    #     torch_img = [torch_img]
    masks = masks.repeat(1,3,1,1)
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
                        loss_l2 = loss_fn_sum(ypred,X)/torch.numel(X)
                        loss_grad = image_gradient_noSqrt(ypred)
                        loss = loss_l2
                    else:  # move to stage 2 loss: ‖f-g * mask(polyp)‖^2 + ‖∇g *mask(1-polyp)‖^2
                        loss_l2 = torch.sum(torch.pow(torch.mul(ypred-X,intermediate),2))/torch.sum(intermediate)
                        gradients = image_gradient_noSqrt(ypred,'No reduction')
                        gradients_masked = torch.mul(gradients, 1-intermediate) #consider only background
                        loss_grad = torch.sum(torch.pow(gradients_masked,2))/torch.sum(1-intermediate)
                        loss = loss_grad + loss_l2

                    loss_batches.append(loss.clone().detach().cpu().numpy())
                    loss_l2_batches.append(loss_l2.clone().detach().cpu().numpy())
                    loss_grad_batches.append(loss_grad.clone().detach().cpu().numpy())
                    original_images_grad.append(image_gradient_noSqrt(X).clone().detach().cpu().numpy())

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

                    loss = loss_fn(ypred,intermediate)
                    loss_l2 = torch.Tensor((1))
                    loss_grad = torch.Tensor((1))
                    loss_batches.append(loss.clone().detach().cpu().numpy())
                    loss_l2_batches.append(loss_l2.clone().detach().cpu().numpy())
                    loss_grad_batches.append(loss_grad.clone().detach().cpu().numpy())
                    original_images_grad.append(image_gradient_noSqrt(X).clone().detach().cpu().numpy())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                if flag:
                    kernel_243 = model.models[2].weight.squeeze()
                    #resize
                    kernel_243 = nn.functional.interpolate(kernel_243,size=intermediate.shape[1:], mode='bilinear')
                    show(ypred, intermediate,kernel_243, phase, index=100 + epoch, save=True)
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