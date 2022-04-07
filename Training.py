import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch import nn
from torchvision import transforms

def image_gradient(images):
    device = torch.device('cuda:0')
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


def training_loop(n_epochs, optimizer, lamda, model, loss_fn, data_loader_dic, device,num_epochs):
    best_val_loss = 100
    for epoch in range(0, n_epochs + 1):

        for phase in ['train', 'val','test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            flag = True
            total_train_images = 0
            #TODO: the Loss here are normalized using np.mean() to get the average loss across all images
            loss_batches = []
            loss_l2_batches = []
            loss_grad_batches = []
            original_images_grad = []



            pbar = tqdm(data_loader_dic[phase], total=len(data_loader_dic[phase]))
            for X,intermediate, y in pbar:
                batch_size = len(X)
                total_train_images += batch_size

                # torch.cuda.empty_cache()
                model.train()
                X = X.to(device).float()
                intermediate = intermediate.to(device).float() #intermediate is the mask with type of float
                y = y.to(device) #this is the mask as Boolean
                ypred = model(X)

                optimizer.zero_grad()
                # show(X)
                # image_gradient(X)
                # Calculating the loss starts here
                with torch.set_grad_enabled(phase == 'train'):
                    loss_l2 = loss_fn(ypred, X)*lamda["l2"]
                    loss_grad = image_gradient(ypred)*lamda["grad"]

                    loss = loss_l2

                    # if (loss.item() <= 0.01):
                    #     scaler += 10
                    #     print("scaler is used to increase the loss=", scaler)

                    loss_batches.append(loss.clone().detach().cpu().numpy())
                    loss_l2_batches.append(loss_l2.clone().detach().cpu().numpy())
                    loss_grad_batches.append(loss_grad.clone().detach().cpu().numpy())
                    original_images_grad.append(image_gradient(X).clone().detach().cpu().numpy())

                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                if flag:
                    show(ypred, X,phase, index=100 + epoch, save=True)
                    flag=False

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

def denoising_loss(created_images, original_images):
    alpha = torch.sum(torch.pow(created_images - original_images, 2)) / (
            original_images.shape[-1] * original_images.shape[-2])
    beta = 0.1 * image_gradient(created_images)
    if alpha > 1000 or beta > 100:
        print((original_images.shape[-1] * original_images.shape[-2]))

    total = beta  # alpha + beta
    print(alpha, beta)
    return total
