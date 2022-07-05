import torch
import glob
import numpy as np
from pprint import pprint
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import collections

def getLoadersBySetName(dataset_name, data_C,target_img_size, train_val_ratio=0,batch_size=7,shuffle=False,):
    #input should be dataset_name="CVC-ClinicDB",  data_C="data_C1", split_ratio=0.5
    #output dataloader1 OR dataloader1, dataloader2 if split_ratio != 0 (i.e., we want train/val dataloaders)
    databases = {'EndoCV': "/content/trainData_EndoCV2021_5_Feb2021",
                 'CVC_ClinicDB': '/content/CVC-ClinicDB',
                 'Kvasir_Seg': '/content/Kvasir-SEG',
                 'ETIS_Larib':'/content/ETIS-LaribPolypDB'}
    #if the dataset_info is list of tuple, then it will be combined before splitting by the function
    #getDataloadersDic(). Good!!
    if not isinstance(data_C, list):#
        data_C = [data_C]

    dataset_info = []
    for child_dir in data_C:#Create a list of dataset_info
        C = child_dir.split('_')[-1]
        imageDir = 'images_'+C #images_C1 or C2 ... etc
        maskDir = 'mask_'+C
        dataset_info.append((databases[dataset_name],
                             child_dir, imageDir, maskDir, target_img_size))

    dataloder_info = (train_val_ratio, batch_size, shuffle)
    Dataloaders_dic = getDataloadersDic(dataset_info, dataloder_info)

    dataloader1 = Dataloaders_dic['train']
    dataloader2 = Dataloaders_dic['val']
    if train_val_ratio==0:
        dataloader1 = None

    return dataloader1, dataloader2


def getDataloadersDic(dataset_info, dataloder_info):
    datasets_list = []
    if not isinstance(dataset_info, list):
        dataset_info = [dataset_info]

    for i, info in enumerate(dataset_info):
        datasets_list.append(SegDataset(*info))

    dataset = ConcatDataset(datasets_list)
    train_val_ratio, batchSize, shuffle = dataloder_info
    trainDataset, valDataset = trainTestSplit(dataset, train_val_ratio)
    # -------------Dataloader configurations---------------------
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=shuffle, drop_last=False)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=shuffle, drop_last=False)
    dataloader_dic = {'train': trainLoader, 'val': valLoader}
    return dataloader_dic


class SegDataset(Dataset):
    def __init__(self, parentDir, dataset_name, imageDir, maskDir, targetSize, augmentation=None, load_to_RAM=False):
        self.imageList = sorted(glob.glob("/".join((parentDir, dataset_name, imageDir, '/*'))), key=deleteTail)
        self.maskList = sorted(glob.glob("/".join((parentDir, dataset_name, maskDir, '/*'))), key=deleteTail)

        mismatch = identifyMismatch(self.imageList, self.maskList)
        print('Number of mismatch for Data{} is {}'.format(dataset_name, mismatch))
        assert(mismatch==0)
        # At this stage we are sure that the mask corresponds to its mask
        self.targetSize = targetSize
        self.tensor_images = []
        self.tensor_masks = []
        self.load_to_RAM = load_to_RAM
        self.augmentation = augmentation
        if self.augmentation == None:
            self.augmentation = transforms.Resize(self.targetSize)

        if self.load_to_RAM:  # load all data to RAM for faster fetching
            print("Loading dataset to RAM...")
            self.tensor_images = [self.get_tensor_image(image_path) for image_path in self.imageList]
            self.tensor_masks = [self.get_tensor_mask(mask_path) for mask_path in self.maskList]
            print("Finish loading dataset to RAM")

    def __getitem__(self, index):
        if self.load_to_RAM:  # if images are loaded to the RAM copy them, otherwise, read them
            x = self.tensor_images[index]
            y_dic = self.tensor_masks[index]
            y = y_dic['seg_target']
            intermediate = y_dic['seg_intermediate']
        else:
            x = self.get_tensor_image(self.imageList[index])
            y_dic = self.get_tensor_mask(self.maskList[index])
            y = y_dic['seg_target']
            intermediate = y_dic['seg_intermediate']
        return x, intermediate, y

    def __len__(self):
        return len(self.imageList)

    def get_tensor_image(self, image_path):
        '''this function get image path and return transformed tensor image'''
        preprocess = transforms.Compose([
            # transforms.Resize((384, 288), 2),
            transforms.Resize(self.targetSize),
            self.augmentation,
            transforms.ToTensor()])
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
        X = Image.open(image_path).convert('RGB')
        X = preprocess(X)
        return X

    def get_tensor_mask(self, mask_path):
        trfresize = transforms.Resize(self.targetSize)
        trftensor = transforms.ToTensor()
        yimg = Image.open(mask_path).convert('L')
        y1 = trftensor(trfresize(yimg))
        mask = y1
        y1 = y1.type(torch.BoolTensor)
        y2 = torch.bitwise_not(y1)
        y = torch.cat([y2, y1], dim=0).float()
        mask_dic = {'seg_target': y, 'seg_intermediate': mask}
        # y.squeeze_()
        return mask_dic


# TTR is Train Test Ratio
def trainTestSplit(dataset, TTR):
    '''This function split train test randomely'''
    if not isinstance(dataset, collections.Sequence):
        dataset = (dataset, dataset)
    print("dataset is splitted randomely")
    dataset_size = len(dataset[0])  # dataset is tuble = (megaDataset_augmented, megaDataset_no_augmented)
    dataset_permutation = np.random.permutation(dataset_size)
    # print(dataset_permutation[:10])
    # trainDataset = torch.utils.data.Subset(dataset, range(0, int(TTR * len(dataset))))
    # valDataset = torch.utils.data.Subset(dataset, range(int(TTR*len(dataset)), len(dataset)))
    #
    trainDataset = torch.utils.data.Subset(dataset[0], dataset_permutation[:int(TTR * dataset_size)])
    valDataset = torch.utils.data.Subset(dataset[1], dataset_permutation[int(TTR * dataset_size):])
    print("training indices first samples{}\n val indices first samples{}".format(trainDataset.indices[:5],
                                                                                  valDataset.indices[:5]))
    # print(trainDataset.dataset[0])
    # exit(0)
    return trainDataset, valDataset


def deleteTail(x):
    # print(x)
    # D:/Databases/CVC-ClinicDB/data_C1/images_C1\\1.png --> images_C1\\1.png (windows) 1.png (linux)
    x = x.split('/')[-1]
    x = x.split('\\')[-1]  # images_C1\\1.png --> 1.png for both (linux) and (windows)
    x = x.split('_mask')[0]  # C3_0110_mask.jpg --> C3_0110
    x = x.split('.')[0]  # C3_0110.jpg --> C3_0110
    # x = x.split('.')[0] # 0110.jpg --> C3_0110
    # print(x)
    return x


def pruneFileNames(pair, dataset_name="CVC-ClinicDB"):
    # img_name, mask_name = pair
    # if dataset_name == "CVC-ClinicDB":
    #     img_name = img_name.split('\\')[-1].split('/')[-1]
    #     mask_name = mask_name.split('\\')[-1].split('/')[-1]
    # else:  # I think this is for EndoCV
    #     img_name = img_name.split('_')[-1].split('.')[0]
    #     mask_name = mask_name.split('_')[-2]
    names = []
    for x in pair:
        x = x.split('/')[-1]
        x = x.split('\\')[-1]  # images_C1\\1.png --> 1.png for both (linux) and (windows)
        x = x.split('_mask')[0]  # C3_0110_mask.jpg --> C3_0110
        x = x.split('.')[0]
        names.append(x)
    return names


def identifyMismatch(imageList, maskList, dataset_name="CVC-ClinicDB", examples=2):
    mismatch = 0
    for pair in zip(imageList, maskList):
        img_name, mask_name = pruneFileNames(pair)
        if img_name != mask_name:
            mismatch += 1
            print('img={} mask={}'.format(img_name,mask_name))
        if examples > 0:
            pprint(pair)
            examples -= 1
    return mismatch

def blure_background_getDataloadersDic(dataset_info, dataloder_info):
    datasets_list = []
    if not isinstance(dataset_info, list):
        dataset_info = [dataset_info]

    for i, info in enumerate(dataset_info):
        datasets_list.append(blure_background_SegDataset(*info))

    dataset = ConcatDataset(datasets_list)
    train_val_ratio, batchSize, shuffle = dataloder_info
    trainDataset, valDataset = trainTestSplit(dataset, train_val_ratio)
    # -------------Dataloader configurations---------------------
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=shuffle, drop_last=False)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=shuffle, drop_last=False)
    dataloader_dic = {'train': trainLoader, 'val': valLoader}
    return dataloader_dic

class blure_background_SegDataset(Dataset):
    def __init__(self, parentDir, dataset_name, imageDir, maskDir, targetSize, augmentation=None, load_to_RAM=False):
        self.imageList = sorted(glob.glob("/".join((parentDir, dataset_name, imageDir, '/*'))), key=deleteTail)
        self.maskList = sorted(glob.glob("/".join((parentDir, dataset_name, maskDir, '/*'))), key=deleteTail)

        mismatch = identifyMismatch(self.imageList, self.maskList)
        print('Number of mismatch for Data{} is {}'.format(dataset_name, mismatch))
        assert(mismatch==0)
        # At this stage we are sure that the mask corresponds to its mask
        self.targetSize = targetSize
        self.tensor_images = []
        self.tensor_masks = []
        self.load_to_RAM = load_to_RAM
        self.augmentation = augmentation
        if self.augmentation == None:
            self.augmentation = transforms.Resize(self.targetSize)

        if self.load_to_RAM:  # load all data to RAM for faster fetching
            print("Loading dataset to RAM...")
            self.tensor_images = [self.get_tensor_image(image_path) for image_path in self.imageList]
            self.tensor_masks = [self.get_tensor_mask(mask_path) for mask_path in self.maskList]
            print("Finish loading dataset to RAM")

    def __getitem__(self, index):
        if self.load_to_RAM:  # if images are loaded to the RAM copy them, otherwise, read them
            x = self.tensor_images[index]
            y_dic = self.tensor_masks[index]
            y = y_dic['seg_target']
            intermediate = y_dic['seg_intermediate']
        else:
            x = self.get_tensor_image(self.imageList[index])
            y_dic = self.get_tensor_mask(self.maskList[index])
            y = y_dic['seg_target']#mask as a boolean (for log loss)
            intermediate = y_dic['seg_intermediate']#it is a mask as a float (for L2 loss)
            x_blure = TF.gaussian_blur(x,11)
            x = x_blure.mul(1-intermediate)+ x.mul(intermediate) #bluring the background and keeping the polyp
        return x, intermediate, y

    def __len__(self):
        return len(self.imageList)

    def get_tensor_image(self, image_path):
        '''this function get image path and return transformed tensor image'''
        preprocess = transforms.Compose([
            # transforms.Resize((384, 288), 2),
            transforms.Resize(self.targetSize),
            self.augmentation,
            transforms.ToTensor()])
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
        X = Image.open(image_path).convert('RGB')
        X = preprocess(X)
        return X

    def get_tensor_mask(self, mask_path):
        trfresize = transforms.Resize(self.targetSize)
        trftensor = transforms.ToTensor()
        yimg = Image.open(mask_path).convert('L')
        y1 = trftensor(trfresize(yimg))
        mask = y1
        y1 = y1.type(torch.BoolTensor)
        y2 = torch.bitwise_not(y1)
        y = torch.cat([y2, y1], dim=0)
        mask_dic = {'seg_target': y, 'seg_intermediate': mask}
        # y.squeeze_()
        return mask_dic