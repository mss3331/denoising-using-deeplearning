import torch
import glob
import numpy as np
from pprint import pprint
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import collections



def getDataloadersDic(dataset_info,dataloder_info):
    dataset = SegDataset(*dataset_info)
    train_val_ratio,batchSize, shuffle= dataloder_info
    trainDataset, valDataset = trainTestSplit(dataset, train_val_ratio)

    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=shuffle, drop_last=False)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=shuffle, drop_last=False)
    dataloader_dic = {'train':trainLoader,'val':valLoader}
    return dataloader_dic

class SegDataset(Dataset):
    def __init__(self, parentDir, dataset_name, imageDir, maskDir, targetSize, augmentation=None, load_to_RAM= False):
        self.imageList = sorted(glob.glob("/".join((parentDir, dataset_name, imageDir,'/*'))), key = deleteTail)
        # self.imageList.sort()
        self.maskList = sorted(glob.glob("/".join((parentDir, dataset_name, maskDir,'/*'))), key = deleteTail)
        # self.maskList.sort()
        mismatch = identifyMismatch(self.imageList,self.maskList)
        print('Number of mismatch for Data{} is {}'.format(dataset_name, mismatch))
        assert(mismatch==0)
        #At this stage we are sure that the mask corresponds to its mask
        self.targetSize = targetSize
        self.tensor_images = []
        self.tensor_masks = []
        self.load_to_RAM = load_to_RAM
        self.augmentation = augmentation
        if self.augmentation == None:
            self.augmentation=transforms.Resize(self.targetSize)

        if self.load_to_RAM:# load all data to RAM for faster fetching
            print("Loading dataset to RAM...")
            self.tensor_images = [self.get_tensor_image(image_path) for image_path in self.imageList]
            self.tensor_masks = [self.get_tensor_mask(mask_path) for mask_path in self.maskList]
            print("Finish loading dataset to RAM")

    def __getitem__(self, index):
        if self.load_to_RAM:#if images are loaded to the RAM copy them, otherwise, read them
            x = self.tensor_images[index]
            y = self.tensor_masks[index]
        else:
            x=self.get_tensor_image(self.imageList[index])
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
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
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
        mask_dic = {'seg_target':y,'seg_intermediate':mask}
        # y.squeeze_()
        return mask_dic
#TTR is Train Test Ratio
def trainTestSplit(dataset, TTR):
    '''This function split train test randomely'''
    if not isinstance(dataset, collections.Sequence):
        dataset = (dataset,dataset)
    print("dataset is splitted randomely")
    dataset_size = len(dataset[0]) #dataset is tuble = (megaDataset_augmented, megaDataset_no_augmented)
    dataset_permutation = np.random.permutation(dataset_size)
    # print(dataset_permutation[:10])
    # trainDataset = torch.utils.data.Subset(dataset, range(0, int(TTR * len(dataset))))
    # valDataset = torch.utils.data.Subset(dataset, range(int(TTR*len(dataset)), len(dataset)))
    #
    trainDataset = torch.utils.data.Subset(dataset[0], dataset_permutation[:int(TTR * dataset_size)] )
    valDataset = torch.utils.data.Subset(dataset[1],dataset_permutation[int(TTR * dataset_size):] )
    print("training indices first samples{}\n val indices first samples{}".format(trainDataset.indices[:5],valDataset.indices[:5]))
    # print(trainDataset.dataset[0])
    # exit(0)
    return trainDataset, valDataset

def deleteTail(x):
  # print(x)
  x = x.split('/')[-1]#D:/Databases/CVC-ClinicDB/data_C1/images_C1\\1.png --> images_C1\\1.png (windows) 1.png (linux)
  x = x.split('\\')[-1]# images_C1\\1.png --> 1.png for both (linux) and (windows)
  x = x.split('_mask')[0] # C3_0110_mask.jpg --> C3_0110
  x = x.split('.')[0] #C3_0110.jpg --> C3_0110
  # x = x.split('.')[0] # 0110.jpg --> C3_0110
  # print(x)
  return x
def pruneFileNames(pair,dataset_name="CVC-ClinicDB"):
    img_name, mask_name = pair
    if dataset_name=="CVC-ClinicDB":
        img_name = img_name.split('\\')[-1].split('/')[-1]
        mask_name = mask_name.split('\\')[-1].split('/')[-1]
    else: #I think this is for EndoCV
        img_name = img_name.split('_')[-1].split('.')[0]
        mask_name = mask_name.split('_')[-2]
    return img_name,mask_name
def identifyMismatch(imageList,maskList,dataset_name="CVC-ClinicDB",examples=2):
    mismatch=0
    for pair in zip(imageList, maskList):
        img_name, mask_name = pruneFileNames(pair)
        if img_name != mask_name:
            mismatch += 1
            # print('img={} mask={}'.format(img_name,mask_name))
        if examples>0:
            pprint(pair)
            examples -= 1
    return mismatch