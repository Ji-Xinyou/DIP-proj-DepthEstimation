from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from load_data_utils import nyu2_paired_path
import torch.nn as nn
import random
import cv2

class nyu2_dataset(Dataset):
    
    '''
    nyu2_dataset: 
        used to train one shot depth estimation
    '''
    
    def __init__(self, 
                 pairs, 
                 _transforms=None):
        
        # self.paths has item like [path_of_xtr, path_of_ytr]
        self.path_pairs = pairs
        self.transforms = _transforms
    
    def __getitem__(self, index):
        path_xtr, path_ytr = self.path_pairs[index]
        
        x_tr = cv2.imread(path_xtr)
        y_tr = cv2.imread(path_ytr)
        y_tr = cv2.cvtColor(y_tr, cv2.COLOR_BGR2GRAY)
                
        if self.transforms:
            x_tr = self.transforms(x_tr)
            y_tr = self.transforms(y_tr)
        
        normalize_by_imagenet = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
            ])
        
        x_tr = normalize_by_imagenet(x_tr)
            
        return (x_tr, y_tr)

    def __len__(self):
        return len(self.path_pairs)
    
def nyu2_dataloaders(batchsize=64, nyu2_path='./nyu2_train'):
    
    '''
    split and return training set, validation set and testing test
    all in format of torch.util.data.Dataloader
    
    Args:
        batchsize (int): the # of entry to be used in one batch of training 
        (or testing)
        nyu2path (str) : the path of nyu2_train dataset
    '''
    
    print("Entering nyu2_dataloaders()")
    print("---------------- Loading Dataloaders ----------------")
    
    # used for trainingset and validation set
    train_val_transforms = transforms.Compose([
        # output is a (224, 224, 3) tensor
        transforms.ToPILImage(),
        transforms.Scale(size=[320, 240]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.RandomCrop([224, 224]),
        transforms.ToTensor(),
        ]
    )
    
    # used for testing set
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Scale(size=[320, 240]),
        transforms.RandomCrop([224, 224]),
        transforms.ToTensor(),
        ]
    )
    
    # preparing the pathpairs for different parts of data
    all_pairs = nyu2_paired_path(nyu2_path=nyu2_path)
    
    # train: val: test = 7: 2: 1
    total_size = len(all_pairs)
    train_size = int(total_size * 0.7)
    ttl_sz_left = total_size - train_size
    val_size = int(total_size * 0.2)
    
    # shuffle the list and assign them to datasets
    random.shuffle(all_pairs)
    
    train_pair = all_pairs[: train_size]
    val_pair = all_pairs[train_size: train_size + val_size]
    test_pair = all_pairs[train_size + val_size: ]
    
    # from pairs -> to datasets
    train_dataset = nyu2_dataset(pairs=train_pair,
                                 _transforms=train_val_transforms)
    val_dataset   = nyu2_dataset(pairs=val_pair,
                                 _transforms=train_val_transforms)
    test_dataset  = nyu2_dataset(pairs=test_pair,
                                 _transforms=test_transforms)
    
    print("-------- Datasets are ready, preparing Dataloaders --------")
    
    # datalodaers, to be enumerated
    train_loader  = DataLoader (dataset=train_dataset,
                                shuffle=True,
                                batch_size=batchsize,
                                num_workers=4)
    val_loader    = DataLoader (dataset=val_dataset,
                                shuffle=True,
                                batch_size=batchsize,
                                num_workers=4)
    test_loader   = DataLoader (dataset=test_dataset,
                                shuffle=True,
                                batch_size=batchsize,
                                num_workers=4)
    
    print("----------------- DataLoaders Ready ----------------")
    print("Exit nyu2_dataloaders()")
    
    return (train_loader, val_loader, test_loader)