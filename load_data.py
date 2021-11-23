from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from load_data_utils import nyu2_paired_path

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
        
        x_tr = Image.open(path_xtr)
        y_tr = Image.open(path_ytr)
        
        if self.transforms:
            x_tr = self.transforms(x_tr)
            y_tr = self.transforms(y_tr)
        
        return (x_tr, y_tr)

    def __len__(self):
        return len(self.path_pairs)
    
def nyu2_dataloaders(batchsize=64, nyu2path='./nyu2_train'):
    
    '''
    split and return training set, validation set and testing test
    all in format of torch.util.data.Dataloader
    
    Args:
        batchsize (int): the # of entry to be used in one batch of training 
        (or testing)
        nyu2path (str) : the path of nyu2_train dataset
    '''
    
    # used for trainingset and validation set
    train_val_transforms = nn.Sequential(
        # output is a (224, 224, 3) tensor
        transforms.Scale(size=[224, 224]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.RandomCrop(size=[224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                  [0.229, 0.224, 0.225])
    )
    
    # used for testing set
    test_transforms = nn.Sequential(
        transforms.Scale(size=[224, 224]),
        transforms.ToTensor(is_test=True),
        transforms.Normalize([0.485, 0.456, 0.406],
                  [0.229, 0.224, 0.225])
    )
    
    all_pairs = nyu2_paired_path(nyu2_path=nyu2path)
    
    # train: val: test = 7: 2: 1
    total_size = len(all_pairs)
    train_size = int(total_size * 0.7)
    ttl_sz_left = total_size - train_size
    val_size = int(total_size * 0.2)
    test_size = ttl_sz_left - val_size
    
    # TODO: split the dataset to three, better shuffle them
    # train_dataset, val_dataset, test_dataset = 
    # random_split()
    
    # datalodaers, to be enumerated
    train_loader  = DataLoader (dataset=train_dataset,
                                batch_size=batchsize)
    val_loader    = DataLoader (dataset=val_dataset,
                                batch_size=batchsize)
    test_loader   = DataLoader (dataset=test_dataset,
                                batch_size=batchsize)
    
    return (train_loader, val_loader, test_loader)