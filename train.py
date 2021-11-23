import argparse
from torch.autograd import Variable
from utils import load_param, save_param

description = "CS386 course project - Depth Estimation In Soccer Games"
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--epochs', default=25, type=int,
                    help='# of epochs')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='initial learning rate')

def train(train_dataloader,
          model,
          optimizer,
          epochs):
    # turn to train mode
    model.train()
    
    # batched_image_size: (batch_size, C, H, W)
    for i, (x_tr, y_tr) in enumerate(train_dataloader):
        x_tr = x_tr.cuda()
        y_tr = y_tr.cuda()
        
        y_pred = model(x_tr)
        
        
        
    