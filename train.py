import argparse
import torch
import torch.optim as optim
from datetime import datetime
from time import time
from torch.autograd import Variable
from utils import load_param, save_param
from model.model import spacialFeatureExtractor
from model.resnet_module import get_resnet50
from model.blocks import Encoder_resnet50
from load_data import nyu2_dataloaders
from loss import compute_loss

description = "CS386 course project - Depth Estimation In Soccer Games"
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--epochs', default=25, type=int,
                    help='# of epochs')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--L2penalty', default=1e-4, type=float,
                    help='weight_decay, i.e. L2normPenalty')

def get_model(**kwargs):
    base_resnet50 = get_resnet50(pretrained=True)
    # encoder output a tuple of each block's output
    if kwargs == None or kwargs['encoder'] == 'resnet50':
        E = Encoder_resnet50(base=base_resnet50)
    model = spacialFeatureExtractor(Encoder=E,
                                    encoder_block_dims=[256, 512, 1024, 2048])

def check_loss_on_val(val_dataloader, model, device):
    model.eval()
    loss = 0
    with torch.no_grad():
        for x_val, y_val in val_dataloader:
            x_val = x_val.to(device=device)
            y_val = y_val.to(device=device)
            y_pred = model(x_val)
            
            #TODO: define loss
            _loss = compute_loss(pred=y_pred,
                                 truth=y_val,
                                 device=device,
                                 alpha=0.5,
                                 lambda=1,
                                 mu=1)
        loss /= len(val_dataloader)
        print("")
            

def train(train_dataloader,
          val_dataloader,
          model,
          optimizer,
          epochs):
    print_every = 50
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    model = model.to(device=device)
    
    start_time = time.time()
    for epoch in epochs:
        # batched_image_size: (batch_size, C, H, W)
        for i, (x_tr, y_tr) in enumerate(train_dataloader):
            # turn to train mode
            model.train()
            
            x_tr = x_tr.to(device=device)
            y_tr = y_tr.to(device=device)
            y_pred = model(x_tr)
            
            # TODO: define loss
            loss = []
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            end_time = time.time()
            if i % print_every == 0:
                print("[Epoch]: %d [Iteration]: %d/%d, [loss]: %.4f, [Time Spent]: %.3f"
                      %(epoch, 
                        i, len(train_dataloader), 
                        loss, 
                        (end_time - start_time)))  

def main():
    # hyperparams
    global args
    args = parser.parse_args()
    
    # ---------------- params ---------------- #
    epochs = args.epochs
    lr = args.lr
    weight_decay = args.L2penalty
    # batchsize should better be more than 32 since BN is used frequently
    batchsize = 32
    # ---------------- params ---------------- #
    
    model = get_model(encoder='resnet50')
    optimizer = optim.Adam(model.parameters(), 
                           lr=lr,
                           weight_decay=weight_decay)
    train_set, val_set, _ = nyu2_dataloaders(batchsize=batchsize,
                                             nyu2_path='./nyu2_train')
    
    # all epochs wrapped in train()
    train(train_dataloader=train_set,
          val_dataloader=val_set,
          model=model,
          optimizer=optimizer,
          epochs=epochs)
    
    # SAVE THE PARAMETERS
    # default is current time, change it whatever you like
    filelabel = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    save_param(model=model,
               pth_path='./model_pth/{}.pth'.format(filelabel))

if __name__ == '__main__':
    main()
    