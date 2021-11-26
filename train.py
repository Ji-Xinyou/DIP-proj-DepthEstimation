import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import time
from utils import load_param, save_param
from load_data import nyu2_dataloaders
from model.model import get_model
from loss import compute_loss

description = "CS386 course project - Depth Estimation In Soccer Games"
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--epochs', default=25, type=int,
                    help='# of epochs')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--L2penalty', default=1e-4, type=float,
                    help='weight_decay, i.e. L2normPenalty')
parser.add_argument('--alpha', default=0.5, type=float,
                    help='loss_params alpha, used in logarithm')
parser.add_argument('--lmbd', default=1, type=float,
                    help='coefficient of loss_grad term')
parser.add_argument('--mu', default=1, type=int,
                    help='coefficient of loss_normal term')

args = parser.parse_args()

loss_params = {
    '_alpha': args.alpha,
    '_lambda': args.lmbd,
    '_mu': args.mu
}

def check_loss_on_set(dataloader, model, device):
    model.eval()
    loss = 0
    with torch.no_grad():
        for x_val, y_val in dataloader:
            x_val = x_val.to(device=device)
            y_val = y_val.to(device=device)
            y_pred = model(x_val)
            
            _loss = compute_loss(pred=y_pred,
                                 truth=y_val,
                                 device=device,
                                 _alpha=loss_params['_alpha'], 
                                 _lambda=loss_params['_lambda'], 
                                 _mu=loss_params['_mu'])
            loss += _loss
        loss /= len(dataloader)
        print("Test on [val]: loss avg: %.4f" 
              % (
                    loss    
                )
              )
            
def train(train_dataloader,
          val_dataloader,
          model,
          optimizer,
          epochs,
          device):
    print_every = 5
    
    model = model.to(device=device)
    
    start_time = time.time()
    
    print("train(): Training on {}".format(device))
    
    for epoch in range(epochs):
        # batched_image_size: (batch_size, C, H, W)
        for i, (x_tr, y_tr) in enumerate(tqdm(train_dataloader)):
            # turn to train mode
            model.train()
            
            x_tr = x_tr.to(device=device)
            y_tr = y_tr.to(device=device)
            y_pred = model(x_tr)
            
            # TODO: define loss
            loss = compute_loss(pred=y_pred,
                                truth=y_tr,
                                device=device,
                                _alpha=loss_params['_alpha'], 
                                _lambda=loss_params['_lambda'], 
                                _mu=loss_params['_mu'])
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            end_time = time.time()
            if i % print_every == 0:
                # print the information of the epoch
                print("[Epoch]: %d/%d [Iteration]: %d/%d, [loss]: %.4f, [Time Spent]: %.3f"
                      %(
                            epoch, epochs, 
                            i, len(train_dataloader), 
                            loss, 
                            (end_time - start_time)
                        )
                      )
                
        # check on validation set each epoch
        check_loss_on_set(dataloader=val_dataloader,
                          model=model,
                          device=device)
    
def main():
    # hyperparams
    global args
    
    # ---------------- params ---------------- #
    epochs = args.epochs
    lr = args.lr
    weight_decay = args.L2penalty
    # batchsize should better be more than 32 since BN is used frequently
    batchsize = 32
    # ---------------- params ---------------- #
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    print("main(): Getting model......")
    model = get_model(encoder='resnet50')
    
    optimizer = optim.Adam(model.parameters(), 
                           lr=lr,
                           weight_decay=weight_decay)
    
    print("main(): Getting dataloaders......")
    train_set, val_set, test_set = nyu2_dataloaders(batchsize=batchsize,
                                             nyu2_path='./nyu2_train')
    
    print("main(): start training......")
    # all epochs wrapped in train()
    train(train_dataloader=train_set,
          val_dataloader=val_set,
          model=model,
          optimizer=optimizer,
          epochs=epochs,
          device=device)
    
    print("Training Session is over, test the model on testset")
    # after training, test it on testset
    check_loss_on_set(dataloader=test_set,
                      model=model,
                      device=device)
    
    # SAVE THE PARAMETERS
    # default is current time, change it whatever you like
    filelabel = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    save_param(model=model,
               pth_path='./model_pth/{}.pth'.format(filelabel))

if __name__ == '__main__':
    main()