from load_data import nyu2_dataloaders
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_loader, val_loader, test_loader = nyu2_dataloaders()
    train_set, val_set, test_set = train_loader.dataset, val_loader.dataset, test_loader.dataset
    for i, (x_tr, y_tr) in enumerate(train_loader):
        # x_tr (C, H, W)
        if i == 0:
            plt.imshow(np.transpose(x_tr[0].numpy(), (1, 2, 0)))
            plt.show()
            plt.imshow(np.transpose(y_tr[0].numpy(), (1, 2, 0)))
            plt.show()