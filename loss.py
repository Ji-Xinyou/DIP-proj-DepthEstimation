import numpy as np
import torch
import torch.nn as nn
import torch.linalg as linalg

class Sobel(nn.Module):
    '''
    Edge detection using Sobel operator:
        input: depth image
        output: 
            out[:, 0, :, :] = dx
            out[:, 1, :, :] = dy
    
    The output of Sobel operator will be used to
    compute terms **loss_grad** and **loss_normal**
    '''
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        # 2(dx, dy) x 1(depth) x (3 x 3) (filter size)
        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x) 
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
        return out
    
def compute_loss(pred, truth, device, **kwargs):
    '''
    Compute the loss of the model
    Inputs:
        pred: output depth of the model
        truth: ground truth depth
        device: cuda or cpu  
        kwargs:
            'alpha': constant added in the logarithm
                default: 0.5
            'lambda': constant multiplied on loss_grad for bounding
                default: 1
            'mu': constant multiplied on loss_normal for bounding
                default: 1
        
    Logic:
        There are three parts of losses
        loss_depth, loss_grad, loss_normal
            diff = truth - pred
            loss_depth: logarithm of L1/L2 norm of diff
            loss_grad: sum of logarithm of L1/L2 norm of diff_dx and diff_dy
            loss_normal: 
    '''
    
    _alpha = kwargs.get('alpha', default=0.5)
    _lambda = kwargs.get('lambda', default=1)
    _mu = kwargs.get('mu', default=1)
    
    # TODO: In the paper, L1 norm is used
    # TODO: Try L2 norm
    
    # first term of loss
    loss_depth = torch.log(torch.abs(truth - pred) + _alpha).mean()
    
    grad_of = Sobel.to(device=device)
    pred_grad, truth_grad = grad_of(pred), grad_of(truth)
    pred_dx = pred_grad[:, 0:, :, :].contiguous().view_as(truth)
    pred_dy = pred_grad[:, 1:, :, :].contiguous().view_as(truth)
    truth_dx = truth_grad[:, 0:, :, :].contiguous().view_as(truth)
    truth_dy = truth_grad[:, 1:, :, :].contiguous().view_as(truth)
    
    # second term of loss
    loss_grad = torch.log(torch.abs(truth_dx - pred_dx) + _alpha).mean() + \
                torch.log(torch.abs(truth_dy - pred_dy) + _alpha).mean()
    
    # (B, 1, H, W)
    normal_z_shape = [truth.size(0), 1, truth.size(2), truth.size(3)]
    pred_normal = torch.cat((-pred_dx, -pred_dy, torch.ones(*normal_z_shape)), 1)
    truth_normal = torch.cat((-truth_dx, -truth_dy, torch.ones(*normal_z_shape)), 1)
    
    # similarity computed in the depth_derivative channel (dim 1)
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-8)
    loss_normal = torch.abs(1 - cos_sim(truth_normal, pred_normal)).mean()
    
    loss = loss_depth + _lambda * loss_grad + _mu * loss_normal
    
    return loss