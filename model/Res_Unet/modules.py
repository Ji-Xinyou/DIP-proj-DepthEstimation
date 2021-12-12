import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_Norm_ReLU(nn.Module):
    '''
    Conbination of Conv -> BN -> Leaky_ReLu
    Args:
        inp: inplane
        outp: outplane
        leaky_alpha: the negative slope of leaky relu
    '''
    
    def __init__(self, inp, outp, leaky_alpha=0.02):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=inp,
                              out_channels=outp,
                              kernel_size=3,
                              padding=1)
        self.norm = nn.BatchNorm2d(outp)
        self.acti = nn.LeakyReLU(negative_slope=leaky_alpha,
                                 inplace=True)
    
    def forward(self, x):
        return self.acti(self.norm(self.conv(x)))

class DoubleConv(nn.Module):
    '''
    Twice of Conv -> BN -> leakyrelu
    Args:
        inp, midp, outp:
            conv_norm_acti_1: inp  ----> midp
            conv_norm_acti_2: midp ----> outp
        leaky_alpha: the negative slope of leaky relu
    '''
    
    def __init__(self, 
                 inp, 
                 outp, 
                 midp=None, 
                 leaky_alpha=0.02):
        super().__init__()
        if not midp:
            midp= outp
        
        self.conv_norm_acti_1 = Conv_Norm_ReLU(inp=inp,
                                               outp=midp,
                                               leaky_alpha=leaky_alpha)
        self.conv_norm_acti_2 = Conv_Norm_ReLU(inp=midp,
                                               outp=outp,
                                               leaky_alpha=leaky_alpha)

    def forward(self, x):
        x = self.conv_norm_acti_1(x)
        x = self.conv_norm_acti_2(x)
        return x

class ResidualConv(nn.Module):
    
    def __init__(self,
                 inp, 
                 outp,  
                 leaky_alpha=0.02):
        super().__init__()
        self.conv_norm_acti = Conv_Norm_ReLU(inp=inp,
                                             outp=outp,
                                             leaky_alpha=leaky_alpha)
    
    def forward(self, x):
        return x + self.conv_norm_acti(x)

class ResidualDoubleConv(nn.Module):
    
    def __init__(self,
                 inp, 
                 outp, 
                 midp=None, 
                 leaky_alpha=0.02):
        super().__init__()
        self.doubleconv = DoubleConv(inp=inp,
                                     outp=outp,
                                     midp=midp,
                                     leaky_alpha=leaky_alpha)
    
    def forward(self, x):
        return x + self.doubleconv(x)
        
class DownSampling(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2),
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

# this function is from https://github.com/milesial/Pytorch-UNet/
class UpSampling(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)