import torch
import torch.nn as nn
import torch.nn.functional as functional

class Encoder_resnet50(nn.Module):
    
    def __init__(self, base):
        super(Encoder_resnet50, self).__init__()
        # encoder is a pretrained resnet, inherit the architecture
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        
        # layer1: out_channel = 64 * 4
        self.layer1 = base.layer1
        # layer2: out_channel = 128 * 4
        self.layer2 = base.layer2
        # layer3: out_channel = 256 * 4
        self.layer3 = base.layer3
        # layer4: out_channel = 512 * 4
        # layer4's output is the input of decoder
        self.layer4 = base.layer4

        self.avgpool = base.avgpool
        self.fc = base.fc
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x_b1 = self.layer1(x)
        x_b2 = self.layer2(x_b1)
        x_b3 = self.layer3(x_b2)
        x_b4 = self.layer4(x_b3)
        
        # return all for multiscale mff
        return x_b1, x_b2, x_b3, x_b4

class upsampling(nn.Module):
    
    '''
    net from Deeper depth prediction with fully convolutional residual
    networks
    
    Args of __init__():
        in_channels: the # of channels of the input
        out_channels: the # of channels of the output (result of upsampling)
    
    Args of forward():
        x: the input
        upsample_size: the output size (H x W) of the feature map
        
    Do notice the upsample is about resolution not # of channels
    '''
    def __init__(self, in_channels, out_channels):
        super(upsampling, self).__init__()
        
        # this conv maintains the shape
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv1_2 = nn.Conv2d(out_channels, out_channels,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels,
                              kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x, upsample_size):
        '''
        In forward pass, the (..., H, W) will be changed
        it will be upsampled to increase the feature maps' resolution
        '''
        x = functional.interpolate(x, 
                                   size=upsample_size, 
                                   mode='bilinear')
        _x = self.conv1(x)
        _x = self.bn1(_x)
        _x = self.relu(_x)
        
        x_branch1 = self.bn1_2(self.conv1_2(_x))
        x_branch2 = self.bn2(self.conv2(x))
        
        merge = self.relu(x_branch1 + x_branch2)
        
        return merge
        
class Decoder(nn.Module):
    '''
    Decoder use a series of upsampling layers,
        compressing the features and enlarging the map's scale
    '''
    
    # resnet50's layer4 has out_channels of 2048
    def __init__(self, in_channels=2048):
        super(Decoder, self).__init__()
        
        # H x W not changed
        self.conv2 = nn.Conv2d(in_channels, in_channels // 2,
                               kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels // 2)
        in_channels //= 2
        
        # in_channels // 2 -> in_channels // 4
        self.up1 = upsampling(in_channels, in_channels // 2)
        in_channels //= 2
        
        # in_channels // 4 -> in_channels // 8
        self.up2 = upsampling(in_channels, in_channels // 2)
        in_channels //= 2
        
        # in_channels // 8 -> in_channels // 16
        self.up3 = upsampling(in_channels, in_channels // 2)
        in_channels //= 2
        
        # in_channels // 16 -> in_channels // 32
        # by default: 2048 -> 64
        self.up4 = upsampling(in_channels, in_channels // 2)
        in_channels //= 2
        
    def forward(self, x_b1, x_b2, x_b3, x_b4):
        # params are used to acquire the upsampling shape
        up1_shape = [x_b3.size(2), x_b3.size(3)]
        up2_shape = [x_b2.size(2), x_b2.size(3)]  
        up3_shape = [x_b1.size(2), x_b1.size(3)]
        # up4 reconstruct to the shape before encoder's first conv2d
        up4_shape = [2 * x_b1.size(2), 2 * x_b1.size(3)]
        
        x = functional.relu(self.bn(self.conv2(x_b4)))
        x = self.up1(x, up1_shape)
        x = self.up2(x, up2_shape)
        x = self.up3(x, up3_shape)
        x = self.up4(x, up4_shape)
        
        return x

class MFF(nn.Module):
    
    '''
    Multiscale Feature Fusion Module
        Take the output of different layers of encoder
        upsample them and concat them, fuse the feature maps together
        Note that the upsampling layer also has conv layers
    '''
    
    def __init__(self, in_channel_list, out_channels):
        super(MFF, self).__init__()
        # by default the # of MFF block has 4 layers
        # concat them channel-wise, so each layer's output
        # has out_channels // 4 channels, we expect out_channels % 4 == 0
        out_channel_each = out_channels // len(in_channel_list)
        
        self.up5 = upsampling(in_channel_list[0], out_channel_each)
        self.up6 = upsampling(in_channel_list[1], out_channel_each)
        self.up7 = upsampling(in_channel_list[2], out_channel_each)
        self.up8 = upsampling(in_channel_list[3], out_channel_each)
        
        # after concat
        self.conv3 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x_b1, x_b2, x_b3, x_b4):
        # the output of decoder
        up4_shape = [2 * x_b1.size(2), 2 * x_b1.size(3)]
        
        mff_1 = self.up5(x_b1, up4_shape)
        mff_2 = self.up6(x_b2, up4_shape)
        mff_3 = self.up7(x_b3, up4_shape)
        mff_4 = self.up8(x_b4, up4_shape)
        
        # out_channels dim
        mff = torch.cat((mff_1, mff_2, mff_3, mff_4), 1)
        
        mff = self.conv3(mff)
        mff = self.bn(mff)
        mff = self.relu(mff)
        
        return mff

class RefineBlock(nn.Module):
    
    '''
    Do the refinement work, 
    take the concat of MFF output and decoder output as input,
    output the one channel (depth) image, i.e. the depth estimation image
    
    dimension recap:
        input:
            default:
                MFF output:     64 channels
                Decoder output: 64 channels
            in args:
                MFF output:     out_channels
                Decoder output: in_channels // 32
            so after concat:
                input dim = 64 + 64 = 128 channels
    '''

    def __init__(self):
        super().__init__()
        # explained in comment above
        input_dim = 128
        
        self.conv1 = nn.Conv2d(input_dim, input_dim,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(input_dim)
        
        self.conv2 = nn.Conv2d(input_dim, input_dim,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(input_dim)
        
        # output layer
        self.conv3 = nn.Conv2d(input_dim, 1,
                               kernel_size=5, stride=1, padding=2,
                               bias=True)
        
    def forward(self, x):
        x = functional.relu(self.bn1(self.conv1(x)))
        x = functional.relu(self.bn2(self.conv2(x)))
        
        x = self.conv3(x)
        
        return x
        