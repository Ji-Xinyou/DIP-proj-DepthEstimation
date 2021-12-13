import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import math

class Conv_Norm_ReLU(nn.Module):
    '''
    Conbination of Conv -> BN -> Leaky_ReLu
    Args:
        inp: inplane
        outp: outplane
        leaky_alpha: the negative slope of leaky relu
    '''
    
    def __init__(self, 
                 inp, 
                 outp, 
                 leaky_alpha=0.02, 
                 kernel_size=3, 
                 stride=1,
                 padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=inp,
                              out_channels=outp,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.norm = nn.BatchNorm2d(outp)
        self.acti = nn.LeakyReLU(negative_slope=leaky_alpha,
                                 inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.acti(x)
        return x

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
                 leaky_alpha=0.02,
                 kernel_size=3, 
                 stride=1,
                 padding=1):
        super().__init__()
        if not midp:
            midp= outp
        
        self.conv_norm_acti_1 = Conv_Norm_ReLU(inp=inp,
                                               outp=midp,
                                               leaky_alpha=leaky_alpha,
                                               kernel_size=kernel_size, 
                                               stride=stride,
                                               padding=padding)
        self.conv_norm_acti_2 = Conv_Norm_ReLU(inp=midp,
                                               outp=outp,
                                               leaky_alpha=leaky_alpha,
                                               kernel_size=kernel_size, 
                                               stride=stride,
                                               padding=padding)

    def forward(self, x):
        x = self.conv_norm_acti_1(x)
        x = self.conv_norm_acti_2(x)
        return x

class ResidualConv(nn.Module):
    
    def __init__(self,
                 inp, 
                 outp,  
                 leaky_alpha=0.02,
                 kernel_size=3, 
                 stride=1,
                 padding=1):
        super().__init__()
        self.conv_norm_acti = Conv_Norm_ReLU(inp=inp,
                                             outp=outp,
                                             leaky_alpha=leaky_alpha,
                                             kernel_size=kernel_size, 
                                             stride=stride,
                                             padding=padding)
    
    def forward(self, x):
        return x + self.conv_norm_acti(x)

class ResidualDoubleConv(nn.Module):
    
    def __init__(self,
                 inp, 
                 outp, 
                 midp=None, 
                 leaky_alpha=0.02,
                 kernel_size=3, 
                 stride=1,
                 padding=1):
        super().__init__()
        self.doubleconv = DoubleConv(inp=inp,
                                     outp=outp,
                                     leaky_alpha=leaky_alpha,
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=padding)
    
    def forward(self, x):
        return x + self.doubleconv(x)
        
class DownSampling(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
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
    

# part of codes are from
# https://github.com/JunjH/Revisiting_Single_Depth_Estimation/

class Bottleneck(nn.Module):
    '''
    Bottleneck is a resnet block
    '''
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x    
    
def get_resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(url, 'pretrained_model/resnet50'))
    return model

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
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # return all for multiscale mff
        return x

def get_resnet50_encoder(**kwargs):
    base_resnet50 = get_resnet50(pretrained=True)
    # encoder output a tuple of each block's output
    E = Encoder_resnet50(base=base_resnet50)
    return E