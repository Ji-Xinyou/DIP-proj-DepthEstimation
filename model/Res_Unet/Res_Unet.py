from .modules import *

class Residual_Encoder(nn.Module):
    '''
    The Residual Encoder of the Depth Estimator
    Args:
        inp: 3 for three channels RGB images
        mid_planes: the inp and outp for residual convs in between
        outp: output planes of the encoder, **need to be matched** with unet decoder
        leaky_alpha: the negative slope of leaky ReLU
    '''
    def __init__(self, 
                 inp,
                 mid_planes,
                 outp,
                 leaky_alpha=0.02):
        super().__init__()
        # inp -> mid_planes[0], mp[0] -> mp[1]， ..., mp[l - 2] -> mp[l - 1], mp[l - 1] -> outp
        self.inconv = ResidualDoubleConv(inp=inp,
                                         outp=mid_planes[0],
                                         leaky_alpha=leaky_alpha)
        self.blocks = nn.ModuleList()
        for i in range(len(mid_planes) - 1):
            in_plane = mid_planes[i]
            out_plane = mid_planes[i + 1]
            self.blocks.append(ResidualConv(inp=in_plane,
                                            outp=out_plane,
                                            leaky_alpha=leaky_alpha))
        self.outconv = ResidualDoubleConv(inp=mid_planes[-1], 
                                          outp=outp,
                                          leaky_alpha=leaky_alpha)
    
    def forward(self, x):
        x = self.inconv(x)
        for block in self.blocks:
            x = block(x)
        x = self.outconv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownSampling(64, 128)
        self.down2 = DownSampling(128, 256)
        self.down3 = DownSampling(256, 512)
        
        factor = 2 if bilinear else 1
        self.down4 = DownSampling(512, 1024 // factor)
        
        self.up1 = UpSampling(1024, 512 // factor, bilinear)
        self.up2 = UpSampling(512, 256 // factor, bilinear)
        self.up3 = UpSampling(256, 128 // factor, bilinear)
        self.up4 = UpSampling(128, 64, bilinear)
        
        self.conv1 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class Encoder_Decoder_Net(nn.Module):
    
    def __init__(self, 
                 e_inp=3, 
                 e_midps=[64, 128, 256, 512],
                 e_outp=64, 
                 d_outp=1, 
                 leaky_alpha=0.02):
        super().__init__()
        self.encoder = Residual_Encoder(inp=e_inp,
                                        mid_planes=e_midps,
                                        outp=e_outp,
                                        leaky_alpha=leaky_alpha)
        # encoder's output channel = decoder's input channel
        self.decoder = UNet(n_channels=e_outp,
                            n_classes=d_outp,
                            bilinear=True)
    
    def forward(self, x):
        return self.decoder(self.encoder(x))