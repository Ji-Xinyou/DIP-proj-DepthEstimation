import torch
import torch.nn as nn
from . import resnet_module
from . import blocks

class spacialFeatureExtractor(nn.Module):
    '''
    The spacial feature extractor part of the network
    '''
    def __init__(self, 
                 Encoder, 
                 encoder_block_dims = [256, 512, 1024, 2048],
                 **kwargs
                 ):
        '''
        Args:
            Encoder: 
                The encoder network of the extractor
            encoder_block_dims: 
                The dimensions of outputs from each block of the encoder
        
        Note:
            1. The input_dim of DECODER is encoder_block_dims[-1]
            2. The in_channel_list of MFF is encoder_block_dims
            3. The out_channels of MFF is encoder_block_dims[-1] // 32
        '''
        super().__init__()
        # if the encoder is a resnet, 
        # the output has init_decoder_channels of feature maps
        
        # by default, use resnet50 as encoder
        self.encoder = Encoder
        
        self.Decoder = blocks.Decoder(in_channels=encoder_block_dims[-1])
        self.MFF     = blocks.MFF(in_channel_list=encoder_block_dims,
                                  out_channels=encoder_block_dims[-1] // 32)
        
        # the input_dim of refineblock is determined
        # by the decoder and mff, which is determined by the encoder
        # if the encoder is not resnet50
        # decoder, mff, refinement block should be changed accordingly
        self.refine  = blocks.RefineBlock()
        
    def forward(self, x):
        x_b1, x_b2, x_b3, x_b4 = self.encoder(x)
        # params are used to acquire the upsampling shape
        up1_shape = [2 * x_b3.size(2), 2 * x_b3.size(3)]
        up2_shape = [2 * x_b2.size(2), 2 * x_b2.size(3)]  
        up3_shape = [2 * x_b1.size(2), 2 * x_b1.size(3)]
        # up4 reconstruct to the shape before encoder's first conv2d
        up4_shape = [4 * x_b1.size(2), 4 * x_b1.size(3)]
        
        x_D = self.Decoder(x_b4, up1_shape, up2_shape, up3_shape, up4_shape)
        x_mff = self.MFF(x_b1, x_b2, x_b3, x_b4)
        
        # concat the x_D and x_mff
        depth = self.refine(torch.cat((x_D, x_mff), 1))
        
        # (B, 1, H, W) image
        return depth

def get_model(**kwargs):
    base_resnet50 = resnet_module.get_resnet50(pretrained=True)
    # encoder output a tuple of each block's output
    if kwargs == None or kwargs['encoder'] == 'resnet50':
        E = blocks.Encoder_resnet50(base=base_resnet50)
    model = spacialFeatureExtractor(Encoder=E,
                                    encoder_block_dims=[256, 512, 1024, 2048])
    return model