import torch
import torch.nn as nn
from resnet_module import get_resnet50
from blocks import Encoder_resnet50, Decoder, MFF, RefineBlock

class spacialFeatureExtractor(nn.Module):
    '''
    The spacial feature extractor part of the network
    '''
    def __init__(self, 
                 Encoder, 
                 encoder_block_dims = [256, 512, 1024, 2048]
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