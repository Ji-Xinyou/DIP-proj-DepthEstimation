'''
I/O:
    Input: image to be inferenced
    Output: depth image
    Relation: out = model(in)
    
    path -> readimg -> image -> transform -> tensor
    tensor -> model
    model: get model from get_model() in train.py, load_param from .pth file
    model -> output tensor -> transpose to H x W x C -> imshow & save
'''
import argparse
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from utils import load_param
from model.model import get_model
