#import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np

class SegmenationModel:
    """ Class of a model for segmentation
        Use an RNN to outputbounding boxes as a vector, and use IOU to compute
        For the model h_t(x, h), the input x is the image, and h is the last internal state vector.
        The output may be a stop or the next state.
    """
    def __init__(self, inputsize):
        self.dims = len(inputsize)
        self.xdim = inputsize[0]
        self.ydim = inputsize[1]

        # hyperparams

        return

    def build(self):
        input
        return

    def load(self, modelfile):
        return

    def save(self, modelfile):
        return
