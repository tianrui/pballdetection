import tensorflow as tf
import numpy as np

class SegmenationModel:
    """ Class of a model for segmentation
        Use an RNN to outputbounding boxes as a vector, and use IOU to compute
    """
    def __init__(self, inputsize):
        self.dims = len(inputsize)
        self.xdim = inputsize[0]
        self.ydim = inputsize[1]

        # hyperparams

        return

    def build(self):
        return

    def load(self, modelfile):
        return

    def save(self, modelfile):
        return
