# COMP.5300 Deep Learning
# Terry Griffin
#
# Routines for converting a three channel input model to a single
# channel input model.
#
# The input size of the first convolutional layer is changed by
# updating the in_channels parameter.
# The weights for the layer are updated by summing across the depth
# dimension and multiplying by 3

import torch
import torch.nn as nn

def resnet_grayscale_model(model):
    conv = model.conv1
    if conv.in_channels > 1:
        with torch.no_grad():
            conv.in_channels = 1
            conv.weight = nn.Parameter(conv.weight.sum(dim=1, keepdim=True) * 3)

def densenet_grayscale_model(model):
    conv = model.features.conv0
    if conv.in_channels > 1:
        with torch.no_grad():
            conv.in_channels = 1
            conv.weight = nn.Parameter(conv.weight.sum(dim=1, keepdim=True)* 3)

