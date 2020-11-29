import torch
import torch.nn as nn
import numpy as np


class FocalLoss(nn.Module):
    def forward(self, classifications, regression, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
