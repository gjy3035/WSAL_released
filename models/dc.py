# -*- coding: utf-8 -*-
import sys
import torch
from torch import nn
import torch.nn.functional as F

from torchvision import models
from config_model import initialize_weights

from config import cfg

import pdb


class domain_classifier(nn.Module):

    def __init__(self, featmap_dim=512, n_channel=512):
        super(domain_classifier, self).__init__()
        self.featmap_dim = featmap_dim
        self.conv1 = nn.Conv2d(n_channel, featmap_dim / 4, 3,
                               stride=2, padding=2)
        self.conv2 = nn.Conv2d(featmap_dim / 4, featmap_dim / 2, 3,
                               stride=2, padding=2)
        self.BN2 = nn.BatchNorm2d(featmap_dim / 2)

        self.fc = nn.Linear(featmap_dim * 4 * 4, 1)

    def forward(self, x):
        """
        Strided convulation layers,
        Batch Normalization after convulation but not at input layer,
        LeakyReLU activation function with slope 0.2.
        """
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.BN2(self.conv2(x)), negative_slope=0.2)
        x = x.view(-1, self.featmap_dim * 4 * 4)
        x = F.sigmoid(self.fc(x))
        return x

class dc_pixel(nn.Module):

    def __init__(self, input_channels):
        super(dc_pixel, self).__init__()
        self.imgsize = cfg.TRAIN.IMG_SIZE

        self.conv1 = nn.Conv2d(input_channels, 512, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.BN2 = nn.BatchNorm2d(512)
        # number of labels: 2: real and fake
        self.conv3 = nn.Conv2d(512, 2, 3, stride=1, padding=1)

    def forward(self, x):
        """
        Strided convulation layers,
        Batch Normalization after convulation but not at input layer,
        LeakyReLU activation function with slope 0.2.
        """
        
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.BN2(self.conv2(x)), negative_slope=0.2)
        
        # x = F.leaky_relu(self.BN3(self.conv3(x)), negative_slope=0.2)
        x = F.upsample(self.conv3(x), self.imgsize, None, 'bilinear')
        return x

class dc_object(nn.Module):

    def __init__(self, num_classes):
        super(dc_object, self).__init__()
        input_channels, pool_w, pool_h, scale = cfg.TRAIN.ROI_POOLED_SIZE
        self.fc1 = nn.Linear(input_channels * pool_w * pool_h, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.classifier = nn.Linear(512, num_classes*2)

    def forward(self, x):
        """
        Strided convulation layers,
        Batch Normalization after convulation but not at input layer,
        LeakyReLU activation function with slope 0.2.
        """
        x = x.view(x.size()[0], -1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = self.classifier(x)        
        return x

class dc_object2(nn.Module):

    def __init__(self, num_classes):
        super(dc_object2, self).__init__()
        input_channels, pool_w, pool_h, scale = cfg.TRAIN.ROI_POOLED_SIZE
        self.conv1 = nn.Conv2d(input_channels, 1024, 3, stride=2, padding=2)
        # self.BN1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.Conv2d(1024, 1024, 3, stride=2, padding=2)
        self.BN2 = nn.BatchNorm2d(1024)
        self.fc1 = nn.Linear(1024 * 4 * 4, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.classifier = nn.Linear(512, num_classes*2)

    def forward(self, x):
        """
        Strided convulation layers,
        Batch Normalization after convulation but not at input layer,
        LeakyReLU activation function with slope 0.2.
        """
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.BN2(self.conv2(x)), negative_slope=0.2)
        # pdb.set_trace()
        x = x.view(x.size()[0], -1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = self.classifier(x)        
        return x