import os
import torch.nn as nn
from config import cfg
import math
# here (https://github.com/pytorch/vision/tree/master/torchvision/models) to find the download link of pretrained models

root = cfg.DATA.ROOT + 'Models/PyTorch Pretrained'
res152_path = os.path.join(root, 'resnet152-b121ed2d.pth')
vgg19_bn_path = os.path.join(root, 'vgg19_bn-c79401a0.pth')
dense201_path = os.path.join(root, 'densenet201-4c113574.pth')


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / n))
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                n = module.weight.size(1)
                module.weight.data.normal_(0, math.sqrt(2. / n))
                module.bias.data.zero_()
