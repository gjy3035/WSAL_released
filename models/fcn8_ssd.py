

# Version 5
import numpy as np
import sys
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from torch.autograd import Variable

from config import cfg
from utils.multibox_loss import MultiBoxLoss
from config_model import vgg19_bn_path, res152_path, dense201_path, initialize_weights
from utils.multibox_layer import MultiBoxLayer
from utils.l2norm import L2Norm

from faster_rcnn.roi_pooling.modules.roi_pool import RoIPool

import pdb


class _FCN8Base(nn.Module):
    def __init__(self):
        super(_FCN8Base, self).__init__()
        self.features3 = None #
        self.features4 = None #  v.s. conv7 1024,32,32
        self.features5 = None #  v.s. conv8_2 512,16,16
        self.fconv3 = None
        self.fconv4 = None
        self.fconv5 = None

        # SSD
        self.norm4 = L2Norm(512, 20)# 512,64,64

        self.ssd_conv5 = None
        self.ssd_conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.ssd_conv7 = nn.Conv2d(1024, 1024, kernel_size=1) # conv7 1024,32,32

        self.ssd_conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.ssd_conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)# conv8_2 512,16,16

        self.ssd_conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.ssd_conv9_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)# conv9_2 256,8,8

        self.ssd_conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.ssd_conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)# conv10_2 256,4,4

        self.ssd_conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.ssd_conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)# conv11_2 256,2,2

        # multibox layer
        self.multibox = MultiBoxLayer()

        # roi pooling 
        self.roi_pool = RoIPool(cfg.TRAIN.ROI_POOLED_SIZE[1], 
                                cfg.TRAIN.ROI_POOLED_SIZE[2], 
                                1.0/16)

    


class FCN8VGG(_FCN8Base):
    def __init__(self, num_classes, pretrained=True, phase='train'):
        super(FCN8VGG, self).__init__()
        vgg = models.vgg19_bn()
        if pretrained:
            vgg.load_state_dict(torch.load(vgg19_bn_path))
        features = list(vgg.features.children())

        self.features3 = nn.Sequential(*features[0:27])
        self.features4_4 = nn.Sequential(*features[27:39])
        self.features4 = nn.Sequential(*features[39:40])
        self.features5 = nn.Sequential(*features[40:])
        self.features5_test = nn.Sequential(*features[40:52])

        if cfg.TRAIN.LOC_P:
            loc_tmp = np.arange(0,2,2.0/64) - 1.0
            loc_x = np.tile(loc_tmp,(64,1))
            loc_y = loc_x.transpose()
            loc = np.concatenate((loc_x[None,:,:],loc_x[None,:,:]),axis=0)[None,:,:,:]
            loc_input = np.repeat(loc,cfg.TRAIN.IMG_BATCH_SIZE,axis=0)
            loc_input = torch.from_numpy(loc_input.astype(np.float32))
            self.loc_input = Variable(loc_input.cuda())
            
            self.fconv3 = nn.Conv2d(258, num_classes, kernel_size=1)
        else:
            self.fconv3 = nn.Conv2d(256, num_classes, kernel_size=1)

        self.fconv4 = nn.Conv2d(1536, num_classes, kernel_size=1)
        self.fconv5 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(2048, 2048, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(2048, num_classes, kernel_size=1)
        )
        initialize_weights(self.fconv3, self.fconv4, self.fconv5)

        self.ssd_conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)

    def forward(self, x, gt=None, train_flag=True):

        # features
        f_y3 = self.features3(x) # for FCN 256,64,64
        f_y4_4 = self.features4_4(f_y3) # 512,64,64
        f_y4 = self.features4(f_y4_4) # for FCN 512, 32, 32
        f_y5 = self.features5(f_y4) # for FCN 512, 16, 16
        # pdb.set_trace()

        if cfg.TRAIN.LOC_P:
            if train_flag:
                f_y3 = torch.cat((f_y3,self.loc_input),1)
            if not train_flag:
                # pdb.set_trace()
                f_y3 = torch.cat((f_y3,self.loc_input[0,:,:,:][None,:,:,:]),1)
        # SSD 
        hs = []
        f_y4_4 = self.norm4(f_y4_4)
        hs.append(f_y4_4) 

        h = F.relu(self.ssd_conv5(f_y4))
        h = F.max_pool2d(h, kernel_size=3, padding=1, stride=1, ceil_mode=True)

        h = F.relu(self.ssd_conv6(h))
        h = F.relu(self.ssd_conv7(h)) # 1024
        hs.append(h) 

        # FCN conv4 + SSD conv7 
        y4 = torch.cat((f_y4,h), 1) # 1536, 32, 32

        # ROI Pooling
        pooled_features = []
        if gt is not None: 
            # pdb.set_trace()
            pooled_features = self.roi_pool(h, gt)

        

        y4 = self.fconv4(y4)

        h = F.relu(self.ssd_conv8_1(h))
        h = F.relu(self.ssd_conv8_2(h))
        hs.append(h) 

        # FCN conv5 + SSD conv8_2 
        y5 = torch.cat((f_y5,h), 1) # 1024, 16, 16
        # y5_f = y5
        y5 = self.fconv5(y5)
        # FCN conv3
        y3 = self.fconv3(f_y3)

        y = y4 + F.upsample(y5, y4.size()[2:], None, 'bilinear')
        y = y3 + F.upsample(y, y3.size()[2:], None, 'bilinear')
        y = F.upsample(y, x.size()[2:], None, 'bilinear')

        if not train_flag:
            return y 

        h = F.relu(self.ssd_conv9_1(h))
        h = F.relu(self.ssd_conv9_2(h))
        hs.append(h)  

        h = F.relu(self.ssd_conv10_1(h))
        h = F.relu(self.ssd_conv10_2(h))
        hs.append(h)  

        h = F.relu(self.ssd_conv11_1(h))
        h = F.relu(self.ssd_conv11_2(h))
        hs.append(h)  

        loc_preds, conf_preds = self.multibox(hs)

        # DONE ROI Pooling
        

        return y, f_y5, loc_preds, conf_preds, pooled_features
             


class FCN8ResNet(_FCN8Base):
    def __init__(self, num_classes, pretrained=True):
        super(FCN8ResNet, self).__init__()
        res = models.resnet152()
        if pretrained:
            res.load_state_dict(torch.load(res152_path))
        self.features3 = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2
        )
        # pdb.set_trace()
        self.features4 = res.layer3
        self.features5 = res.layer4

        if cfg.TRAIN.LOC_P:
            loc_tmp = np.arange(0,2,2.0/64) - 1.0
            loc_x = np.tile(loc_tmp,(64,1))
            loc_y = loc_x.transpose()
            loc = np.concatenate((loc_x[None,:,:],loc_x[None,:,:]),axis=0)[None,:,:,:]
            loc_input = np.repeat(loc,cfg.TRAIN.IMG_BATCH_SIZE,axis=0)
            loc_input = torch.from_numpy(loc_input.astype(np.float32))
            self.loc_input = Variable(loc_input.cuda())
            
            self.fconv3 = nn.Conv2d(514, num_classes, kernel_size=1)
        else:
            self.fconv3 = nn.Conv2d(512, num_classes, kernel_size=1)

        self.fconv4 = nn.Conv2d(2048, num_classes, kernel_size=1)
        self.fconv5 = nn.Conv2d(2560, num_classes, kernel_size=7)
        initialize_weights(self.fconv3, self.fconv4, self.fconv5)

        self.ssd_conv5 = nn.Conv2d(1024, 512, kernel_size=3, padding=1, dilation=1)
        

    def forward(self, x, gt=None, train_flag=True):

        # features
        f_y3 = self.features3(x) # 512,64,64
        f_y4 = self.features4(f_y3) # 1024,32,32
        f_y5 = self.features5(f_y4) # 2048,16,16
        
        # SSD 
        hs = []
        hs.append(self.norm4(f_y3))

        if cfg.TRAIN.LOC_P:
            if train_flag:
                f_y3 = torch.cat((f_y3,self.loc_input),1)
            if not train_flag:
                # pdb.set_trace()
                f_y3 = torch.cat((f_y3,self.loc_input[0,:,:,:][None,:,:,:]),1)

        h = F.relu(self.ssd_conv5(f_y4))
        h = F.max_pool2d(h, kernel_size=3, padding=1, stride=1, ceil_mode=True)

        h = F.relu(self.ssd_conv6(h))
        h = F.relu(self.ssd_conv7(h))
        hs.append(h)  # conv7

        y4 = torch.cat((f_y4,h), 1) # 2048, 32, 32
        pooled_features = []
        if gt is not None:
            # pdb.set_trace()
            pooled_features = self.roi_pool(h, gt)
        y4 = self.fconv4(y4)

        h = F.relu(self.ssd_conv8_1(h))
        h = F.relu(self.ssd_conv8_2(h))
        hs.append(h)  # conv8_2

        # FCN conv5 + SSD conv8_2 
        y5 = torch.cat((f_y5,h), 1) # 2560, 16, 16
        # y5_f = y5
        y5 = self.fconv5(y5)
        # FCN conv3
        y3 = self.fconv3(f_y3)

        y = y4 + F.upsample(y5, y4.size()[2:], None, 'bilinear')
        y = y3 + F.upsample(y, y3.size()[2:], None, 'bilinear')
        y = F.upsample(y, x.size()[2:], None, 'bilinear')

        if not train_flag:
            return y 

        h = F.relu(self.ssd_conv9_1(h))
        h = F.relu(self.ssd_conv9_2(h))
        hs.append(h)  # conv9_2


        h = F.relu(self.ssd_conv10_1(h))
        h = F.relu(self.ssd_conv10_2(h))
        hs.append(h)  # conv10_2

        h = F.relu(self.ssd_conv11_1(h))
        h = F.relu(self.ssd_conv11_2(h))
        hs.append(h)  # conv11_2

        loc_preds, conf_preds = self.multibox(hs)

        # DONE ROI Pooling



        return y, f_y5, loc_preds, conf_preds, pooled_features
