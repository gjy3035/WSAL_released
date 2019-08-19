# -*- coding: utf-8 -*-
import os
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss

from config import cfg
from datasets.encoder import DataEncoder
from loadingData import load_dataset
from utils.loss import CrossEntropyLoss2d
from train_net import train_adversarial
from utils.training import *
from utils.multibox_loss import MultiBoxLoss
import pdb

# torch.cuda.set_device(cfg.TRAIN.GPU_ID)

def main():
    if not cfg.TRAIN.MULTI_GPU:
        torch.cuda.set_device(cfg.TRAIN.GPU_ID[0])

    i_tb = 0
    # loading data
    src_loader, tgt_loader, restore_transform = load_dataset()
    data_encoder = DataEncoder()

    
    ext_model = None
    dc_model = None
    obc_model = None
    # initialize models
    if cfg.TRAIN.COM_EXP == 5: # Full model
        ext_model, dc_model, obc_model, cur_epoch = init_model(cfg.TRAIN.NET)
    elif cfg.TRAIN.COM_EXP == 6: # FCN + SSD + OBC
        ext_model, _, obc_model, cur_epoch = init_model(cfg.TRAIN.NET)
    elif cfg.TRAIN.COM_EXP == 4: # FCN + SSD + DC
        ext_model, dc_model, _, cur_epoch = init_model(cfg.TRAIN.NET)
    elif cfg.TRAIN.COM_EXP == 3: # FCN + SSD
        ext_model, _, __, cur_epoch = init_model(cfg.TRAIN.NET)
    
    # set criterion and optimizer, training
    if ext_model is not None: 
        weight = torch.ones(cfg.DATA.NUM_CLASSES)
        weight[cfg.DATA.NUM_CLASSES - 1] = 0
        spvsd_cri = CrossEntropyLoss2d(cfg.TRAIN.LABEL_WEIGHT).cuda() # traditional Loss for the FCN-8s
        unspvsd_cri = CrossEntropyLoss2d(cfg.TRAIN.LABEL_WEIGHT).cuda() # traditional Loss for the FCN-8s
        det_cri = MultiBoxLoss()
        # the ext_opt will be set in the train_net.py, because the ssd learning rate is stepsise        

    if dc_model is not None:
        dc_cri = CrossEntropyLoss2d().cuda()    
        dc_invs_cri = CrossEntropyLoss2d().cuda()
        dc_opt = optim.Adam(dc_model.parameters(), lr=cfg.TRAIN.DC_LR, betas=(0.5, 0.999))        

    if obc_model is not None:
        obc_cri = CrossEntropyLoss().cuda()
        obc_invs_cri = CrossEntropyLoss().cuda()
        obc_opt = optim.Adam(obc_model.parameters(), lr=cfg.TRAIN.OBC_LR, betas=(0.5, 0.999))

    if cfg.TRAIN.COM_EXP == 6:
        train_adversarial(cur_epoch, i_tb, data_encoder, src_loader, tgt_loader, restore_transform, 
                        ext_model, spvsd_cri, unspvsd_cri, det_cri, 
                        obc_model=obc_model, obc_cri=obc_cri, obc_invs_cri=obc_invs_cri, obc_opt=obc_opt)
        

    if cfg.TRAIN.COM_EXP == 5:
        train_adversarial(cur_epoch, i_tb, data_encoder, src_loader, tgt_loader, restore_transform, 
                        ext_model, spvsd_cri, unspvsd_cri, det_cri, 
                        dc_model=dc_model,  dc_cri=dc_cri, dc_invs_cri=dc_invs_cri, dc_opt=dc_opt, 
                        obc_model=obc_model, obc_cri=obc_cri, obc_invs_cri=obc_invs_cri, obc_opt=obc_opt)
    if cfg.TRAIN.COM_EXP == 4:
        train_adversarial(cur_epoch, i_tb, data_encoder, src_loader, tgt_loader, restore_transform, 
                        ext_model, spvsd_cri, unspvsd_cri, det_cri, 
                        dc_model=dc_model,  dc_cri=dc_cri, dc_invs_cri=dc_invs_cri, dc_opt=dc_opt)
    if cfg.TRAIN.COM_EXP == 3:
        train_adversarial(cur_epoch, i_tb, data_encoder, src_loader, tgt_loader, restore_transform, 
                        ext_model, spvsd_cri, unspvsd_cri, det_cri)

if __name__ == '__main__':
    main()