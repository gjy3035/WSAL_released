from __future__ import division
import os

import numpy as np
from PIL import Image
from encoder import DataEncoder
import scipy.io as sio

import torch

import pdb


def txtToNpy(rootPath, list_filename,dst_h,dst_w):

    list_file = rootPath + '/' + list_filename

    dst_label_path = rootPath + '/ssd_labels/'
    img_path = rootPath + '/images/'
    if not os.path.exists(dst_label_path):
    	os.mkdir(dst_label_path)

    data_encoder = DataEncoder()
   
    with open(list_file) as f:
        lines = f.readlines()

    i_lines = 0
    for line in lines:
    	print i_lines
    	i_lines = i_lines + 1
        splited = line.strip().split()
        imgName = splited[0]

        num_objs = int(splited[1])
        box = []
        label = []
        for i in range(num_objs):
            xmin = splited[2+5*i]
            ymin = splited[3+5*i]
            xmax = splited[4+5*i]
            ymax = splited[5+5*i]
            c = splited[6+5*i]
            box.append([float(xmin),float(ymin),float(xmax),float(ymax)])
            label.append(int(c))

        box = np.array(box)
        label = np.array(label)
        img = Image.open(img_path + imgName)
        w,h = img.size
        

        scale_w = dst_w/w
        scale_h = dst_h/h
        # pdb.set_trace()
        if box.shape[1] !=4:
            pdb.set_trace()
        box[:,[0,2]] = box[:,[0,2]]*scale_w
        box[:,[1,3]] = box[:,[1,3]]*scale_h

        box_lr = box.copy()# for flipping
        ori_box = box.copy()# for saving
        ori_label = label.copy()# for saving
        ori_label_for_genlr = label.copy()# for flipping
        

        # No flipping
        # Normalize
        box[:,[0,2]] = box[:,[0,2]]/dst_w
        box[:,[1,3]] = box[:,[1,3]]/dst_h

        box = (torch.from_numpy(box)).float()
        label = torch.from_numpy(label)
        # pdb.set_trace()

        box_ssd, label_ssd = data_encoder.encode(box, label)
        # sio.savemat(dst_label_path + imgName.split('.png')[0] + '.mat', {'box':box, 'label':label})
        box_ssd = box_ssd.numpy()
        label_ssd = label_ssd.numpy()
        





        # Flipping
        xmin = dst_w - box_lr[:,2]
        xmax = dst_h - box_lr[:,0]
        box_lr[:,0] = xmin
        box_lr[:,2] = xmax

        ori_box_lr = box_lr.copy() # for saving 
        ori_label_lr = ori_label.copy() # for saving 



        box_lr[:,[0,2]] = box_lr[:,[0,2]]/dst_w
        box_lr[:,[1,3]] = box_lr[:,[1,3]]/dst_h

        box_lr = (torch.from_numpy(box_lr)).float()
        ori_label_for_genlr = torch.from_numpy(ori_label_for_genlr)

        box_ssd_lr, label_ssd_lr = data_encoder.encode(box_lr, ori_label_for_genlr)
        box_ssd_lr = box_ssd_lr.numpy()
        label_ssd_lr = label_ssd_lr.numpy()
        


        sio.savemat(dst_label_path + imgName.split('.png')[0] + '.mat', {'box_ssd':box_ssd, 'label_ssd':label_ssd, \
        																	'box_ssd_lr':box_ssd_lr, 'label_ssd_lr':label_ssd_lr, \
        																	'ori_box':ori_box, 'ori_label':ori_label, \
        																	'ori_box_lr':ori_box_lr, 'ori_label_lr':ori_label_lr})
        # pdb.set_trace()
        # xxx=1

txtToNpy('/home/optimal/GJY/GTA5', 'GTA5_all.txt',512,512)