import os

import numpy as np
from PIL import Image
from torch.utils import data

from .config_City import processed_train_path, processed_val_path
from ..encoder import DataEncoder
from ..shuffleData import getShuffleIdx
from config import cfg

import torch

def default_loader(path):
    return Image.open(path)

class CityScapes(data.Dataset):
    def __init__(self, mode, list_filename, simul_transform=None, transform=None, target_transform=None):

        rootPath = []

        if mode=='train':
            rootPath = processed_train_path
        elif mode == 'val':
            rootPath = processed_val_path
        else:
            print 'Error of Dataset Mode!'

        self.img_root = os.path.join(rootPath, 'img')
        self.mask_root = os.path.join(rootPath, 'mask')
        list_file = rootPath + '/' + list_filename

        self.simul_transform = simul_transform
        self.transform = transform
        self.target_transform = target_transform

        self.data_encoder = DataEncoder()

        self.fnames = []
        self.boxes = []
        self.labels = []
        # self.ori_boxes = []
        
        with open(list_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])

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
            self.boxes.append(torch.Tensor(box))
            # self.ori_boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.img_loader = default_loader

    def __getitem__(self, idx):

        fname = self.fnames[idx]
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]
        ori_labels = self.labels[idx].clone()

        img = self.img_loader(os.path.join(self.img_root,fname))
        maskName = fname.split('leftImg8bit.png')[0] + 'gtFine_labelIds.png'
        mask = self.img_loader(os.path.join(self.mask_root,maskName))

        # flip and rescale
        if self.simul_transform is not None:
            img, mask, boxes = self.simul_transform(img, mask, boxes)

        ori_boxes = boxes.clone()
        # Scale bbox locaitons to [0,1]
        w,h = img.size
        boxes = boxes/torch.Tensor([w,h,w,h]).expand_as(boxes)

        # Encode bbx & objects labels.
        boxes, labels = self.data_encoder.encode(boxes, labels)

        # gen roi data for roipooling 
        shuffle_idx = getShuffleIdx(ori_boxes.size()[0])
        shuffle_idx = torch.from_numpy(shuffle_idx.astype(np.int64))
        ori_boxes = torch.index_select(ori_boxes, 0, shuffle_idx)
        ori_labels = torch.index_select(ori_labels, 0, shuffle_idx)

        # Normalize
        if self.transform is not None:
            img = self.transform(img)
        # change the seg labels 255->19    
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask, boxes, labels, ori_boxes, ori_labels

    def __len__(self):
        return self.num_samples