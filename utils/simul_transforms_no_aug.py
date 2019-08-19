from __future__ import division
import numbers
import random

from PIL import Image, ImageOps


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, bbx):
        for t in self.transforms:
            img, mask, bbx = t(img, mask, bbx)
        return img, mask, bbx


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, bbx):
        if random.random() < 0.5:
            w, h = img.size
            xmin = w - bbx[:,2]
            xmax = w - bbx[:,0]
            bbx[:,0] = xmin
            bbx[:,2] = xmax
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), bbx
        return img, mask, bbx


class FreeScale(object):
    def __init__(self, size, interpolation=Image.NEAREST):
        self.size = size  # (h, w)
        self.interpolation = interpolation

    def __call__(self, img, mask, bbx):
        w, h = img.size
        scale_w = self.size[1]/w
        scale_h = self.size[0]/h
        if bbx.shape[1] !=4:
        	pdb.set_trace()
        bbx[:,[0,2]] = bbx[:,[0,2]]*scale_w
        bbx[:,[1,3]] = bbx[:,[1,3]]*scale_h
        return img.resize((self.size[1], self.size[0]), self.interpolation), mask.resize(self.size, self.interpolation), bbx
