from __future__ import division
import numbers
import random
import numpy as np

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

# ========================= Augmentation ============================
# numpy operation

class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        # image = np.array(image)
        image = image[:, :, self.swaps]
        # image = Image.fromarray(image.astype("uint8"))
        return image

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        
        if random.random() < 0.5:
            image = np.array(image)
            image[:, :, 1] = image[:, :, 1] * random.uniform(self.lower, self.upper)
            image = Image.fromarray(image.astype("uint8"))
        return image, boxes, labels



class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        image = np.array(image)
        if random.random() < 0.5:
            image[:, :, 0] = image[:, :, 0] + random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] = image[:, :, 0][image[:, :, 0] > 360.0] - 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] = image[:, :, 0][image[:, :, 0] < 0.0] + 360.0

        image = Image.fromarray(image.astype("uint8"))
        return image, boxes, labels

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        
        if random.random() < 0.5:
            image = np.array(image)
            swap = self.perms[random.randint(0,len(self.perms)-1)]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
            image = Image.fromarray(image.astype("uint8"))
        return image, boxes, labels

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        
        if random.random() < 0.5:
            image = np.array(image)
            alpha = random.uniform(self.lower, self.upper)
            image = image * alpha
            image = Image.fromarray(image.astype("uint8"))
        return image, boxes, labels

class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        
        if random.random() < 0.5:
            image = np.array(image)
            delta = random.uniform(-self.delta, self.delta)
            image = image + delta
            image = Image.fromarray(image.astype("uint8"))    
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='RGB', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'RGB' and self.transform == 'HSV':
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image = image.convert('HSV')
        elif self.current == 'HSV' and self.transform == 'RGB':
            # image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
            image = image.convert('RGB')
        else:
            raise NotImplementedError
        return image, boxes, labels


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='RGB'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.random() < 0.5:
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)