import numpy as np
from PIL import Image
'''
from .config import palette


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask
'''
from .config_GTA import ignored_label, label_colors


def colorize_mask(mask):
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    cmap = np.zeros((h, w, 3))

    for i in xrange(h):
        for j in xrange(w):
            v = mask[i, j]
            if v == ignored_label:
                continue
            cmap[i, j, :] = label_colors[v]

    return Image.fromarray(cmap.astype(np.uint8), 'RGB')
