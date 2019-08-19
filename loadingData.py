from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms as standard_transforms
import utils.simul_transforms as simul_transforms
import utils.transforms as expanded_transforms

from config import cfg
from datasets.GTA5 import GTA5
from datasets.SYN import SYN
from datasets.cityscapes import CityScapes


def load_dataset():
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if cfg.TRAIN.DATA_AUG:
        source_simul_transform = simul_transforms.Compose([
            simul_transforms.FreeScale(cfg.TRAIN.IMG_SIZE),
            simul_transforms.RandomHorizontallyFlip(),
            simul_transforms.PhotometricDistort()
        ])
        target_simul_transform = simul_transforms.Compose([
            simul_transforms.FreeScale(cfg.TRAIN.IMG_SIZE),
            simul_transforms.RandomHorizontallyFlip(),
            simul_transforms.PhotometricDistort()
        ])
    else:
        source_simul_transform = simul_transforms.Compose([
            simul_transforms.FreeScale(cfg.TRAIN.IMG_SIZE),
            simul_transforms.RandomHorizontallyFlip(),

        ])
        target_simul_transform = simul_transforms.Compose([
            simul_transforms.FreeScale(cfg.TRAIN.IMG_SIZE),
            simul_transforms.RandomHorizontallyFlip(),
        ])\

    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = standard_transforms.Compose([
        expanded_transforms.MaskToTensor(),
        expanded_transforms.ChangeLabel(cfg.DATA.IGNORE_LABEL, cfg.DATA.NUM_CLASSES - 1)
    ])
    restore_transform = standard_transforms.Compose([
        expanded_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    print '='*50
    print 'Prepare Data...'
    source_set = []
    if cfg.TRAIN.SOURCE_DOMAIN=='GTA5':
        source_set = GTA5('train', list_filename = 'GTA5_'+ cfg.DATA.SSD_GT + '.txt', simul_transform=source_simul_transform, transform=img_transform,
                           target_transform=target_transform)
    elif cfg.TRAIN.SOURCE_DOMAIN=='SYN':
    	source_set = SYN('train', list_filename = 'SYN_'+ cfg.DATA.SSD_GT + '.txt', simul_transform=source_simul_transform, transform=img_transform,
                           target_transform=target_transform)

    source_loader = DataLoader(source_set, batch_size=cfg.TRAIN.IMG_BATCH_SIZE, num_workers=16, shuffle=True, drop_last=True)
    
    target_set = CityScapes('train', list_filename = 'cityscapes_'+ cfg.DATA.SSD_GT + '.txt',simul_transform=target_simul_transform, transform=img_transform,
                         target_transform=target_transform)
    target_loader = DataLoader(target_set, batch_size=cfg.TRAIN.IMG_BATCH_SIZE, num_workers=16, shuffle=True, drop_last=True)

    return source_loader, target_loader, restore_transform

    

def load_val_dataset():
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    source_simul_transform = simul_transforms.Compose([
        simul_transforms.FreeScale(cfg.VAL.IMG_SIZE)
    ])
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    target_transform = standard_transforms.Compose([
        expanded_transforms.MaskToTensor(),
        expanded_transforms.ChangeLabel(cfg.DATA.IGNORE_LABEL, cfg.DATA.NUM_CLASSES - 1)
    ])
    restore_transform = standard_transforms.Compose([
        expanded_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    print '='*50
    print 'Prepare Data...'
    val_set = CityScapes('val', list_filename = 'cityscapes_all.txt', simul_transform=source_simul_transform, \
                            transform=img_transform, target_transform=target_transform)
    target_loader = DataLoader(val_set, batch_size=cfg.VAL.IMG_BATCH_SIZE, num_workers=16, shuffle=True)

    return source_loader, target_loader, restore_transform
