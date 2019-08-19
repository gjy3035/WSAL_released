import os
from easydict import EasyDict as edict
import torch

__C = edict()

cfg = __C

__C.DATA = edict()

__C.DATA.NUM_CLASSES = 19 + 1

__C.DATA.IGNORE_LABEL = 255
# Training options


# devices: opt603a, optimal, qzq, gpu03
__C.DATA.SERVER = 'gpu03'
__C.DATA.ROOT = '/home/' + __C.DATA.SERVER + '/GJY/'

if __C.DATA.SERVER == 'gpu03':
    __C.DATA.ROOT = '/mount/gjy/Dataset/'
# detction gt: all: 200_400, 
#              300_500, 
#              small
__C.DATA.SSD_GT = 'all'



__C.TRAIN = edict()
__C.TRAIN.NET = 'res' 
__C.TRAIN.GPU_ID = [5]

__C.TRAIN.MULTI_GPU = None
if len(__C.TRAIN.GPU_ID)>1:
	__C.TRAIN.MULTI_GPU = True
else:
	__C.TRAIN.MULTI_GPU = False

# four important experiments
# 
# 1: FCN
# 2: FCN + DC
# 3: FCN + SSD
# 4: FCN + SSD + DC
# 5: FCN + SSD + DC + OBC
# 6: FCN + SSD + OBC
__C.TRAIN.COM_EXP = 6

cfg.TRAIN.RESUME = False


cfg.TRAIN.LOC_P = False

__C.TRAIN.SOURCE_DOMAIN = 'GTA5' # SYN and GTA5

__C.TRAIN.DATA_AUG = False

__C.TRAIN.LOSS_V2 = True
__C.TRAIN.IMG_SIZE = (512,512)

__C.TRAIN.EXT_LR = 1e-4
__C.TRAIN.SEG_LR = 1e-2
__C.TRAIN.SEG_LR_DECAY_RATE = 0.95
__C.TRAIN.SEG_LR_DECAY_EPOCH = 2
__C.TRAIN.SSD_LR = 1e-2
__C.TRAIN.SSD_LR_DECAY_RATE = 0.95
__C.TRAIN.SSD_LR_DECAY_EPOCH = 2
__C.TRAIN.DC_LR = 1e-4
__C.TRAIN.OBC_LR = 1e-4 
__C.TRAIN.WEIGHT_DECAY = 5e-4
__C.TRAIN.LOSS_WEIGHT = (1,1,1,1)


__C.TRAIN.LABEL_WEIGHT = torch.FloatTensor([
        0.08191489349313398, 0.47552527366794356, 0.13232434015378922, 1.5234863249754125, \
        1.520698973125806, 2.473327791287872, 8.21338514895267, 5.243302978265823, \
        0.18682776226615075, 1.4705083045570722, 0.6879584793023933, 1.9785913894963796, \
        7.793313357127503, 0.41677927582766033, 1.3817035726303235, 1.1992261505958535, \
        0.6276003920043414, 5.352514059058637, 4.093544867718577,0])

__C.TRAIN.IMG_BATCH_SIZE = 2
__C.TRAIN.OB_BATCH_SIZE = 40

__C.TRAIN.MAX_EPOCH = 1000

__C.TRAIN.ROI_POOLED_SIZE = []
if __C.TRAIN.NET=='res':
    __C.TRAIN.ROI_POOLED_SIZE = [1024,7,7,1.0/16]
elif __C.TRAIN.NET=='vgg':
	__C.TRAIN.ROI_POOLED_SIZE = [1024,7,7,1.0/16]



__C.TRAIN.SNAPSHOT = ''

__C.TRAIN.PRINT_FREQ = 40

__C.TRAIN.SHOW_SEG_DET_FREQ = 200

__C.TRAIN.SAVE_EPOCH = 2


__C.TRAIN.EXP_NAME = 'v5Exp' + str(__C.TRAIN.COM_EXP) + '_Net' + __C.TRAIN.NET[0] + \
    '_S_' + __C.TRAIN.SOURCE_DOMAIN[0] + \
    '_' + str(__C.TRAIN.LOSS_WEIGHT) + '_' + __C.DATA.SERVER + \
    '_Aug'+ str(__C.TRAIN.DATA_AUG)[0] + '_DET' + __C.DATA.SSD_GT[0] + \
    '_decay_' + str(__C.TRAIN.SSD_LR_DECAY_EPOCH) + \
    'loc'+ str(__C.TRAIN.LOC_P)[0]
print __C.TRAIN.EXP_NAME


__C.TRAIN.CKPT_PATH = './ckpt'
# make directory for models
if not os.path.exists(__C.TRAIN.CKPT_PATH):
    os.mkdir(__C.TRAIN.CKPT_PATH)
if not os.path.exists(os.path.join(__C.TRAIN.CKPT_PATH, __C.TRAIN.EXP_NAME)):
    os.mkdir(os.path.join(__C.TRAIN.CKPT_PATH, __C.TRAIN.EXP_NAME))

# make directory for loggers
__C.TRAIN.LOGROOT_PATH = './NetV2_exp'   
if not os.path.exists(__C.TRAIN.LOGROOT_PATH):
    os.mkdir(__C.TRAIN.LOGROOT_PATH)
__C.TRAIN.LOG_PATH = os.path.join(__C.TRAIN.LOGROOT_PATH, cfg.TRAIN.EXP_NAME)  
if not os.path.exists(__C.TRAIN.LOG_PATH):
    os.mkdir(__C.TRAIN.LOG_PATH)



# Validation options
__C.VAL = edict()
__C.VAL.GPU_ID = 3
__C.VAL.IMG_SIZE = __C.TRAIN.IMG_SIZE

# make directory for results
__C.VAL.IMG_RESULTS = './img_results'
if not os.path.exists(__C.VAL.IMG_RESULTS):
    os.mkdir(__C.VAL.IMG_RESULTS)

__C.VAL.EXP_PATH = os.path.join(__C.VAL.IMG_RESULTS, __C.TRAIN.EXP_NAME)
if not os.path.exists(__C.VAL.EXP_PATH):
    os.mkdir(__C.VAL.EXP_PATH)

__C.VAL.PRED_RESULTS = os.path.join(__C.VAL.EXP_PATH, 'pred')
if not os.path.exists(__C.VAL.PRED_RESULTS):
	os.mkdir(__C.VAL.PRED_RESULTS)

__C.VAL.GT_RESULTS = os.path.join(__C.VAL.EXP_PATH, 'GT')	
if not os.path.exists(__C.VAL.GT_RESULTS):
	os.mkdir(__C.VAL.GT_RESULTS)


