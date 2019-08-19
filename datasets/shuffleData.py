import numpy as np
from config import cfg

def getShuffleIdx(lenOfData):

    idx_for_obc = np.random.permutation(lenOfData)

    if cfg.TRAIN.OB_BATCH_SIZE > lenOfData:
        while len(idx_for_obc) < cfg.TRAIN.OB_BATCH_SIZE:
            idx_for_obc = np.hstack((idx_for_obc,idx_for_obc))


    idx_for_obc = idx_for_obc[0:cfg.TRAIN.OB_BATCH_SIZE]

    return idx_for_obc