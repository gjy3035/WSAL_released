import math
import sys
import os

from config import cfg
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.backends import cudnn
from datasets.GTA5.utils_GTA import colorize_mask
from models.fcn8_ssd import FCN8ResNet,FCN8VGG
from models.dc import dc_pixel,dc_object,dc_object2
from datasets.encoder import DataEncoder
import torch.optim as optim
from PIL import ImageDraw


import pdb

def init_model(key):
    

    resume = cfg.TRAIN.RESUME
    etractNet=[]
    dcNet=[]
    
    if key=='res':
        etractNet = FCN8ResNet(num_classes=cfg.DATA.NUM_CLASSES)
        dcNet = dc_pixel(input_channels=2048)
    elif key=='dense':
        etractNet = FCN8DenseNet(num_classes=cfg.DATA.NUM_CLASSES)
        dcNet = dc_pixel(input_channels=1920)
    elif key=='vgg':
        etractNet = FCN8VGG(num_classes=cfg.DATA.NUM_CLASSES) 
        dcNet = dc_pixel(input_channels=512)
    else:
        print 'Invalid Networks!!!'
    obc_Net = dc_object(num_classes=cfg.DATA.NUM_CLASSES)

    max_epoch = 0
    if resume:
        ckpt_path = './ckpt/' + cfg.TRAIN.EXP_NAME + '/'
        extName,max_epoch = find_newest_model(ckpt_path)
        if os.path.exists(ckpt_path + extName):
            etractNet.load_state_dict(torch.load(ckpt_path + extName))
            print 'Successfully load ' + extName + '.'
        else:
            print 'No ext resume!!!'

        if os.path.exists(ckpt_path + 'dc.pth'):
            dcNet.load_state_dict(torch.load(ckpt_path + 'dc.pth'))
            print 'Successfully load dc.pth.'
        else:
            print 'No dc resume!!!'

        if os.path.exists(ckpt_path + 'obc.pth'):    
            obc_Net.load_state_dict(torch.load(ckpt_path + 'obc.pth'))
            print 'Successfully load dc.pth.'
        else:
            print 'No obc resume!!!'

    if cfg.TRAIN.MULTI_GPU:
        etractNet = torch.nn.DataParallel(etractNet, device_ids=cfg.TRAIN.GPU_ID).cuda()
        dcNet = torch.nn.DataParallel(dcNet, device_ids=cfg.TRAIN.GPU_ID).cuda()
        obc_Net = torch.nn.DataParallel(obc_Net, device_ids=cfg.TRAIN.GPU_ID).cuda()
    else:
        etractNet = etractNet.cuda()
        dcNet = dcNet.cuda()
        obc_Net = obc_Net.cuda()
    

    cudnn.benchmark = True

    etractNet.train()
    dcNet.train()
    obc_Net.train()
    
    return etractNet, dcNet, obc_Net, max_epoch

def find_newest_model(path):
    epoch = []
    for model_name in os.listdir(path):
        if model_name.endswith('_ext.pth'):
            i_epoch = int(model_name.split('_')[1])
            epoch.append(i_epoch)
    max_epoch = max(epoch)
    newest_model = 'epoch_'+ str(max_epoch) + '_ext.pth'

    return newest_model,max_epoch


def set_extract_optimizer(etractNet, i_epoch):
    pretrained_lr = cfg.TRAIN.EXT_LR
    new_lr = cfg.TRAIN.SEG_LR
    ssd_lr = cfg.TRAIN.SSD_LR
    weight_decay = cfg.TRAIN.WEIGHT_DECAY

    ssd_lr = ssd_lr * (cfg.TRAIN.SSD_LR_DECAY_RATE **(i_epoch / cfg.TRAIN.SSD_LR_DECAY_EPOCH))
    new_lr = new_lr * (cfg.TRAIN.SEG_LR_DECAY_RATE **(i_epoch / cfg.TRAIN.SEG_LR_DECAY_EPOCH))
    extracter_optimizer = optim.SGD([
        {'params': [param for name, param in etractNet.named_parameters() if
                    name[-4:] == 'bias' and 'fconv' in name], 'lr': 2 * new_lr},
        {'params': [param for name, param in etractNet.named_parameters() if
                    name[-4:] != 'bias' and 'fconv' in name], 'lr': new_lr, 'weight_decay': weight_decay},
        {'params': [param for name, param in etractNet.named_parameters() if
                    name[-4:] == 'bias' and 'ssd' in name], 'lr': 2 * ssd_lr},
        {'params': [param for name, param in etractNet.named_parameters() if
                    name[-4:] != 'bias' and 'ssd' in name],'lr': ssd_lr, 'weight_decay': weight_decay},
        {'params': [param for name, param in etractNet.named_parameters() if
                    name[-4:] == 'bias' and 'fconv' not in name and 'ssd' not in name], 'lr': 2 * pretrained_lr},
        {'params': [param for name, param in etractNet.named_parameters() if
                    name[-4:] != 'bias' and 'fconv' not in name and 'ssd' not in name],'lr': pretrained_lr, 'weight_decay': weight_decay},

    ], momentum=0.9, nesterov=True)
    return extracter_optimizer


def calculate_mean_iu(predictions, gts):
    sum_iu = 0
    class_iu = np.zeros([cfg.DATA.NUM_CLASSES])
    for i in xrange(cfg.DATA.NUM_CLASSES):
        n_ii = t_i = sum_n_ji = 1e-9
        for p, gt in zip(predictions, gts):
            n_ii += np.sum(gt[p == i] == i)
            t_i += np.sum(gt == i)
            sum_n_ji += np.sum(p == i)

        class_iu[i] = float(n_ii) / (t_i + sum_n_ji - n_ii)
        sum_iu += float(n_ii) / (t_i + sum_n_ji - n_ii)
    mean_iu = sum_iu / (cfg.DATA.NUM_CLASSES)
    return mean_iu, class_iu

def logger(i_epoch,i_iter,i_tb,writer,ext_loss,sp_loss,unsp_loss,det_loss,src_m_iu,tgt_m_iu,
            dc_loss=None,dc_ins_loss=None,
            obc_loss=None,obc_ins_loss=None):

    batch_size = cfg.TRAIN.IMG_BATCH_SIZE
    writer.add_scalar('loss_ext', ext_loss.data[0], i_tb*batch_size*cfg.TRAIN.PRINT_FREQ)
    writer.add_scalar('loss_sp', sp_loss.data[0], i_tb*batch_size*cfg.TRAIN.PRINT_FREQ)
    writer.add_scalar('loss_unsp', unsp_loss.data[0], i_tb*batch_size*cfg.TRAIN.PRINT_FREQ)
    writer.add_scalar('loss_det', det_loss.data[0], i_tb*batch_size*cfg.TRAIN.PRINT_FREQ)    
    writer.add_scalar('meanIU_src', src_m_iu, i_tb*batch_size*cfg.TRAIN.PRINT_FREQ)
    writer.add_scalar('meanIU_tgt', tgt_m_iu, i_tb*batch_size*cfg.TRAIN.PRINT_FREQ)
    
    if dc_loss is not None:
        writer.add_scalar('loss_dc', dc_loss.data[0], i_tb*batch_size*cfg.TRAIN.PRINT_FREQ)    
        writer.add_scalar('loss_dc_ins', dc_ins_loss.data[0], i_tb*batch_size*cfg.TRAIN.PRINT_FREQ)
        
    if obc_loss is not None:    
        writer.add_scalar('loss_obc', obc_loss.data[0], i_tb*batch_size*cfg.TRAIN.PRINT_FREQ)
        writer.add_scalar('loss_obc_ins', obc_ins_loss.data[0], i_tb*batch_size*cfg.TRAIN.PRINT_FREQ)

    # print infos in terminial
    if cfg.TRAIN.COM_EXP == 5:
        print '[ep %d], [it %d], [Loss: dc %.4f, ext_all %.4f, sp %.4f, det %.4f, ivs1 %.4f, ivs2 %.4f], [src_miu %.4f], [tgt_miu %.4f]' \
            % (i_epoch + 1, i_iter + 1, dc_loss.data[0], ext_loss.data[0], sp_loss.data[0], det_loss.data[0], \
            dc_ins_loss.data[0], obc_ins_loss.data[0],src_m_iu,tgt_m_iu)
            
    elif cfg.TRAIN.COM_EXP == 4:
        print '[ep %d], [it %d], [Loss: dc %.4f, ext_all %.4f, sp %.4f, det %.4f, ivs1 %.4f], [src_miu %.4f], [tgt_miu %.4f]' \
            % (i_epoch + 1, i_iter + 1, dc_loss.data[0], ext_loss.data[0], sp_loss.data[0], det_loss.data[0], \
                dc_ins_loss.data[0], src_m_iu,tgt_m_iu)

    elif cfg.TRAIN.COM_EXP == 3:
        print '[ep %d], [it %d], [Loss: ext_all %.4f, sp %.4f, det %.4f], [src_miu %.4f], [tgt_miu %.4f]' \
        % (i_epoch + 1, i_iter + 1, ext_loss.data[0], sp_loss.data[0], det_loss.data[0], src_m_iu,tgt_m_iu)
    
    elif cfg.TRAIN.COM_EXP == 6:
        print '[ep %d], [it %d], [Loss: ext_all %.4f, sp %.4f, det %.4f, ivs2 %.4f], [src_miu %.4f], [tgt_miu %.4f]' \
            % (i_epoch + 1, i_iter + 1, ext_loss.data[0], sp_loss.data[0], det_loss.data[0], \
         obc_ins_loss.data[0],src_m_iu,tgt_m_iu)




def show_img(i_tb, writer,src_label_color,src_pred_color,tgt_label_color,tgt_pred_color, 
            src_det_img, tgt_det_img):

    batch_size = cfg.TRAIN.IMG_BATCH_SIZE
    writer.add_image('gt_src', src_label_color)
    writer.add_image('pred_src', src_pred_color)
    writer.add_image('det_src', np.array(src_det_img))

    writer.add_image('gt_tgt', tgt_label_color)
    writer.add_image('pred_tgt', tgt_pred_color)    
    writer.add_image('det_tgt', np.array(tgt_det_img))
    

def eval_ext_logs(outputs,labels):
    outputs = outputs[:, :cfg.DATA.NUM_CLASSES - 1, :, :]
    # pdb.set_trace()

    prediction = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()
    mean_iu, class_iu = calculate_mean_iu(prediction, labels.data.cpu().numpy())

    return mean_iu, class_iu 

def gen_labeled_map(outputs,labels=None):
    map_size = cfg.TRAIN.IMG_SIZE
    dst_label_shape = (map_size[0],map_size[1],1)
    dst_RGB_shape = (map_size[0],map_size[1],3)
    # pdb.set_trace()
    prediction = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()
    pred_colormap = np.array(colorize_mask(prediction[0])).reshape(dst_RGB_shape)
    if labels is None:
    	return pred_colormap
    # pdb.set_trace()
    label_colormap = np.array(colorize_mask(labels.data[0].cpu().numpy())).reshape(dst_RGB_shape)

    return pred_colormap, label_colormap


# Done
def draw_bbx(loc_preds, conf_preds, img, restore):

    data_encoder = DataEncoder()
    bbx, _, _ = \
    data_encoder.decode(loc_preds.data.squeeze(0), F.softmax(conf_preds.squeeze(0)).data)

    map_size = cfg.TRAIN.IMG_SIZE
    bbx = bbx * map_size[0]
    # pdb.set_trace()
    img = restore(img)
    # print img.size()
    imgDraw = ImageDraw.Draw(img)
    for i_bbx in range(0,bbx.shape[0]):
        imgDraw.rectangle(bbx[i_bbx,:].tolist(),outline = "red")
    del imgDraw

    return img

def gen_obc_labels(src_obj_labels,tgt_obj_labels):
    len_src_obj = src_obj_labels.size()[0]*src_obj_labels.size()[1]
    len_tgt_obj = tgt_obj_labels.size()[0]*tgt_obj_labels.size()[1]

    src_obj_labels = src_obj_labels.resize_(len_src_obj,1)
    tgt_obj_labels = tgt_obj_labels.resize_(len_tgt_obj,1)

    obc_labels = torch.zeros((src_obj_labels.size()[0] + tgt_obj_labels.size()[0]))
    obc_ivs_labels = torch.zeros(obc_labels.shape)

    for i_src, label in enumerate(src_obj_labels):
        obc_labels[i_src] = label.numpy()[0]
        obc_ivs_labels[i_src] = cfg.DATA.NUM_CLASSES + label.numpy()[0]
    for i_tgt, label in enumerate(tgt_obj_labels):
        obc_labels[src_obj_labels.size()[0] + i_tgt] = cfg.DATA.NUM_CLASSES + label.numpy()[0]
        obc_ivs_labels[src_obj_labels.size()[0] + i_tgt] = label.numpy()[0]

    obc_labels = Variable(obc_labels.long().cuda())
    obc_ivs_labels = Variable(obc_ivs_labels.long().cuda())
    return obc_labels, obc_ivs_labels

def gen_dc_pixels_label():
    batch_size = cfg.TRAIN.IMG_BATCH_SIZE
    dc_labels = np.zeros((2 * batch_size, cfg.TRAIN.IMG_SIZE[0],cfg.TRAIN.IMG_SIZE[1]))
    dc_ivs_labels = np.zeros(dc_labels.shape)
    dc_labels[:batch_size,:,:] = 1
    dc_ivs_labels[batch_size:,:,:] = 1
    dc_labels = torch.from_numpy(dc_labels.astype(np.int64))
    dc_labels = Variable(dc_labels.cuda())
    dc_ivs_labels = torch.from_numpy(dc_ivs_labels.astype(np.int64))
    dc_ivs_labels = Variable(dc_ivs_labels.cuda())
    return dc_labels, dc_ivs_labels



def forward_ext_model(ext_model, ext_src_inputs, ext_tgt_inputs,
                        src_bbx,tgt_bbx):
    pred_tgt_outputs, dc_tgt_inputs, tgt_loc_preds, tgt_conf_preds, tgt_pooled_features \
                        = ext_model(Variable(ext_tgt_inputs.cuda()),gt=Variable(tgt_bbx.cuda()))

    pred_src_outputs, dc_src_inputs, src_loc_preds, src_conf_preds, src_pooled_features \
                    = ext_model(Variable(ext_src_inputs.cuda()),gt=Variable(src_bbx.cuda()))

    loc_preds = torch.cat((src_loc_preds, tgt_loc_preds),0)
    conf_preds = torch.cat((src_conf_preds, tgt_conf_preds),0)
    dc_inputs = torch.cat((dc_src_inputs, dc_tgt_inputs),0)
    pooled_features = torch.cat((src_pooled_features, tgt_pooled_features),0)
    return pred_src_outputs, pred_tgt_outputs, loc_preds, conf_preds, dc_inputs, pooled_features


def preapre_bbx_for_roipol(bbx):
    len_obj = bbx.size()[0]*bbx.size()[1]

    
    img_batch_size = cfg.TRAIN.IMG_BATCH_SIZE
    obj_batch_size = cfg.TRAIN.OB_BATCH_SIZE
    
    if len_obj != obj_batch_size*img_batch_size:
        print 'Batch Size is ERROR!'

    bbx = bbx.resize_(len_obj,4)
    
    batch_ind = torch.zeros(len_obj,1)
    for i_img in range(0,img_batch_size):
        batch_ind[i_img*obj_batch_size:(i_img+1)*obj_batch_size,0] = i_img
    # pdb.set_trace()
    bbx = torch.cat((batch_ind,bbx),1)
    return bbx


     

