import os
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.transforms as standard_transforms
from tensorboard import SummaryWriter
import utils.transforms as expanded_transforms

from PIL import Image

from config import cfg
from datasets.cityscapes.config_City import processed_val_path, ignored_label
from utils.training import *
from utils.timer import Timer

def eval_net(i_epoch,i_iter,i_tb,writer,ext_model):
    ext_model.eval()
    # processed_val_img_path = os.path.join(processed_val_path, 'img')
    processed_val_img_path = os.path.join(processed_val_path, 'img')
    processed_val_mask_path = os.path.join(processed_val_path, 'mask')
    valSet = []
    for img_name in [img_name.split('leftImg8bit.png')[0] for img_name in os.listdir(processed_val_img_path)]:
        item = (processed_val_img_path + '/' + img_name + 'leftImg8bit.png', processed_val_mask_path + '/'+ img_name + 'gtFine_labelIds.png')
        valSet.append(item)


    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    restore_transform = standard_transforms.Compose([
        expanded_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])
    print '='*50
    print 'Start validating...'
    all_pred = np.zeros((0, cfg.VAL.IMG_SIZE[0], cfg.VAL.IMG_SIZE[1]))
    all_labels = np.zeros((0, cfg.VAL.IMG_SIZE[0], cfg.VAL.IMG_SIZE[1]))

    i_img = 0
    all_pred_list = []
    all_labels_list = []
    _t = {'iter' : Timer()}  
    for val_data in valSet:        

        img_path, mask_path = val_data
        img = Image.open(img_path)
        img = img.resize(cfg.VAL.IMG_SIZE,Image.NEAREST)
        img_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        img = img_transform(img)

        labels = Image.open(mask_path)
        labels = labels.resize(cfg.VAL.IMG_SIZE)
        label_transform = standard_transforms.Compose([expanded_transforms.MaskToTensor(),
                                expanded_transforms.ChangeLabel(ignored_label, cfg.DATA.NUM_CLASSES - 1)])
        labels = label_transform(labels)
        labels = labels[None,:,:]            
        img = Variable(img[None,:,:,:],volatile=True).cuda()

        
        _t['iter'].tic()
        # forward ext model
        # pred_val_outputs = forward_ext_model(ext_tgt_inputs = img)                          
        pred_val_outputs = ext_model(img,train_flag=False)
        _t['iter'].toc(average=True)
        
        if i_img % 50 ==0:
        	print 'i_img: {:d}, net_forward: {:.3f}s'.format(i_img,_t['iter'].average_time)

        pred_map = pred_val_outputs.data.cpu().max(1)[1].squeeze_(1).numpy()
        
        all_pred_list.append(pred_map.tolist())
        all_labels_list.append(labels.numpy().tolist())
        
        i_img = i_img + 1

    all_pred_np = np.array(all_pred_list)
    all_labels_np = np.array(all_labels_list)
    
    all_pred = all_pred_np.reshape((-1,all_pred_np.shape[2],all_pred_np.shape[3]))
    all_labels = all_labels_np.reshape((-1,all_labels_np.shape[2],all_labels_np.shape[3]))
         
    tgt_m_iu, tgt_class_iu = calculate_mean_iu_test(all_pred, all_labels)
    # pdb.set_trace()
    batch_size = cfg.TRAIN.IMG_BATCH_SIZE
    writer.add_scalar('meanIU_tgt_val', tgt_m_iu, i_tb*batch_size*cfg.TRAIN.PRINT_FREQ)
    print tgt_m_iu
    print tgt_class_iu
    ext_model.train()
    print '='*50
    print 'COntinue training...'   



def train_adversarial(cur_epoch, i_tb, data_encoder, src_loader, tgt_loader, restore_transform, 
                        ext_model, spvsd_cri, unspvsd_cri, det_cri, 
                        dc_model=None,  dc_cri=None, dc_invs_cri=None, dc_opt=None, 
                        obc_model=None, obc_cri=None, obc_invs_cri=None, obc_opt=None):

    writer = SummaryWriter(cfg.TRAIN.LOG_PATH)
    ext_opt = set_extract_optimizer(ext_model,cur_epoch)

    eval_net(cur_epoch,0,cur_epoch*37,writer,ext_model)

    print '='*50
    print 'Start Training FCN-8s...'
    for i_epoch in range(cur_epoch, cfg.TRAIN.MAX_EPOCH):

        i_tb = i_epoch*37 # 2960/40/2

        if i_epoch%10 ==0 or i_epoch > 40:
            ext_opt = set_extract_optimizer(ext_model,i_epoch)
        
        # for i_iter, src_data in enumerate(src_loader, 0):
        for i_iter, tgt_data in enumerate(tgt_loader, 0):
            # 
            # prepare data
            _t = {'pre_data':Timer(), 'net' : Timer()}
            _t['pre_data'].tic()
            ext_src_inputs, src_labels, src_bbx_ssd, src_obj_labels_ssd, src_bbx, src_obj_labels \
                                        = iter(src_loader).next() # src_data

            ext_tgt_inputs, tgt_labels, tgt_bbx_ssd, tgt_obj_labels_ssd, tgt_bbx, tgt_obj_labels \
                                        = tgt_data # iter(tgt_loader).next()            
            _t['pre_data'].toc(average=False)
            _t['net'].tic()
            
            src_labels = Variable(src_labels.cuda(),requires_grad=False)  
            src_bbx_ssd = Variable(src_bbx_ssd.cuda(),requires_grad=False)
            src_obj_labels_ssd = Variable(src_obj_labels_ssd.cuda(),requires_grad=False)             
            tgt_labels = Variable(tgt_labels.cuda(),requires_grad=False)
            tgt_bbx_ssd = Variable(tgt_bbx_ssd.cuda(),requires_grad=False)
            tgt_obj_labels_ssd = Variable(tgt_obj_labels_ssd.cuda(),requires_grad=False)



            # ext model
            tgt_bbx = preapre_bbx_for_roipol(tgt_bbx)
            src_bbx = preapre_bbx_for_roipol(src_bbx)
            
            pred_tgt_outputs, dc_tgt_inputs, tgt_loc_preds, tgt_conf_preds, tgt_pooled_features \
                    = ext_model(Variable(ext_tgt_inputs.cuda()),gt=Variable(tgt_bbx.cuda()))

            pred_src_outputs, dc_src_inputs, src_loc_preds, src_conf_preds, src_pooled_features \
                            = ext_model(Variable(ext_src_inputs.cuda()),gt=Variable(src_bbx.cuda()))           

            # concat ssd ouputs and labels
            loc_preds = torch.cat((src_loc_preds, tgt_loc_preds),0)
            conf_preds = torch.cat((src_conf_preds, tgt_conf_preds),0)
            loc_gt = torch.cat((src_bbx_ssd, tgt_bbx_ssd),0)
            conf_gt = torch.cat((src_obj_labels_ssd, tgt_obj_labels_ssd),0)           

            # ext model
            ext_opt.zero_grad()
            # pdb.set_trace()
            sp_loss = spvsd_cri(pred_src_outputs,src_labels)
            unsp_loss = unspvsd_cri(pred_tgt_outputs,tgt_labels)
            det_loss = det_cri(loc_preds, loc_gt, conf_preds, conf_gt)

            ext_loss = sp_loss*cfg.TRAIN.LOSS_WEIGHT[0] + det_loss*cfg.TRAIN.LOSS_WEIGHT[1]

            # dc model           
            if cfg.TRAIN.COM_EXP in [2,4,5]:
                # generate the dc labels
                dc_labels, dc_ivs_labels = gen_dc_pixels_label()
                dc_inputs = torch.cat((dc_src_inputs, dc_tgt_inputs),0)
                # pdb.set_trace()
                dc_outputs = dc_model(dc_inputs)

                dc_opt.zero_grad()
                dc_loss = dc_cri(dc_outputs, dc_labels)
                dc_loss.backward(retain_graph=True)                
                dc_opt.step()
                dc_ins_loss = dc_invs_cri(dc_outputs,dc_ivs_labels)

                if cfg.TRAIN.LOSS_V2:
                    ext_loss = ext_loss + (dc_loss+dc_ins_loss)*0.5*cfg.TRAIN.LOSS_WEIGHT[2]
                else:
                    ext_loss = ext_loss + dc_ins_loss*cfg.TRAIN.LOSS_WEIGHT[2]

            # obc model
            if cfg.TRAIN.COM_EXP in [5]:
                # generate the obc inputs and labels
                pooled_features = torch.cat((src_pooled_features, tgt_pooled_features),0) 
                obc_labels, obc_ivs_labels = gen_obc_labels(src_obj_labels,tgt_obj_labels) 
                obc_outputs = obc_model(pooled_features)

                obc_opt.zero_grad()
                obc_loss = obc_cri(obc_outputs, obc_labels)
                obc_loss.backward(retain_graph=True)
                obc_opt.step() 
                obc_ins_loss = obc_invs_cri(obc_outputs,obc_ivs_labels)

                if cfg.TRAIN.LOSS_V2:
                    ext_loss = ext_loss + (obc_loss+obc_ins_loss)*0.5*cfg.TRAIN.LOSS_WEIGHT[3]
                else:
                    ext_loss = ext_loss + obc_ins_loss*cfg.TRAIN.LOSS_WEIGHT[3]

            # only obc model
            if cfg.TRAIN.COM_EXP in [6]:
                # generate the obc inputs and labels
                pooled_features = torch.cat((src_pooled_features, tgt_pooled_features),0) 
                obc_labels, obc_ivs_labels = gen_obc_labels(src_obj_labels,tgt_obj_labels) 
                obc_outputs = obc_model(pooled_features)

                obc_opt.zero_grad()
                obc_loss = obc_cri(obc_outputs, obc_labels)
                obc_loss.backward(retain_graph=True)
                obc_opt.step() 
                obc_ins_loss = obc_invs_cri(obc_outputs,obc_ivs_labels)
                if cfg.TRAIN.LOSS_V2:
                    ext_loss = ext_loss + (obc_loss+obc_ins_loss)*0.5*cfg.TRAIN.LOSS_WEIGHT[3]
                else:
                    ext_loss = ext_loss + obc_ins_loss*cfg.TRAIN.LOSS_WEIGHT[3]
                
            ext_loss.backward()
            ext_opt.step()
            # _t['net'].toc(average=False)
            # print 'pre_data: {:.3f}s, net: {:.3f}s'.format(_t['pre_data'].average_time,_t['net'].average_time)

            # calculate miu and log data
            if (i_iter + 1) % cfg.TRAIN.PRINT_FREQ == 0:
                i_tb +=1
                src_m_iu, src_class_iu = eval_ext_logs(pred_src_outputs,src_labels)
                tgt_m_iu, tgt_class_iu = eval_ext_logs(pred_tgt_outputs,tgt_labels)

                _t['net'].toc(average=False)
                print 'pre_data: {:.3f}s, net: {:.3f}s'.format(_t['pre_data'].average_time,_t['net'].average_time)

                if cfg.TRAIN.COM_EXP == 6:
                    logger(i_epoch,i_iter,i_tb,writer,ext_loss,sp_loss,unsp_loss,det_loss,src_m_iu,tgt_m_iu,
                            obc_loss=obc_loss,obc_ins_loss=obc_ins_loss)

                if cfg.TRAIN.COM_EXP == 5:
                    logger(i_epoch,i_iter,i_tb,writer,ext_loss,sp_loss,unsp_loss,det_loss,src_m_iu,tgt_m_iu,
                            dc_loss=dc_loss,dc_ins_loss=dc_ins_loss,
                            obc_loss=obc_loss,obc_ins_loss=obc_ins_loss)

                elif cfg.TRAIN.COM_EXP == 4:
                    logger(i_epoch,i_iter,i_tb,writer,ext_loss,sp_loss,unsp_loss,det_loss,src_m_iu,tgt_m_iu,
                            dc_loss=dc_loss,dc_ins_loss=dc_ins_loss)

                elif cfg.TRAIN.COM_EXP == 3:
                    logger(i_epoch,i_iter,i_tb,writer,ext_loss,sp_loss,unsp_loss,det_loss,src_m_iu,tgt_m_iu)               
                # eval_net(i_epoch,i_iter,i_tb,writer,ext_model)
                if (i_iter + 1) % cfg.TRAIN.PRINT_FREQ == 0:
                    src_pred_color, src_label_color = gen_labeled_map(pred_src_outputs,labels=src_labels)
                    tgt_pred_color, tgt_label_color = gen_labeled_map(pred_tgt_outputs,labels=tgt_labels)
                    src_det_img = draw_bbx(src_loc_preds[0].cpu(),src_conf_preds[0].cpu(), ext_src_inputs[0], restore_transform)
                    tgt_det_img = draw_bbx(tgt_loc_preds[0].cpu(),tgt_conf_preds[0].cpu(), ext_tgt_inputs[0], restore_transform)

                    show_img(i_tb, writer,src_label_color,src_pred_color,tgt_label_color,tgt_pred_color, 
                                src_det_img, tgt_det_img)
               
        eval_net(i_epoch,i_iter,i_tb,writer,ext_model)
        # save model
        snapshot_name = 'epoch_%d' % (i_epoch + 1)

        torch.save(ext_model.state_dict(), os.path.join(
                    cfg.TRAIN.CKPT_PATH, cfg.TRAIN.EXP_NAME, snapshot_name + '_ext.pth'))
        if cfg.TRAIN.COM_EXP in [2,4,5]:
            torch.save(dc_model.state_dict(), os.path.join(
                        cfg.TRAIN.CKPT_PATH, cfg.TRAIN.EXP_NAME, 'dc.pth'))# replace old model
        if cfg.TRAIN.COM_EXP in [5]:
            torch.save(obc_model.state_dict(), os.path.join(
                        cfg.TRAIN.CKPT_PATH, cfg.TRAIN.EXP_NAME, 'obc.pth'))# replace old model

        