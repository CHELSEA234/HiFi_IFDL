# ------------------------------------------------------------------------------
# Author: Xiao Guo (guoxia11@msu.edu)
# CVPR2023: Hierarchical Fine-Grained Image Forgery Detection and Localization
# ------------------------------------------------------------------------------
import os
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ReduceLROnPlateau
from kmeans_pytorch import kmeans
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn import metrics

Softmax_m = nn.Softmax(dim=1)
device = torch.device('cuda:0')

def device_ids_return(cuda_list):
    '''return the device id'''
    if  len(cuda_list) == 1:
        device_ids = [0]
    elif len(cuda_list) == 2:
        device_ids = [0,1]
    elif len(cuda_list) == 3:
        device_ids = [0,1,2]
    elif len(cuda_list) == 4:
        device_ids = [0,1,2,3]
    elif len(cuda_list) == 7:
        device_ids = [0,1,2,3,4,5,6]
    return device_ids

def findLastCheckpoint(save_dir):
    if os.path.exists(save_dir):
        file_list = os.listdir(save_dir)
        result = 0
        for file in file_list:
            try:
                num = int(file.split('.')[0].split('_')[-1])
                result = max(result, num)
            except:
                continue
        return result
    else:
        os.mkdir(save_dir)
        return 0

def get_confusion_matrix(y_true, y_pred):
    return metrics.confusion_matrix(y_true, y_pred)

def compute_cls_acc_f1(label_lst, pred_lst, iter_num, tb_writer, descr='Val/level3_1'):
    F1  = metrics.f1_score(label_lst, pred_lst, average='macro')
    acc = metrics.accuracy_score(label_lst, pred_lst)
    tb_writer.add_scalar(f'{descr}_F1', F1, iter_num)
    tb_writer.add_scalar(f'{descr}_acc', acc, iter_num)
    print(f"In {descr}, the image-level Acc: {acc:.3f}, F1: {F1:.3f}.")
    print("******************************************************")
    return F1, acc

def compute_infer(label_lst, pred_lst, iter_num, tb_writer, descr='Inference'):
    F1  = metrics.f1_score(label_lst, pred_lst, average='macro')
    acc = metrics.accuracy_score(label_lst, pred_lst)
    acc_0, acc_1, thre, AUC = evaluate(label_lst, pred_lst, pos_label=1)
    tb_writer.add_scalar(f'{descr}_F1', F1, iter_num)
    tb_writer.add_scalar(f'{descr}_acc', acc, iter_num)
    # print(f"In {descr}, the image-level Acc: {acc:.3f}, F1: {F1:.3f}, AUC: {AUC:.3f}.")
    # print("******************************************************")
    return F1, acc, AUC

def tb_writer_display(writer, iter_num, lr_scheduler, epoch, 
                      seg_accu, loc_map_loss, manipul_loss, natural_loss, binary_loss,
                      loss_1, loss_2, loss_3, loss_4):
    writer.add_scalar('Train/seg_accu', seg_accu, iter_num)
    writer.add_scalar('Train/map_loss', loc_map_loss, iter_num)
    writer.add_scalar('Train/binary_map_loss', binary_loss, iter_num)
    writer.add_scalar('Train/manip_loss', manipul_loss, iter_num)
    writer.add_scalar('Train/natur_loss', natural_loss, iter_num)
    writer.add_scalar('Train/loss_1', loss_1, iter_num)
    writer.add_scalar('Train/loss_2', loss_2, iter_num)
    writer.add_scalar('Train/loss_3', loss_3, iter_num)
    writer.add_scalar('Train/loss_4', loss_3, iter_num)
    for count, gp in enumerate(lr_scheduler.optimizer.param_groups,1):
        writer.add_scalar('progress/lr_%d'%count, gp['lr'], iter_num)
    writer.add_scalar('progress/epoch', epoch, iter_num)
    writer.add_scalar('progress/curr_patience',lr_scheduler.num_bad_epochs,iter_num)
    writer.add_scalar('progress/patience',lr_scheduler.patience,iter_num)

def one_hot_label(vector, Softmax_m=Softmax_m):
    x = Softmax_m(vector)
    x = torch.argmax(x, dim=1)
    return x

def one_hot_label_new(vector, Softmax_m=Softmax_m):
    '''
        compute the probability for being as the synthesized image (TODO: double check).
    '''
    x = Softmax_m(vector)
    indices = torch.argmax(x, dim=1)
    prob = 1 - x[:,0]
    indices = list(indices.cpu().numpy())
    prob = list(prob.cpu().numpy())
    return indices, prob

def level_1_convert(input_lst):
    res_lst = []
    for _ in input_lst:
        if _ == 0:
            res_lst.append(0)
        else:
            res_lst.append(1)
    return res_lst

def confusion_matrix_display(label_lst, res_lst, display_lst, display_name):
    confusion_matrix = metrics.confusion_matrix(label_lst, res_lst)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, 
                                                display_labels = display_lst)
    cm_display.plot()
    plt.savefig(f'{display_name}.png')
    confusion_matrix = metrics.confusion_matrix(label_lst, res_lst, normalize='true')
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, 
                                                display_labels = display_lst)
    cm_display.plot()
    plt.savefig(f'{display_name}_normalized.png')

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
        print(f"Making folder {folder_name}.")
    else:
        print(f"Folder {folder_name} exists.")

def class_weight(mask, mask_idx):
    '''balance the weight on real and forgery pixel.'''
    mask_balance = torch.ones_like(mask).to(torch.float)
    if (mask == 1).sum():
        mask_balance[mask == 1] = 0.5 / ((mask == 1).sum().to(torch.float) / mask.numel())
        mask_balance[mask == 0] = 0.5 / ((mask == 0).sum().to(torch.float) / mask.numel())
    else:
        pass
        # print(f'Mask{mask_idx} balance is not working!')
    return mask.to(device), mask_balance.to(device)

def setup_optimizer(args, SegNet, FENet):
    '''setup the optimizier, which applies different learning rate on different layers.'''
    '''different hyper-parameters are changed towards HiFi-IFDL dataset.'''
    params_dict_list = []
    params_dict_list.append({'params' : SegNet.module.getmask.parameters(), 'lr' : args.learning_rate*1.2})
    params_dict_list.append({'params' : SegNet.module.branch_cls_level_3.parameters(), 'lr' : args.learning_rate})
    params_dict_list.append(
                        {'params' : SegNet.module.branch_cls_level_2.parameters(), 'lr' : args.learning_rate})
    params_dict_list.append(
                        {'params' : SegNet.module.branch_cls_level_1.parameters(), 'lr' : args.learning_rate})
    params_dict_list.append({'params' : FENet.module.stage4[0].fuse_layers.parameters(), 'lr' : args.learning_rate})
    params_dict_list.append({'params' : FENet.module.stage3[0].fuse_layers.parameters(), 'lr' : args.learning_rate})
    params_dict_list.append({'params' : FENet.module.stage2[0].fuse_layers.parameters(), 'lr' : args.learning_rate})
    params_dict_list.append({'params' : FENet.module.transition1.parameters(), 'lr' : args.learning_rate})
    params_dict_list.append({'params' : FENet.module.conv_1x1_merge.parameters(), 'lr' : args.learning_rate})
    ## newly-added layer will have the larger learning rate.
    ## newly-added layer will have the larger learning rate.
    ## 0.75 ==> 1
    params_dict_list.append({'params' : FENet.module.conv1fre.parameters(), 'lr' : args.learning_rate*args.lr_backbone})
    params_dict_list.append({'params' : FENet.module.bn1fre.parameters(), 'lr' : args.learning_rate*args.lr_backbone})
    params_dict_list.append({'params' : FENet.module.conv2fre.parameters(), 'lr' : args.learning_rate*args.lr_backbone})
    params_dict_list.append({'params' : FENet.module.bn2fre.parameters(), 'lr' : args.learning_rate*args.lr_backbone})

    optimizer    = torch.optim.Adam(params_dict_list, lr=args.learning_rate*0.75, weight_decay=1e-06)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.step_factor, min_lr=1e-08,
                                     patience=args.patience, verbose=True)

    return optimizer, lr_scheduler

def restore_weight_helper(model, model_dir, initial_epoch):
    '''load model given the model_dir that has the model weights.'''
    try:
        weight_path = '{}/{}.pth'.format(model_dir, initial_epoch)
        state_dict = torch.load(weight_path, map_location='cuda:0')['model']
        model.load_state_dict(state_dict)
        print('{} weight-loading succeeds: {}'.format(model_dir, weight_path))
    except:
        print('{} weight-loading fails'.format(model_dir))

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("{}_params: {}".format(model_dir, pytorch_total_params))
    return model

def restore_optimizer(optimizer, model_dir):
    '''restore the optimizer.'''
    try:
        weight_path = '{}/{}.pth'.format(model_dir, initial_epoch)
        state_dict = torch.load(weight_path, map_location='cuda:0')
        print('Optimizer weight-loading succeeds.')
        optimizer.load_state_dict(state_dict['optimizer'])
    except:
        # print('{} Optimizer weight-loading fails.')
        pass
    return optimizer

def composite_obj(args, loss, loss_1, loss_2, loss_3, loss_4, loss_binary):
    ''' 'base', 'fg', 'local', 'full' '''
    if args.ablation == 'full':     # fine-grained + localization
        # loss_total = 100*loss + loss_1 + loss_2 + loss_3 + 100*loss_4 + loss_binary
        loss_total = 100*loss + loss_1 + loss_2 + loss_3 + 100*loss_4 + loss_binary
    elif args.ablation == 'base':   # one-shot
        loss_total = loss_4
    elif args.ablation == 'fg':     # only fine-grained
        loss_total = loss_1 + loss_2 + loss_3 + loss_4
    elif args.ablation == 'local':  # only loclization
        loss_total = loss + loss_1 + loss_2 + loss_3 + loss_4
    else:
        assert False
    return loss_total

def composite_obj_step(args, loss_4_sum, map_loss_sum):
    ''' return loss for the scheduler '''
    if args.ablation == 'full':
        schedule_step_loss = loss_4_sum + map_loss_sum
    elif args.ablation == 'base':
        schedule_step_loss = loss_4_sum
    elif args.ablation == 'fg':
        schedule_step_loss = loss_4_sum
    elif args.ablation == 'local':
        schedule_step_loss = map_loss_sum
    else:
        assert False
    return schedule_step_loss

def train_log_dump(args, seg_correct, seg_total, map_loss_sum, mani_lss_sum, natu_lss_sum,
                    binary_map_loss_sum, loss_1_sum, loss_2_sum, loss_3_sum,
                    loss_4_sum, epoch, step, writer, iter_num, lr_scheduler):
    '''compute and output the different training loss & seg accuarcy.'''
    seg_accu = seg_correct / seg_total * 100
    loc_map_loss = map_loss_sum / args.dis_step
    manipul_loss = mani_lss_sum / args.dis_step
    natural_loss = natu_lss_sum / args.dis_step
    binary_loss  = binary_map_loss_sum / args.dis_step
    loss_1 = loss_1_sum / args.dis_step
    loss_2 = loss_2_sum / args.dis_step
    loss_3 = loss_3_sum / args.dis_step
    loss_4 = loss_4_sum / args.dis_step
    print(f'[Epoch: {epoch+1}, Step: {step + 1}] batch_loc_acc: {seg_accu:.2f}')
    print(f'cls1_loss: {loss_1:.3f}, cls2_loss: {loss_2:.3f}, cls3_loss: {loss_3:.3f}, '+
          f'cls4_loss: {loss_4:.3f}, map_loss:   {loc_map_loss:.3f}, '+
          f'manip_loss: {manipul_loss:.3f}, natur_loss: {natural_loss:.3f}, '+
          f'binary_map_loss: {binary_loss:.3f}') 
    '''write the tensorboard.'''
    tb_writer_display(writer, iter_num, lr_scheduler, epoch, seg_accu, 
                      loc_map_loss, manipul_loss, natural_loss, binary_loss,
                      loss_1, loss_2, loss_3, loss_4)