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
from torchvision.utils import make_grid
from einops import rearrange
from PIL import Image

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
    params_dict_list.append({'params' : SegNet.module.parameters(), 'lr' : args.learning_rate})
    freq_list = []
    para_list = []
    for name, param in FENet.named_parameters():
        if 'fre' in name:
            freq_list += [param]
        else:
            para_list += [param]
    params_dict_list.append({'params' : freq_list, 'lr' : args.learning_rate*args.lr_backbone})
    params_dict_list.append({'params' : para_list, 'lr' : args.learning_rate})

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
        loss_total = 100*loss + loss_1 + loss_2 + loss_3 + 100*loss_4 + loss_binary
    elif args.ablation == 'base':   # one-shot
        loss_total = loss_4
    elif args.ablation == 'fg':     # only fine-grained
        loss_total = loss_1 + loss_2 + loss_3 + loss_4
    elif args.ablation == 'local':  # only loclization
        loss_total = loss + 10e-6*(loss_1 + loss_2 + loss_3 + loss_4)
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

def viz_log(args, mask, pred_mask, image, iter_num, step, mode='train'):
    '''viz training, val and inference.'''
    mask = torch.unsqueeze(mask, dim=1)
    pred_mask = torch.unsqueeze(pred_mask, dim=1)
    mask_viz = torch.cat([mask]*3, axis=1)
    pred_mask = torch.cat([pred_mask]*3, axis=1)
    image = torch.nn.functional.interpolate(image,  # for viz.
                                          size=(256, 256), 
                                          mode='bilinear')
    fig_viz = torch.cat([mask_viz, image, pred_mask], axis=0)
    grid = make_grid(fig_viz, nrow=mask_viz.shape[0])   # nrow in fact is the column number.
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    img_h = Image.fromarray(grid.astype(np.uint8))
    # os.makedirs(f"./viz_{mode}_{args.learning_rate}/", exist_ok=True)
    os.makedirs(f"./viz_{mode}/", exist_ok=True)
    if mode == 'train':
        # img_h.save(f"./viz_{mode}_{args.learning_rate}/iter_{iter_num}.jpg")
        img_h.save(f"./viz_{mode}/iter_{iter_num}.jpg")
    else:
        # img_h.save(f"./viz_{mode}_{args.learning_rate}/iter_{iter_num}_step_{step}.jpg")
        img_h.save(f"./viz_{mode}/iter_{iter_num}_step_{step}.jpg")

def process_mask(mask, pred_mask):
    '''process the mask'''
    pred_mask = torch.unsqueeze(pred_mask, dim=1)
    mask = torch.unsqueeze(mask, dim=1)
    pred_mask = torch.cat([pred_mask]*3, axis=1)
    mask = torch.cat([mask]*3, axis=1)

    pred_mask = nn.functional.interpolate(pred_mask, 
                                        size=(256, 256), mode='bilinear')
    mask = nn.functional.interpolate(mask, 
                                    size=(256, 256), mode='bilinear')

    return pred_mask, mask

def viz_logs_scale(args, iter_num, mask_128, mask_64, mask_32, mask2, mask3, mask4, mode='train'):
    '''visualize the mask and predicted mask.'''
    pred_mask_128, mask128 = process_mask(mask_128, mask2)
    pred_mask_64, mask64 = process_mask(mask_64, mask3)
    pred_mask_32, mask32 = process_mask(mask_32, mask4)

    fig_viz = torch.cat([pred_mask_32, mask32, pred_mask_64, mask64, 
                        pred_mask_128, mask128], axis=0)
    grid = make_grid(fig_viz, nrow=pred_mask_32.shape[0])
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    img_h = Image.fromarray(grid.astype(np.uint8))
    os.makedirs(f"./viz_{mode}_{args.learning_rate}/", exist_ok=True)
    img_h.save(f"./viz_{mode}_{args.learning_rate}/iter_{iter_num}_pred.jpg")

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