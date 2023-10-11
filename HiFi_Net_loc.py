# ------------------------------------------------------------------------------
# Author: Xiao Guo (guoxia11@msu.edu)
# CVPR2023: Hierarchical Fine-Grained Image Forgery Detection and Localization
# ------------------------------------------------------------------------------
from utils.utils import *
from IMD_dataloader import *
from utils.custom_loss import IsolatingLossFunction, load_center_radius
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_cfg_defaults
from models.NLCDetection_loc import NLCDetection

from sklearn import metrics
from sklearn.metrics import roc_auc_score

from torchvision.utils import make_grid
from einops import rearrange
from PIL import Image
from sklearn import metrics

import os
import csv
import time
import torch
import torch.nn as nn
import argparse
import numpy as np

device = torch.device('cuda:0')
device_ids = [0]

def config(args):
    '''Set up input configurations.'''
    args.crop_size = [args.crop_size, args.crop_size]
    # cuda_list = args.list_cuda
    global device 
    device = torch.device('cuda:0')
    # global device_ids
    # device_ids = device_ids_return(cuda_list)

    args.save_dir    = 'lr_' + str(args.learning_rate)+'_loc'
    FENet_dir, SegNet_dir = args.save_dir+'/HRNet', args.save_dir+'/NLCDetection'
    FENet_cfg = get_cfg_defaults()
    FENet  = get_seg_model(FENet_cfg).to(device) # load the pre-trained model inside.
    SegNet = NLCDetection().to(device)

    FENet  = nn.DataParallel(FENet, device_ids=device_ids)
    SegNet = nn.DataParallel(SegNet, device_ids=device_ids)

    writer = None

    return args, writer, FENet, SegNet, FENet_dir, SegNet_dir

def restore_weight(args, FENet, SegNet, FENet_dir, SegNet_dir):
    '''load FENet, SegNet and optimizer.'''
    params      = list(FENet.parameters()) + list(SegNet.parameters()) 
    optimizer   = torch.optim.Adam(params, lr=args.learning_rate)
    initial_epoch = findLastCheckpoint(save_dir=SegNet_dir)

    # load FENet and SegNet weight:
    FENet  = restore_weight_helper(FENet,  FENet_dir,  initial_epoch)
    SegNet = restore_weight_helper(SegNet, SegNet_dir, initial_epoch)
    optimizer  = restore_optimizer(optimizer, SegNet_dir)

    return optimizer, initial_epoch

def Inference_loc(
                args, FENet, SegNet, LOSS_MAP, tb_writer, 
                iter_num=None, 
                save_tag=False, 
                localization=True
                ):
    '''
        the inference pipeline for the pre-trained model.
        the image-level detection will dump to the csv file.
        the pixel-level localization will be saved as in the npy file.
    '''

    for val_tag in [0,1,2,3,4]:

        val_data_loader, data_label = eval_dataset_loader_init(args, val_tag)
        print(f"working on the dataset: {data_label}.")
        F1_lst, auc_lst = [], []
        with torch.no_grad():
            FENet.eval()
            SegNet.eval()
            for step, val_data in enumerate(tqdm(val_data_loader)):
                image, mask, cls, image_names = val_data
                image, mask = image.to(device), mask.to(device)
                mask = torch.squeeze(mask, axis=1)

                # model
                try:
                    output = FENet(image)
                    mask1_fea, mask_binary, out0, out1, out2, out3 = SegNet(output, image)
                except:
                    print(f"does not work on the ", image_names)
                    continue
                if args.loss_type == 'dm':
                    loss_map, loss_manip, loss_nat = LOSS_MAP(mask1_fea, mask)
                    pred_mask = LOSS_MAP.dis_curBatch.squeeze(dim=1)
                    pred_mask_score = LOSS_MAP.dist.squeeze(dim=1)
                elif args.loss_type == 'ce':
                    pred_mask_score = mask_binary
                    pred_mask = torch.zeros_like(mask_binary)
                    pred_mask[mask_binary > 0.5] = 1
                    pred_mask[mask_binary <= 0.5] = 0
                viz_log(args, mask, pred_mask, image, iter_num, f"{step}_{val_tag}", mode='eval')

                mask = torch.unsqueeze(mask, axis=1)
                for img_idx, cur_img_name in enumerate(image_names):

                    mask_ = torch.unsqueeze(mask[img_idx,0], 0)
                    pred_mask_ = torch.unsqueeze(pred_mask[img_idx], 0)
                    pred_mask_score_ = torch.unsqueeze(pred_mask_score[img_idx], 0)

                    mask_ = mask_.cpu().clone().cpu().numpy().reshape(-1)
                    pred_mask_ = pred_mask_.cpu().clone().cpu().numpy().reshape(-1)
                    pred_mask_score_ = pred_mask_score_.cpu().clone().cpu().numpy().reshape(-1)

                    F1_a  = metrics.f1_score(mask_, pred_mask_, average='macro')
                    auc_a = metrics.roc_auc_score(mask_, pred_mask_score_)

                    pred_mask_[np.where(pred_mask_ == 0)] = 1
                    pred_mask_[np.where(pred_mask_ == 1)] = 0

                    F1_b  = metrics.f1_score(mask_, pred_mask_, average='macro')
                    if F1_a > F1_b:
                        F1 = F1_a
                    else:
                        F1 = F1_b
                    F1_lst.append(F1)
                    AUC_score = auc_a if auc_a > 0.5 else 1-auc_a
                    auc_lst.append(AUC_score)
                    
        print("F1: ", np.mean(F1_lst))
        print("AUC: ", np.mean(auc_lst))

def main(args):
    ## Set up the configuration.
    args, writer, FENet, SegNet, FENet_dir, SegNet_dir = config(args)

    ## load FENet and SegNet weight:
    if args.loss_type == 'ce':
        FENet  = restore_weight_helper(FENet,  "weights/HRNet",  225000)
        SegNet = restore_weight_helper(SegNet, "weights/NLCDetection", 225000)    
    elif args.loss_type == 'dm':
        FENet  = restore_weight_helper(FENet,  "weights/HRNet",  315000)
        SegNet = restore_weight_helper(SegNet, "weights/NLCDetection", 315000)
    else:
        raise ValueError

    ## Set up the loss function.
    center, radius = load_center_radius(args, FENet, SegNet, 
                                        train_data_loader=None, 
                                        center_radius_dir='./center_loc')
    CE_loss  = nn.CrossEntropyLoss().to(device)
    BCE_loss = nn.BCELoss(reduction='none').to(device)
    LOSS_MAP = IsolatingLossFunction(center,radius).to(device)

    Inference_loc(
                args, 
                FENet, 
                SegNet,
                LOSS_MAP,
                tb_writer=writer, 
                iter_num=99999, 
                save_tag=True, 
                localization=True
                )
    print("after saving the points...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--list_cuda', nargs='+', help='<Required> Set flag')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--lr_gamma', type=float, default=2.0)
    parser.add_argument('--lr_backbone', type=float, default=0.9)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--step_factor', type=float, default=0.95)
    parser.add_argument('--dis_step', type=int, default=50)
    parser.add_argument('--val_step', type=int, default=500)

    ## train hyper-parameters
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--val_num', type=int, default=200, help='val sample number.')
    parser.add_argument('--train_num', type=int, default=360000, help='train sample number.')
    parser.add_argument('--train_tag', type=int, default=0)
    parser.add_argument('--val_tag', type=int, default=0)
    parser.add_argument('--val_all', type=int, default=1)
    parser.add_argument('--ablation', type=str, default='local', 
                            choices=['base', 'fg', 'local', 'full'], 
                            help='exp for one-shot, fine_grain, plus localization, plus pconv')
    parser.add_argument('--val_loc_tag', action='store_true')
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--debug_mode', action='store_true')
    parser.set_defaults(val_loc_tag=True)
    parser.set_defaults(fine_tune=True)

    parser.add_argument('--train_ratio', nargs='+', default="0.4 0.4 0.2", help='deprecated')
    parser.add_argument('--path', type=str, default="", help='deprecated')
    parser.add_argument('--train_bs', type=int, default=10, help='batch size in the training.')
    parser.add_argument('--val_bs', type=int, default=10, help='batch size in the validation.')
    parser.add_argument('--percent', type=float, default=1.0, help='label dataset.')
    parser.add_argument('--loss_type', type=str, default='ce',
                            choices=['ce', 'dm'], help='ce or deep metric.')

    ## inference hyperparameters:
    parser.add_argument('--initial_epoch', type=int, default=70500)
    args = parser.parse_args()
    main(args)
