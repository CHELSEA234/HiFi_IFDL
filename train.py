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

## eval package
from sklearn import metrics
from sklearn.metrics import roc_auc_score

## visualize package
from sklearn import metrics
import os
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
    cuda_list = args.list_cuda
    global device 
    device = torch.device('cuda:0')
    global device_ids
    device_ids = device_ids_return(cuda_list)

    args.save_dir    = 'lr_' + str(args.learning_rate)+'_loc'
    FENet_dir, SegNet_dir = args.save_dir+'/HRNet', args.save_dir+'/NLCDetection'
    FENet_cfg = get_cfg_defaults()
    FENet  = get_seg_model(FENet_cfg).to(device) # load the pre-trained model inside.
    SegNet = NLCDetection().to(device)

    FENet  = nn.DataParallel(FENet, device_ids=device_ids)
    SegNet = nn.DataParallel(SegNet, device_ids=device_ids)

    make_folder(args.save_dir)
    make_folder(FENet_dir)
    make_folder(SegNet_dir)
    writer = SummaryWriter(f'tb_logs/{args.save_dir}')

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

def save_weight(FENet, SegNet, FENet_dir, SegNet_dir, optimizer, epoch):
    # Save checkpoint
    FENet_checkpoint = {'model': FENet.state_dict(),
                        'optimizer': optimizer.state_dict()}
    torch.save(FENet_checkpoint, '{0}/{1}.pth'.format(FENet_dir, epoch + 1))

    SegNet_checkpoint = {'model': SegNet.state_dict(),
                         'optimizer': optimizer.state_dict()}
    torch.save(SegNet_checkpoint, '{0}/{1}.pth'.format(SegNet_dir, epoch + 1))

def validation(
            args, FENet, SegNet, LOSS_MAP, tb_writer, 
            iter_num=None, 
            save_tag=False, 
            localization=True
            ):
    """ standard validation. """
    val_data_loader = infer_dataset_loader_init(args)
    val_num_per_epoch = len(val_data_loader)
    F1_lst, auc_lst = [], []

    with torch.no_grad():
        FENet.eval()
        SegNet.eval()
        for step, val_data in enumerate(tqdm(val_data_loader)):
            image, masks, cls0, cls1, cls2, cls3, image_names = val_data
            mask1, mask2, mask3, mask4 = masks
            image = image.to(device)
            mask1, mask2, mask3, mask4 = mask1.to(device), mask2.to(device), mask3.to(device), mask4.to(device)
            cls0, cls1, cls2, cls3 = cls0.to(device), cls1.to(device), cls2.to(device), cls3.to(device)

            # model 
            output = FENet(image)
            mask1_fea, mask_binary, out0, out1, out2, out3 = SegNet(output, image)
            if args.loss_type == 'dm':
                loss_map, loss_manip, loss_nat = LOSS_MAP(mask1_fea, mask1)
                pred_mask = LOSS_MAP.dis_curBatch.squeeze(dim=1)
                pred_mask_score = LOSS_MAP.dist.squeeze(dim=1)
            elif args.loss_type == 'ce':
                pred_mask_score = mask_binary
                pred_mask = torch.zeros_like(mask_binary)
                pred_mask[mask_binary > 0.5] = 1
                pred_mask[mask_binary <= 0.5] = 0
            viz_log(args, mask1, pred_mask, image, iter_num, step, mode='val')
            mask = torch.unsqueeze(mask1, axis=1)
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

            if step == 5 and args.debug_mode:
                break
            elif step == 10:
                break

    FENet.train()
    SegNet.train()
    print("...computing the pixel-wise scores/metrics here...")
    print(f"the scr_auc is: {np.mean(auc_lst):.3f}.")
    print(f"the macro is: {np.mean(F1_lst):.3f}")

def Inference(
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
    for val_tag in [0,1,2,3]:
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
                output = FENet(image)
                mask1_fea, mask_binary, out0, out1, out2, out3 = SegNet(output, image)
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

                if step == 5 and args.debug_mode:
                    break
                elif step == 180:
                    break

        FENet.train()
        SegNet.train()
        print("F1: ", np.mean(F1_lst))
        print("AUC: ", np.mean(auc_lst))

def main(args):
    ## Set up the configuration.
    args, writer, FENet, SegNet, FENet_dir, SegNet_dir = config(args)

    ## Dataloader: 
    train_data_loader = train_dataset_loader_init(args)
    train_num_per_epoch = int(args.train_num/args.train_bs)

    ## Model and Optimizer:
    optimizer, lr_scheduler = setup_optimizer(args, SegNet, FENet)
    optimizer, initial_iter = restore_weight(args, FENet, SegNet, FENet_dir, SegNet_dir)
    initial_epoch = int(initial_iter/train_num_per_epoch)

    ## Set up the loss function.
    center, radius = load_center_radius(args, FENet, SegNet, train_data_loader)
    CE_loss  = nn.CrossEntropyLoss().to(device)
    BCE_loss = nn.BCELoss(reduction='none').to(device)
    if args.loss_type == 'ce':
        LOSS_MAP = None
    elif args.loss_type == 'dm':
        LOSS_MAP = IsolatingLossFunction(center,radius).to(device)

    for epoch in range(0, args.num_epochs):
        start_time = time.time()
        seg_total, seg_correct, seg_loss_sum = [0]*3
        map_loss_sum, mani_lss_sum, natu_lss_sum, binary_map_loss_sum = [0]*4
        loss_1_sum, loss_2_sum, loss_3_sum, loss_4_sum = [0]*4

        for step, train_data in enumerate(train_data_loader):
            iter_num = epoch * train_num_per_epoch + step
            image, masks, cls0, cls1, cls2, cls3 = train_data
            mask1, mask2, mask3, mask4 = masks
            image = image.to(device)
            mask1, mask2, mask3, mask4 = mask1.to(device), mask2.to(device), mask3.to(device), mask4.to(device)
            cls0, cls1, cls2, cls3 = cls0.to(device), cls1.to(device), cls2.to(device), cls3.to(device)
            mask1, mask1_balance = class_weight(mask1, 1)
            mask2, mask2_balance = class_weight(mask2, 1)
            mask3, mask3_balance = class_weight(mask3, 1)
            mask4, mask4_balance = class_weight(mask4, 1)

            # model 
            output = FENet(image)
            mask1_fea, mask_binary, out0, out1, out2, out3 = SegNet(output, image)
            
            # objective
            loss_4 = CE_loss(out3, cls3)    # label: 0 --> 13
            forgery_cls = ~(cls0.eq(0)) # mask real images, only compute the loss_4.
            if np.sum(forgery_cls.cpu().numpy()) != 0:
                loss_1 = CE_loss(out0[forgery_cls,:], cls0[forgery_cls])    # label: 0 --> 2
                loss_2 = CE_loss(out1[forgery_cls,:], cls1[forgery_cls])    # label: 0 --> 4
                loss_3 = CE_loss(out2[forgery_cls,:], cls2[forgery_cls])    # label: 0 --> 6
            else:
                loss_1 = torch.tensor(0.0, requires_grad=True).to(device)
                loss_2 = torch.tensor(0.0, requires_grad=True).to(device)
                loss_3 = torch.tensor(0.0, requires_grad=True).to(device)

            # print(mask_binary.size())
            # print(mask1.size())
            # import sys;sys.exit(0)
            loss_binary_map = torch.mean(BCE_loss(mask_binary, mask1.to(torch.float)) * mask1_balance)

            ## the composite loss containing loss_dm and loss_pixel.
            if args.loss_type == 'dm':
                loss, loss_manip, loss_nat = LOSS_MAP(mask1_fea, mask1)
            elif args.loss_type == 'ce':
                loss = loss_binary_map
                loss_manip = loss_binary_map
                loss_nat = loss_binary_map
            loss_total = composite_obj(args, loss, loss_1, loss_2, loss_3, loss_4, loss_binary_map)
            
            ## backpropagate.
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            ## compute logs.
            if args.loss_type == 'dm':
                pred_mask1 = LOSS_MAP.dis_curBatch.squeeze(dim=1)
            elif args.loss_type == 'ce':
                pred_mask1 = torch.zeros_like(mask_binary)
                pred_mask1[mask_binary > 0.5] = 1
                pred_mask1[mask_binary <= 0.5] = 0
            seg_correct += (pred_mask1 == mask1).sum().item()
            seg_total   += int(torch.ones_like(mask1).sum().item())
            map_loss_sum += loss.item()
            mani_lss_sum += loss_manip.item()
            natu_lss_sum += loss_nat.item()
            binary_map_loss_sum += loss_binary_map.item()
            loss_1_sum += loss_1.item()
            loss_2_sum += loss_2.item()
            loss_3_sum += loss_3.item()
            loss_4_sum += loss_4.item()

            if step % args.dis_step == 0:
                train_log_dump(
                            args, seg_correct, seg_total, map_loss_sum, mani_lss_sum, 
                            natu_lss_sum, binary_map_loss_sum, loss_1_sum, loss_2_sum, 
                            loss_3_sum, loss_4_sum, epoch, step, writer, iter_num,
                            lr_scheduler
                            )
                schedule_step_loss = composite_obj_step(args, loss_4_sum, map_loss_sum)
                lr_scheduler.step(schedule_step_loss)
                ## reset
                seg_total, seg_correct, seg_loss_sum = [0]*3
                loss_1_sum, loss_2_sum, loss_3_sum, loss_4_sum = [0]*4
                map_loss_sum, mani_lss_sum, natu_lss_sum, binary_map_loss_sum = [0]*4
                viz_log(args, mask1, pred_mask1, image, iter_num, step, mode='train')

            if (iter_num+1) % args.val_step == 0:
                validation(
                        args, 
                        FENet, 
                        SegNet, 
                        LOSS_MAP,
                        tb_writer=writer, 
                        iter_num=iter_num, 
                        save_tag=True, 
                        localization=True
                        )
                Inference(
                        args, 
                        FENet, 
                        SegNet, 
                        LOSS_MAP, 
                        tb_writer=writer, 
                        iter_num=iter_num, 
                        save_tag=True, 
                        localization=True
                        )
                print(f"...save the iteration number: {iter_num}.")
                save_weight(FENet, SegNet, FENet_dir, SegNet_dir, optimizer, iter_num)

            if args.debug_mode and step == 2:
                args.val_step = 1
                break

        if args.debug_mode and epoch == 1:
            print("Finish two complete epoches.")
            import sys;sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--list_cuda', nargs='+', help='<Required> Set flag')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4)
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
    args = parser.parse_args()
    main(args)
