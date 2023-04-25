# ------------------------------------------------------------------------------
# Author: Xiao Guo (guoxia11@msu.edu)
# CVPR2023: Hierarchical Fine-Grained Image Forgery Detection and Localization
# ------------------------------------------------------------------------------
from utils.utils import *
from utils.custom_loss import IsolatingLossFunction, load_center_radius
from IMD_dataloader import train_dataset_loader_init, viz_dataset_loader_init
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_cfg_defaults
from models.NLCDetection_pconv import NLCDetection
from sklearn import metrics
from sklearn.metrics import roc_auc_score

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

	args.save_dir 	 = 'lr_' + str(args.learning_rate)+'_loc'
	FENet_dir, SegNet_dir = args.save_dir+'/HRNet', args.save_dir+'/NLCDetection'
	FENet_cfg = get_cfg_defaults()
	FENet  = get_seg_model(FENet_cfg).to(device) # load the pre-trained model inside.
	SegNet = NLCDetection(args).to(device)

	FENet  = nn.DataParallel(FENet, device_ids=device_ids)
	SegNet = nn.DataParallel(SegNet, device_ids=device_ids)

	make_folder(args.save_dir)
	make_folder(FENet_dir)
	make_folder(SegNet_dir)
	writer = SummaryWriter(f'tb_logs/{args.save_dir}')

	return args, writer, FENet, SegNet, FENet_dir, SegNet_dir

def restore_weight(args, FENet, SegNet, FENet_dir, SegNet_dir):
	'''load FENet, SegNet and optimizer.'''
	params 		= list(FENet.parameters()) + list(SegNet.parameters()) 
	optimizer 	= torch.optim.Adam(params, lr=args.learning_rate)
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

def validation(args, FENet, SegNet, LOSS_MAP, tb_writer, iter_num=None, save_tag=False, localization=True):
	val_data_loader = viz_dataset_loader_init(args, val_tag=0)
	val_num_per_epoch = len(val_data_loader)
	AUC_score_lst, F1_score_lst = [], []
	mask_save_path = f'/user/guoxia11/cvlshare/cvl-guoxia11/CVPR2023_camera_ready_v4/mask_result_{args.learning_rate}/'
	os.makedirs(mask_save_path, exist_ok=True)
	mask_array_lst, pred_array_lst, scr_array_lst = [], [], []
	with torch.no_grad():
		FENet.eval()
		SegNet.eval()
		for step, val_data in enumerate(tqdm(val_data_loader)):
			if step % 70 != 0:
				continue
			image, masks, cls0, cls1, cls2, cls3, img_names = val_data
			cls0, cls1, cls2, cls3 = cls0.to(device), cls1.to(device), cls2.to(device), cls3.to(device)
			image, mask = image.to(device), masks[0].to(device)
			mask, mask_balance = class_weight(mask, 1)
			
			output = FENet(image)
			mask1_fea, mask1_binary, out0, out1, out2, out3 = SegNet(output, image)
			loss_map, loss_manip, loss_nat = LOSS_MAP(mask1_fea, mask)
			pred_mask = LOSS_MAP.dis_curBatch.squeeze(dim=1)
			pred_mask_score = LOSS_MAP.dist.squeeze(dim=1)

			mask_array = mask.cpu().numpy()
			pred_array = pred_mask.cpu().numpy()
			scor_array = pred_mask_score.cpu().numpy()

			mask_array = list(np.reshape(mask_array, -1))
			pred_array = list(np.reshape(pred_array, -1))
			scr_array  = list(np.reshape(scor_array, -1))

			mask_array_lst.extend(mask_array)
			pred_array_lst.extend(pred_array)
			scr_array_lst.extend(scr_array)

	print('the image used is ', len(mask_array_lst))
	print("...computing the pixel-wise scores/metrics here...")
	scr_auc = roc_auc_score(mask_array_lst, scr_array_lst)
	print(f"the scr_auc is: {scr_auc:.3f}.")
	F1 = metrics.f1_score(mask_array_lst, pred_array_lst, average='macro')
	print(f"the macro is: {F1:.3f}")

def main(args):
	## Set up the configuration.
	args, writer, FENet, SegNet, FENet_dir, SegNet_dir = config(args)

	## Dataloader: 
	train_data_loader = train_dataset_loader_init(args)
	train_num_per_epoch = int(args.train_num/args.train_bs)

	## Model and Optimizer:
	optimizer, lr_scheduler = setup_optimizer(args, SegNet, FENet)
	optimizer, initial_iter = restore_weight(args, FENet, SegNet, FENet_dir, SegNet_dir)
	# initial_iter = 0
	initial_epoch = int(initial_iter/train_num_per_epoch)

	## Set up the loss function.
	center, radius = load_center_radius(args, FENet, SegNet, train_data_loader)
	CE_loss  = nn.CrossEntropyLoss().to(device)
	BCE_loss = nn.BCELoss(reduction='none').to(device)
	LOSS_MAP = IsolatingLossFunction(center,radius).to(device)

	for epoch in range(0, args.num_epochs):
		start_time = time.time()
		seg_total, seg_correct, seg_loss_sum = [0]*3
		map_loss_sum, mani_lss_sum, natu_lss_sum, binary_map_loss_sum = [0]*4
		loss_1_sum, loss_2_sum, loss_3_sum, loss_4_sum = [0]*4

		for step, train_data in enumerate(train_data_loader):
			iter_num = epoch * train_num_per_epoch + step
			image, masks, cls0, cls1, cls2, cls3 = train_data
			image, mask1 = image.to(device), masks[0].to(device)
			cls0, cls1, cls2, cls3 = cls0.to(device), cls1.to(device), cls2.to(device), cls3.to(device)
			mask1, mask1_balance = class_weight(mask1, 1)

			# zero the parameter gradients
			optimizer.zero_grad()

			# output the tensor
			output = FENet(image)
			mask1_fea, mask_binary, out0, out1, out2, out3 = SegNet(output, image)
			
			# objective
			loss_4 = CE_loss(out3, cls3)	# label: 0 --> 13
			forgery_cls = ~(cls0.eq(0)) # mask real images, only compute the loss_4.
			if np.sum(forgery_cls.cpu().numpy()) != 0:
				loss_1 = CE_loss(out0[forgery_cls,:], cls0[forgery_cls])	# label: 0 --> 2
				loss_2 = CE_loss(out1[forgery_cls,:], cls1[forgery_cls])	# label: 0 --> 4
				loss_3 = CE_loss(out2[forgery_cls,:], cls2[forgery_cls])	# label: 0 --> 6
			else:
				loss_1 = torch.tensor(0.0, requires_grad=True).to(device)
				loss_2 = torch.tensor(0.0, requires_grad=True).to(device)
				loss_3 = torch.tensor(0.0, requires_grad=True).to(device)
			loss_binary_map = BCE_loss(mask_binary, mask1.to(torch.float)) * mask1_balance
			loss_binary_map = torch.mean(loss_binary_map)

			loss, loss_manip, loss_nat = LOSS_MAP(mask1_fea, mask1)
			loss_total = composite_obj(args, loss, loss_1, loss_2, loss_3, loss_4, loss_binary_map)
			loss_total.backward()
			optimizer.step()

			pred_mask1 = LOSS_MAP.dis_curBatch.squeeze(dim=1)
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
				train_log_dump(args, seg_correct, seg_total, map_loss_sum, mani_lss_sum, 
								natu_lss_sum, binary_map_loss_sum, loss_1_sum, loss_2_sum, 
								loss_3_sum, loss_4_sum, epoch, step, writer, iter_num,
								lr_scheduler)
				schedule_step_loss = composite_obj_step(args, loss_4_sum, map_loss_sum)
				lr_scheduler.step(schedule_step_loss)
				## reset
				seg_total, seg_correct, seg_loss_sum = [0]*3
				loss_1_sum, loss_2_sum, loss_3_sum, loss_4_sum = [0]*4
				map_loss_sum, mani_lss_sum, natu_lss_sum, binary_map_loss_sum = [0]*4				
			
			if iter_num % args.val_step == 0:
				print(f"...save the iteration number: {iter_num}.")
				save_weight(FENet, SegNet, FENet_dir, SegNet_dir, optimizer, iter_num)
				validation(args, FENet, SegNet, LOSS_MAP, tb_writer=writer, iter_num=iter_num, save_tag=True, 
						   localization=True)
				print("after saving the points...")

			if args.debug_mode:
				print("Finish one complete epoches.")
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
  parser.add_argument('--train_num', type=int, default=1710000, help='train sample number.')
  parser.add_argument('--train_tag', type=int, default=0)
  parser.add_argument('--val_tag', type=int, default=0)
  parser.add_argument('--val_all', type=int, default=1)
  parser.add_argument('--ablation', type=str, default='full', choices=['base', 'fg', 'local', 'full'], 
  					  help='exp for one-shot, fine_grain, plus localization, plus pconv')
  parser.add_argument('--val_loc_tag', action='store_true')
  parser.add_argument('--fine_tune', action='store_true')
  parser.add_argument('--debug_mode', action='store_true')
  parser.set_defaults(val_loc_tag=True)
  parser.set_defaults(fine_tune=True)

  parser.add_argument('--train_ratio', nargs='+', default="0.4 0.4 0.2", help='if 4 values, authentic, splice, copymove, removal; \
  						if 3 values, then splice, copymove, removal')
  parser.add_argument('--path', type=str, default="/user/guoxia11/cvl/semi_supervise_local/2021/05_25_SCCM/Dataset/pretrain_dataset")
  parser.add_argument('--train_bs', type=int, default=10, help='batch size in the training.')
  parser.add_argument('--val_bs', type=int, default=10, help='batch size in the validation.')
  parser.add_argument('--percent', type=float, default=1.0, help='label dataset.')
  args = parser.parse_args()
  main(args)