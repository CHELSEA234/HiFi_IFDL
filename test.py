# ------------------------------------------------------------------------------
# Author: Xiao Guo (guoxia11@msu.edu)
# CVPR2023: Hierarchical Fine-Grained Image Forgery Detection and Localization
# ------------------------------------------------------------------------------
from utils.utils import *
from utils.custom_loss import IsolatingLossFunction, load_center_radius
from IMD_dataloader import train_dataset_loader_init, infer_dataset_loader_init
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_cfg_defaults
from models.NLCDetection_pconv import NLCDetection
from sklearn import metrics
from sklearn.metrics import roc_auc_score

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

def Inference(
			args, FENet, SegNet, LOSS_MAP, tb_writer, 
			iter_num=None, save_tag=False, localization=True,
			mask_save_path=f'/user/guoxia11/cvlshare/cvl-guoxia11/CVPR2023_github/mask_result/'
			):
	'''
		the inference pipeline for the pre-trained model.
		the image-level detection will dump to the csv file.
		the pixel-level localization will be saved as in the npy file.
	'''
	val_data_loader = infer_dataset_loader_init(args, val_tag=0)
	val_num_per_epoch = len(val_data_loader)
	AUC_score_lst, F1_score_lst = [], []
	mask_GT_lst, mask_pred_lst, mask_scr_lst = [], [], []
	img_GT_list, img_pred_list, img_scr_list = [], [], []

	## localization: forgery mask is saved in the npy file.
	mask_save_path = mask_save_path
	os.makedirs(mask_save_path, exist_ok=True)

	## detection: different scores are saved in the csv file.
	csv_file_name = f'result_{args.learning_rate}.csv'
	csv_file = open(csv_file_name, mode='w')
	csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	csv_writer.writerow(['Image_names', 'Sc_4', 'RS_4', 'GT_4',
										'Sc_3', 'RS_3', "GT_3",
										'Sc_2', "RS_2", "GT_2",
										'Sc_1', "RS_1", "GT_1"])

	with torch.no_grad():
		FENet.eval()
		SegNet.eval()
		for step, val_data in enumerate(tqdm(val_data_loader)):
			image, masks, cls0, cls1, cls2, cls3, img_names = val_data
			cls0, cls1, cls2, cls3 = cls0.to(device), cls1.to(device), cls2.to(device), cls3.to(device)
			image, mask = image.to(device), masks[0].to(device)
			mask, mask_balance = class_weight(mask, 1)
			output = FENet(image)
			mask1_fea, mask1_binary, out0, out1, out2, out3 = SegNet(output, image)

			## detection.
			res3, prob3 = one_hot_label_new(out3)
			res2, prob2 = one_hot_label_new(out2)
			res1, prob1 = one_hot_label_new(out1)
			res0, prob0 = one_hot_label_new(out0)
			
			cls3_np = list(cls3.cpu().numpy())
			cls2_np = list(cls2.cpu().numpy())
			cls1_np = list(cls1.cpu().numpy())
			cls0_np = list(cls0.cpu().numpy())
			for i in range(len(cls3_np)):
				write_list = [img_names[i], 
								prob3[i], res3[i], cls3_np[i], 
								prob2[i], res2[i], cls2_np[i],
								prob1[i], res1[i], cls1_np[i],
								prob0[i], res0[i], cls0_np[i]
							  ]
				csv_writer.writerow(write_list)
			csv_file.flush()
	
			##############################################################################
			## The following is the examplar code for dumping the mask into numpy files.
			## The final version of localization measurement will be updated.
			## localization. 
			# loss_map, loss_manip, loss_nat = LOSS_MAP(mask1_fea, mask)
			# pred_mask = LOSS_MAP.dis_curBatch.squeeze(dim=1)
			# pred_mask_score = LOSS_MAP.dist.squeeze(dim=1)

			# np.save(f'{mask_save_path}/{step}_mask.npy', mask.cpu().numpy())
			# np.save(f'{mask_save_path}/{step}_pred_mask.npy', pred_mask.cpu().numpy())
			# np.save(f'{mask_save_path}/{step}_pred_mask_score.npy', pred_mask_score.cpu().numpy())

	csv_file.close()

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
	LOSS_MAP = IsolatingLossFunction(center,radius).to(device)

	Inference(
			args, FENet, SegNet, LOSS_MAP, tb_writer=writer, iter_num=initial_iter, save_tag=True, 
			localization=True
			)

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

	parser.add_argument('--train_ratio', nargs='+', default="0.4 0.4 0.2", help='deprecated')
	parser.add_argument('--path', type=str, default="", help='deprecated')
	parser.add_argument('--train_bs', type=int, default=10, help='batch size in the training.')
	parser.add_argument('--val_bs', type=int, default=10, help='batch size in the validation.')
	parser.add_argument('--percent', type=float, default=1.0, help='label dataset.')
	args = parser.parse_args()
	main(args)