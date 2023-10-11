# ------------------------------------------------------------------------------
# Author: Xiao Guo (guoxia11@msu.edu)
# CVPR2023: Hierarchical Fine-Grained Image Forgery Detection and Localization
# ------------------------------------------------------------------------------
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm

device = torch.device('cuda:0')
device_ids = [0]

class IsolatingLossFunction(torch.nn.Module):
	def __init__(self, c, R, p=2, threshold_val=1.85):
		super().__init__()
		self.c = c.clone().detach() # Center of the hypershpere, c ∈ ℝ^d (d-dimensional real-valued vector)
		self.R = R.clone().detach() # Radius of the hypersphere, R ∈ ℝ^1 (Real-valued)
		self.p = p                  # norm value (p-norm), p ∈ ℝ^1 (Default 2)
		self.margin_natu = (0.15)*self.R    
		self.margin_mani = (2.5)*self.R
		self.threshold   = threshold_val*self.R

		print('\n')
		print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
		print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
		print(f'The Radius manipul is {self.margin_natu}.')
		print(f'The Radius expansn is {self.margin_mani}.')
		print(f'The Radius threshold is {self.threshold}.')
		print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
		print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
		print('\n')
		self.pdist = torch.nn.PairwiseDistance(p=self.p) # Creating a Pairwise Distance object
		self.dis_curBatch = 0
		self.dis = 0

	def forward(self, model_output, label, threshold_new=None, update_flag=None):
		'''output the distance mask and compute the loss.'''
		bs, feat_dim, w, h = model_output.size()
		model_output = model_output.permute(0,2,3,1)
		model_output = torch.reshape(model_output, (-1, feat_dim))
		dist = self.pdist(model_output, self.c)
		self.dist = dist
		pred_mask = torch.gt(self.dist, self.threshold).to(torch.float32)
		pred_mask = torch.reshape(pred_mask, (bs,w,h,1)).permute(0,3,1,2)
		self.dist = torch.reshape(self.dist, (bs,w,h,1)).permute(0,3,1,2)
		self.dis_curBatch = pred_mask.to(device).to(torch.float32)

		label = torch.reshape(label, (bs*w*h,1))
		label_sum = label.sum().item()

		label_nat  = torch.eq(label,0)
		label_mani = torch.eq(label,1)
		assert dist.size() == label_nat[:,0].size() 
		assert dist.size() == label_mani[:,0].size()

		label_nat_sum  = label_nat.sum().item()
		label_mani_sum = label_mani.sum().item()

		dist_nat  = torch.masked_select(dist, label_nat[:,0])
		dist_mani = torch.max(torch.tensor(0).to(device).float(),
							  torch.sub(self.margin_mani, 
										torch.masked_select(dist, label_mani[:,0]))
							  )

		loss_nat  = dist_nat.sum()/label_nat_sum if label_nat_sum != 0 else \
					torch.tensor(0).to(device).float()
		loss_mani = dist_mani.sum()/label_mani_sum if label_mani_sum != 0 else \
					torch.tensor(0).to(device).float()
		loss_total = loss_nat + loss_mani

		return loss_total.to(device), loss_mani.to(device), loss_nat.to(device)

	def inference(self, model_output):
		'''output the distance for the final binary mask.'''
		bs, feat_dim, w, h = model_output.size()
		model_output = model_output.permute(0,2,3,1)
		model_output = torch.reshape(model_output, (-1, feat_dim))
		dist = self.pdist(model_output, self.c)
		self.dist = dist
		pred_mask = torch.gt(self.dist, self.threshold).to(torch.float32)
		pred_mask = torch.reshape(pred_mask, (bs,w,h,1)).permute(0,3,1,2)
		self.dist = torch.reshape(self.dist, (bs,w,h,1)).permute(0,3,1,2)
		self.dis_curBatch = pred_mask.to(device).to(torch.float32)
		return self.dis_curBatch.squeeze(dim=1), self.dist.squeeze(dim=1)

def center_radius_init(args, FENet, SegNet, train_data_loader, debug=True, center=None):
	'''the center is the mean-value of pixel features of the real pixels'''
	sample_num = 0
	center = torch.zeros(18).to(device)
	FENet.eval()
	SegNet.eval()
	with torch.no_grad():
		for batch_id, train_data in enumerate(tqdm(train_data_loader, desc="compute center")):  
			image, masks, cls, fcls, scls, tcls = train_data
			if batch_id % 10 != 0:
				continue
			mask_cls = fcls.eq(0)
			image_selected = image[mask_cls,:]
			if image_selected.size()[0] == 0:
				continue
			else:
				sample_num += image_selected.size()[0]
			mask1 = masks[0].to(device)
			image_selected = image_selected.to(device)
			cls = cls.to(device)
			mask1_fea = FENet(image_selected)
			mask1_fea, _, _, _, _, _ = SegNet(mask1_fea, image_selected)
			mask1_fea = torch.mean(mask1_fea,(0,2,3))
			center += mask1_fea

	center = center/sample_num
	pdist  = torch.nn.PairwiseDistance(2)
	radius = torch.tensor(0, dtype=torch.float32).to(device)
	with torch.no_grad():
		for batch_id, train_data in enumerate(tqdm(train_data_loader, desc="compute radius")):  
			if batch_id % 10 != 0:
				continue    
			image, masks, cls, fcls, scls, tcls = train_data
			mask1 = masks[0].to(device)
			image = image.to(device)
			fcls = fcls.to(device)
			mask_cls = fcls.eq(0)
			image_selected = image[mask_cls,:]
			if image_selected.size()[0] == 0:
				continue
			mask1_fea = FENet(image_selected)
			mask1_fea, _, _, _, _, _ = SegNet(mask1_fea, image_selected)
			bs, channel, h, w = mask1_fea.size()
			mask1_fea = mask1_fea.permute(0,2,3,1)      
			mask1_fea = torch.reshape(mask1_fea, (bs*w*h, -1))
			dist = pdist(mask1_fea, center)
			dist_max = torch.max(dist)
			if radius < dist_max:
				radius = dist_max
	return center, radius

def load_center_radius(args, FENet, SegNet, train_data_loader, center_radius_dir='center'):
	'''loading the pre-computed center and radius.'''
	center_radius_path = os.path.join(center_radius_dir, 'radius_center.pth')
	if os.path.exists(center_radius_path):
		load_dict_center_radius = torch.load(center_radius_path)
		center = load_dict_center_radius['center']
		radius = load_dict_center_radius['radius']
		center, radius = center.to(device), radius.to(device)
	else:
		os.makedirs(center_radius_dir, exist_ok=True)
		center, radius = center_radius_init(args, FENet, SegNet, train_data_loader, debug=True)
		torch.save({'center': center, 'radius': radius}, center_radius_path)
	return center, radius

def load_center_radius_api(center_radius_dir='center'):
	'''loading the pre-computed center and radius.'''
	center_radius_path = os.path.join(center_radius_dir, 'radius_center.pth')
	load_dict_center_radius = torch.load(center_radius_path)
	center = load_dict_center_radius['center']
	radius = load_dict_center_radius['radius']
	center, radius = center.to(device), radius.to(device)
	return center, radius