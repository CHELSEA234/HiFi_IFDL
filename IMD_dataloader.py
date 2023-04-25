# ------------------------------------------------------------------------------
# Author: Xiao Guo, Xiaohong Liu.
# CVPR2023: Hierarchical Fine-Grained Image Forgery Detection and Localization
# ------------------------------------------------------------------------------
from torch.utils.data import DataLoader

def train_dataset_loader_init(args):
	from utils.load_tdata import TrainData
	train_data_loader = DataLoader(TrainData(args), batch_size=args.train_bs, 
									shuffle=True, num_workers=8)
	return train_data_loader

def infer_dataset_loader_init(args, val_tag):
	from utils.load_vdata import ValData
	val_dataset = ValData(args)
	val_data_loader = DataLoader(val_dataset, batch_size=64,
								 shuffle=False, num_workers=8)
	return val_data_loader

def viz_dataset_loader_init(args, val_tag):
	from utils.load_viz import VizData
	viz_data_loader = DataLoader(VizData(args), batch_size=64,
								 shuffle=False, num_workers=8)
	return viz_data_loader

def unseen_dataset_loader_init(args, val_tag):
	from utils.load_unseen import UnseenData
	val_data_loader = DataLoader(UnseenData(args), batch_size=64,
								 shuffle=False, num_workers=8)
	return val_data_loader	