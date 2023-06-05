# ------------------------------------------------------------------------------
# Author: Xiao Guo, Xiaohong Liu.
# CVPR2023: Hierarchical Fine-Grained Image Forgery Detection and Localization
# ------------------------------------------------------------------------------
from torch.utils.data import DataLoader

def train_dataset_loader_init(args):
	from utils.load_tdata import TrainData
	train_dataset = TrainData(args)
	train_data_loader = DataLoader(
								train_dataset, 
								batch_size=args.train_bs, 
								shuffle=True, 
								num_workers=8
								)
	return train_data_loader

def infer_dataset_loader_init(args, val_tag):
	from utils.load_vdata import ValData
	val_dataset = ValData(args)
	val_data_loader = DataLoader(
								val_dataset, 
								batch_size=64,
								shuffle=False, 
								num_workers=8
								)
	return val_data_loader