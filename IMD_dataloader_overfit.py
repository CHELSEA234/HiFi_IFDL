from torch.utils.data import DataLoader
from utils.load_vdata_overfit import *

def eval_dataset_loader_init(args, val_tag, batch_size=1):
	
	if val_tag == 0:
		data_label = 'columbia'
		val_data_loader = DataLoader(ValColumbia(args), batch_size=batch_size, shuffle=False,
									 num_workers=0)
	elif val_tag == 1:
		data_label = 'coverage'
		val_data_loader = DataLoader(ValCoverage(args), batch_size=batch_size, shuffle=False,
									 num_workers=0)
	elif val_tag == 2:
		data_label = 'casia'
		val_data_loader = DataLoader(ValCasia(args), batch_size=batch_size, shuffle=False,
									 num_workers=0)
	elif val_tag == 3:
		data_label = 'NIST16'
		val_data_loader = DataLoader(ValNIST16(args), batch_size=batch_size, shuffle=False,
									 num_workers=0)
	elif val_tag == 4:
		data_label = 'IMD2020'
		val_data_loader = DataLoader(ValIMD2020(args), batch_size=batch_size, shuffle=False,
									 num_workers=0)
	return val_data_loader, data_label