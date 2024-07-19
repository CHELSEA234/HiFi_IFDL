# coding: utf-8
# author: Hierarchical Fine-Grained Image Forgery Detection and Localization
import os
import numpy as np
import subprocess
import logging
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import datetime

from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
source_path = os.path.join('./sequence')
sys.path.append(source_path)
from rnn_stratified_dataloader import get_dataloader
from models.HiFiNet_deepfake import HiFiNet_deepfake
from torch_utils import eval_model,display_eval_tb,train_logging,lrSched_monitor
from runjobs_utils import init_logger,Saver,DataConfig,torch_load_model

logger = init_logger(__name__)
logger.setLevel(logging.INFO)

starting_time = datetime.datetime.now()

## Deterministic training
_seed_id = 100
torch.backends.cudnn.deterministic = True
torch.manual_seed(_seed_id)

datasets = ['original', 'Deepfakes', 'FaceSwap', 'NeuralTextures', 'Face2Face']
# datasets = ['original', 'Deepfakes']
manipulations_names = [n for c, n in enumerate(datasets) if n != 'original']
manipulations_dict = {n : c  for c, n in enumerate(manipulations_names) }
manipulations_dict['original'] = 255

for key, value in manipulations_dict.items():
	print(key, value)
ctype = 'c40'

# Create the parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training (default: 32)')
parser.add_argument('--window_size', type=int, default=5, help='size of the sliding window (default: 5)')
parser.add_argument('--dataset_name', type=str, default="FF++", help='size of the sliding window (default: 5)')
parser.add_argument('--gpus', type=int, default=4, help='input batch size for training (default: 32)')
parser.add_argument('--feat_dim', type=int, default=270, help='input dim to rnn. (default: 32)')
parser.add_argument('--valid_epoch', type=int, default=2, help='val epoch')
parser.add_argument('--display_step', type=int, default=50, help='display the loss value.')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='the used learning rate')

# Parse the arguments
args = parser.parse_args()
## Hyper-params #######################
hparams = {
            'epochs': 50, 'batch_size': args.batch_size, 
            'basic_lr': args.learning_rate, 'fine_tune': True, 'use_laplacian': True, 
            'step_factor': 0.1, 'patience': 20, 'weight_decay': 1e-06, 'lr_gamma': 2.0, 'use_magic_loss': True, 
            'feat_dim': args.feat_dim, 'drop_rate': 0.2, 
            'skip_valid': False, 'rnn_type': 'LSTM', 'rnn_hidden_size': 256, 
            'num_rnn_layers': 1, 'rnn_drop_rate': 0.2, 
            'bidir': False, 'merge_mode': 'concat', 'perc_margin_1': 0.95, 'perc_margin_2': 0.95, 'soft_boundary': False, 
            'dist_p': 2, 'radius_param': 0.84, 'strat_sampling': True, 'normalize': True, 'window_size': args.window_size, 'hop': 1, 
            'valid_epoch': args.valid_epoch, 'display_step': args.display_step, 'use_sched_monitor': True
            }
batch_size = hparams['batch_size']
basic_lr = hparams['basic_lr']
fine_tune = hparams['fine_tune']
use_laplacian = hparams['use_laplacian']
step_factor = hparams['step_factor']
patience = hparams['patience']
weight_decay = hparams['weight_decay']
lr_gamma = hparams['lr_gamma']
use_magic_loss = hparams['use_magic_loss']
feat_dim = hparams['feat_dim']
drop_rate = hparams['drop_rate']
rnn_type = hparams['rnn_type']
rnn_hidden_size = hparams['rnn_hidden_size']
num_rnn_layers = hparams['num_rnn_layers']
rnn_drop_rate = hparams['rnn_drop_rate']
bidir = hparams['bidir']
merge_mode = hparams['merge_mode']
perc_margin_1 = hparams['perc_margin_1']
perc_margin_2 = hparams['perc_margin_2']
dist_p = hparams['dist_p']
radius_param = hparams['radius_param']
strat_sampling = hparams['strat_sampling']
normalize = hparams['normalize']
window_size = hparams['window_size']
hop = hparams['hop']
soft_boundary = hparams['soft_boundary']
use_sched_monitor = hparams['use_sched_monitor']
########################################
workers_per_gpu = 6
dataset_name = f"{args.dataset_name}"
exp_name = f"exp_FF_c40_bs_{batch_size}_lr_{basic_lr}_ws_{window_size}"
model_name = exp_name
model_path = os.path.join(f'./{dataset_name}', model_name)
print(f'Window_size: {args.window_size}; Dataset: {dataset_name}; Batch_Size: {batch_size}; LR: {basic_lr}.')
os.makedirs('./log', exist_ok=True)
log_file_path = f"log/{exp_name}.txt"
with open(log_file_path, "a+") as log_file:
    log_file.write(
        f'Dataset Name: {dataset_name} \n' 
        f'Window_size: {args.window_size}'
    )

# Create the model path if doesn't exists
if not os.path.exists(model_path):
    subprocess.call(f"mkdir -p {model_path}", shell=True)

## Data Generation
img_path = "/user/guoxia11/cvlshare/cvl-guoxia11/FaceForensics_HiFiNet"
balanced_minibatch_opt = True

if dataset_name == 'FF++':
    train_generator, train_dataset = get_dataloader(
                                                img_path, datasets, ctype, manipulations_dict, window_size, hop, 
                                                use_laplacian, normalize, strat_sampling, balanced_minibatch_opt, 
                                                'train', batch_size, workers=workers_per_gpu*args.gpus
                                                )
    test_generator, test_dataset = get_dataloader(
                                                img_path, datasets, ctype, manipulations_dict, window_size, hop, 
                                                use_laplacian, normalize, strat_sampling, False, 
                                                'test', batch_size, workers=workers_per_gpu*args.gpus
                                                )
    # print("the dataset length is: ", len(train_dataset))
    # print("the dataloader length is: ", len(train_generator))
    del train_dataset
    del test_dataset
elif dataset_name == "CelebDF":        
    pass    ## TODO: will be released in the near future. 
elif dataset_name == 'DFW':
    pass    ## TODO: will be released in the near future. 

## Model definition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HiFiNet_deepfake(use_laplacian=True, drop_rate=drop_rate, use_magic_loss=False,
                        pretrained=True, rnn_drop_rate=rnn_drop_rate,
                        feat_dim=feat_dim, rnn_hidden_size=rnn_hidden_size, 
                        num_rnn_layers=num_rnn_layers,
                        bidir=bidir)
model = model.to(device)
model = torch.nn.DataParallel(model).cuda()

## Fine-tuning functions
params_to_optimize = model.parameters()
optimizer = torch.optim.Adam(params_to_optimize, lr=basic_lr, weight_decay=weight_decay)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=step_factor, min_lr=1e-09, patience=patience, verbose=True)
criterion = nn.CrossEntropyLoss()

## Re-loading the model in case
epoch_init=epoch=ib=ib_off=before_train=0
load_model_path = os.path.join(model_path,'current_model.pth')
val_loss = np.inf
if os.path.exists(load_model_path):
    logger.info(f'Loading weights, optimizer and scheduler from {load_model_path}...')
    _, _, _, _ = torch_load_model(model, optimizer, load_model_path)

## Saver object and data config
data_config = DataConfig(model_path, model_name)
saver = Saver(model, optimizer, lr_scheduler, data_config, starting_time, hours_limit=23, mins_limit=0)
sched_monitor = lrSched_monitor(model, lr_scheduler, data_config)

## Writer summary for tb
tb_folder = os.path.join(model_path, 'tb_logs',model_name)
writer = SummaryWriter(tb_folder)
log_string_config = '  '.join([k+':'+str(v) for k,v in hparams.items()])
writer.add_text('config : %s' % model_name, log_string_config, 0)

if epoch_init == 0:
    model.zero_grad()
## Start training
tot_iter = 0
total_loss = 0
total_accu = 0
for epoch in range(epoch_init,hparams['epochs']):
    logger.info(f'Epoch ############: {epoch}')
    for ib, (img_batch_mmodal, true_labels, manip_type) in enumerate(train_generator,1):
        img_batch = img_batch_mmodal.float().to(device)
        true_labels = true_labels.long().to(device)
        optimizer.zero_grad()
        pred_labels = model(img_batch)
        loss = criterion(pred_labels, true_labels)
        total_loss += loss.item()
        log_probs = F.softmax(pred_labels, dim=-1)
        res_probs = torch.argmax(log_probs, dim=-1)
        summation = torch.sum(res_probs == true_labels)
        accu = summation / img_batch.shape[0]
        total_accu += accu
        loss.backward()
        optimizer.step()
        tot_iter += 1
        if tot_iter % hparams['display_step'] == 0:
            train_logging(
                        'loss/train_loss_iter', writer, logger, epoch, saver, 
                        tot_iter, total_loss/hparams['display_step'], 
                        total_accu/hparams['display_step'], lr_scheduler
                        )
            with open(log_file_path, "a+") as log_file:
                log_file.write(
                    f"Epoch: {epoch}, Iteration: {tot_iter}, "
                    f"Train Loss: {total_loss/hparams['display_step']:.4f}, "
                    f"Accuracy: {total_accu/hparams['display_step']:.4f}\n"
                )
            total_loss = 0
            total_accu = 0
    saver.save_model(epoch,tot_iter,sys.maxsize,before_train,force_saving=True)

    if (epoch % hparams['valid_epoch'] == 0) or (epoch == hparams['epochs']):
        metrics = eval_model(model,dataset_name,test_generator,criterion,device,desc='valid',val_metrics=None,debug_mode=False)
        # metrics = eval_model(model,dataset_name,test_generator,criterion,device,desc='valid',val_metrics=None,debug_mode=True)
        val_loss = metrics.get_avg_loss()
        saver.save_model(epoch,ib+ib_off,val_loss,before_train,best_only=True)
        # display_eval_tb(writer,metrics,epoch,desc='valid')
        display_eval_tb(writer,metrics,epoch,desc='test')
        lr_scheduler.step(val_loss)
        sched_monitor.monitor()
        for i, grp in enumerate(sched_monitor.scheduler.optimizer.param_groups):
            if 'lr' in grp.keys():
                print("the first grp learning rate is: ", grp['lr'])
                break
        file_path = f"./{exp_name}.txt"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'a') as f:
            f.write(f"AUC: {metrics.roc.auc}\n")
            f.write(f"Best Accuracy: {metrics.best_valid_acc} (Threshold: {metrics.best_valid_thr})\n")
            for fpr_value in [0.1, 0.01]:
                tpr_fpr, score_for_tpr_fpr = metrics.roc.get_tpr_at_fpr(fpr_value)
                f.write(f"TPR at FPR={fpr_value*100}%: {tpr_fpr} (Score: {score_for_tpr_fpr})\n")
            f.write(f"Average Loss: {metrics.get_avg_loss()}\n")
            f.write("#" * 100)