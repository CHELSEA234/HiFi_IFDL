# coding: utf-8
# author: Hierarchical Fine-Grained Image Forgery Detection and Localization
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn import metrics
import numpy as np
from runjobs_utils import init_logger
import logging
import torch.nn.functional as F
import os
from collections import OrderedDict
import csv

logger = init_logger(__name__)
logger.setLevel(logging.INFO)

class ROC(object):
    def __init__(self):
        self.fpr = None
        self.tpr = None
        self.auc = None
        self.scores = None
        self.ap_0 = None
        self.ap_1 = None
        self.weighted_ap = None
        
        self.predictions = []
        self.gt = []
        self.best_acc = None

    def get_trunc_auc(self,fpr_value):
        abs_fpr = np.absolute(self.fpr - fpr_value)
        idx_min = np.argmin(abs_fpr)
        area_curve = sum(self.tpr[idx_min])
        tot_area = sum(np.ones_like(self.tpr)[idx_min])
        if tot_area == 0:
            raise ZeroDivisionError('when computing truncated ROC aread')
        t_auc = area_curve/tot_area
        return t_auc

    def get_tpr_at_fpr(self,fpr_value):
        abs_fpr = np.absolute(self.fpr - fpr_value)
        idx_min = np.argmin(abs_fpr)
        fpr_value_target = self.fpr[idx_min]
        idx = np.max(np.where(self.fpr == fpr_value_target))
        return self.tpr[idx], self.scores[idx]
        
    def eval(self):
        self.fpr, self.tpr, self.scores = metrics.roc_curve(self.gt,self.predictions,drop_intermediate=True)
        self.auc = metrics.auc(self.fpr,self.tpr)

    def compute_best_accuracy(self,n_samples=200):
        '''find the best threshold for the accuracy.'''
        acc_thrs = []
        min_thr = min(self.predictions)
        max_thr = max(self.predictions)
        all_thrs = np.linspace(min_thr,max_thr,n_samples).tolist()
        for t in all_thrs:
            acc = self.compute_acc(self.predictions,self.gt,t)
            acc_thrs.append((t,acc))
        acc_thrs_arr = np.array(acc_thrs)
        idx_max = acc_thrs_arr[:,1].argmax()
        best_thr = acc_thrs_arr[idx_max,0]
        self.best_acc = acc_thrs_arr[idx_max,1]
        return best_thr, self.best_acc

    def compute_acc(self,list_scores,list_labels,thr):
        labels = np.array(list_labels)
        scores_th = (np.array(list_scores) >= thr).astype(np.int32)
        acc = (scores_th==labels).sum()/labels.size
        return acc
    
    def get_precision(self,criterion,thr):
        '''compute the best precision'''
        pred_labels = []
        for d in self.predictions:
            if (d < thr):
                pred_labels.append(0)
            elif (d >= thr):
                pred_labels.append(1)
        self.ap_0 = metrics.precision_score(self.gt, pred_labels, average='binary', pos_label=0)
        self.ap_1 = metrics.precision_score(self.gt, pred_labels, average='binary', pos_label=1)
        self.weighted_ap = metrics.precision_score(self.gt, pred_labels, average='weighted')

class Metrics(object):
    def __init__(self):
        self.tp = 0
        self.tot_samples = 0
        self.loss = 0.0
        self.loss_samples = 0
        self.roc = ROC()
        
        self.best_valid_acc = 0.0
        self.best_valid_thr = 0.0

        self.tuned_acc_thrs = (0,0)
        
    def update(self,tp,loss_value,samples):
        self.tp+=tp
        self.tot_samples+=samples
        self.loss+=loss_value
        self.loss_samples+=1

    def get_avg_loss(self):
        if self.loss_samples == 0:
            raise ZeroDivisionError('not enough sample to avg loss')
        return self.loss/self.loss_samples

def count_matching_samples(preds,true_labels,criterion,use_magic_loss=True):
    acc = 0
    if use_magic_loss:
        for l,d in zip(true_labels,preds):
            if (l == criterion.class_label and d < criterion.R) \
            or (l != criterion.class_label and d >= criterion.R):
                acc += 1
    else:
        matching_idx = (preds.argmax(dim=1)==true_labels)
        acc = matching_idx.sum().item()
    return acc

def eval_model(model,dataset_name,valid_joined_generator,criterion,
               device,desc='valid',val_metrics=None,
               debug_mode=False):
    model.eval()
    print(f"with the eval model and the debug mode {debug_mode}.")
    with torch.no_grad():
        metrics = Metrics()
        for jb, val_batch in tqdm(enumerate(valid_joined_generator,1),
                                  total=len(valid_joined_generator),
                                  desc=desc):
            if jb % 8 != 0 and debug_mode:
                continue
            ## Getting Input
            val_img_batch_mmodal, val_true_labels, image_names = val_batch
            n_samples = val_img_batch_mmodal.shape[0]
            val_img_batch_mmodal = val_img_batch_mmodal.float().to(device)      
            val_true_labels = val_true_labels.long().to(device)
            ## Inference
            val_preds = model(val_img_batch_mmodal)

            ## Computing loss
            val_loss = criterion(val_preds, val_true_labels)
            log_probs = F.softmax(val_preds, dim=-1)
            res_probs = torch.argmax(log_probs, dim=-1)
            fixed_labels = 1 - val_true_labels
                    
            ## acc/matching_samples. 
            matching_num = count_matching_samples(val_preds,val_true_labels,criterion,use_magic_loss=False)
            # metrics.roc.predictions.extend(res_probs.tolist())
            metrics.roc.predictions.extend(log_probs[:,0].tolist())
            ## Inverting the labels
            metrics.roc.gt.extend(fixed_labels[:].tolist())
            metrics.update(matching_num,val_loss.item(),n_samples)
            
    ## Getting the Results
    metrics.roc.eval()
    print("the auc is: %.5f"%metrics.roc.auc)
    best_acc = best_thr = None
    best_thr, best_acc = metrics.roc.compute_best_accuracy()
    metrics.best_valid_acc = best_acc
    metrics.best_valid_thr = best_thr
    print("the accuracy is: %.5f: "%best_acc)
    print("the threshold is: %.5f: "%best_thr)
    fpr_values = [0.1,0.01]    
    for fpr_value in fpr_values:
        tpr_fpr, score_for_tpr_fpr = metrics.roc.get_tpr_at_fpr(fpr_value)
        print('tpr_fpr_%.1f: '%(fpr_value*100.0), "%.5f"%tpr_fpr)
    ## Setting the model back to train mode
    model.train()
    return metrics

def display_eval_tb(writer,metrics,tot_iter,desc='test',old_metrics=False):
    avg_loss = metrics.get_avg_loss()
    acc = metrics.roc.best_acc
    auc = metrics.roc.auc
    writer.add_scalar('%s/loss'%desc, avg_loss, tot_iter)
    writer.add_scalar('%s/acc'%desc, acc, tot_iter)                      
    writer.add_scalar('%s/auc'%desc, auc, tot_iter)
    fpr_values = [0.1,0.01]    
    for fpr_value in fpr_values:
        tpr_fpr, score_for_tpr_fpr = metrics.roc.get_tpr_at_fpr(fpr_value)
        writer.add_scalar('%s/tpr_fpr_%.0f'%(desc,(fpr_value*100.0)), tpr_fpr, tot_iter)

def train_logging(string, writer, logger, epoch, saver, tot_iter, loss, accu, lr_scheduler):
    _, hours, mins = saver.check_time()
    logger.info("[Epoch %d] | h:%d m:%d | iteration: %d, loss: %f, accu: %f", epoch, hours, mins, tot_iter,
                loss, accu)
    
    writer.add_scalar(string, loss, tot_iter )
    for count, gp in enumerate(lr_scheduler.optimizer.param_groups,1):
        writer.add_scalar('progress/lr_%d'%count, gp['lr'], tot_iter)
    writer.add_scalar('progress/epoch', epoch, tot_iter)
    writer.add_scalar('progress/curr_patience',lr_scheduler.num_bad_epochs,tot_iter)
    writer.add_scalar('progress/patience',lr_scheduler.patience,tot_iter)

class lrSched_monitor(object):
    """
    This class is used to monitor the learning rate scheduler's behavior
    during training. If the learning rate decreases then this class re-initializes
    the last best state of the model and starts training from that point of time.
    
    Parameters
    ----------
    model : torch model
    scheduler : learning rate scheduler object from training
    data_config : this object holds model_path and model_name, used to load the last best model.
    """
    def __init__(self, model, scheduler, data_config):
        self.model = model
        self.scheduler = scheduler
        self.model_name = data_config.model_name
        self.model_path = data_config.model_path
        self._last_lr = [0]*len(scheduler.optimizer.param_groups)
        self.prev_lr_mean = self.get_lr_mean()
    
    ## Get the current mean learning rate from the optimizer
    def get_lr_mean(self):
        lr_mean = 0
        for i, grp in enumerate(self.scheduler.optimizer.param_groups):
            if 'lr' in grp.keys():
                lr_mean += grp['lr']
                self._last_lr[i] = grp['lr']
        return lr_mean/(i+1)       
        
    ## This is the function that is to be called right after lr_scheduler.step(val_loss)    
    def monitor(self):
        if self.scheduler.num_bad_epochs == self.scheduler.patience:
            self.prev_lr_mean = self.get_lr_mean()
        elif self.get_lr_mean() < self.prev_lr_mean:
            self.load_best_model()
            self.prev_lr_mean = self.get_lr_mean()
    
    ## This function loads the last best model once the learning rate decreases
    def load_best_model(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            ckpt = torch.load(os.path.join(self.model_path,'best_model.pth'))
            self.model.load_state_dict(ckpt['model_state_dict'], strict=True)
            self.scheduler.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        else:
            print(f'Loading the best model from {self.model_path}')
            if device.type == 'cpu':
                ckpt = torch.load(os.path.join(self.model_path,'best_model.pth'), map_location='cpu')
            else:
                ckpt = torch.load(os.path.join(self.model_path,'best_model.pth'))
            ## Model State Dict
            state_dict = ckpt['model_state_dict']
            ## Since the model files are saved on dataparallel we use the below hack to load the weights on a model in cpu or a model on single gpu.
            keys = state_dict.keys()
            values = state_dict.values()
            new_keys = []
            for key in keys:
                new_key = key.replace('module.','')    # remove the 'module.'
                new_keys.append(new_key)

            new_state_dict = OrderedDict(list(zip(new_keys, values))) # create a new OrderedDict with (key, value) pairs
            self.model.load_state_dict(new_state_dict, strict=True)
            
            ## Optimizer State Dict
            optim_state_dict = ckpt['optimizer_state_dict']
            # Since the model files are saved on dataparallel we use the below hack to load the optimizer state in cpu or a model on single gpu.
            keys = optim_state_dict.keys()
            values = optim_state_dict.values()
            new_keys = []
            for key in keys:
                new_key = key.replace('module.','')    # remove the 'module.'
                new_keys.append(new_key)

            new_optim_state_dict = OrderedDict(list(zip(new_keys, values))) # create a new OrderedDict with (key, value) pairs
            self.scheduler.optimizer.load_state_dict(new_optim_state_dict)
        
        ## Reduce the learning rate
        for i, grp in enumerate(self.scheduler.optimizer.param_groups):
            grp['lr'] = self._last_lr[i]
