# coding: utf-8
# author: Hierarchical Fine-Grained Image Forgery Detection and Localization
import datetime
import logging
import sys
import torch
import os
import datetime

def init_logger(name):
    logger = logging.getLogger(name)
    h = logging.StreamHandler(sys.stdout)
    h.flush = sys.stdout.flush
    logger.addHandler(h)
    return logger

logger = init_logger(__name__)
logger.setLevel(logging.INFO)

def torch_load_model(model, optimizer, load_model_path,strict=True):
    loaded_file = torch.load(load_model_path)
    model.load_state_dict(loaded_file['model_state_dict'], strict=strict)
    # model.load_state_dict(loaded_file['model_state_dict'], strict=False)
    iteration = loaded_file['iter']
    scheduler = loaded_file['scheduler']
    epoch = loaded_file['epoch']
    val_loss = 1.0
    if 'val_loss' in loaded_file:
        val_loss = loaded_file['val_loss']
    # optimizer.load_state_dict(loaded_file['optimizer_state_dict'])    
    return iteration, epoch, scheduler, val_loss

class DataConfig(object):
    def __init__(self, model_path, model_name):
        self.model_path = model_path
        self.model_name = model_name

class Saver(object):
    def __init__(self, model, optimizer, scheduler, data_config,
                 starting_time, hours_limit=23, mins_limit=0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_val_loss = sys.maxsize
        self.data_config = data_config
        
        self.hours_limit = hours_limit
        self.mins_limit = mins_limit
        self.starting_time = starting_time

    def save_model(self,epoch,ib,val_loss,before_train,best_only=False,force_saving=False):
        # if (val_loss  <= self.best_val_loss and not(before_train)) or force_saving:
        if val_loss <= self.best_val_loss or force_saving:
            ## preserving best_loss
            if val_loss  <= self.best_val_loss:
                self.best_val_loss = val_loss
                
            if best_only:
                saving_list = [os.path.join(self.data_config.model_path,'best_model.pth')]

            if force_saving:
                saving_list = [os.path.join(self.data_config.model_path,'current_model.pth')]
            print("===================================")
            print(f"saving model list is: ", saving_list)
            print("===================================")
            for ss in saving_list:
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict':
                            self.optimizer.state_dict() if self.optimizer is not None else None,
                            'iter' : ib,
                            'scheduler' : self.scheduler,
                            'val_loss' : val_loss,
                            },
                           ss
                )

    def check_time(self):
        this_time = datetime.datetime.now()
        days, hours, mins = self.days_hours_minutes(
            this_time - self.starting_time)
        return days, hours, mins

    def days_hours_minutes(self, td):
        return td.days, td.seconds//3600, (td.seconds//60) % 60