# ------------------------------------------------------------------------------
# Author: Xiao Guo (guoxia11@msu.edu)
# CVPR2023: Hierarchical Fine-Grained Image Forgery Detection and Localization
# ------------------------------------------------------------------------------
from utils.utils import *
from utils.custom_loss import IsolatingLossFunction, load_center_radius_api
from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_cfg_defaults
from models.NLCDetection_api import NLCDetection
from PIL import Image

import torch
import torch.nn as nn
import numpy as np
import argparse
import imageio as imageio

class HiFi_Net():
    '''
        FENET is the multi-branch feature extractor.
        SegNet contains the classification and localization modules.
        LOSS_MAP is the classification loss function class.
    '''
    def __init__(self):
        device = torch.device('cuda:0')
        device_ids = [0]

        FENet_cfg = get_cfg_defaults()
        FENet  = get_seg_model(FENet_cfg).to(device) # load the pre-trained model inside.
        SegNet = NLCDetection().to(device)
        FENet  = nn.DataParallel(FENet)
        SegNet = nn.DataParallel(SegNet)

        self.FENet  = restore_weight_helper(FENet,  "weights/HRNet",  750001)
        self.SegNet = restore_weight_helper(SegNet, "weights/NLCDetection", 750001)
        self.FENet.eval()
        self.SegNet.eval()

        center, radius = load_center_radius_api()
        self.LOSS_MAP = IsolatingLossFunction(center,radius).to(device)

    def _transform_image(self, image_name):
        '''transform the image.'''
        image = imageio.imread(image_name)
        image = Image.fromarray(image)
        image = image.resize((256,256), resample=Image.BICUBIC)
        image = np.asarray(image)
        image = image.astype(np.float32) / 255.
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        image = torch.unsqueeze(image, 0)
        return image

    def _normalized_threshold(self, res, prob, threshold=0.5, verbose=False):
        '''to interpret detection result via omitting the detection decision.'''
        if res > threshold:
            decision = "Forged"
            prob = (prob - threshold) / threshold
        else:
            decision = 'Real'
            prob = (threshold - prob) / threshold
        print(f'Image being {decision} with the confidence {prob*100:.1f}.')

    def detect(self, image_name, verbose=False):
        """
            Para: image_name is string type variable for the image name.
            Return:
                res: binary result for real and forged.
                prob: the prob being the forged image.
        """
        with torch.no_grad():
            img_input = self._transform_image(image_name)
            output = self.FENet(img_input)
            mask1_fea, mask1_binary, out0, out1, out2, out3 = self.SegNet(output, img_input)
            res, prob = one_hot_label_new(out3)
            res = level_1_convert(res)[0]
            if not verbose:
                return res, prob[0]
            else:
                self._normalized_threshold(res, prob[0])

    def localize(self, image_name):
        """
            Para: image_name is string type variable for the image name.
            Return:
                binary_mask: forgery mask.
        """
        with torch.no_grad():
            img_input = self._transform_image(image_name)
            output = self.FENet(img_input)
            mask1_fea, mask1_binary, out0, out1, out2, out3 = self.SegNet(output, img_input)
            pred_mask, pred_mask_score = self.LOSS_MAP.inference(mask1_fea)   # inference
            pred_mask_score = pred_mask_score.cpu().numpy()
            ## 2.3 is the threshold used to seperate the real and fake pixels.
            ## 2.3 is the dist between center and pixel feature in the hyper-sphere.
            ## for center and pixel feature please refer to "IsolatingLossFunction" in custom_loss.py
            pred_mask_score[pred_mask_score<2.3] = 0.
            pred_mask_score[pred_mask_score>=2.3] = 1.
            binary_mask = pred_mask_score[0]        
            return binary_mask


def inference(img_path):
    HiFi = HiFi_Net()   # initialize
    
    ## detection
    res3, prob3 = HiFi.detect(img_path)
    # print(res3, prob3) 1 1.0
    HiFi.detect(img_path, verbose=True)
    
    ## localization
    binary_mask = HiFi.localize(img_path)
    binary_mask = Image.fromarray((binary_mask*255.).astype(np.uint8))
    binary_mask.save('pred_mask.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='asset/sample_1.jpg')
    args = parser.parse_args()
    inference(args.img_path)
