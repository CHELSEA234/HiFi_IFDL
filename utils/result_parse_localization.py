'''localization result parser file.'''
import numpy as np
import csv
import os
import cv2
import argparse

from sklearn import metrics
from sklearn.metrics import roc_auc_score
from glob import glob
from tqdm import tqdm

def resize_resolution(input_arr, bs=32, img_size=16, GT=False, label=False):
    '''resize the image for the computation.'''
    '''GX: https://stackoverflow.com/a/65160547'''
    resized_tensor = np.empty((bs, img_size, img_size), dtype=input_arr.dtype)
    for (k, image) in enumerate(input_arr):
        resized_tensor[k] = cv2.resize(image, dsize=(img_size, img_size))
        if GT and label:
            resized_tensor[k][resized_tensor[k]<=0.5] = 0.
            resized_tensor[k][resized_tensor[k]>0.5]  = 1.
        elif GT and not label:
            ## 2.3 is the magic number used to separate the real and fake pixels.
            resized_tensor[k][resized_tensor[k]<=2.3] = 0.
            resized_tensor[k][resized_tensor[k]>2.3]  = 1.
    return resized_tensor


def main(args):
    file_name_lst = glob(args.npy_path + '/*') 
    file_name_lst.sort()
    file_num = int(len(file_name_lst) / 3)
    mask_array_lst, pred_array_lst, scr_array_lst = [], [], []

    for idx in tqdm(range(file_num)):
        mask_file = args.npy_path + '/' + str(idx) + '_mask.npy'
        pred_file = args.npy_path + '/' + str(idx) + '_pred_mask.npy'
        scr_file  = args.npy_path + '/' + str(idx) + '_pred_mask_score.npy'

        mask_array = np.load(mask_file)
        pred_array = np.load(pred_file)
        scr_array  = np.load(scr_file)

        mask_array = resize_resolution(mask_array, bs=mask_array.shape[0], GT=True, label=True)
        pred_array = resize_resolution(pred_array, bs=mask_array.shape[0], GT=True, label=True)
        scr_array  = resize_resolution(scr_array, bs=mask_array.shape[0])

        mask_array = list(np.reshape(mask_array, -1))
        pred_array = list(np.reshape(pred_array, -1))
        scr_array  = list(np.reshape(scr_array,  -1))

        mask_array_lst.extend(mask_array)
        pred_array_lst.extend(pred_array)
        scr_array_lst.extend(scr_array)

    print("...computing the pixel-wise scores/metrics here...")
    print('the pixel used is ', len(mask_array_lst))
    scr_auc = roc_auc_score(mask_array_lst, scr_array_lst)
    print(f"the scr_auc is: {scr_auc:.3f}.")
    F1 = metrics.f1_score(mask_array_lst, pred_array_lst, average='macro')
    print(f"the macro is: {F1:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_path', type=str, 
                        default='/user/guoxia11/cvlshare/cvl-guoxia11/CVPR2023_camera_ready_v5/mask_result_0.0003')
    args = parser.parse_args()
    main(args)