# ------------------------------------------------------------------------------
# Author: Xiao Guo (guoxia11@msu.edu)
# CVPR2023: Hierarchical Fine-Grained Image Forgery Detection and Localization
# ------------------------------------------------------------------------------
from os.path import isfile, join
from PIL import Image
from torchvision import transforms
import numpy as np
import abc
import cv2
import torch.utils.data as data
import torch.nn.functional as F
import random
random.seed(1234567890)
from random import randrange
import torch.nn as nn
import torch
import imageio
import time
import math
import torch

class BaseData(data.Dataset):
    '''
        The dataset used for the IFDL dataset.
    '''
    def __init__(self, args):
        super(BaseData, self).__init__()
        self.crop_size = args.crop_size
        self.file_path = '/user/guoxia11/cvlshare/cvl-guoxia11/IMDL/REAL'
        self.file_path_fake = '/user/guoxia11/cvlshare/cvl-guoxia11/IMDL/FAKE'     

        # Real and Fake images.
        self.image_names = []
        self.image_class = self._img_list_retrieve()
        for idx, _ in enumerate(self.image_class):
            self.image_names += _

    def __getitem__(self, index):
        res = self.get_item(index)
        return res

    def __len__(self):
        return len(self.image_names)

    @abc.abstractmethod
    def _img_list_retrieve():
        pass

    def _resize_func(self, input_img):
        '''resize the input image into the crop size.'''
        input_img = Image.fromarray(input_img)
        input_img = input_img.resize(self.crop_size, resample=Image.BICUBIC)
        input_img = np.asarray(input_img)
        return input_img

    def get_image(self, image_name, aug_index=None):
        '''transform the image.'''
        image = imageio.imread(image_name)
        if image.shape[-1] == 4:
            image = self.rgba2rgb(image)
        image = self._resize_func(image)
        image = image.astype(np.float32) / 255.
        image = torch.from_numpy(image)
        return image.permute(2, 0, 1)

    def rgba2rgb(self, rgba, background=(255, 255, 255)):
        '''
            turn rgba to rgb.
        '''
        row, col, ch = rgba.shape
        rgb = np.zeros((row, col, 3), dtype='float32')
        r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
        a = np.asarray(a, dtype='float32') / 255.0

        R, G, B = background
        rgb[:, :, 0] = r * a + (1.0 - a) * R
        rgb[:, :, 1] = g * a + (1.0 - a) * G
        rgb[:, :, 2] = b * a + (1.0 - a) * B

        return np.asarray(rgb, dtype='uint8')

    def generate_4masks(self, mask):
        '''generate 4 masks at different scale.'''
        crop_height, crop_width = self.crop_size
        ma_height, ma_width = mask.shape[:2]
        mask_pil = Image.fromarray(mask)

        if ma_height != crop_height or ma_width != crop_width:
            mask_pil = mask_pil.resize(self.crop_size, resample=Image.BICUBIC)

        (width2, height2) = (mask_pil.width // 2, mask_pil.height // 2)
        (width3, height3) = (mask_pil.width // 4, mask_pil.height // 4)
        (width4, height4) = (mask_pil.width // 8, mask_pil.height // 8)

        mask2 = mask_pil.resize((width2, height2))
        mask3 = mask_pil.resize((width3, height3))
        mask4 = mask_pil.resize((width4, height4))

        mask = np.asarray(mask_pil)
        mask = mask.astype(np.float32) / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        mask2 = np.asarray(mask2).astype(np.float32) / 255
        mask2[mask2 > 0.5] = 1
        mask2[mask2 <= 0.5] = 0

        mask3 = np.asarray(mask3).astype(np.float32) / 255
        mask3[mask3 > 0.5] = 1
        mask3[mask3 <= 0.5] = 0

        mask4 = np.asarray(mask4).astype(np.float32) / 255
        mask4[mask4 > 0.5] = 1
        mask4[mask4 <= 0.5] = 0

        mask = torch.from_numpy(mask)
        mask2 = torch.from_numpy(mask2)
        mask3 = torch.from_numpy(mask3)
        mask4 = torch.from_numpy(mask4)

        # print(mask.size(), mask2.size(), mask3.size(), mask4.size())

        return mask, mask2, mask3, mask4

    def get_mask(self, image_name, cls, aug_index=None):
        '''given the cls, we return the mask.'''
        # authentic
        if cls in [0,1,2,3,4]:
            mask = self.load_mask('', real=True, aug_index=aug_index)
            return_res = [0,0,0,0]
        
        # splice
        elif cls == 5:
            if '.jpg' in image_name:
                mask_name = image_name.replace('fake', 'mask').replace('.jpg', '.png')
            else:
                mask_name = image_name.replace('fake', 'mask').replace('.tif', '.png')
            mask = self.load_mask(mask_name, aug_index=aug_index)
            return_res = [1,1,1,cls - 4]
        
        # inpainting
        elif cls == 6:
            mask_name = image_name.replace('/fake/', '/mask/').replace('.jpg', '.png')
            mask = self.load_mask(mask_name, aug_index=aug_index)
            return_res = [1,1,1,cls - 4]

        # copy-move
        elif cls == 7:
            mask_name = image_name.replace('.png', '_mask.png')
            mask_name = mask_name.replace('CopyMove', 'CopyMove_mask')
            mask = self.load_mask(mask_name, aug_index=aug_index)
            return_res = [1,1,1,cls - 4]

        # faceshifter
        elif cls == 8:  
            image_id  = image_name.split('/')[-1].split('.')[0]
            mask_name = image_name.replace(image_id, f'mask/{image_id}_mask')
            mask = self.load_mask(mask_name, aug_index=aug_index)
            return_res = [1,2,2,cls - 4]

        # STGAN
        elif cls == 9: 
            image_id  = image_name.split('/')[-1].split('.')[0]
            mask_name = image_name.replace('fake', 'mask').replace(image_id, f'{image_id}_label')
            mask = self.load_mask(mask_name, aug_index=aug_index)
            return_res = [1,2,2,cls - 4]

        ## they are star2, hisd, stylegan2, stylegan3, ddpm, ddim, latent, guided
        elif cls in [10,11,12,13,14,15,16,17]:  
            mask = self.load_mask('', real=False, full_syn=True, aug_index=aug_index)
            if cls in [10,11]:
                return_res = [2,3,3,cls-4]
            elif cls in [12,13]:
                return_res = [2,3,4,cls-4]
            elif cls in [14,15]:
                return_res = [2,4,5,cls-4]
            elif cls in [16,17]:
                return_res = [2,4,6,cls-4]
        else:
            print(cls, index)
            raise Exception('class is not defined!')

        return mask, return_res

    def load_mask(self, mask_name, real=False, full_syn=False, gray=True, aug_index=None):
        '''binarize the mask, given the mask_name.'''
        if real:
            mask = np.zeros(self.crop_size)
        else:
            if not full_syn:
                mask = imageio.imread(mask_name) if not gray else np.asarray(Image.open(mask_name).convert('RGB').convert('L'))
                mask = mask.astype(np.float32)
            else:
                mask = np.ones(self.crop_size)

        mask = self.generate_4masks(mask)
        return mask

    def get_cls(self, image_name):
        '''return the forgery/authentic cls given the image_name.'''
        if '/authentic/' in image_name:
            return_cls = 0
        elif '/REAL/LSUN/' in image_name:
            return_cls = 0
        elif '/afhq_v2/' in image_name:
            return_cls = 1
        elif '/CelebAHQ/' in image_name:
            return_cls = 2
        elif '/FFHQ/' in image_name:
            return_cls = 3
        elif '/Youtube' in image_name:
            return_cls = 4
        elif '/splice' in image_name:
            return_cls = 5
        elif '/Inpainting' in image_name:
            return_cls = 6
        elif '/CopyMove' in image_name:
            return_cls = 7
        elif '/FaShifter' in image_name:
            return_cls = 8
        elif '/STGAN' in image_name:
            return_cls = 9
        elif '/Star2' in image_name:
            return_cls = 10
        elif '/HiSD' in image_name:
            return_cls = 11
        elif '/STYL2' in image_name:
            return_cls = 12
        elif '/STYL3' in image_name:
            return_cls = 13
        elif '/DDPM_' in image_name:
            return_cls = 14
        elif '/DDIM_' in image_name:
            return_cls = 15
        elif '/D_latent' in image_name:
            return_cls = 16
        elif '/GLIDE/' in image_name:
            return_cls = 17
        else:
            print(image_name)
            raise ValueError 
        return return_cls

class TrainData(BaseData):
    '''
        The dataset used for the IFDL dataset.
    '''
    def __init__(self, args):
        self.is_train = True
        self.val_num  = 90000
        super(TrainData, self).__init__(args)

    def img_retrieve(self, file_text, file_folder, real=True):
        '''
            Parameters:
                file_text: str, text file for images.
                file_folder: str, images folder.
            Returns:
                the image list.
        '''
        result_list = []
        val_num   = self.val_num * 3 if file_text in ["Youtube", "FaShifter"] else self.val_num
        data_path = self.file_path if real else self.file_path_fake

        data_text = join(data_path, file_text)
        data_path = join(data_path, file_folder)

        file_handler = open(data_text)
        contents = file_handler.readlines()
        if self.is_train:
            contents_lst = contents[:val_num]
        else:
            contents_lst = contents[val_num:]

        for content in contents_lst:
            if '.npy' not in content and 'mask' not in content:
                img_name = content.strip()
                img_name = join(data_path, img_name)
                result_list.append(img_name)
        file_handler.close()

        ## only truncate the val_num images. 
        if len(result_list) < val_num:
            mul_factor  = (val_num//len(result_list)) + 2
            result_list = result_list * mul_factor
        result_list = result_list[-val_num:]
        return result_list

    def get_item(self, index):
        '''
            given the index, this function returns the image with the forgery mask
            this function calls get_image, get_mask for the image and mask torch tensor.
        '''
        image_name = self.image_names[index]
        cls = self.get_cls(image_name)

        # image and mask
        aug_index = randrange(0, 8)
        image = self.get_image(image_name, aug_index)
        mask, return_res = self.get_mask(image_name, cls, aug_index)

        return image, mask, return_res[0], return_res[1], return_res[2], return_res[3]

    def _img_list_retrieve(self):
        '''Returns image list for different authentic and forgery image.'''
        authentic_names = self.img_retrieve('authentic.txt', 'authentic')
        splice_names     = self.img_retrieve('splice_randmask.txt', 'splice_randmask/fake',False)
        inpainting_names = self.img_retrieve('Inpainting.txt', 'Inpainting/fake', False)
        copymove_names   = self.img_retrieve('copy_move.txt', 'CopyMove', False)
        STGAN_names   = self.img_retrieve('STGAN.txt', 'STGAN/fake', False)
        FaShifter_names  = self.img_retrieve('FaShifter.txt', 'FaShifter', False)
        return [authentic_names, splice_names, inpainting_names, copymove_names, STGAN_names, FaShifter_names]

class ValData(BaseData):
    '''
        The dataset used for the IFDL dataset.
    '''
    def __init__(self, args):
        self.is_train  = False
        self.val_num   = 900
        super(ValData, self).__init__(args)

    def img_retrieve(self, file_text, file_folder, real=True):
        '''
            Parameters:
                file_text: str, text file for images.
                file_folder: str, images folder.
            Returns:
                the image list.
        '''
        result_list = []
        val_num   = self.val_num * 3 if file_text in ["Youtube", "FaShifter"] else self.val_num
        data_path = self.file_path if real else self.file_path_fake

        data_text = join(data_path, file_text)
        data_path = join(data_path, file_folder)

        file_handler = open(data_text)
        contents = file_handler.readlines()
        for content in contents[-val_num:]:
            if '.npy' not in content and 'mask' not in content:
                img_name = content.strip()
                img_name = join(data_path, img_name)
                result_list.append(img_name)
        file_handler.close()

        ## only truncate the val_num images. 
        if len(result_list) < val_num:
            mul_factor  = (val_num//len(result_list)) + 2
            result_list = result_list * mul_factor
        result_list = result_list[-val_num:]
        return result_list
    
    def get_item(self, index):
        '''
            given the index, this function returns the image with the forgery mask
            this function calls get_image, get_mask for the image and mask torch tensor.
        '''
        image_name = self.image_names[index]
        cls = self.get_cls(image_name)

        # image
        image = self.get_image(image_name)
        mask, return_res = self.get_mask(image_name, cls)

        return image, mask, return_res[0], return_res[1], return_res[2], return_res[3], image_name

    def _img_list_retrieve(self):
        '''Returns image list for different authentic and forgery image.'''
        STGAN_names   = self.img_retrieve('STGAN.txt', 'STGAN/fake', False)
        FaShifter_names  = self.img_retrieve('FaShifter.txt', 'FaShifter', False)
        return [STGAN_names, FaShifter_names]
