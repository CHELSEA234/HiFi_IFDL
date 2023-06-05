# ------------------------------------------------------------------------------
# Author: Xiao Guo (guoxia11@msu.edu)
# CVPR2023: Hierarchical Fine-Grained Image Forgery Detection and Localization
# ------------------------------------------------------------------------------
import numpy as np
import torch.utils.data as data
from os.path import isfile, join
from PIL import Image
from torchvision import transforms
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

class TrainData(data.Dataset):
    '''
        The dataset used for the IFDL dataset.
    '''
    def __init__(self, args):
        super(TrainData, self).__init__()
        path, crop_size, train_num, train_ratio, val_num = \
            args.path, args.crop_size, args.train_num, args.train_ratio, args.val_num
        self.is_train = True
        val_num = 90000
        self.val_num = val_num
        self.crop_size = crop_size
        self.file_path = '/user/guoxia11/cvlshare/cvl-guoxia11/IMDL/REAL'
        self.file_path_fake = '/user/guoxia11/cvlshare/cvl-guoxia11/IMDL/FAKE'
        self.train_num   = train_num
        self.train_ratio = train_ratio

        # Real and Fake images.
        self.image_names = []
        authentic_names, afhq_v2_names, CelebAHQ_names, FFHQ_names, Youtube_names, splice_names, \
        inpainting_names, copymove_names, FaShifter_names, STGAN_names, Star2_names, HiSD_names, \
        STYL2_names, STYL3_names, DDPM_names, DDIM_names, guided_Diffusion_names, \
        latent_Diffusion_names = self._img_list_retrieve()
        self.image_class = [authentic_names, afhq_v2_names, CelebAHQ_names, FFHQ_names, Youtube_names,
                            splice_names, inpainting_names, copymove_names,
                            FaShifter_names, STGAN_names, 
                            Star2_names, HiSD_names,
                            STYL2_names, STYL3_names,
                            DDPM_names, DDIM_names,
                            latent_Diffusion_names, guided_Diffusion_names 
                            ]
        for idx, _ in enumerate(self.image_class):
            self.image_names += _
        assert len(self.image_names) == self.train_num, (len(self.image_names), self.train_num)

    def __getitem__(self, index):
        res = self.get_item(index)
        return res

    def __len__(self):
        return len(self.image_names)

    def _resize_func(self, input_img):
        '''resize the input image into the crop size.'''
        input_img = Image.fromarray(input_img)
        input_img = input_img.resize(self.crop_size, resample=Image.BICUBIC)
        input_img = np.asarray(input_img)
        return input_img

    def _img_list_retrieve(self):
        '''Returns image list for different authentic and forgery image.'''
        authentic_names = self.img_retrieve('authentic.txt', 'authentic')
        LSUN_names      = self.img_retrieve('LSUN.txt', 'LSUN')
        authentic_names.extend(LSUN_names)
        afhq_v2_names   = self.img_retrieve('afhq_v2.txt', 'afhq_v2')
        CelebAHQ_names  = self.img_retrieve('CelebAHQ.txt', 'CelebAHQ')
        FFHQ_names      = self.img_retrieve('FFHQ.txt', 'FFHQ')
        Youtube_names   = self.img_retrieve('Youtube.txt', 'Youtube')
        
        splice_names     = self.img_retrieve('splice_randmask.txt', 'splice_randmask/fake',False)
        inpainting_names = self.img_retrieve('Inpainting.txt', 'Inpainting/fake', False)
        copymove_names   = self.img_retrieve('copy_move.txt', 'CopyMove', False)

        FaShifter_names  = self.img_retrieve('FaShifter.txt', 'FaShifter', False)
        STGAN_names   = self.img_retrieve('STGAN.txt', 'STGAN/fake', False)
        Star2_names   = self.img_retrieve('Star2.txt', 'Star2', False)
        HiSD_names    = self.img_retrieve('HiSD.txt', 'HiSD', False)        
        STYL2_names   = self.img_retrieve('STYL2.txt', 'STYL2', False)
        STYL3_names   = self.img_retrieve('STYL3.txt', 'STYL3', False)

        # DDPM
        DDPM_names = []
        with open(join(self.file_path_fake, 'DDPM_cat.txt')) as f:
            content = f.readlines()
            for content in content[:31500]:
                content = content.strip()
                img_name = join(self.file_path_fake, 'DDPM_cat', content)
                DDPM_names.append(img_name)

        with open(join(self.file_path_fake, 'DDPM_church.txt')) as f:
            content = f.readlines()
            for content in content[:31500]:
            # for content in content[:int(self.val_num/2)]:
                content = content.strip()
                img_name = join(self.file_path_fake, 'DDPM_church', content)
                DDPM_names.append(img_name)

        with open(join(self.file_path_fake, 'DDPM_bedroom.txt')) as f:
            content = f.readlines()
            for content in content[:27000]:
            # for content in content[:int(self.val_num/2)]:
                content = content.strip()
                img_name = join(self.file_path_fake, 'DDPM_bedroom', content)
                DDPM_names.append(img_name)

        # DDIM
        DDIM_names = []
        with open(join(self.file_path_fake, 'DDIM_bedroom.txt')) as f:
            content = f.readlines()
            for content in content[:int(self.val_num/2)]:
                content = content.strip()
                img_name = join(self.file_path_fake, 'DDIM_bedroom', content)
                DDIM_names.append(img_name)

        with open(join(self.file_path_fake, 'DDIM_church.txt')) as f:
            content = f.readlines()
            for content in content[:int(self.val_num/2)]:
                content = content.strip()
                img_name = join(self.file_path_fake, 'DDIM_church', content)
                DDIM_names.append(img_name)

        # guided diffusion
        guided_Diffusion_names = []
        with open(join(self.file_path_fake, 'GLIDE.txt')) as f:
            content = f.readlines()
            for content in content[:self.val_num]:
                content = content.strip()
                img_name = join(self.file_path_fake, 'GLIDE', content)
                guided_Diffusion_names.append(img_name)

        # latent diffusion
        latent_Diffusion_names = []
        with open(join(self.file_path_fake, 'D_latent_bed.txt')) as f:
            contents = f.readlines()
            for content in contents[:int(self.val_num/2)]:
                content = content.strip()
                img_name = join(self.file_path_fake, 'D_latent_bed', content)
                latent_Diffusion_names.append(img_name)
                
        with open(join(self.file_path_fake, 'D_latent_church.txt')) as f:
            contents = f.readlines()
            for content in contents[:int(self.val_num/2)]:
                content = content.strip()
                img_name = join(self.file_path_fake, 'D_latent_church', content)
                latent_Diffusion_names.append(img_name)

        return authentic_names, afhq_v2_names, CelebAHQ_names, FFHQ_names, Youtube_names, splice_names, \
                inpainting_names, copymove_names, FaShifter_names, STGAN_names, Star2_names, HiSD_names, \
                STYL2_names, STYL3_names, DDPM_names, DDIM_names, guided_Diffusion_names, latent_Diffusion_names

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
            contents_lst = contents[-val_num:]

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
        # print("image number: ", file_folder, len(result_list))
        return result_list

    def load_mask(self, mask_name, real=False, full_syn=False, gray=True, aug_index=None):
        '''binarize the mask, given the mask_name.'''
        if real:
            mask = np.zeros(self.crop_size)
        else:
            if not full_syn:
                mask = imageio.imread(mask_name) if not gray else imageio.imread(mask_name, pilmode='L')
                mask = mask.astype(np.float32) / 255.
            else:
                mask = np.ones(self.crop_size)

        crop_height, crop_width = self.crop_size
        ma_height, ma_width = mask.shape[:2]
        mask = mask.astype(np.float32)
        if ma_height != crop_height or ma_height != crop_width:
            mask = self._resize_func(mask)

        mask = mask.astype(np.float32)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        mask = torch.from_numpy(mask)
        return mask

    def get_image(self, image_name, aug_index):
        '''transform the image.'''
        image = imageio.imread(image_name)
        image = self._resize_func(image)
        image = image.astype(np.float32) / 255.
        image = torch.from_numpy(image)
        return image.permute(2, 0, 1)

    def get_mask(self, image_name, cls, aug_index):
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
            mask_name = image_name.replace('fake', 'mask').replace('.jpg', '.png')
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

        return image, [mask], return_res[0], return_res[1], return_res[2], return_res[3]

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
        elif '/Youtube/' in image_name:
            return_cls = 4
        elif '/splice_randmask/' in image_name:
            return_cls = 5
        elif '/Inpainting/' in image_name:
            return_cls = 6
        elif '/CopyMove/' in image_name:
            return_cls = 7
        elif '/FaShifter/' in image_name:
            return_cls = 8
        elif '/STGAN/' in image_name:
            return_cls = 9
        elif '/Star2/' in image_name:
            return_cls = 10
        elif '/HiSD/' in image_name:
            return_cls = 11
        elif '/STYL2/' in image_name:
            return_cls = 12
        elif '/STYL3/' in image_name:
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
