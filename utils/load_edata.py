from PIL import Image
from torchvision import transforms
from os.path import join
import abc
import numpy as np
import torch
import torch.utils.data as data
import imageio
import os

class BaseData(data.Dataset):
    '''
        The dataset used for the IFDL dataset.
    '''
    def __init__(self, args):
        super(BaseData, self).__init__()
        self.crop_size = args.crop_size
        ## demo dataset:
        self.mani_data_dir = './data_dir'
        ## the full dataset:
        # self.mani_data_dir = './data'
        self.image_names = []
        self.image_class = []
        self.mask_names  = []

    def __getitem__(self, index):
        res = self.get_item(index)
        return res

    def __len__(self):
        return len(self.image_names)

    def generate_mask(self, mask):
        '''
            generate the corresponding binary mask.
        '''
        mask = mask.astype(np.float32) / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)
        return mask

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
        return np.asarray(rgb, dtype='uint8') # the output value is uint8 that belongs to [0,255]

    def get_image(self, image_name):
        '''
            return the image with the tensor.
        '''
        image = imageio.imread(image_name)
        if len(image.shape) == 2:
            image = imageio.imread(image_name, as_gray=False, pilmode="RGB")
        if image.shape[-1] == 4:
            image = self.rgba2rgb(image)
        image = torch.from_numpy(image.astype(np.float32) / 255)
        return image.permute(2, 0, 1)

    def get_mask(self, mask_name):
        '''
            return the binary mask.
        '''   
        mask = Image.open(mask_name).convert('L')
        mask = mask.resize(self.crop_size, resample=Image.BICUBIC)
        mask = np.asarray(mask)
        mask = self.generate_mask(mask) 
        return mask

    @abc.abstractmethod
    def get_item(self, index):
        '''
            blur
            image = Image.fromarray(image)
            image = image.filter(ImageFilter.GaussianBlur(radius=7))
            image = np.asarray(image)

            resize
            image = Image.fromarray(image)
            image = image.resize((int(image.width*0.25), int(image.height*0.25)), resample=Image.BILINEAR)
            image = np.asarray(image)

            noise
            import skimage
            image = skimage.util.random_noise(image/255., mode='gaussian', mean=0, var=15/255) * 255

            jpeg compression
            im = Image.open(image_name)
            temp_name = './temp/' + image_name.split('/')[-1][:-3] + 'jpg'
            im.save(temp_name, 'JPEG', quality=50)
            image = Image.open(temp_name)
            image = np.asarray(image)
        '''
        pass

class ValColumbia(BaseData):
    def __init__(self, args):
        super(ValColumbia, self).__init__(args)
        ddir = os.path.join(self.mani_data_dir, 'columbia')
        with open(join(ddir, 'vallist.txt')) as f:
            contents = f.readlines()
            for content in contents:
                _ = os.path.join(ddir, '4cam_splc', content.strip())
                self.image_names.append(_)
        self.image_class = [1] * len(self.image_names)

    def get_item(self, index):
        image_name = self.image_names[index]
        cls = self.image_class[index]

        # image
        image = self.get_image(image_name)

        # mask
        if '4cam_splc' in image_name:
            mask_name = image_name.replace('4cam_splc', 'mask').replace('.tif', '.jpg')
            mask = self.get_mask(mask_name)
        else:
            mask = np.zeros((1, 256, 256), dtype='float32')

        return image, mask, cls, image_name

class ValCoverage(BaseData):
    def __init__(self, args):
        super(ValCoverage, self).__init__(args)
        ddir = os.path.join(self.mani_data_dir, 'Coverage')
        with open(join(ddir, 'fake.txt')) as f:
            contents = f.readlines()
            for content in contents:
                _ = os.path.join(ddir, 'image', content.strip())
                self.image_names.append(_)
        self.image_class = [2] * len(self.image_names)

    def get_item(self, index):
        image_name = self.image_names[index]
        cls = self.image_class[index]

        ## read image.
        image = self.get_image(image_name)

        # mask
        mask_name = image_name.replace('image', 'mask').replace('t.tif', 'forged.tif')
        mask = self.get_mask(mask_name)

        return image, mask, cls, image_name

class ValCasia(BaseData):
    def __init__(self, args):
        super(ValCasia, self).__init__(args)
        ddir = os.path.join(self.mani_data_dir, 'CASIA/CASIA1')
        with open(join(ddir, 'fake.txt')) as f:
            contents = f.readlines()
            for content in contents:
                tag = content.split('/')[-1].split('_')[1]
                if tag == 'D':
                    self.image_class.append(1)
                elif tag == 'S':
                    self.image_class.append(2)
                else:
                    raise Exception('unknown class: {}'.format(content))
                self.image_names.append(os.path.join(ddir, 'fake', content.strip()))

        ddir = os.path.join(self.mani_data_dir, 'CASIA/CASIA2')
        with open(join(ddir, 'fake.txt')) as f:
            contents = f.readlines()
            for content in contents:
                tag = content.split('/')[-1].split('_')[1]
                if tag == 'D':
                    self.image_class.append(1)
                elif tag == 'S':
                    self.image_class.append(2)
                else:
                    raise Exception('unknown class: {}'.format(content))
                self.image_names.append(os.path.join(ddir, 'fake', content.strip()))

    def get_item(self, index):
        image_name = self.image_names[index]
        cls = self.image_class[index]

        # image
        image = self.get_image(image_name)

        # mask
        if '.jpg' in image_name:
            mask_name = image_name.replace('fake', 'mask').replace('.jpg', '_gt.png')
        else:
            mask_name = image_name.replace('fake', 'mask').replace('.tif', '_gt.png')
        mask = self.get_mask(mask_name)

        return image, mask, cls, image_name

class ValNIST16(BaseData):
    def __init__(self, args):
        super(ValNIST16, self).__init__(args)
        ddir = os.path.join(self.mani_data_dir, 'NIST16')
        file_name = 'alllist.txt'
        with open(join(ddir, file_name)) as f:
            contents = f.readlines()
            for content in contents:
                image_name, mask_name = content.split(' ')
                self.image_names.append(join(ddir, image_name))
                self.mask_names.append(join(ddir, mask_name.strip()))

    def get_item(self, index):
        image_name = self.image_names[index]
        mask_name = self.mask_names[index]

        if 'splice' in mask_name:
            cls = 1
        elif 'manipulation' in mask_name:
            cls = 2
        elif 'remove' in mask_name:
            cls = 3
        else:
            cls = 0

        # image
        image = self.get_image(image_name)
        if image.size()[2]*image.size()[1] >= 1000*1000:
            image = imageio.imread(image_name)
            if image.shape[-1] == 4:
                image = self.rgba2rgb(image)
            image = Image.fromarray(image)
            image = image.resize((1000, 1000), resample=Image.BICUBIC)
            image = np.asarray(image)
            image = torch.from_numpy(image.astype(np.float32) / 255)
            image = image.permute(2, 0, 1)

        # mask
        mask = self.get_mask(mask_name)
        mask = torch.abs(mask - 1)

        return image, mask, cls, image_name

class ValIMD2020(BaseData):
    def __init__(self, args):
        super(ValIMD2020, self).__init__(args)
        ddir = os.path.join(self.mani_data_dir, 'IMD2020')
        file_name = 'fake.txt'
        with open(join(ddir, file_name)) as f:
            contents = f.readlines()
            for content in contents:
                image_name = content.strip()
                if '.jpg' in image_name:
                    mask_name = image_name.replace('.jpg', '_mask.png')
                else:
                    mask_name = image_name.replace('.png', '_mask.png')
                self.image_names.append(join(ddir, 'fake_img', image_name))
                self.mask_names.append(join(ddir, 'mask', mask_name))
        self.image_class = [2] * len(self.image_names)

    def get_item(self, index):
        image_name = self.image_names[index]
        mask_name = self.mask_names[index]
        cls = self.image_class[index]
        try:
            image = self.get_image(image_name)
        except:
            print(f"Fail at {image_name}.")
        mask = self.get_mask(mask_name)

        return image, mask, cls, image_name