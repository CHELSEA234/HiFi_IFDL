# coding: utf-8
# author: Hierarchical Fine-Grained Image Forgery Detection and Localization, CVPR2023
# based on the sample strategy proposed in Two-branch Recurrent Network for Isolating Deepfakes in Videos, ECCV2020
import torch
import torchvision
import h5py
import os
import glob
import numpy as np
import json
import numpy as np

from torch.utils import data

# Image transformation
def get_image_transformation(use_laplacian=False, normalize=True):
    transforms = []
    if normalize:
        transforms.extend(
                        [torchvision.transforms.ToPILImage(), # Next line takes PIL images as input (ToPILImage() preserves the values in the input array or tensor)
                         torchvision.transforms.ToTensor(), # To bring the pixel values in the range [0,1]
                         torchvision.transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))]
        )

        return torchvision.transforms.Compose(transforms)
    else:
        transforms.extend(
                        [torchvision.transforms.ToPILImage(), # Next line takes PIL images as input (ToPILImage() preserves the values in the input array or tensor)
                         torchvision.transforms.ToTensor()] # To bring the pixel values in the range [0,1]
        )
        return torchvision.transforms.Compose(transforms)

# Main dataloader 
def get_dataloader(img_path,train_dataset_names,ctype,manipulations_dict,window_size=10,hop=1,use_laplacian=False,normalize=True,strat_sampling=False,balanced_minibatch=False,mode='train',bs=32,workers=4):
    """
    This is a dataloader for Face Forensics++ dataset stored in HDF5 file format.
    
    The structure of the files should be as shown below:
    filename.h5 -> keys (video names. Ex, 000_003 for manipulated and 000 for original) -> each video will further have 'n' number of
    frames. f[key][i] to acces 'ith' frame of 'key' video.
    Example of filename: FF++_Deepfakes_c40.h5, FF++_Face2Face_c23.h5, FF++_original_c0.h5, etc.
    
    Parameters
    ----------
    img_path : str
        The location of h5 files on hard drive.
    train_dataset_names : list
        The datasets that are to be loaded.
        
    returns
    -------
    out: torch.utils.data.dataloader.DataLoader
        A generator that can be used to get the required batches of sequential
        samples of data.
        
    Examples
    --------
    img_path = '/research/cvlshare/cvl-guoxia11/FaceForensics++'
    train_dataset_names = ['original', 'Deepfakes']
    ctype = 'c40'
    manipulations_dict = {0:'Deepfakes',255:'original'}
    window_size = 10
    hop = 5
    use_laplacian = True
    normalize = True
    strat_sampling = True
    mode='train'
    bs=32
    workers=0
    train_generator = get_dataloader(img_path,train_dataset_names,ctype,manipulations_dict,window_size,hop,use_laplacian,normalize,strat_sampling,mode,bs,workers)
    """
    transform = get_image_transformation(use_laplacian=False, normalize=normalize)
    params = {'batch_size': bs,
              'shuffle': (mode=='train'),
              'num_workers': workers,
              'drop_last' : (mode=='train')
            }
    if mode == 'test' or mode == 'val':
        strat_sampling = False

    datalist_dict = get_img_list(img_path, train_dataset_names, ctype, mode, window_size, hop, strat_sampling, balanced_minibatch)

    datasets = { dataset_key : ForensicFaceDatasetRNN(img_list, img_path, dataset_key, ctype,
                                                        manipulations_dict, window_size, hop=hop,
                                                        use_laplacian=use_laplacian, 
                                                        strat_sampling=strat_sampling,
                                                        transform=transform)
                 for dataset_key, img_list in datalist_dict.items()
                }
    joined_dataset = data.ConcatDataset([dataset for keys, dataset in datasets.items() ])
    joined_generator = data.DataLoader(joined_dataset,**params,pin_memory=True)
    return joined_generator, joined_dataset


# Generate a dictionary with "dataset": [dataset-video_id-frame_start]
def get_img_list(img_path, datasets, ctype, split, window_size, hop, strat_sampling, balanced_minibatch, repeat_num=6):
    # Get the video_ids based on the split
    if split == 'train':
        with open('/research/cvl-guoxia11/deepfake_AIGC/FaceForensics/dataset/splits/train.json', 'r') as f_json:
            img_folders = json.load(f_json)
    elif split == 'val':
        with open('/research/cvl-guoxia11/deepfake_AIGC/FaceForensics/dataset/splits/val.json', 'r') as f_json:
            img_folders = json.load(f_json)
    elif split == 'test':
        with open('/research/cvl-guoxia11/deepfake_AIGC/FaceForensics/dataset/splits/test.json', 'r') as f_json:
            img_folders = json.load(f_json)

    data_dict = {}
    for dataset in datasets:
        data_list = []
        data_filename = glob.glob(f'{img_path}/*{dataset}*{ctype}*.h5')[0] # Find the correct data file in the img_path
        f = h5py.File(data_filename, 'r') # Load the data file in f
        tmp_img_folders = []
        if dataset == "original":
            tmp_img_folder = [x for sublist in img_folders for x in sublist]
            if split == 'train' and strat_sampling and balanced_minibatch:
                for i in range(4*repeat_num):
                    tmp_img_folders.extend(tmp_img_folder) # Oversample by 4, then it has 2880 sequences.
            else:
                tmp_img_folders = tmp_img_folder
        else:
            _ = list(map(lambda x:["_".join([x[0],x[1]]),"_".join([x[1],x[0]])], img_folders))
            tmp_img_folder = [x for sublist in _ for x in sublist]
            if split == 'train' and strat_sampling and balanced_minibatch:
                for i in range(repeat_num):
                    tmp_img_folders.extend(tmp_img_folder) # Oversample by 4, then it has 2880 sequences.
            else:
                tmp_img_folders = tmp_img_folder

        for folder in tmp_img_folders:
            if strat_sampling:
                frame_limit = f[folder].shape[0]
                if frame_limit > window_size*hop:
                    ## we record: the dataset name, the video id (folder) and total number of frames (frame_limit)
                    data_list.append(f'{dataset}-{folder}-{frame_limit}')
            else:
                # Get the indices of the starting frame of each chunk of frames
                if f[folder].shape[0] > window_size*hop:
                    frame_start_indices = np.arange(0, f[folder].shape[0]-(window_size*hop), window_size*hop)
                for frame_index in frame_start_indices:
                    data_list.append(f'{dataset}-{folder}-{frame_index}')
        f.close()
        data_dict[dataset] = data_list
    return data_dict

class ForensicFaceDatasetRNN(data.Dataset):
    def __init__(self, list_ids, img_path, dataset_name, ctype, manipulations_dict, window_size, hop, use_laplacian=False, strat_sampling=False, transform=[]):
        super(ForensicFaceDatasetRNN, self).__init__()
        self.list_ids = list_ids
        self.transform = transform
        self.use_laplacian = use_laplacian
        self.strat_sampling = strat_sampling
        self.dataset_name = dataset_name
        self.dname_to_id = manipulations_dict
        self.window_size = window_size
        self.hop = hop
        self.h5_handler = None
        self.data_filename = self.get_dbfile_path(f'{img_path}/*{dataset_name}*{ctype}*.h5')
        if not os.path.exists(self.data_filename):
            raise RunTimeError('%s not found' % (self.data_filename))
        if self.hop < 1:
            raise ValueError(f'Minimum value of hop is 1. And you provided {self.hop}')
        
    def __len__(self):
        return len(self.list_ids)
    
    def get_dbfile_path(self,path_pattern):
        list_files = glob.glob(path_pattern)
        n_files = len(list_files)
        if n_files >=2:
            raise RuntimeError(f'Found multiple files in {path_pattern}')
        elif n_files == 0:
            raise RuntimeError(f'Files not found in {path_pattern}')
        else:
            return list_files[0]
        
    def __getitem__(self, index):
        if self.h5_handler is None:
            self.h5_handler = h5py.File(self.data_filename, 'r', swmr=True)
        file_id = self.list_ids[index].split('-')

        data_folder = file_id[1]
        if self.strat_sampling:
            frame_limit = file_id[2]
            ## now we random sample a frame within the video
            frame_id = np.random.randint(0,int(frame_limit)-(self.window_size*self.hop))
        else:
            frame_id = file_id[2]

        frames = self.h5_handler[data_folder][int(frame_id):int(frame_id)+(self.window_size*self.hop):self.hop]

        ## Now handling the label
        label = 1.0 if self.dataset_name == "original" else 0.0
            
        '''
            ## visualization example:
            import cv2
            print(f"the frames are: ", frames.shape)
            # output_frames = self.transform(frames)
            for _ in range(10):
                frame = frames[_]
                # print(f"the frame is: ", frame.shape)
                # print("output frames: ", frame.shape)
                image_data = frame.astype(np.uint8)
                image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                # cv2.imshow('demo.png', image_data)
                cv2.imwrite(f'demo_{_}_{self.dataset_name}.png', image_data)
        '''
        frames = torch.stack(list(map(self.transform,frames)))
        image_names = '~'.join([f"{data_folder}/{int(frame_id) + i * self.hop}" for i in range(self.window_size)])

        return frames, label, image_names