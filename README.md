# HiFi_IFDL

This is the source code for image editing detection and localization. Our work has been accepted by CVPR $2023$, titled as "*Hierarchical Fine-Grained Image Forgery Detection and Localization*". [[Arxiv]](https://arxiv.org/pdf/2303.17111.pdf)

Authors: [Xiao Guo](https://scholar.google.com/citations?user=Gkc-lAEAAAAJ&hl=en), [Xiaohong Liu](https://jhc.sjtu.edu.cn/~xiaohongliu/), [Iacopo Masi](https://iacopomasi.github.io/), [Xiaoming Liu](http://cvlab.cse.msu.edu/)

<p align="center">
  <img src="https://github.com/CHELSEA234/HiFi_IFDL/blob/main/figures/overview_4.png" alt="drawing" width="1000"/>
</p>

### <a name="update"></a> Updates.
- The first version dataset can be acquired via this link: [Dataset Link](https://drive.google.com/drive/folders/1fwBEmW30-e0ECpCNNG3nRU6I9OqJfMAn?usp=sharing)
- The DAPRA sponsored image forensic demo can be viewed at this link: [Demo](https://drive.google.com/file/d/1q5ruko3bS4g-fuvq28C6SfzeSUrtLES6/view?usp=sharing)
- We release new pre-trained weights for the localization task, please refer the following contents.
- The extended version of our work has been submitted to one of Machine Learning Journals.
- **this github will keep updated, please stay tuned~**

### Short 5 Min Video 
[![Please Click the Figure](https://github.com/CHELSEA234/HiFi_IFDL/blob/main/figures/architecture.png)](https://www.youtube.com/watch?v=FwS3X5xcj8A&list=LL&index=5)

### Quick Usage on Manipulation Localization:
- To create your own environment by:
  ```
  conda env create -f environment.yml
  ```
- Go to [localization_weights_link](https://drive.google.com/drive/folders/1cxCoE2hjcDj4lLrJmGEbskzPRJfoDIMJ?usp=sharing) to download the weights from, and then put them in `weights`.
- To apply the pre-trained model on images in the `./data_dir` and then obtain results in `./viz_eval`, please run
  ```
  bash HiFi_Net_loc.sh
  ```
- More quantitative and qualitative results can be found at: [csv](https://drive.google.com/drive/folders/12iS0ILb6ndXtdWjonByrgnejzuAvwCqp?usp=sharing) and [qualitative results](https://drive.google.com/drive/folders/1iZp6ciOHSbGq4EsC_AYl7zVK24gBtrd1?usp=sharing).

### Quick Usage on the IFDL dataset:
- Go to [HiFi_IFDL_weights_link](https://drive.google.com/drive/folders/1v07aJ2hKmSmboceVwOhPvjebFMJFHyhm?usp=sharing) to download the weights, and then put them in `weights`. 
- The quick usage on HiFi_Net:
```python
  from HiFi_Net import HiFi_Net 
  from PIL import Image
  import numpy as np

  HiFi = HiFi_Net()   # initialize
  img_path = 'asset/sample_1.jpg'

  ## detection
  res3, prob3 = HiFi.detect(img_path)
  # print(res3, prob3) 1 1.0
  HiFi.detect(img_path, verbose=True)

  ## localization
  binary_mask = HiFi.localize(img_path)
  binary_mask = Image.fromarray((binary_mask*255.).astype(np.uint8))
  binary_mask.save('pred_mask.png')
```

### Quick Start of Source Code
The quick view on the code structure:
```bash
./HiFi_IFDL
    ├── train.py
    ├── train.sh 
    ├── HiFi_Net_loc.py (localization files)
    ├── HiFi_Net_loc.sh (localization evaluation)
    ├── HiFi_Net.py (API for the user input image.)
    ├── IMD_dataloader.py (call dataloaders in the utils folder)
    ├── model (model module folder)
    │      ├── NLCDetection_pconv.py (partial convolution, localization and classification modules)
    │      ├── seg_hrnet.py (feature extrator based on HRNet)
    │      ├── LaPlacianMs.py (laplacian filter on the feature map)
    │      ├── GaussianSmoothing.py (self-made smoothing functions)
    │      └── ...   
    ├── utils (utils, dataloader, and localization loss class.)
    │      ├── custom_loss.py (localization loss class and the real pixel center initialization)
    │      ├── utils.py
    │      ├── load_data.py (loading traininga and val dataset.)
    │      └── load_edata.py (loading inference dataset.)
    ├── asset (folder contains sample images with their ground truth and predictions.)
    ├── weights (put the pre-trained weights in.)
    ├── center (The pre-computed `.pth` file for the HiFi-IFDL dataset.)
    └── center_loc (The pre-computed `.pth` file for the localization task (Tab.3 in the paper).)
```

### Question and Answers.
Q1. Why train and val dataset are in the same path? 

A1. For each forgery method, we save both train and val in the SAME folder, from which we use a text file to obtain the training and val images. The text file contains list of image names, and first `val_num` are used for training and last "val_num" for validation. Specifically refer to [code](https://github.com/CHELSEA234/HiFi_IFDL/blob/main/utils/load_data.py#L271) for details. What is more, we build up the code on the top of the PSCC-Net, which adapts the same style of loading data, please compare [code1](https://github.com/proteus1991/PSCC-Net/blob/main/utils/load_tdata.py#L88) with [code2](https://github.com/proteus1991/PSCC-Net/blob/main/utils/load_tdata.py#L290).

Q2. What is the dataset naming for STGAN and faceshifter section?

A2. Please check the STGAN.txt in this [link](https://drive.google.com/drive/folders/1OIUv7OGxfAyerMnmKvrNnN_5CmIDcNxo?usp=sharing), which contains all manipulated/modified images we have used for training and validation. This txt file will be loaded by this line of [code](https://github.com/CHELSEA234/HiFi_IFDL/blob/main/utils/load_data.py#L163), which says about the corresponding masks. Lastly, I am not sure if I have release the authentic images, if I do not, you can simply find them in the public celebAHQ dataset. I will try to offer the rigid naming for the dataset in the near future. 

Q3. The training script does not make the loss down.

A3. As indicated in [Updates Section](#update), before October, we only release the code for the reference. Now, we have updated one of training codes we have used as one **example** which however may not have all components proposed and reflect a specific combination of hyper-parameters presented in the paper. If you want to learn more on the training, please take a look at this [screenshot](https://github.com/CHELSEA234/HiFi_IFDL/blob/main/figures/tb_viz.png), which I collect from the previous experiment. This figure demonstrates the loss decreases after the careful choosing a set of hyper-parameters, and it can take much time for map_loss to get converged. 

### Reference
If you would like to use our work, please cite:
```Bibtex
@inproceedings{hifi_net_xiaoguo,
  author = { Xiao Guo and Xiaohong Liu and Zhiyuan Ren and Steven Grosz and Iacopo Masi and Xiaoming Liu },
  title = { Hierarchical Fine-Grained Image Forgery Detection and Localization },
  booktitle = { CVPR },
  year = { 2023 },
}
```
