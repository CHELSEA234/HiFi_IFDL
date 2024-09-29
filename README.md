# HiFi_IFDL

This is the source code for our CVPR $2023$: "*Hierarchical Fine-Grained Image Forgery Detection and Localization*." [[Arxiv]](https://arxiv.org/pdf/2303.17111.pdf)

Authors: [Xiao Guo](https://scholar.google.com/citations?user=Gkc-lAEAAAAJ&hl=en), [Xiaohong Liu](https://jhc.sjtu.edu.cn/~xiaohongliu/), [Zhiyuan Ren](https://scholar.google.com/citations?user=Z1ltuXEAAAAJ&hl=en), [Steven Grosz](https://scholar.google.com/citations?user=I1wOjTYUyYAC&hl=en), [Iacopo Masi](https://iacopomasi.github.io/), [Xiaoming Liu](http://cvlab.cse.msu.edu/)

<p align="center">
  <img src="https://github.com/CHELSEA234/HiFi_IFDL/blob/main/figures/overview_4.png" alt="drawing" width="1000"/>
</p>

### <a name="update"></a> Updates.
- [Sep 2024] 👏 The International Journal of Computer Vision (**IJCV**) has accepted the extended version of HiFi-Net, stay tuned~
- [Aug 2024] The HiFi-Net is integrated into the DeepFake-o-meter v2.0 platform, which is a user-friendly public detection tool designed by the **University at Buffalo**. [[DeepFake-o-meter v2.0]](https://zinc.cse.buffalo.edu/ubmdfl/deep-o-meter/home_login) [[ArXiv]](https://arxiv.org/pdf/2404.13146)
- [Jul. 2024] 👏 **ECCV2024** "Deepfake Explainer" paper [[ArXiv]](https://arxiv.org/pdf/2402.00126) reports HiFi-Net's deep fake detection performance and the source code is released [[link]](https://github.com/CHELSEA234/HiFi_IFDL/edit/main/applications/deepfake_detection).
- [Sep 2023] The first version dataset can be acquired via this link: [Dataset Link](https://drive.google.com/drive/folders/1fwBEmW30-e0ECpCNNG3nRU6I9OqJfMAn?usp=sharing)
- [Sep 2023] The DAPRA-sponsored image forensic demo can be viewed at this link: [Demo](https://drive.google.com/file/d/1q5ruko3bS4g-fuvq28C6SfzeSUrtLES6/view?usp=sharing)
- [June 2023] The extended version of our work has been submitted to one of the ~~Machine Learning Journals~~ IJCV.
- **This GitHub will keep updated, please stay tuned~**

### Short 5 Min Video 
[![Please Click the Figure](https://github.com/CHELSEA234/HiFi_IFDL/blob/main/figures/architecture.png)](https://www.youtube.com/watch?v=FwS3X5xcj8A&list=LL&index=5)

### Usage on Manipulation Localization (_e.g._, Columbia, Coverage, CASIA, NIST16 and IMD2020)
- To create your environment by
  ```
  conda env create -f environment.yml
  ```
  or mannually install `pytorch 1.11.0` and `torchvision 0.12.0` in `python 3.7.16`.
- Go to [localization_weights_link](https://drive.google.com/drive/folders/1cxCoE2hjcDj4lLrJmGEbskzPRJfoDIMJ?usp=sharing) to download the weights from, and then put them in `weights`.
- To apply the pre-trained model on images in the `./data_dir` and then obtain results in `./viz_eval`, please run
  ```
  bash HiFi_Net_loc.sh
  ```
- More quantitative and qualitative results can be found at: [csv](https://drive.google.com/drive/folders/12iS0ILb6ndXtdWjonByrgnejzuAvwCqp?usp=sharing) and [qualitative results](https://drive.google.com/drive/folders/1iZp6ciOHSbGq4EsC_AYl7zVK24gBtrd1?usp=sharing).
- If you would like to generate the above result. Download $5$ datasets via [link](https://drive.google.com/file/d/1RYXTg0Q82KEvkeOtaaR5AZ0FBx5219SY/view?usp=sharing) and unzip it by `tar -xvf data.tar.gz`. Then, uncomment this [line](https://github.com/CHELSEA234/HiFi_IFDL/blob/main/utils/load_edata.py#L21) and run `HiFi_Net_loc.sh`. 

### Usage on Detecting and Localization for the general forged content including GAN and diffusion-generated images:
- This reproduces detection and localization results in the HiFi-IFDL dataset (Tab. 2 and Supplementary Fig.1)
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
A quick view of the code structure:
```bash
./HiFi_IFDL
    ├── HiFi_Net_loc.py (localization files)
    ├── HiFi_Net_loc.sh (localization evaluation)
    ├── HiFi_Net.py (API for the user input image.)
    ├── IMD_dataloader.py (call dataloaders in the utils folder)
    ├── model (model module folder)
    │      ├── NLCDetection_pconv.py (partial convolution, localization, and classification modules)
    │      ├── seg_hrnet.py (feature extractor based on HRNet)
    │      ├── LaPlacianMs.py (laplacian filter on the feature map)
    │      ├── GaussianSmoothing.py (self-made smoothing functions)
    │      └── ...   
    ├── utils (utils, dataloader, and localization loss class.)
    │      ├── custom_loss.py (localization loss class and the real pixel center initialization)
    │      ├── utils.py
    │      ├── load_data.py (loading training and val dataset.)
    │      └── load_edata.py (loading inference dataset.)
    ├── asset (folder contains sample images with their ground truth and predictions.)
    ├── weights (put the pre-trained weights in.)
    ├── center (The pre-computed `.pth` file for the HiFi-IFDL dataset.)
    └── center_loc (The pre-computed `.pth` file for the localization task (Tab.3 in the paper).)
```

### Question and Answers.
Q1. Why train and val datasets are in the same path? 

A1. For each forgery method, we save both train and val in the SAME folder, from which we use a text file to obtain the training and val images. The text file contains a list of image names, and the first `val_num` are used for training and the last "val_num" for validation. Specifically, refer to [code](https://github.com/CHELSEA234/HiFi_IFDL/blob/main/utils/load_data.py#L271) for details. What is more, we build up the code on the top of the PSCC-Net, which adapts the same style of loading data, please compare [code1](https://github.com/proteus1991/PSCC-Net/blob/main/utils/load_tdata.py#L88) with [code2](https://github.com/proteus1991/PSCC-Net/blob/main/utils/load_tdata.py#L290).

Q2. What is the dataset naming for STGAN and the face-shifter section?

A2. Please check the STGAN.txt in this [link](https://drive.google.com/drive/folders/1OIUv7OGxfAyerMnmKvrNnN_5CmIDcNxo?usp=sharing), which contains all manipulated/modified images we have used for training and validation. This txt file will be loaded by this line of [code](https://github.com/CHELSEA234/HiFi_IFDL/blob/main/utils/load_data.py#L163), which says about the corresponding masks. Lastly, I am not sure if I have release the authentic images, if I do not, you can simply find them in the public celebAHQ dataset. I will try to offer the rigid naming for the dataset in the near future. 

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
