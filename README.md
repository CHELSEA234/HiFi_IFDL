# HiFi_IFDL

This is the source code for image editing detection and localization, as well as the diffusion model attribution. Our work has been accepted by CVPR $2023$, titled as "*Hierarchical Fine-Grained Image Forgery Detection and Localization*". [[Arxiv]](https://arxiv.org/pdf/2303.17111.pdf)

Authors: [Xiao Guo](https://scholar.google.com/citations?user=Gkc-lAEAAAAJ&hl=en), [Xiaohong Liu](https://jhc.sjtu.edu.cn/~xiaohongliu/), [Iacopo Masi](https://iacopomasi.github.io/), [Xiaoming Liu](http://cvlab.cse.msu.edu/)

<p align="center">
  <img src="https://github.com/CHELSEA234/HiFi_IFDL/blob/main/figures/overview_4.png" alt="drawing" width="1000"/>
</p>

### Updates.
- Per many email requests, we release our first version source code for the reference. 
- ~~Apologize that this is not the finalized version code, which will be released soon along with the dataset~~
- Update the inference interface and test.py
- The first version dataset can be acquired via this link: [Dataset Link](https://drive.google.com/drive/folders/1fwBEmW30-e0ECpCNNG3nRU6I9OqJfMAn?usp=sharing)
- **the DAPRA sponsored image forensic demo will be released soon**
- **this github will keep updated, please stay tuned~**

### Short 5 Min Video 
[![Please Click the Figure](https://github.com/CHELSEA234/HiFi_IFDL/blob/main/figures/architecture.png)](https://www.youtube.com/watch?v=FwS3X5xcj8A&list=LL&index=5)

### Quick Usage:
- To create your own environment by:
  ```
  conda env create -f environment.yml
  ```
- Go to this [link](https://drive.google.com/drive/folders/1v07aJ2hKmSmboceVwOhPvjebFMJFHyhm?usp=sharing) to download the weights and put them in `weights`. 
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
    ├── train.sh (call train.py to run the training)
    ├── test.py
    ├── test.sh (call test.py to produce numerical results on the testing dataset)
    ├── test.py
    ├── HiFi_Net.py (API for the user input image.)
    ├── IMD_dataloader.py (call train and val dataloaders.)
    ├── model (model module folder)
    │      ├── NLCDetection_pconv.py (partial convolution, localization and classification modules)
    │      ├── seg_hrnet.py (feature extrator based on HRNet)
    │      ├── LaPlacianMs.py (laplacian filter on the feature map)
    │      ├── GaussianSmoothing.py (self-made smoothing functions)
    │      └── ...   
    ├── utils (utils, dataloader, and localization loss class.)
    │      ├── custom_loss.py (localization loss class and the real pixel center initialization)
    │      ├── utils.py
    │      ├── load_tdata.py (loading dataset/ will be improved :) )
    │      ├── load_vdata.py (loading val/testing dataset/ will be improved :) )
    │      ├── load_vdata.py (loading val/testing dataset/ will be improved :) )
    │      └── result_parse_localization.py (produce the numerical localization results.)
    │      └── result_parse_detection.py (produce the numerical detection results.)
    ├── asset (folder contains sample images with their ground truth and predictions.)
    ├── weights (put the pre-trained weights in.)
    └── center (The pre-computed `.pth` file to represent the center used in the localization loss.)
```

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
