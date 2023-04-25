# HiFi_IFDL

This is the source code for image editing detection and localization, as well as the diffusion model attribution. Our work has been accepted by CVPR $2023$, titled as "*Hierarchical Fine-Grained Image Forgery Detection and Localization*". 

<p align="center">
  <img src="https://github.com/CHELSEA234/HiFi_IFDL/blob/main/figures/overview_4.png" alt="drawing" width="1000"/>
</p>

### Updates.
- Per many email requests, we release our first version source code for the reference. 
- **Apologize that this is not the finalized version code, which will be released soon along with the dataset**.

### Quick Start.
The quick view on the code structure:
```bash
./HiFi_IFDL
    ├── train.py
    ├── train.sh (call train.py to run the training)
    ├── model (model module folder)
    ├── utils (utils, dataloader, and localization loss class.)
    └── center (The pre-computed `.pth` file to represent the center used in the localization loss.)
```
<p align="center">
  <img src="https://github.com/CHELSEA234/HiFi_IFDL/blob/main/figures/architecture.png" alt="drawing" width="1000"/>
</p>

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
