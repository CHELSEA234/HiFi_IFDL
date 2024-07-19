# HiFi_Deepfake

We apply the HiFi_Net for the deepfake detection as the following diagram:

<p align="center">
  <img src="https://github.com/CHELSEA234/HiFi_IFDL/blob/main/figures/HiFi_deepfake.png" alt="drawing" width="1000"/>
</p>

### Reported Performance
<center>
  
| Dataset | AUC | Accuracy | EER | TPR@FPR=**$10$**% |TPR@FPR=**$1$**% | 
|:----:|:----:|:----:|:----:|:----:|:----:|
|FF++(c40)|$92.10$|$89.16$|N/A|$74.44$|$40.85$
|CelebDF|$68.80$|$67.20$|$36.13$|N/A|N/A
|WildDeepfake|$65.22$|$66.29$|$38.65$|N/A|N/A
  
</center>

More results please refer to the table $3$ of our ECCV2024 paper [[ArXiv]](https://arxiv.org/pdf/2402.00126)

### The Pre-trained Weights and User-friendly Preprocessed Dataset:
1. The pre-trained weights on FF++ can be download via [[link]](https://drive.google.com/drive/folders/1AElYlVxsahgGIua3m3Kj2VhSc3S7ADLJ?usp=sharing)
2. We offer a preprocessed FF++ dataset in the HDF5 file format [[link]](https://drive.google.com/drive/folders/1ovuurFCkBfmcMq7HKO5ph36U1QyL75UA?usp=sharing), supporting faster I/O. The dataset follows the naming ```FF++_{manipulation_type}_{compression rate}.h5``` and is structured as follows:
```
FF++_Deepfakes_c23.h5:
FF++_Deepfakes_c40.h5
FF++_Face2Face_c23.h5
FF++_Face2Face_c40.h5
...
```

### Quick Start
1. Setup the environment using ```environment.yml```, then put the pre-trained weights in ```FF++``` folder.
2. Download the entire dataset or a small portion of datasets, for example ```FF++_original_c40.h5``` and ```FF++_Deepfakes_c40.h5```.
3. Run `bash test.sh` after setting up the data path [here](https://github.com/CHELSEA234/HiFi_IFDL/blob/main/applications/deepfake_detection/test.py#L106).
4. If you choose to run the small portion dataset (e.g., ```FF++_original_c40.h5``` and ```FF++_Deepfakes_c40.h5```), please comment this [link](https://github.com/CHELSEA234/HiFi_IFDL/blob/main/applications/deepfake_detection/test.py#L34)

### Quick View of Code
```bash
./deepfake_detection
    ├── test.py (the inference code.)
    ├── test.sh (run the inference code.)
    ├── dataset_test.py (dataset tutorial)
    ├── dataset_test.sh (dataset tutorial)
    ├── train.py (the train code.)
    ├── train.sh (run the train code.)
    ├── exp_FF_c40_bs_32_lr_0.0001_ws_10.txt (The training log file.)
    ├── FF++ (Please download the pre-trained weights and put it here)
    ├── sequence (model module folder)
    │      ├── rnn_stratified_dataloader.py (datalaoder)
    │      ├── runjobs_utils.py (the first utility)
    │      ├── torch_utils.py (the second utility)
    │      └── models
    │            ├── run_model.sh (model tutorial)
    │            ├── LaPlacianMs.py
    │            ├── HiFiNet_deepfake.py
    │            └── ...
    └── environment.yml
```
