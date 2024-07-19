source ~/.bashrc
conda activate HiFi_Net_deepfake
CUDA_NUM=2
CUDA_VISIBLE_DEVICES=$CUDA_NUM python HiFiNet_deepfake.py