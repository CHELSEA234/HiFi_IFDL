source ~/.bashrc
conda activate HiFi_Net_clip
CUDA_NUM=2
CUDA_VISIBLE_DEVICES=$CUDA_NUM python train.py --list_cuda 0 --learning_rate 0.000005 --train_bs 8 --num_epochs 25 --patience 65 --debug_mode
