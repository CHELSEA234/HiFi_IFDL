source ~/.bashrc
conda activate IFDL
CUDA_NUM=3
# CUDA_VISIBLE_DEVICES=$CUDA_NUM python train.py --list_cuda 0 --train_bs 16 --num_epochs 20 --patience 65 --debug_mode
CUDA_VISIBLE_DEVICES=$CUDA_NUM python train.py --list_cuda 0 --learning_rate 0.001 --train_bs 30 --num_epochs 20 --patience 65