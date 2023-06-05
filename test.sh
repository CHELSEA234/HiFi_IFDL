source ~/.bashrc
conda activate IFDL
CUDA_NUM=5
CUDA_VISIBLE_DEVICES=$CUDA_NUM python test.py --list_cuda 0 --train_bs 16 --num_epochs 20 --patience 65