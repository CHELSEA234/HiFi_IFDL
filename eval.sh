#!/bin/bash
source ~/.bashrc
conda activate HiFi_Net
CUDA_NUM=1
VAL_TAG=3
LR_RATE=5e-5
initial_epochs=16000

# Define an array with the initial_epoch values
CUDA_VISIBLE_DEVICES=$CUDA_NUM python eval_pretrain.py --list_cuda 0 \
                                                    --train_bs 16 \
                                                    --num_epochs 20 \
                                                    --patience 65 \
                                                    -lr $LR_RATE \
                                                    --initial_epoch $initial_epoch \
                                                    --val_tag $VAL_TAG