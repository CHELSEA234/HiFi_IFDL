source ~/.bashrc
conda activate HiFi_Net_deepfake
CUDA_NUM=0,1,3,4,5,6
CUDA_VISIBLE_DEVICES=$CUDA_NUM python train.py \
                                --dataset_name FF++ \
                                --batch_size 32 \
                                --window_size 10 \
                                --gpus 6 \
                                --valid_epoch 1 \
                                --feat_dim 1000 \
                                --learning_rate 1e-4 \
                                --display_step 150
