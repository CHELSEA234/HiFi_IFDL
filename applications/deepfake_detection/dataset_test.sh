source ~/.bashrc
conda activate HiFi_Net_deepfake
CUDA_NUM=7
CUDA_VISIBLE_DEVICES=$CUDA_NUM python dataset_test.py \
                                    --dataset_name FF++ \
                                    --batch_size 32 \
                                    --window_size 10 \
                                    --gpus 1 \
                                    --valid_epoch 1 \
                                    --feat_dim 1000 \
                                    --learning_rate 1e-4 \
                                    --display_step 100