# train
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file acc_configs/gpu4.yaml train.py  big --workspace workspace_train --resume ./path_to_your_stage_1_models


# test
CUDA_VISIBLE_DEVICES=0 python test.py big --resume ./path_to_your_stage_2_models --workspace workspace_test

