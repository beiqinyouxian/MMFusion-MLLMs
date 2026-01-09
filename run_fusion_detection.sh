#!/bin/bash

# sleep 21400
# export CUDA_VISIBLE_DEVICES=3,2,1,0
# 
export CUDA_VISIBLE_DEVICES=0,1,2,3


DATASET='test'
SPLITBY="car,person,bike,curve,car_stop,guardrail" #MSRS,MFNet
# SPLITBY="road,sidewalk,structure_and_power_line_tower,traffic_light,blue_road_tracffic_sign,tree,sky,person,car,truck,bus,Motorcycle"
EVAL_TYPE="Fusion"  # Orin 和 Fusion 和 Enhance(Orin+Fusion)
ATTENTION_TYPE="rel" # orin rel
SCENE="ALL" #DAY或者NIGHT或者ALL
# MODEL_PATH="/data/VLM/llava-v1.5-7b-hf"
# MODEL_PATH="/data/VLM/llava-1.5-13b-hf"
# MODEL_PATH="/data/VLM/Qwen2.5-VL-3B-Instruct"
# MODEL_PATH="/data/VLM/Qwen2.5-VL-7B-Instruct"

# MODEL_PATH='/data/VLM/Qwen2-VL-2B'
# MODEL_PATH='/data/VLM/Qwen2-VL-2B-Instruct'

# MODEL_PATH='/data/VLM/Qwen2-VL-7B'
# MODEL_PATH='/data/VLM/Qwen2-VL-7B-Instruct'
# MODEL_PATH="/data/VLM/deepseek-vl2-tiny"
# MODEL_PATH="/data/VLM/deepseek-vl2-small"
   



# DATA_ROOT="/data/dataset/MMFusion-dataset/MSRS/"
# DATA_ROOT="/data/dataset/MMFusion-dataset/DN-FMB_dataset/"
# DATA_ROOT="/data/dataset/MMFusion-dataset/MFNet/"


THRESHOLD=0 
echo "=========================================="
echo "运行检测"
echo "=========================================="

for HEAD in {2..2}; do
    echo "Running with NUM of HEAD=$HEAD"
    python Enhance_Fusion_detection.py --threshold $THRESHOLD --num $HEAD --dataset $DATASET --objects $SPLITBY --scene $SCENE --eval_type $EVAL_TYPE --attention_type $ATTENTION_TYPE --model_path $MODEL_PATH --data_root $DATA_ROOT 2>&1
done