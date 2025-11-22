#!/bin/bash


# export CUDA_VISIBLE_DEVICES=0,1,2,3

export CUDA_VISIBLE_DEVICES=3,2,1,0
# 创建结果目录

DATASET='test'
SPLITBY="car,person,bike,curve,car_stop,guardrail" #MSRS,MFNet
# SPLITBY="road,sidewalk,structure_and_power_line_tower,traffic_light,blue_road_tracffic_sign,tree,sky,person,car,truck,bus,Motorcycle"
EVAL_TYPE="Auto"  # 全层级全注意力头提取
ATTENTION_TYPE="rel" # orin rel

# MODEL_PATH="/data/VLM/llava-v1.5-7b-hf"
# MODEL_PATH="/data/VLM/llava-1.5-13b-hf"
# MODEL_PATH="/data/VLM/Qwen2.5-VL-3B-Instruct"
# MODEL_PATH="/data/VLM/Qwen2.5-VL-7B-Instruct"


# MODEL_PATH='/data/VLM/Qwen2-VL-2B'
# MODEL_PATH='/data/VLM/Qwen2-VL-2B-Instruct'

# MODEL_PATH='/data/VLM/Qwen2-VL-7B'
# MODEL_PATH='/data/VLM/Qwen2-VL-7B-Instruct' 
# MODEL_PATH="/data/VLM/deepseek-vl2-tiny"
# MODEL_PATH="/data/VLM/deepseek-vl-7b-chat"
# MODEL_PATH="/data/VLM/InternVL2_5-4B"   



DATA_ROOT="/data/dataset/MSRS/"
# DATA_ROOT="/data/dataset/FMB_dataset/"
# DATA_ROOT="/data/dataset/MFNet/"


THRESHOLD=0 # 暂时无用
echo "=========================================="
echo "运行检测"
echo "=========================================="

python Auto_modal_param.py --threshold $THRESHOLD --dataset $DATASET --objects $SPLITBY --eval_type $EVAL_TYPE --attention_type $ATTENTION_TYPE --model_path $MODEL_PATH --data_root $DATA_ROOT 2>&1 
