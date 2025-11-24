import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from model_methods.deepseekvl2_methods import *
def split_model(model_name):
    device_map = {}
    # 3b,16b,27b
    model_splits = {        
        '/data/VLM/deepseek-vl2-tiny': [3, 4, 4, 4], # 2 GPU
        # '/data/VLM/deepseek-vl2-small': [13, 14], # 2 GPU 
        '/data/VLM/deepseek-vl2-small': [6, 8, 8, 8], # 2 GPU 
        # '/data/VLM/deepseek-vl2': [10,10,10], # 3 GPU
        '/data/VLM/deepseek-vl2': [6, 8, 8, 8], # 4 GPU
    }
    num_layers_per_gpu = model_splits[model_name]
    num_layers = sum(num_layers_per_gpu)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language.model.layers.{layer_cnt}'] = i 
            layer_cnt += 1
    device_map['vision'] = 0
    device_map['projector'] = 0
    device_map['image_newline'] = 0
    device_map['view_seperator'] = 0
    device_map['language.model.embed_tokens'] = 0
    device_map['language.model.norm'] = 0
    device_map['language.lm_head'] = 0
    device_map[f'language.model.layers.{num_layers - 1}'] = 0
    return device_map
 
def load_model(model_path, dtype=torch.bfloat16):
    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
 
    # csdn2k
    device_map = split_model(model_path)
    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=dtype,
        device_map=device_map
    ).eval()
    return tokenizer, vl_gpt, vl_chat_processor


if __name__=="__main__":
    _,vl_gpt,vl_chat_processor=load_model("/data/VLM/deepseek-vl2-small")
    print(vl_gpt)
    import time

    print("开始")

    # 延时 2 秒
    time.sleep(2000)

    print("结束")