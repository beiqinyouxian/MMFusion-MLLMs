from matplotlib.pylab import true_divide
import torch
import os
import random
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from tqdm import tqdm
import model_methods
from transformers import AutoModel, AutoTokenizer
model_type_to_module = {
    "qwen2.5vl": "model_methods.qwen2_5_methods",
    "qwen2vl": "model_methods.qwen2_methods",
    "llava": "model_methods.llava_methods",
    "deepseek_vl2": "model_methods.deepseekvl2_methods",
    "deepseek_vl": "model_methods.deepseekvl_methods",
    "internvl2_5": "model_methods.internvl2_5_methods"
}

def load_VLM_model(model_path, model_type):
    print(f"Loading {model_type} model from {model_path}")
    if model_type == "qwen2.5vl":
        qwen_model = model_methods.qwen2_5_methods.Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
                device_map="auto"
            )
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
       
        qwen_processor = model_methods.qwen2_5_methods.AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels, trust_remote_code=True, padding_side='left', use_fast=True)
        qwen_processor.image_processor.size["longest_edge"] = max_pixels
        return qwen_model, qwen_processor
    elif model_type == "qwen2vl":
        qwen2_model = model_methods.qwen2_methods.Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            device_map="auto"
        )
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        qwen2_processor = model_methods.qwen2_methods.AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels, trust_remote_code=True, use_fast=True)
        return qwen2_model, qwen2_processor
    elif model_type == "llava":
        llava_model = model_methods.llava_methods.LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            device_map="auto"
        )
        llava_processor = model_methods.llava_methods.LlavaProcessor.from_pretrained(model_path, patch_size=14, use_fast=True)
        return llava_model, llava_processor
    elif model_type == "deepseek_vl2":
        vl2_chat_processor: model_methods.deepseekvl2_methods.DeepseekVLV2Processor = model_methods.deepseekvl2_methods.DeepseekVLV2Processor.from_pretrained(model_path)
        device_map = model_methods.deepseekvl2_methods.split_model(model_path)
        vl2_gpt: model_methods.deepseekvl2_methods.DeepseekVLV2ForCausalLM = model_methods.deepseekvl2_methods.AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            device_map=device_map
        ).eval()
        return vl2_gpt, vl2_chat_processor

    elif model_type == "deepseek_vl":
        vl_chat_processor: model_methods.deepseekvl_methods.VLChatProcessor = model_methods.deepseekvl_methods.VLChatProcessor.from_pretrained(model_path)
        tokenizer = vl_chat_processor.tokenizer
        vl_gpt: model_methods.deepseekvl_methods.MultiModalityCausalLM = model_methods.deepseekvl_methods.AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
        return vl_gpt, vl_chat_processor
    elif model_type == "internvl2_5":
        device_map = model_methods.internvl2_5_methods.split_model('InternVL2_5-4B')
        internvl2_5_model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
                device_map=device_map).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        return internvl2_5_model, tokenizer
    else:
        print("ERROR in loading. Please check the path of the model.")
        exit()



from utils import *
from PIL import Image, ImageDraw
import cv2
import numpy as np


def generate_all_head_attention_map(model, processor, image, object, label, attention_type, model_type):
   
    if model_type == "llava":
        if "7" in args.model_path:
            LAYERS=32
            HEADS=32
        elif "13" in args.model_path:
            LAYERS=40
            HEADS=40
        general_question = 'Write a general description of the image.'
        prompt = f"<image>\nUSER: {object}\nASSISTANT:"
        general_prompt = f"<image>\nUSER: {general_question} Answer the question using a single word or phrase.\nASSISTANT:"
        res_threshold=1024
        if image.size[0] > res_threshold or image.size[1] > res_threshold:
            print("image size is greater than 1024, use high_res")
            exit()
        else:
            if attention_type == "orin":
                att_results,eval_results = norm_res(model_methods.llava_methods.auto_param_orin_attention_llava, image, prompt, general_prompt, model, processor, label, LAYERS, HEADS)
            elif attention_type == "rel":
                att_results,eval_results = norm_res(model_methods.llava_methods.auto_param_rel_attention_llava, image, prompt, general_prompt, model, processor, label, LAYERS, HEADS)

    elif model_type == "qwen2.5vl":
        if "3" in args.model_path:
            LAYERS=36
            HEADS=16
        elif "7" in args.model_path:
            LAYERS=28
            HEADS=28
        prompt = f"{object}"
        general_prompt = f"Write a general description of the image. Answer the question using a single word or phrase."
        if attention_type=="orin":
            att_results,eval_results = norm_res(model_methods.qwen2_5_methods.auto_param_orin_attention_qwen2_5, image, prompt, general_prompt, model, processor, label, LAYERS, HEADS)
        elif attention_type=="rel":
            att_results,eval_results = norm_res(model_methods.qwen2_5_methods.auto_param_rel_attention_qwen2_5, image, prompt, general_prompt, model, processor, label, LAYERS, HEADS)
    
    elif model_type == "qwen2vl":
        if "7" in args.model_path:
            LAYERS=28
            HEADS=28
        elif "2" in args.model_path:
            LAYERS=28
            HEADS=12
        prompt = f"{object}"
        general_prompt = f"Write a general description of the image. Answer the question using a single word or phrase."
        if attention_type=="orin":
            att_results,eval_results = norm_res(model_methods.qwen2_methods.auto_param_orin_attention_qwen2, image, prompt, general_prompt, model, processor, label, LAYERS, HEADS)
        elif attention_type=="rel":
            att_results,eval_results = norm_res(model_methods.qwen2_methods.auto_param_rel_attention_qwen2, image, prompt, general_prompt, model, processor, label, LAYERS, HEADS)

    elif model_type == "deepseek_vl2":
        if "tiny" in args.model_path:
            LAYERS=12
            HEADS=10
        elif "small" in args.model_path:
            LAYERS=26
            HEADS=16
        prompt = f"{object}"
        general_prompt = f"Write a general description of the image. Answer the question using a single word or phrase."
        if attention_type=="orin":
            att_results,eval_results = norm_res(model_methods.deepseekvl2_methods.auto_param_orin_attention_deepseekvl2, image, prompt, general_prompt, model, processor, label, LAYERS, HEADS)
        elif attention_type=="rel":
            att_results,eval_results = norm_res(model_methods.deepseekvl2_methods.auto_param_rel_attention_deepseekvl2, image, prompt, general_prompt, model, processor, label, LAYERS, HEADS)
    else:
        print("ERROR in generate_attention_map. Please check the type of the model.")
        exit()

    return att_results,eval_results

def get_fusion_attention_map(model, processor, ir_img, vi_img, label, object, attention_type, model_type):
    model_att_results={}
    model_eval_results={}
    
    model_att_results["ir"], model_eval_results["ir"] = generate_all_head_attention_map(model, processor, ir_img, object, label, attention_type, model_type)
    model_att_results["vi"], model_eval_results["vi"] = generate_all_head_attention_map(model, processor, vi_img, object, label, attention_type, model_type)
    
    return model_att_results, model_eval_results


import json

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return super(CustomEncoder, self).default(obj)

def append_to_json_lines(new_dict, filename):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(new_dict, ensure_ascii=False, cls=CustomEncoder) + '\n')



class SegmentationLabelProcessor:
    def __init__(self,dataset_path, scene):
        if "MSRS" in dataset_path:
            self.dataset_type= "MSRS" 
        elif "MFNet" in dataset_path:
            self.dataset_type= "MFNet"
        elif "FMB" in dataset_path:
            self.dataset_type= "FMB"
        else:
            print("ERROR in dataset_path. Please check the dataset_path.")
            exit()
        self.palette = self.get_palette()
        self.ir_path = os.path.join(dataset_path,"ir")
        self.vi_path = os.path.join(dataset_path,"vi")
        self.label_path = os.path.join(dataset_path,"Segmentation_labels")
        self.file_list = os.listdir(self.label_path)
        if scene == "DAY":
            self.file_list = [f for f in self.file_list if f.endswith('D.png') or f.endswith('D.jpg') or f.endswith('D.jpeg')]
        elif scene == "NIGHT":
            self.file_list = [f for f in self.file_list if f.endswith('N.png') or f.endswith('N.jpg') or f.endswith('N.jpeg')]
        elif scene == "ALL":
            pass
        else:
            print(f"Warning: Unknown scene parameter '{scene}', using all files.")

    def __getitem__(self,item):
        file_name = self.file_list[item]
        label = self.imread(os.path.join(self.label_path,file_name)) 
        ir_img = Image.open(os.path.join(self.ir_path,file_name)).convert("RGB")
        vi_img = Image.open(os.path.join(self.vi_path,file_name)).convert("RGB")
        return label, ir_img, vi_img, file_name

    def __len__(self):
        return len(self.file_list)
    def imread(self, path):
        label = np.array(Image.open(path))
        return label
    
    def get_palette(self):
        unlabelled = [0, 0, 0]
        car = [64, 0, 128]
        person = [64, 64, 0]
        bike = [0, 128, 192]
        curve = [0, 0, 192]
        car_stop = [128, 128, 0]
        guardrail = [64, 64, 128]
        color_cone = [192, 128, 128]
        bump = [192, 64, 0]
        
        return np.array([
            unlabelled, car, person, bike, curve, 
            car_stop, guardrail, color_cone, bump
        ])
    
    def extract_mask(self, label, class_id):
        if self.dataset_type == "MSRS" or self.dataset_type == "MFNet":
            mask = np.zeros_like(label, dtype=np.uint8)
            mask[label == class_id] = 1
        elif self.dataset_type == "FMB":
            mask = 1 - (label == class_id).astype(np.uint8)
        else:
            print("ERROR in extract_mask. Please check the dataset_type.")
            exit()
        return mask
    
    def check_class_exists(self, label, class_id):
        unique_classes = np.unique(label)
        return class_id in unique_classes

if __name__ == "__main__":

    random.seed(42)
    
    import argparse
    parser = argparse.ArgumentParser(description='Process command line arguments')

    parser.add_argument('--threshold', type=float, help='Threshold value')
    parser.add_argument('--dataset', help='Dataset name')
    parser.add_argument('--objects', help='Split method')
    parser.add_argument('--scene', help='Scene type')
    parser.add_argument('--eval_type', help='Evaluation type')
    parser.add_argument('--attention_type', help='Attention type')
    parser.add_argument('--model_path', help='Path to the model')
    parser.add_argument('--data_root', help='Root directory for data')

    args = parser.parse_args()

    print("threshold:", args.threshold)
    print("dataset:", args.dataset)
    print("objects:", args.objects)
    print("scene:", args.scene)
    print("attention_type:", args.attention_type)
    print("eval_type:", args.eval_type)
    print("model_path:", args.model_path)
    print("data_root:", args.data_root)


    if args.model_path:
        lower_str = args.model_path.lower()
        model_mapping = {
            "qwen2.5-vl": "qwen2.5vl",
            "qwen2-vl": "qwen2vl",
            "llava": "llava",
            "deepseek-vl2": "deepseek_vl2",
            "deepseek-vl": "deepseek_vl",
            "internvl2_5": "internvl2_5"
        }
        results = {key: key in lower_str for key in model_mapping}
        if "deepseek-vl2" in lower_str:
            results["deepseek-vl"] = False

        print("results:", results)

        model_type = next((model_mapping[key] for key, found in results.items() if found), None)

        if model_type:
            print(f"Model type detected: {model_type}")
        else:  
            print("ERROR in loading. Please check the path of the model.")
            exit()
    else:
        print("Warning: model_path not provided, skipping results generation")
        exit()


    if model_type in model_type_to_module:
        module_name = model_type_to_module[model_type]
        globals()[module_name] = __import__(module_name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model, processor = load_VLM_model(args.model_path, model_type)

    objects = args.objects.split(',')

    dataset_path = os.path.join(args.data_root,args.dataset)    
    dataset_processor = SegmentationLabelProcessor(dataset_path, args.scene)
    dataset_name = os.path.basename(os.path.normpath(args.data_root))
    res_img={}
    result=[]
    model_name = (args.model_path).split('/')[-1]
    vi_json_file = f'./{args.scene}_auto_param_result/{model_name}_{model_type}_{dataset_name}_{args.eval_type}_{args.attention_type}_vi.json'
    ir_json_file = f'./{args.scene}_auto_param_result/{model_name}_{model_type}_{dataset_name}_{args.eval_type}_{args.attention_type}_ir.json'
    if os.path.exists(vi_json_file):
        os.remove(vi_json_file)
    if os.path.exists(ir_json_file):
        os.remove(ir_json_file)
    LIMIT = 100
    print(f"length of dataset_processor:{len(dataset_processor)}")
    NUM = len(dataset_processor) if len(dataset_processor) < LIMIT else LIMIT
    for item in tqdm(range(NUM), desc="Processing images"):
        label, ir_img, vi_img, file_name = dataset_processor[item]
        for i in range(len(objects)+1):
            if dataset_processor.check_class_exists(label, i):
                label_mask = dataset_processor.extract_mask(label, i)
                object = objects[i-1]
                object = object.replace("_", " ")

                if args.eval_type == "Auto":
                    atten_map,eval_results = get_fusion_attention_map(model, processor, ir_img, vi_img, label_mask, object, args.attention_type, model_type)
                    eval_results_vi={}
                    eval_results_ir={}
                    eval_results_vi[f"{file_name}_{object}"] = eval_results["vi"]
                    eval_results_ir[f"{file_name}_{object}"] = eval_results["ir"]
                    append_to_json_lines(eval_results_vi, vi_json_file)
                    append_to_json_lines(eval_results_ir, ir_json_file)
                else:
                    print("ERROR in attention_type. Please check the attention_type.")
                    exit()
            else:
                continue
            

        
  
