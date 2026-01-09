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
        vl2_chat_processor: DeepseekVLV2Processor = model_methods.deepseekvl2_methods.DeepseekVLV2Processor.from_pretrained(model_path)
        device_map = model_methods.deepseekvl2_methods.split_model(model_path)
        vl2_gpt: DeepseekVLV2ForCausalLM = model_methods.deepseekvl2_methods.AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            device_map=device_map
        ).eval()
        return vl2_gpt, vl2_chat_processor

    elif model_type == "deepseek_vl":
        vl_chat_processor: VLChatProcessor = model_methods.deepseekvl_methods.VLChatProcessor.from_pretrained(model_path)
        tokenizer = vl_chat_processor.tokenizer
        vl_gpt: MultiModalityCausalLM = model_methods.deepseekvl_methods.AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
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


def generate_specific_attention_map(model, processor, image, object, label, attention_type, model_type, image_type, scene):
    vi_items=[]
    ir_items=[]
    items=[]
    numofheads=int(args.num)
    model_name = (args.model_path).split('/')[-1]
    if model_type == "llava":
        if "7" in args.model_path:
            LAYERS=32
            HEADS=32
          
        elif "13" in args.model_path:
            LAYERS=40
            HEADS=40
      
        if model_name == "llava-v1.5-7b-hf":
            if scene == "DAY":
                vi_items=["11_17", "14_26", "14_3", "17_18", "14_24", "17_11"][:numofheads]
                ir_items=["14_26", "10_29", "11_17", "14_3", "17_18", "18_10"][:numofheads]
            elif scene == "NIGHT":
                vi_items=["11_17", "14_3", "10_29", "14_20", "14_26", "14_24"][:numofheads]
                ir_items=["14_26", "10_29", "14_3", "11_17", "14_24", "14_20"][:numofheads]
            else:
                print("ERROR in scene. Please check the scene.")
                exit()

        elif model_name == "llava-1.5-13b-hf":

            
            if scene == "DAY":
                vi_items=["13_17", "16_30", "11_37", "13_21", "15_39", "16_31"][:numofheads]
                ir_items=["16_30", "13_17", "15_39", "15_2", "11_37", "16_2"][:numofheads]
            elif scene == "NIGHT":
                vi_items=["13_17", "16_30", "13_21", "16_31", "11_37", "14_21"][:numofheads]
                ir_items=["16_30", "13_21", "15_39", "13_17", "15_2", "11_37"][:numofheads]
            else:
                print("ERROR in scene. Please check the scene.")
                exit()
        if image_type=="vi":
            items=vi_items
        else:
            items=ir_items

        general_question = 'Write a general description of the image.'
        prompt = f"<image>\nUSER: {object}\nASSISTANT:"
        general_prompt = f"<image>\nUSER: {general_question} Answer the question using a single word or phrase.\nASSISTANT:"
        res_threshold=1024
        if image.size[0] > res_threshold or image.size[1] > res_threshold:
            print("image size is greater than 1024, use high_res")
            exit()
        else:
            if attention_type == "orin":
                att_results = specific_norm_res(model_methods.llava_methods.auto_param_orin_attention_llava, image, prompt, general_prompt, model, processor, label, LAYERS, HEADS, items)
            elif attention_type == "rel":
                att_results = specific_norm_res(model_methods.llava_methods.auto_param_rel_attention_llava, image, prompt, general_prompt, model, processor, label, LAYERS, HEADS, items)

    elif model_type == "qwen2.5vl":
        if "3" in args.model_path:
            LAYERS=36
            HEADS=16
            
        elif "7" in args.model_path:
            LAYERS=28
            HEADS=28
            
        if model_name == "Qwen2.5-VL-3B-Instruct":

            if scene == "DAY":
                vi_items=["27_7", "25_8", "27_2", "20_4", "25_14", "26_12"][:numofheads]
                ir_items=["27_2", "27_7", "27_0", "20_7", "20_4", "20_1"][:numofheads]
            elif scene == "NIGHT":
                vi_items=["27_7", "20_4", "25_8", "27_2", "27_0", "20_1"][:numofheads]
                ir_items=["27_7", "27_2", "20_4", "20_7", "27_0", "22_15"][:numofheads]
            else:
                print("ERROR in scene. Please check the scene.")
                exit()
      
        elif model_name == "Qwen2.5-VL-7B-Instruct":
            if scene == "DAY":
                vi_items=["19_20", "19_22", "19_23", "19_21", "19_16", "19_25"][:numofheads]
                ir_items=["19_20", "19_21", "17_27", "19_25", "21_26", "19_16"][:numofheads]
            elif scene == "NIGHT":
                vi_items=["19_20", "19_16", "19_22", "19_17", "19_23", "19_21"][:numofheads]
                ir_items=["19_17", "19_21", "19_23", "19_20", "21_25", "19_22"][:numofheads]
            else:
                print("ERROR in scene. Please check the scene.")
                exit()
        if image_type=="vi":
            items=vi_items
        else:
            items=ir_items

        prompt = f"{object}"
        general_prompt = f"Write a general description of the image. Answer the question using a single word or phrase."
        if attention_type=="orin":
            att_results = specific_norm_res(model_methods.qwen2_5_methods.auto_param_orin_attention_qwen2_5, image, prompt, general_prompt, model, processor, label, LAYERS, HEADS, items)
        elif attention_type=="rel":
            att_results = specific_norm_res(model_methods.qwen2_5_methods.auto_param_rel_attention_qwen2_5, image, prompt, general_prompt, model, processor, label, LAYERS, HEADS, items)
    
    elif model_type == "qwen2vl":
        if "7" in args.model_path:
            LAYERS=28
            HEADS=28
        elif "2" in args.model_path:
            LAYERS=28
            HEADS=12

        if model_name == "Qwen2-VL-2B":
            if scene == "DAY":
                vi_items=["17_1", "17_4", "19_9", "17_5", "21_11", "20_1"][:numofheads]
                ir_items=["14_9", "20_0", "17_2", "19_6", "14_10", "19_8"][:numofheads]
            elif scene == "NIGHT":
                vi_items=["17_1", "17_4", "17_2", "17_5", "21_11", "19_8"][:numofheads]
                ir_items=["20_0", "14_10", "14_9", "19_6", "20_11", "17_2"][:numofheads]
            else:
                print("ERROR in scene. Please check the scene.")
                exit()

        elif model_name == "Qwen2-VL-2B-Instruct":
            if scene == "DAY":
                vi_items=["14_7", "17_2", "19_9", "17_5", "20_5", "19_6"][:numofheads]
                ir_items=["20_5", "14_7", "14_9", "19_9", "17_5", "20_10"][:numofheads]
            elif scene == "NIGHT":
                vi_items=["14_7", "20_5", "19_9", "20_1", "19_6", "17_4"][:numofheads]
                ir_items=["20_5", "19_9", "20_10", "14_7", "14_9", "17_2"][:numofheads]
            else:
                print("ERROR in scene. Please check the scene.")
                exit()
        elif model_name == "Qwen2-VL-7B":
            if scene == "DAY":
                vi_items=["19_20", "19_25", "19_23", "19_16", "20_21", "16_1"][:numofheads]
                ir_items=["19_17", "19_20", "19_22", "19_16", "19_15", "20_19"][:numofheads]
            elif scene == "NIGHT":
                vi_items=["19_20", "19_16", "19_25", "16_1", "20_21", "19_15"][:numofheads]
                ir_items=["19_20", "14_0", "16_9", "19_24", "19_17", "14_8"][:numofheads]
            else:
                print("ERROR in scene. Please check the scene.")
                exit()
        elif model_name == "Qwen2-VL-7B-Instruct":
            if scene == "DAY":
                vi_items=["19_16", "19_21", "19_26", "19_15", "18_10", "19_25"][:numofheads]
                ir_items=["16_1", "19_16", "16_20", "14_0", "14_4", "17_27"][:numofheads]
            elif scene == "NIGHT":
                vi_items=["19_16", "19_21", "11_24", "19_25", "18_10", "20_21"][:numofheads]
                ir_items=["16_1", "16_20", "17_27", "16_0", "14_0", "19_16"][:numofheads]
            else:
                print("ERROR in scene. Please check the scene.")
                exit()

        
        if image_type=="vi":
            items=vi_items
        else:
            items=ir_items

        prompt = f"{object}"
        general_prompt = f"Write a general description of the image. Answer the question using a single word or phrase."
        if attention_type=="orin":
            att_results = specific_norm_res(model_methods.qwen2_methods.auto_param_orin_attention_qwen2, image, prompt, general_prompt, model, processor, label, LAYERS, HEADS, items)
        elif attention_type=="rel":
            att_results = specific_norm_res(model_methods.qwen2_methods.auto_param_rel_attention_qwen2, image, prompt, general_prompt, model, processor, label, LAYERS, HEADS, items)

    elif model_type == "deepseek_vl2":
        if "tiny" in args.model_path:
            LAYERS=12
            HEADS=10
        elif "small" in args.model_path:
            LAYERS=26
            HEADS=16
            
        if model_name == "deepseek-vl2-tiny":
            if scene == "DAY":
                vi_items=["9_6", "1_1", "4_2", "2_6", "10_6", "8_4"][:numofheads]
                ir_items=["9_6", "4_2", "1_1", "10_6", "2_6", "8_4"][:numofheads]
            elif scene == "NIGHT":
                vi_items=["9_6", "8_4", "10_9", "1_1", "10_2", "7_8"][:numofheads]
                ir_items=["9_6", "4_2", "7_7", "11_3", "8_4", "0_2"][:numofheads]
            else:
                print("ERROR in scene. Please check the scene.")
                exit()

        elif model_name == "deepseek-vl2-small":
            if scene == "DAY":
                vi_items=["6_5", "4_5", "9_3", "6_2", "18_4", "6_12"][:numofheads]
                ir_items=["6_5", "4_5", "9_3", "6_2", "18_4", "18_4"][:numofheads]
            elif scene == "NIGHT":
                vi_items=["4_5", "6_5", "18_13", "4_6", "17_7", "6_2"][:numofheads]
                ir_items=["4_5", "6_5", "25_0", "19_12", "18_13", "6_2"][:numofheads]
            else:
                print("ERROR in scene. Please check the scene.")
                exit()

        if image_type=="vi":
            items=vi_items
        else:
            items=ir_items
            
        prompt = f"{object}"
        general_prompt = f"Write a general description of the image. Answer the question using a single word or phrase."
        if attention_type=="orin":
            att_results = specific_norm_res(model_methods.deepseekvl2_methods.auto_param_orin_attention_deepseekvl2, image, prompt, general_prompt, model, processor, label, LAYERS, HEADS, items)
        elif attention_type=="rel":
            att_results = specific_norm_res(model_methods.deepseekvl2_methods.auto_param_rel_attention_deepseekvl2, image, prompt, general_prompt, model, processor, label, LAYERS, HEADS, items)
    else:
        print("ERROR in generate_attention_map. Please check the type of the model.")
        exit()

    return att_results

def get_fusion_attention_map(model, processor, ir_img, vi_img, label_mask, object, attention_type, model_type, scene):

    model_att_results={}
    model_eval_results={}
    
    
    model_att_results["vi"] = generate_specific_attention_map(model, processor, ir_img, object, label_mask, attention_type, model_type,image_type="vi", scene=scene)
    model_att_results["ir"] = generate_specific_attention_map(model, processor, vi_img, object, label_mask, attention_type, model_type, image_type="ir", scene=scene)
    
    fusion_attention_map = composite_attn_map(model_att_results)
    if args.threshold==0:
        threshold_attention_map = auto_otsu(fusion_attention_map)
    else:
        threshold_attention_map = constant_threshold(fusion_attention_map,threshold=args.threshold)
    eval_processor = Evaluation_processor(label_mask, threshold_attention_map)
    model_eval_results = eval_processor.calculate_metrics()
    model_eval_results.append(eval_processor.calculate_Entropy(fusion_attention_map))
    return threshold_attention_map, model_eval_results


def get_enhance_attention_map(model, processor, ir_img, vi_img, label_mask, object, attention_type, model_type, scene):
    model_att_results={}
    model_eval_results={}
    model_att_results["orin"] , _ = generate_lvlm_map(model, processor, ir_img, vi_img, label_mask, object, model_type)
   
    
    model_att_results["vi"] = generate_specific_attention_map(model, processor, ir_img, object, label_mask, attention_type, model_type,image_type="vi", scene=scene)
    model_att_results["ir"] = generate_specific_attention_map(model, processor, vi_img, object, label_mask, attention_type, model_type, image_type="ir", scene=scene)
    
    fusion_attention_map = composite_attn_map(model_att_results)

    if args.threshold==0:
        threshold_attention_map = auto_otsu(fusion_attention_map)
    else:
        threshold_attention_map = constant_threshold(fusion_attention_map,threshold=args.threshold)
    eval_processor = Evaluation_processor(label_mask, threshold_attention_map)
    model_eval_results = eval_processor.calculate_metrics()
    model_eval_results.append(eval_processor.calculate_Entropy(fusion_attention_map))
    fusion_attention_map=None
    return fusion_attention_map, model_eval_results

def create_mask_from_bbox(image_size, bbox_list, model_type):
    image_height, image_width = image_size
    
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    try:
        for bbox in bbox_list:
         
            xmin, ymin, xmax, ymax = bbox
            if xmin <= 1 and ymin <= 1 and xmax <= 1 and ymax <= 1:
                xmin = xmin*image_width
                ymin = ymin*image_height
                xmax = xmax*image_width
                ymax = ymax*image_height
           
            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(image_width, int(xmax))
            ymax = min(image_height, int(ymax))
            
            if xmax <= xmin or ymax <= ymin:
                continue
            mask[ymin:ymax, xmin:xmax] = 1
    except Exception as e:
        mask[0:1, 0:1] = 0
        print("ERROR in create_mask_from_bbox. Please check the bbox.")
    return mask

import re
def generate_lvlm_map(model, processor, ir_image, vi_image, label, object, model_type):
    width, height = ir_image.size
    if model_type == "llava":
        prompt = f"<image>\nUSER: Locate the {object} in the provided pair of images(visual and infrared images). Describe the position of all objects in bbox format. If there are multiple target objects in the image, please provide a list of bboxes. Do not include any extra text, only provide the bboxes.Locate the [target object] in the provided image. Describe the position of all objects in bbox format. If there are multiple target objects in the image, please present them in the form of a two-dimensional list. Do not include any extra text, only provide the bboxes.\nASSISTANT:"
        vi_inputs = processor(vi_image, prompt, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
        ir_inputs = processor(ir_image, prompt, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    
        vi_generate_ids = model.generate(**vi_inputs, max_new_tokens=300)
        vi_result = processor.batch_decode(vi_generate_ids, skip_special_tokens=True)
        vi_result = str(vi_result)

        ir_generate_ids = model.generate(**ir_inputs, max_new_tokens=300)
        ir_result = processor.batch_decode(ir_generate_ids, skip_special_tokens=True)
        ir_result = str(ir_result)
        result = vi_result + ir_result


    if model_type == "qwen2vl":
        vi_image_str = encode_base64(vi_image)
        ir_image_str = encode_base64(ir_image)
        prompt= f'Locate the {object} in the provided image. Describe the position of all objects in bbox format. If there are multiple target objects in the image, please provide a list of bboxes(example: [[x1,y1,x2,y2],[x3,y3,x4,y4]]). Do not include any extra text, only provide the bboxes.'
        messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{vi_image_str}'}, {"type": "image", "image": f'data:image;base64,{ir_image_str}'}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = model_methods.qwen2_methods.process_vision_info(messages)
        inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True, return_tensors="pt").to(model.device, torch.bfloat16)
        generated_ids = model.generate(**inputs, max_new_tokens=300)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        result = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result = str(result)
          
    if model_type == "qwen2.5vl":
        vi_image_str = encode_base64(vi_image)
        ir_image_str = encode_base64(ir_image)
        prompt= f'Locate the {object} in the provided image. Describe the position of all objects in bbox format. If there are multiple target objects in the image, please provide a list of bboxes. Do not include any extra text, only provide the bboxes.'
        messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{vi_image_str}'}, {"type": "image", "image": f'data:image;base64,{ir_image_str}'}, {"type": "text", "text": prompt}]}]
        inputs = model_methods.qwen2_5_methods.prepare_qwen2_5_input(messages, processor).to(model.device, torch.bfloat16)
        generated_ids = model.generate(**inputs, max_new_tokens=300)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        result = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result = str(result)
 
    if model_type == "deepseek_vl2":
        data_url1 = model_methods.deepseekvl2_methods.pil_image_to_data_url(vi_image)
        data_url2 = model_methods.deepseekvl2_methods.pil_image_to_data_url(ir_image)
       
        conversation = [
            {
                "role": "<|User|>",
                "content": "This is visual-light image: <image>\n"
                        "This is infrared image: <image>\n"
                        f"Locate the {object} in the provided image. Describe the position of all objects in bbox format. If there are multiple target objects in the image, please provide a list of bboxes. Do not include any extra text, only provide the bboxes.",
                "images": [
                    data_url1,data_url2,
                ],
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        pil_images=[vi_image,ir_image]
        prepare_inputs = processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(model.device)

        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

        outputs = model.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        result = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)

    if model_type == "deepseek_vl":
        data_url1 = model_methods.deepseekvl_methods.pil_image_to_data_url(vi_image)
        data_url2 = model_methods.deepseekvl_methods.pil_image_to_data_url(ir_image)
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>Locate the {object} in the provided image. Describe the position of all objects in bbox format. If there are multiple target objects in the image, please provide a list of bboxes. Do not include any extra text, only provide the bboxes."
                        f"<image_placeholder>Locate the {object} in the provided image. Describe the position of all objects in bbox format. If there are multiple target objects in the image, please provide a list of bboxes. Do not include any extra text, only provide the bboxes.",
                "images": [
                    data_url1,data_url2,
                        ],
            },
            {"role": "Assistant", "content": ""},
        ]
        pil_images = model_methods.deepseekvl_methods.load_pil_images(conversation)
        prepare_inputs = processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(model.device)

        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        result = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    if model_type == "internvl2_5":
        pixel_values = model_methods.internvl2_5_methods.load_image(vi_image, max_num=12).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=256, do_sample=True)

        pixel_values1 = model_methods.internvl2_5_methods.load_image(vi_image, max_num=12).to(torch.bfloat16).cuda()
        pixel_values2 = model_methods.internvl2_5_methods.load_image(ir_image, max_num=12).to(torch.bfloat16).cuda()
        pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
        num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

        question = f'Image-1: <image>\nImage-2: <image>\n Locate the {object} in the provided image in bboxs.Such as [x1,y1,x2,y2].'
        response, history = model.chat(processor, pixel_values, question, generation_config,
                                    num_patches_list=num_patches_list,
                                    history=None, return_history=True)
        result = str(response)

    try:
        bboxes = re.findall(r'\[([\d.\s,]+)\]', result)
        result = [list(map(float, bbox.replace(' ', '').split(','))) for bbox in bboxes]
    except:
        print("ERROR in generate_lvlm_map. Please check the result.")
        result = [0,0,0,0]

    att_map = create_mask_from_bbox((height, width), result, model_type) 
    eval_processor = Evaluation_processor(label, att_map)
    model_eval_results = eval_processor.calculate_metrics()
    entropy=eval_processor.calculate_Entropy(att_map)
    if type(entropy) == np.float64 or type(entropy) == int:
        model_eval_results.append(entropy)
    else:
        print("ERROR in calculate_Entropy. Please check the entropy.")
        print(type(entropy))
        model_eval_results.append(0)
    return att_map,model_eval_results

def visualize_atten_map(image_path, atten_map, ref_map, file_name, object, image_type):
    atten_map = atten_map.astype(np.uint8)
    ref_map = ref_map.astype(np.uint8)
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    def create_overlay(array, color=(255, 0, 0), alpha=0.5):
        overlay = np.zeros((*array.shape, 4), dtype=np.uint8)
        overlay[array == 1] = [*color, int(255 * alpha)]
        return overlay
    def blend_with_original(original, overlay):
        original_rgba = cv2.cvtColor(original, cv2.COLOR_RGB2RGBA)
        overlay_alpha = overlay[:, :, 3] / 255.0
        blended = original_rgba.copy()
        for c in range(3):
            blended[:, :, c] = (1 - overlay_alpha) * original_rgba[:, :, c] + overlay_alpha * overlay[:, :, c]
        return blended
    overlay1 = create_overlay(ref_map, color=(255, 0, 0), alpha=0.5)
    overlay2 = create_overlay(atten_map, color=(0, 255, 0), alpha=0.5)
    blended1 = blend_with_original(original_img, overlay1)
    blended2 = blend_with_original(original_img, overlay2)

    combined = np.hstack([blended1, blended2])
    cv2.imwrite(f'./picture/{object}_{image_type}_{file_name}', cv2.cvtColor(combined, cv2.COLOR_RGBA2BGRA))


def visualize_GT_map(vi_image_path, ir_image_path, ref_map, file_name, object):
    ref_map = ref_map.astype(np.uint8)
    vi_img = cv2.imread(vi_image_path)
    ir_img = cv2.imread(ir_image_path)
    vi_img = cv2.cvtColor(vi_img, cv2.COLOR_BGR2RGB)
    ir_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB)
    def create_overlay(array, color=(255, 0, 0), alpha=0.5):
        overlay = np.zeros((*array.shape, 4), dtype=np.uint8)
        overlay[array == 1] = [*color, int(255 * alpha)]
        return overlay
    def blend_with_original(original, overlay):
        original_rgba = cv2.cvtColor(original, cv2.COLOR_RGB2RGBA)
        overlay_alpha = overlay[:, :, 3] / 255.0
        blended = original_rgba.copy()
        for c in range(3):
            blended[:, :, c] = (1 - overlay_alpha) * original_rgba[:, :, c] + overlay_alpha * overlay[:, :, c]
        return blended
    overlay1 = create_overlay(ref_map, color=(0, 255, 0), alpha=0.5)
    overlay2 = create_overlay(ref_map, color=(0, 255, 0), alpha=0.5)
    blended1 = blend_with_original(vi_img, overlay1)
    blended2 = blend_with_original(ir_img, overlay2)

    combined = np.hstack([blended1, blended2])
    cv2.imwrite(f'./picture/{object}_{file_name}', cv2.cvtColor(combined, cv2.COLOR_RGBA2BGRA))

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
    parser.add_argument('--num', help='Num of Heads')
    parser.add_argument('--dataset', help='Dataset name')
    parser.add_argument('--objects', help='Split method')
    parser.add_argument('--scene', help='Scene type')
    parser.add_argument('--eval_type', help='Evaluation type')
    parser.add_argument('--attention_type', help='Attention type')
    parser.add_argument('--model_path', help='Path to the model')
    parser.add_argument('--data_root', help='Root directory for data')

    args = parser.parse_args()

    print("threshold:", args.threshold)
    print("num of heads:", args.num)
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
    res_img={}
    result=[]
    json_file = f'./results/{model_type}_{args.dataset}_{args.eval_type}_{args.attention_type}.json'
    LIMIT = 1000000
    print(f"length of dataset_processor:{len(dataset_processor)}")
    NUM = len(dataset_processor) if len(dataset_processor) < LIMIT else LIMIT
    for item in tqdm(range(NUM), desc="Processing images"):
        label, ir_img, vi_img, file_name = dataset_processor[item]
        if args.scene == "DAY":
            SCENE = "DAY"
        elif args.scene == "NIGHT":
            SCENE = "NIGHT"
        elif args.scene == "ALL":
            if "D" in file_name:
                SCENE = "DAY"
            elif "N" in file_name:
                SCENE = "NIGHT"
            else:
                print("ERROR in scene. Please check the scene.")
                exit()

        for i in range(len(objects)+1):
            if dataset_processor.check_class_exists(label, i):
                label_mask = dataset_processor.extract_mask(label, i)
                object = objects[i-1]
                object = object.replace("_", " ")
                if args.eval_type == "Orin":
                    atten_map,eval_results = generate_lvlm_map(model, processor, ir_img, vi_img, label_mask, object, model_type)

                elif args.eval_type == "Fusion":
                    atten_map,eval_results = get_fusion_attention_map(model, processor, ir_img, vi_img, label_mask, object, args.attention_type, model_type, SCENE)
                elif args.eval_type == "Enhance":
                    atten_map,eval_results = get_enhance_attention_map(model, processor, ir_img, vi_img, label_mask, object, args.attention_type, model_type, SCENE)

                else:
                    print("ERROR in attention_type. Please check the attention_type.")
                    exit()


                result.append(eval_results)

                visualize_atten_map(dataset_processor.vi_path+f'/{file_name}', atten_map, label_mask, file_name, object, "VI")
                visualize_atten_map(dataset_processor.ir_path+f'/{file_name}', atten_map, label_mask, file_name, object, "IR")
                
            else:
                continue
            

    model_name = (args.model_path).split('/')[-1]
    if args.eval_type == "Enhance":
        folder = "enhance_results"
    elif args.eval_type == "Fusion":
        folder = "fusion_results"
    elif args.eval_type == "Orin":
        folder = "orin_results"
    os.makedirs(f"./{folder}", exist_ok=True)
    with open(f"./{folder}/{model_name}_{args.num}_{dataset_processor.dataset_type}_output_{args.dataset}_{args.eval_type}_{args.attention_type}.txt", "a") as file:
        sum_param1 = 0
        sum_param2 = 0
        sum_param3 = 0
        sum_param4 = 0
        n = len(result)
        print(f"all_inference:{n}")
        file.write(str(f"all_inference:{n}\n"))
        count=0
        for row in result:
            sum_param1 += row[0]
            sum_param2 += row[1]
            sum_param3 += row[2]
            sum_param4 += row[3]


        avg_param1 = round(100 * sum_param1 / n, 2)
        avg_param2 = round(100 * sum_param2 / n, 2)
        avg_param3 = round(100 * sum_param3 / n, 2)
        avg_param4 = round(100 * sum_param4 / n, 2)

        print(f"model:{args.model_path},avg_param1:{avg_param1},avg_param2:{avg_param2},avg_param3:{avg_param3},avg_param4:{avg_param4}")
        file.write(str([args.model_path,avg_param1,avg_param2,avg_param3,avg_param4])+ '\n')
  

            
