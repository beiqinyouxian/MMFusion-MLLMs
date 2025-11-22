import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from skimage.measure import block_reduce
from utils import *
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
# currently select 22 but feel free to try other layers
ATT_LAYER = 24



def rel_attention_qwen2(image, prompt, general_prompt, model, processor):

    """
    Compute relative attention scores for Qwen2VL.
    
    This function computes the relative attention scores between the input and general inputs
    for Qwen2.5VL. It first computes the attention scores for the input and general inputs, 
    and finally computes the relative attention scores.
    """

    conversation = [{"role": "user","content": [{"type": "image",},{"type": "text", "text": prompt},],}]
    general_conversation = [{"role": "user","content": [{"type": "image",},{"type": "text", "text": general_prompt},],}]
    # Preprocess the inputs
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    general_text_prompt = processor.apply_chat_template(general_conversation, add_generation_prompt=True)

    inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(model.device, torch.bfloat16)
    general_inputs = processor(text=[general_text_prompt], images=[image], padding=True, return_tensors="pt").to(model.device, torch.bfloat16)

    att_shape = (inputs['image_grid_thw'][0, 1:] / 2).cpu().numpy().astype(int).tolist()

    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')

    pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1
    pos_end = inputs['input_ids'].tolist()[0].index(vision_end_token_id)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        general_outputs = model(**general_inputs, output_attentions=True)

        att = outputs['attentions'][ATT_LAYER][0, :, -1, pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()
        general_att = general_outputs['attentions'][ATT_LAYER][0, :, -1, pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()

        att_map = att / general_att

        att_map = att_map.reshape(att_shape)

        return att_map

def orin_attention_qwen2(image, prompt, general_prompt, model, processor):

    """
    Compute relative attention scores for Qwen2.5VL.
    
    This function computes the relative attention scores between the input and general inputs
    for Qwen2.5VL. It first computes the attention scores for the input and general inputs, 
    and finally computes the relative attention scores.
    """

    conversation = [{"role": "user","content": [{"type": "image",},{"type": "text", "text": prompt},],}]
    # Preprocess the inputs
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

    inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(model.device, torch.bfloat16)
    att_shape = (inputs['image_grid_thw'][0, 1:] / 2).cpu().numpy().astype(int).tolist()
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1
    pos_end = inputs['input_ids'].tolist()[0].index(vision_end_token_id)
    outputs = model(**inputs, output_attentions=True)
    att = outputs['attentions'][ATT_LAYER][0, :, -1, pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()
    att_map = att.reshape(att_shape)
    return att_map

def auto_param_rel_attention_qwen2(image, prompt, general_prompt, model, processor, LAYERS, HEADS):

    """
    Compute relative attention scores for Qwen2.5VL.
    
    This function computes the relative attention scores between the input and general inputs
    for Qwen2.5VL. It first computes the attention scores for the input and general inputs, 
    and finally computes the relative attention scores.
    """

    conversation = [{"role": "user","content": [{"type": "image",},{"type": "text", "text": prompt},],}]
    general_conversation = [{"role": "user","content": [{"type": "image",},{"type": "text", "text": general_prompt},],}]
    # Preprocess the inputs
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    general_text_prompt = processor.apply_chat_template(general_conversation, add_generation_prompt=True)

    inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(model.device, torch.bfloat16)
    general_inputs = processor(text=[general_text_prompt], images=[image], padding=True, return_tensors="pt").to(model.device, torch.bfloat16)

    att_shape = (inputs['image_grid_thw'][0, 1:] / 2).cpu().numpy().astype(int).tolist()

    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')

    pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1
    pos_end = inputs['input_ids'].tolist()[0].index(vision_end_token_id)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        general_outputs = model(**general_inputs, output_attentions=True)
       
        temp_result={}
        for LAYER in range(LAYERS):
            for HEAD in range(HEADS):
                att = outputs['attentions'][LAYER][0, HEAD, -1, pos:pos_end].to(torch.float32).detach().cpu().numpy()
                general_att = general_outputs['attentions'][LAYER][0, HEAD, -1, pos:pos_end].to(torch.float32).detach().cpu().numpy()
                epsilon = 1e-7
                general_att_safe = np.where(general_att == 0, epsilon, general_att)
                att_map = att / general_att_safe
                # att_map = att / general_att
                att_map = att_map.reshape(att_shape)
                temp_result[f"{LAYER}_{HEAD}"] = att_map
    
    return temp_result
 


def auto_param_orin_attention_qwen2(image, prompt, general_prompt, model, processor, LAYERS, HEADS):

    """
    Compute relative attention scores for Qwen2.5VL.
    
    This function computes the relative attention scores between the input and general inputs
    for Qwen2.5VL. It first computes the attention scores for the input and general inputs, 
    and finally computes the relative attention scores.
    """

    conversation = [{"role": "user","content": [{"type": "image",},{"type": "text", "text": prompt},],}]
    # Preprocess the inputs
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

    inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(model.device, torch.bfloat16)
    att_shape = (inputs['image_grid_thw'][0, 1:] / 2).cpu().numpy().astype(int).tolist()
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1
    pos_end = inputs['input_ids'].tolist()[0].index(vision_end_token_id)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

       
        temp_result={}
        for LAYER in range(LAYERS):
            for HEAD in range(HEADS):
                att = outputs['attentions'][LAYER][0, HEAD, -1, pos:pos_end].to(torch.float32).detach().cpu().numpy()

  
                att_map = att.reshape(att_shape)
                temp_result[f"{LAYER}_{HEAD}"] = att_map
    
    return temp_result