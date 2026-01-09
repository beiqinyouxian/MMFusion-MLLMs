import torchvision.transforms.functional as TF
import numpy as np
import os
from scipy.ndimage import median_filter
from skimage.measure import block_reduce

from io import BytesIO
import base64

def encode_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str



def high_pass_filter(image, resolusion, km=7, kh=3, sigma=None, reduce=True, block=14):
    image = TF.resize(image, (resolusion, resolusion))
    image = TF.to_tensor(image).unsqueeze(0)
    l = TF.gaussian_blur(image, kernel_size=(kh, kh), sigma=sigma).squeeze().detach().cpu().numpy()
    h = image.squeeze().detach().cpu().numpy() - l
    h_brightness = np.sqrt(np.square(h).sum(axis=0))
    h_brightness = median_filter(h_brightness, size=km)
    if reduce:
        h_brightness = block_reduce(h_brightness, block_size=(block, block), func=np.sum)

    return h_brightness

def bbox_from_att_image_adaptive(att_map, image_size, bbox_size=336):
    ratios = [1, 1.2, 1.4, 1.6, 1.8, 2]

    max_att_poses = []
    differences = []
    block_nums = []

    for ratio in ratios:

        block_size = image_size[0] / att_map.shape[1], image_size[1] / att_map.shape[0]

        block_num = min(int(bbox_size*ratio/block_size[0]), att_map.shape[1]), min(int(bbox_size*ratio/block_size[1]), att_map.shape[0])
        if att_map.shape[1]-block_num[0] < 1 and att_map.shape[0]-block_num[1] < 1:
            if ratio == 1:
                return 0, 0, image_size[0], image_size[1]
            else:
                continue
        block_nums.append((block_num[0], block_num[1]))
        
        sliding_att = np.zeros((att_map.shape[0]-block_num[1]+1, att_map.shape[1]-block_num[0]+1))
        max_att = -np.inf
        max_att_pos = (0, 0)

        for x in range(att_map.shape[1]-block_num[0]+1): 
            for y in range(att_map.shape[0]-block_num[1]+1): 
                att = att_map[y:y+block_num[1], x:x+block_num[0]].sum()
                sliding_att[y, x] = att
                if att > max_att:
                    max_att = att
                    max_att_pos = (x, y)
        
        adjcent_atts = []
        if max_att_pos[0] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]-1])
        if max_att_pos[0] < sliding_att.shape[1]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]+1])
        if max_att_pos[1] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1]-1, max_att_pos[0]])
        if max_att_pos[1] < sliding_att.shape[0]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1]+1, max_att_pos[0]])
        difference = (max_att - np.mean(adjcent_atts)) / (block_num[0] * block_num[1])
        differences.append(difference)
        max_att_poses.append(max_att_pos)
    max_att_pos = max_att_poses[np.argmax(differences)]
    block_num = block_nums[np.argmax(differences)]
    selected_bbox_size = bbox_size * ratios[np.argmax(differences)]
    
    x_center = int(max_att_pos[0] * block_size[0] + block_size[0] * block_num[0] / 2)
    y_center = int(max_att_pos[1] * block_size[1] + block_size[1] * block_num[1] / 2)
    
    x_center = selected_bbox_size//2 if x_center < selected_bbox_size//2 else x_center
    y_center = selected_bbox_size//2 if y_center < selected_bbox_size//2 else y_center
    x_center = image_size[0] - selected_bbox_size//2 if x_center > image_size[0] - selected_bbox_size//2 else x_center
    y_center = image_size[1] - selected_bbox_size//2 if y_center > image_size[1] - selected_bbox_size//2 else y_center

    x1 = max(0, x_center - selected_bbox_size//2)
    y1 = max(0, y_center - selected_bbox_size//2)
    x2 = min(image_size[0], x_center + selected_bbox_size//2)
    y2 = min(image_size[1], y_center + selected_bbox_size//2)

    return x1, y1, x2, y2

def high_res_split_threshold(image, res_threshold=512):
    vertical_split = int(np.ceil(image.size[1] / res_threshold))
    horizontal_split = int(vertical_split * image.size[0] / image.size[1])

    split_num = (horizontal_split, vertical_split)
    split_size = int(np.ceil(image.size[0] / split_num[0])), int(np.ceil(image.size[1] / split_num[1]))
    
    split_images = []
    for j in range(split_num[1]):
        for i in range(split_num[0]):
            split_image = image.crop((i*split_size[0], j*split_size[1], (i+1)*split_size[0], (j+1)*split_size[1]))
            split_images.append(split_image)
    
    return split_images, vertical_split, horizontal_split

from PIL import Image
def resize_to_square(image,image_size):
    output_size=max(image_size)
    image.thumbnail((output_size, output_size))
    square_img = Image.new('RGB', (output_size, output_size), (255, 255, 255))
    square_img.paste(image, ((output_size - image.width) // 2, (output_size - image.height) // 2))
    return square_img

def square_to_orin(image, image_size):
    if image_size[0]>=image_size[1]:
        side_cut=int((image_size[0]-image_size[1])/2)
        cropped_img = image.crop((0, side_cut, image_size[0],image_size[0]-side_cut))
    else:
        side_cut=int((image_size[1]-image_size[0])/2)
        cropped_img = image.crop((side_cut, 0, image_size[1]-side_cut, image_size[1]))
    return cropped_img

def square_array_to_orin(array, image_size):
    if image_size[0]==image_size[1]:
        return array
    
    if  image_size[0]>image_size[1]:
        if image_size[0]%2 ==0 and image_size[1]%2 ==0:
            side_cut=int((image_size[0]-image_size[1])/2)
            cropped_arr = array[side_cut:-side_cut, :]

        elif image_size[0]%2 != 0 and image_size[1]%2 != 0:
            side_cut=int((image_size[0]-image_size[1])/2)
            cropped_arr = array[side_cut:-side_cut, :]
        else:
            side_cut=int((image_size[0]-image_size[1])/2)
            if side_cut==0:
                cropped_arr = array[1:, :] 
            else:
                cropped_arr = array[side_cut+1:-side_cut, :]
    else:
        if image_size[0]%2 ==0 and image_size[1]%2 ==0:
            side_cut=int((image_size[1]-image_size[0])/2)
            cropped_arr = array[:, side_cut:-side_cut]
        elif image_size[0]%2 != 0 and image_size[1]%2 != 0:
            side_cut=int((image_size[1]-image_size[0])/2)
            cropped_arr = array[:, side_cut:-side_cut]
        else:
            side_cut=int((image_size[1]-image_size[0])/2)
            if side_cut==0:
                cropped_arr = array[:, 1:]
            else:
                cropped_arr = array[:, side_cut+1:-side_cut]
    if cropped_arr.shape[0]!=image_size[0] or cropped_arr.shape[1]!=image_size[1]:
        if cropped_arr.shape[0]>image_size[1]:
            cropped_arr=cropped_arr[:image_size[1],:]
        if cropped_arr.shape[1]>image_size[0]:
            cropped_arr=cropped_arr[:,:image_size[0]]
        else:
            cropped_arr=cropped_arr[:image_size[1],:image_size[0]]
    return cropped_arr


def resize_to_pil_size(arr, target_size):
    resized_arr = np.resize(arr, (target_size[1], target_size[0]))
    return resized_arr

from skimage.transform import resize
def resize_with_interpolation(arr, target_size):
    resized_arr = resize(arr, (target_size[1], target_size[0]), anti_aliasing=True)
    return resized_arr

from scipy.ndimage import gaussian_filter
def high_res(map_func, image, prompt, general_prompt, model, processor, LAYER=None, HEAD=None, res_threshold=1024):
    image=resize_to_square(image,image.size)
    split_images, num_vertical_split, num_horizontal_split = high_res_split_threshold(image,res_threshold)
    att_maps = []
    for split_image in split_images:
        if LAYER is None or HEAD is None:
            att_map = map_func(split_image, prompt, general_prompt, model, processor)
        else:
            att_map = map_func(split_image, prompt, general_prompt, model, processor, LAYER, HEAD)
        att_map = resize_with_interpolation(att_map,split_image.size)
        att_maps.append(att_map)
    block_att = np.block([att_maps[j:j+num_horizontal_split] for j in range(0, num_horizontal_split * num_vertical_split, num_horizontal_split)])
    block_att = gaussian_filter(block_att, sigma=1, mode='reflect')
    return block_att


def min_max_scale(array):
    if array.size == 0:
        print(f"警告：生成的注意力图为空，数组内容: {array}")
        return None
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val - min_val == 0:
        return np.zeros_like(array)
    else:
        return (array - min_val) / (max_val - min_val)

def composite_attn_map(atten_maps):
    if type(atten_maps) == dict:
        atten_map = np.sum(list(atten_maps.values()), axis=0)/len(atten_maps)
    else:
        atten_map = np.sum(atten_maps, axis=0)/len(atten_maps)
    if atten_map.size == 0:
        print("Input array is empty")
    att_map = min_max_scale(atten_map)
    return att_map

class Evaluation_processor:
    def __init__(self, ref_map, att_map):
        self.ref_map = ref_map
        self.att_map = att_map
        self._validate_inputs()
    
    def _validate_inputs(self):
        if self.ref_map.ndim != 2 or self.att_map.ndim != 2:
            raise ValueError("输入必须是2D numpy数组")
        
        if not np.all(np.logical_or(self.ref_map == 0, self.ref_map == 1)):
            raise ValueError("ref_map必须只包含0和1")
            
        if not np.all(np.logical_or(self.att_map == 0, self.att_map == 1)):
            raise ValueError("att_map必须只包含0和1")
    def calculate_Entropy(self, attention_map):
        mul = np.multiply(self.ref_map, attention_map)
        sum_mul = np.sum(mul)
        sum_att = np.sum(attention_map)
        if sum_att == 0:
            return 0
        result = sum_mul/sum_att
        return result
    def calculate_metrics(self):
        tp = np.sum(np.logical_and(self.ref_map == 1, self.att_map == 1))
        fp = np.sum(np.logical_and(self.ref_map == 0, self.att_map == 1))
        fn = np.sum(np.logical_and(self.ref_map == 1, self.att_map == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return [precision,recall,f1]

from auto_threshold import *

def norm_res(map_func, image, prompt, general_prompt, model, processor, label, LAYERS, HEADS):

    image_square=resize_to_square(image,image.size)
    att_maps = map_func(image_square, prompt, general_prompt, model, processor, LAYERS, HEADS)
    temp_result={}
    eval_result={}
    for key, value in att_maps.items():
        att_item = resize_with_interpolation(value,image_square.size)
        block_att = gaussian_filter(att_item, sigma=1, mode='reflect')

        right_att_map = square_array_to_orin(block_att,image.size)
        atten_map = auto_otsu(min_max_scale(right_att_map))
        temp_result[f"{key}"]  = atten_map
        eval_processor = Evaluation_processor(label, atten_map)
        ce = eval_processor.calculate_Entropy(right_att_map)
        eval_result[f"{key}"]  = (eval_processor.calculate_metrics()) + [ce]
    return temp_result,eval_result

def specific_norm_res(map_func, image, prompt, general_prompt, model, processor, label, LAYERS, HEADS, largest_items):

    image_square=resize_to_square(image,image.size)
    att_maps = map_func(image_square, prompt, general_prompt, model, processor, LAYERS, HEADS)
 
    att_maps_list=[]
    for key, value in att_maps.items():
        if f'{key}' in largest_items:
            att_item = resize_with_interpolation(value,image_square.size)
            block_att = gaussian_filter(att_item, sigma=1, mode='reflect')


            right_att_map = square_array_to_orin(block_att,image.size)
            att_map = min_max_scale(right_att_map)
            att_maps_list.append(att_map)
        else:
            continue
    
    attention_map=composite_attn_map(att_maps_list)
   
    return attention_map









    



   

        

