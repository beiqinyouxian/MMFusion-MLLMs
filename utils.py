import torchvision.transforms.functional as TF
import numpy as np
import os
from scipy.ndimage import median_filter
from skimage.measure import block_reduce

from io import BytesIO
import base64

def encode_base64(image):
    """
    Encodes a PIL image to a base64 string.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str



def high_pass_filter(image, resolusion, km=7, kh=3, reduce=True):
    """
    Applies a high-pass filter to an image to highlight edges and fine details.
    
    This function resizes the image, applies a Gaussian blur to create a low-frequency version,
    subtracts it from the original to get high-frequency components, and then applies median filtering.
    
    Args:
        image: Input PIL image
        resolusion: Target resolution to resize the image to
        km: Kernel size for median filtering (default: 7)
        kh: Kernel size for Gaussian blur (default: 3)
        reduce: Whether to reduce the output size using block reduction (default: True)
        
    Returns:
        h_brightness: A 2D numpy array representing the high-frequency components of the image
    """

    image = TF.resize(image, (resolusion, resolusion))
    image = TF.to_tensor(image).unsqueeze(0)
    l = TF.gaussian_blur(image, kernel_size=(kh, kh)).squeeze().detach().cpu().numpy()
    h = image.squeeze().detach().cpu().numpy() - l
    h_brightness = np.sqrt(np.square(h).sum(axis=0))
    h_brightness = median_filter(h_brightness, size=km)
    if reduce:
        h_brightness = block_reduce(h_brightness, block_size=(14, 14), func=np.sum)

    return h_brightness

def bbox_from_att_image_adaptive(att_map, image_size, bbox_size=336):
    """
    Generates an adaptive bounding box for original image from an attention map.
    
    This function finds the region with the highest attention in the attention map
    and creates a bounding box around it. It tries different crop ratios and selects
    the one that produces the sharpest attention difference.
    
    Args:
        att_map: A 2D numpy array representing the attention map (e.g., 24x24 for LLaVA or 16x16 for BLIP)
        image_size: Tuple of (width, height) of the original image
        bbox_size: Base size for the bounding box (default: 336)
        
    Returns:
        tuple: (x1, y1, x2, y2) coordinates of the bounding box in the original image
    """

    # the ratios corresponds to the bounding box we are going to crop the image
    ratios = [1, 1.2, 1.4, 1.6, 1.8, 2]

    max_att_poses = []
    differences = []
    block_nums = []

    for ratio in ratios:
        # perform a bbox_size*r width and bbox_size*r height crop, where bbox_size is the size of the model's original image input resolution. (336 for LLaVA, 224 for BLIP)

        # the size of each block in the attention map, in the original image
        block_size = image_size[0] / att_map.shape[1], image_size[1] / att_map.shape[0]

        # if I want a bbox_size*r width and bbox_size*r height crop from the original image, the number of blocks I need (x, y)
        block_num = min(int(bbox_size*ratio/block_size[0]), att_map.shape[1]), min(int(bbox_size*ratio/block_size[1]), att_map.shape[0])
        if att_map.shape[1]-block_num[0] < 1 and att_map.shape[0]-block_num[1] < 1:
            if ratio == 1:
                return 0, 0, image_size[0], image_size[1]
            else:
                continue
        block_nums.append((block_num[0], block_num[1]))
        
        # attention aggregation map
        sliding_att = np.zeros((att_map.shape[0]-block_num[1]+1, att_map.shape[1]-block_num[0]+1))
        max_att = -np.inf
        max_att_pos = (0, 0)

        # sliding window to find the block with the highest attention
        for x in range(att_map.shape[1]-block_num[0]+1): 
            for y in range(att_map.shape[0]-block_num[1]+1): 
                att = att_map[y:y+block_num[1], x:x+block_num[0]].sum()
                sliding_att[y, x] = att
                if att > max_att:
                    max_att = att
                    max_att_pos = (x, y)
        
        # we have the position of max attention, we can calculate the difference between the max attention and the average of its adjacent attentions, to see if it is sharp enough, the more difference, the sharper
        # we choose the best ratio r according to their attention difference
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
    """
    Splits a high-resolution image into smaller patches.
    
    This function divides a large image into smaller patches to process them individually,
    which is useful for handling high-resolution images that might be too large for direct processing.
    
    Args:
        image: Input PIL image
        res_threshold: Maximum resolution threshold before splitting (default: 1024)
        
    Returns:
        tuple: (split_images, vertical_split, horizontal_split)
            - split_images: List of PIL image patches
            - vertical_split: Number of vertical splits
            - horizontal_split: Number of horizontal splits
    """

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
# 原图缩放为正方形
def resize_to_square(image,image_size):
    output_size=max(image_size)
    # 按最长边等比缩放
    image.thumbnail((output_size, output_size))
    # 创建正方形画布（背景可选）
    square_img = Image.new('RGB', (output_size, output_size), (255, 255, 255))
    # # 居中粘贴缩放后的图片
    square_img.paste(image, ((output_size - image.width) // 2, (output_size - image.height) // 2))
    # square_img.save('output.jpg')
    return square_img

def square_to_orin(image, image_size):
    #image_size(102,768)
    if image_size[0]>=image_size[1]:
        side_cut=int((image_size[0]-image_size[1])/2)
        cropped_img = image.crop((0, side_cut, image_size[0],image_size[0]-side_cut))
    else:
        side_cut=int((image_size[1]-image_size[0])/2)
        cropped_img = image.crop((side_cut, 0, image_size[1]-side_cut, image_size[1]))
    return cropped_img

def square_array_to_orin(array, image_size):
    #image_size(1024,768)
    if image_size[0]==image_size[1]:
        return array
    
    if  image_size[0]>image_size[1]:
        if image_size[0]%2 ==0 and image_size[1]%2 ==0:
            side_cut=int((image_size[0]-image_size[1])/2)
            cropped_arr = array[side_cut:-side_cut, :]  # 裁剪掉上下各 128 行

        elif image_size[0]%2 != 0 and image_size[1]%2 != 0:
            side_cut=int((image_size[0]-image_size[1])/2)
            cropped_arr = array[side_cut:-side_cut, :]  # 裁剪掉上下各 128 行
        else:
            side_cut=int((image_size[0]-image_size[1])/2)
            if side_cut==0:
                cropped_arr = array[1:, :] 
            else:
                cropped_arr = array[side_cut+1:-side_cut, :]  # 裁剪掉上下各 128 行
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
    # 为防止出现原图边长为奇数情况，导致裁剪后边长为偶数，导致最后结果不一致
    if cropped_arr.shape[0]!=image_size[0] or cropped_arr.shape[1]!=image_size[1]:
        if cropped_arr.shape[0]>image_size[1]:
            cropped_arr=cropped_arr[:image_size[1],:]
        if cropped_arr.shape[1]>image_size[0]:
            cropped_arr=cropped_arr[:,:image_size[0]]
        else:
            cropped_arr=cropped_arr[:image_size[1],:image_size[0]]
        # print("cropped_arr.shape",cropped_arr.shape)
        # print("image_size",image_size)
    return cropped_arr

#非差值缩放返回原图

# 假设arr是2D NumPy数组，target_size是目标PIL图片的尺寸（width, height）
def resize_to_pil_size(arr, target_size):
    # 调整数组大小（注意：numpy.resize会按顺序填充数据，可能失真）
    resized_arr = np.resize(arr, (target_size[1], target_size[0]))  # (height, width)
    return resized_arr

#差值缩放返回原图
from skimage.transform import resize
def resize_with_interpolation(arr, target_size):
    # 使用skimage的resize（默认插值为双线性）
    # length=min(target_size[1], target_size[0])
    resized_arr = resize(arr, (target_size[1], target_size[0]), anti_aliasing=True)
    # 转换为0-255整数并确保类型为uint8
    # resized_arr = (resized_arr * 255).astype(np.uint8)
    return resized_arr

from scipy.ndimage import gaussian_filter
def high_res(map_func, image, prompt, general_prompt, model, processor, LAYER=None, HEAD=None, res_threshold=1024):
    """
    Applies an attention mapping function to high-resolution images by splitting and recombining.
    
    This function splits a high-resolution image into smaller patches, applies the specified
    attention mapping function to each patch, and then recombines the results into a single
    attention map.
    
    Args:
        map_func: The attention mapping function to apply to each patch
        image: Input PIL image
        prompt: Text prompt for the attention function
        general_prompt: General text prompt for baseline comparison
        model: Model instance (LLaVA or BLIP)
        processor: Processor for the corresponding model
        
    Returns:
        block_att: A 2D numpy array representing the combined attention map for the entire image
    """
    
    image=resize_to_square(image,image.size)
    split_images, num_vertical_split, num_horizontal_split = high_res_split_threshold(image,res_threshold)
    att_maps = []
    for split_image in split_images:
        if LAYER is None or HEAD is None:
            att_map = map_func(split_image, prompt, general_prompt, model, processor)
        else:
            att_map = map_func(split_image, prompt, general_prompt, model, processor, LAYER, HEAD)
        # att_map = att_map / att_map.mean()
        # 差值缩放
        att_map = resize_with_interpolation(att_map,split_image.size)
        att_maps.append(att_map)
    block_att = np.block([att_maps[j:j+num_horizontal_split] for j in range(0, num_horizontal_split * num_vertical_split, num_horizontal_split)])
    # 高斯模糊
    block_att = gaussian_filter(block_att, sigma=1, mode='reflect')
    return block_att


def min_max_scale(array):
     # 检查array是否为空
    if array.size == 0:
        print(f"警告：生成的注意力图为空，数组内容: {array}")
        return None
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val - min_val == 0:  # 全0或全相同值
        return np.zeros_like(array)
    else:
        return (array - min_val) / (max_val - min_val)

def composite_attn_map(atten_maps): # 将一个注意力map列表所有位置对应相加求和，然后再归一化
    if type(atten_maps) == dict:
        atten_map = np.sum(list(atten_maps.values()), axis=0)/len(atten_maps)
    else:
        atten_map = np.sum(atten_maps, axis=0)/len(atten_maps)
    if atten_map.size == 0:
        print("Input array is empty")
    # print("Input array shape:", atten_map.shape)  # 检查形状
    att_map = min_max_scale(atten_map)
    return att_map

class Evaluation_processor:
    def __init__(self, ref_map, att_map):
        """
        初始化评估处理器
        :param ref_map: 参考标注图(2D numpy array), 值为0或1
        :param att_map: 预测关注图(2D numpy array), 值为0或1
        """
        self.ref_map = ref_map
        self.att_map = att_map
        self._validate_inputs()
    
    def _validate_inputs(self):
        """验证输入是否为2D numpy array且值仅为0或1"""
        if self.ref_map.ndim != 2 or self.att_map.ndim != 2:
            raise ValueError("输入必须是2D numpy数组")
        
        if not np.all(np.logical_or(self.ref_map == 0, self.ref_map == 1)):
            raise ValueError("ref_map必须只包含0和1")
            
        if not np.all(np.logical_or(self.att_map == 0, self.att_map == 1)):
            raise ValueError("att_map必须只包含0和1")
    # 交叉熵损失
    def calculate_Entropy(self, attention_map):
        mul = np.multiply(self.ref_map, attention_map)
        sum_mul = np.sum(mul)
        sum_att = np.sum(attention_map)
        if sum_att == 0:
            return 0
        result = sum_mul/sum_att
        return result
    def calculate_metrics(self):
        """计算精确率、召回率和F1指标"""
        # 计算混淆矩阵组件
        tp = np.sum(np.logical_and(self.ref_map == 1, self.att_map == 1))
        fp = np.sum(np.logical_and(self.ref_map == 0, self.att_map == 1))
        fn = np.sum(np.logical_and(self.ref_map == 1, self.att_map == 0))
        
        # 计算精确率、召回率和F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # return {
        #     'precision': precision,
        #     'recall': recall,
        #     'f1': f1
        # }
        return [precision,recall,f1]

from auto_threshold import *

def norm_res(map_func, image, prompt, general_prompt, model, processor, label, LAYERS, HEADS):

    image_square=resize_to_square(image,image.size)
    att_maps = map_func(image_square, prompt, general_prompt, model, processor, LAYERS, HEADS)
    temp_result={}
    eval_result={}
    for key, value in att_maps.items():
        # 插值缩放
        att_item = resize_with_interpolation(value,image_square.size)
        # 高斯模糊
        block_att = gaussian_filter(att_item, sigma=1, mode='reflect')

        right_att_map = square_array_to_orin(block_att,image.size)
        atten_map = auto_otsu(min_max_scale(right_att_map))
        temp_result[f"{key}"]  = atten_map
        # -------------------------------------------
        # temp_result = {} # Avoid out of memory
        # -------------------------------------------
        eval_processor = Evaluation_processor(label, atten_map)
        # 计算评估指标
        # eval_result[f"{key}"]  = (eval_processor.calculate_metrics()).append(eval_processor.calculate_Entropy(right_att_map))
        ce = eval_processor.calculate_Entropy(right_att_map)
        eval_result[f"{key}"]  = (eval_processor.calculate_metrics()) + [ce]
        # print(eval_result[f"{key}"])        
    return temp_result,eval_result

def specific_norm_res(map_func, image, prompt, general_prompt, model, processor, label, LAYERS, HEADS, largest_items):

    image_square=resize_to_square(image,image.size)
    att_maps = map_func(image_square, prompt, general_prompt, model, processor, LAYERS, HEADS)
 
    att_maps_list=[]
    for key, value in att_maps.items():
        if f'{key}' in largest_items:
            # 插值缩放
            att_item = resize_with_interpolation(value,image_square.size)
            # 高斯模糊
            block_att = gaussian_filter(att_item, sigma=1, mode='reflect')

            right_att_map = square_array_to_orin(block_att,image.size)
            # 进行归一化 
            att_map = min_max_scale(right_att_map)
            att_maps_list.append(att_map)
        else:
            continue
    
    attention_map=composite_attn_map(att_maps_list)
   
    return attention_map

# import json
# def load_json_files(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return json.load(f)


# def rank_guassian_filter(img, kernel_size=3):
#     """
#     Apply a rank-based Gaussian-weighted filter for robust activation map denoising.

#     Parameters:
#     img : np.ndarray
#         Input 2D grayscale image.
#     kernel_size : int
#         Size of the square kernel (must be odd).

#     Returns:
#     filtered_img : np.ndarray
#         Denoised image after applying the Gaussian weighted rank filter.

#     Note:
#         The sigma (std) of is refined to coefficient of variation for robust results
#     """

#     filtered_img = np.zeros_like(img)
#     pad_width = kernel_size // 2
#     padded_img = np.pad(img, pad_width, mode='reflect')
#     ax = np.array(range(kernel_size ** 2)) - kernel_size ** 2 // 2

#     for i in range(pad_width, img.shape[0] + pad_width):
#         for j in range(pad_width, img.shape[1] + pad_width):
#             window = padded_img[i - pad_width:i + pad_width + 1,
#                                 j - pad_width:j + pad_width + 1]

#             sorted_window = np.sort(window.flatten())
#             mean = sorted_window.mean()
#             if mean > 0:
#                 sigma = sorted_window.std() / mean # std -> cov
#                 kernel = np.exp(-(ax**2) / (2 * sigma**2))
#                 kernel = kernel / np.sum(kernel)
#                 value = (sorted_window * kernel).sum()
#             else:
#                 value = 0
#             filtered_img[i - pad_width, j - pad_width] = value
    
#     return filtered_img


# import heapq

# def norm_weight_res(map_func, image, prompt, general_prompt, model, processor, LAYERS, HEADS, head_num, param_path):
   
#     param=load_json_files(param_path)
#     largest_items = heapq.nlargest(head_num, param.items(), key=lambda item: item[1])
#     largest_items = dict(largest_items)
#     # print(largest_items)
#     image=resize_to_square(image,image.size)
#     att_maps = map_func(image, prompt, general_prompt, model, processor, LAYERS, HEADS)
#     temp_result=[]
#     for key, value in att_maps.items():
#         if f'{key}' in largest_items:
#             # 插值缩放
#             # print("yes")
#             att_item = resize_with_interpolation(value,image.size)
#             # 高斯模糊
#             # block_att = gaussian_filter(att_item, sigma=1, mode='reflect')
#             block_att = rank_guassian_filter(att_item)

#             temp_result.append(block_att*largest_items[f'{key}']) 
#         else:
#             continue
        
#     stacked = np.stack(temp_result, axis=0)  # axis=0表示在第一个维度堆叠
#     # 此时stacked的形状为 (3, 2, 3)

#     # 步骤2：在第一个维度（axis=0）上计算平均值
#     mean_arr = np.mean(stacked, axis=0) 

#     # print(mean_arr.shape)
#     return mean_arr