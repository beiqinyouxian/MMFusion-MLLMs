
import cv2
import numpy as np

def constant_threshold(image, threshold=0.5):
    _, binary = cv2.threshold(image, threshold, 1, cv2.THRESH_BINARY)
    return binary

def auto_otsu(image):
    image_uint8 = (image * 255).astype(np.uint8)
    
    _, binary_uint8 = cv2.threshold(image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    binary = binary_uint8 / 255.0
    return binary

def auto_otsu_softmax(image):
    image_flat = image.flatten()
    exp_image = np.exp(image_flat - np.max(image_flat))  
    softmax_image = exp_image / np.sum(exp_image)
    softmax_image = softmax_image.reshape(image.shape)
    softmax_uint8 = (softmax_image * 255).astype(np.uint8)
    _, binary_uint8 = cv2.threshold(softmax_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = binary_uint8 / 255.0
    return binary

def adaptive_threshold(image):
    image_uint8 = (image * 255).astype(np.uint8)
    
    binary_uint8 = cv2.adaptiveThreshold(image_uint8, 255, 
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    
    binary = binary_uint8 / 255.0
    return binary

def local_otsu(image, block_size=64):
    h, w = image.shape
    result = np.zeros_like(image)
    
    image_uint8 = (image * 255).astype(np.uint8)
    result_uint8 = np.zeros_like(image_uint8)
    
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)
            
            block = image_uint8[y:y_end, x:x_end]
            if block.size == 0:
                continue
                
            _, block_binary = cv2.threshold(
                block, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            result_uint8[y:y_end, x:x_end] = block_binary
    
    result = result_uint8 / 255.0
    return result