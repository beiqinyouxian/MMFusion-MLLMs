#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本功能：为DN-FMB数据集中的图像文件添加白天(D)或黑天(N)后缀
遍历 /data/dataset/DN-FMB_dataset 下的 test 和 train 文件夹中的 vi 文件夹
根据图像亮度自动判断白天/黑天，并重命名文件
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def calculate_brightness(image_path):
    """
    计算图像的平均亮度
    
    参数:
        image_path: 图像文件路径
    
    返回:
        平均亮度值 (0-255)
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 计算平均亮度
        brightness = np.mean(gray)
        return brightness
    except Exception as e:
        print(f"读取图像 {image_path} 时出错: {e}")
        return None


def is_day_image(image_path, threshold=100):
    """
    判断图像是白天还是黑天
    
    参数:
        image_path: 图像文件路径
        threshold: 亮度阈值，大于此值认为是白天，默认100
    
    返回:
        True: 白天, False: 黑天
    """
    brightness = calculate_brightness(image_path)
    if brightness is None:
        return None
    return brightness > threshold


def rename_image_with_suffix(image_path, suffix):
    """
    为图像文件名添加后缀（在扩展名之前）
    
    参数:
        image_path: 原始图像路径
        suffix: 要添加的后缀 ('D' 或 'N')
    
    返回:
        新的文件路径，如果已存在后缀则返回None
    """
    path = Path(image_path)
    # 检查文件名是否已经包含后缀
    stem = path.stem
    if stem.endswith('D') or stem.endswith('N'):
        return None
    
    # 构建新文件名
    new_stem = f"{stem}{suffix}"
    new_path = path.parent / f"{new_stem}{path.suffix}"
    
    # 如果新文件名已存在，跳过
    if new_path.exists():
        print(f"警告: {new_path} 已存在，跳过重命名 {image_path}")
        return None
    
    return new_path


def process_vi_folder(vi_folder_path, brightness_threshold=100, dry_run=False):
    """
    处理单个vi文件夹中的所有图像
    
    参数:
        vi_folder_path: vi文件夹路径
        brightness_threshold: 亮度阈值
        dry_run: 如果为True，只显示将要执行的操作，不实际重命名
    """
    vi_path = Path(vi_folder_path)
    if not vi_path.exists():
        print(f"警告: 文件夹 {vi_folder_path} 不存在，跳过")
        return
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    # 获取所有图像文件
    image_files = [f for f in vi_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"在 {vi_folder_path} 中未找到图像文件")
        return
    
    print(f"\n处理文件夹: {vi_folder_path}")
    print(f"找到 {len(image_files)} 个图像文件")
    
    day_count = 0
    night_count = 0
    skipped_count = 0
    
    for img_file in tqdm(image_files, desc="处理图像"):
        # 检查是否已有后缀
        if img_file.stem.endswith('D') or img_file.stem.endswith('N'):
            skipped_count += 1
            continue
        
        # 判断白天/黑天
        is_day = is_day_image(img_file, brightness_threshold)
        
        if is_day is None:
            skipped_count += 1
            continue
        
        suffix = 'D' if is_day else 'N'
        new_path = rename_image_with_suffix(img_file, suffix)
        
        if new_path is None:
            skipped_count += 1
            continue
        
        if dry_run:
            print(f"  [预览] {img_file.name} -> {new_path.name} ({'白天' if is_day else '黑天'})")
        else:
            try:
                img_file.rename(new_path)
                if is_day:
                    day_count += 1
                else:
                    night_count += 1
            except Exception as e:
                print(f"重命名失败 {img_file.name}: {e}")
                skipped_count += 1
    
    print(f"完成: 白天={day_count}, 黑天={night_count}, 跳过={skipped_count}")


def main():
    """
    主函数：遍历数据集文件夹并处理所有vi文件夹
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='为DN-FMB数据集中的图像文件添加白天(D)或黑天(N)后缀',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 预览模式（不实际重命名）
  python rename_day_night_images.py --dry-run
  
  # 执行重命名，使用默认亮度阈值100
  python rename_day_night_images.py
  
  # 自定义亮度阈值和数据集路径
  python rename_day_night_images.py --threshold 120 --dataset-path /path/to/dataset
        """
    )
    
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='/data/dataset/DN-FMB_dataset',
        help='数据集根目录路径（默认: /data/dataset/DN-FMB_dataset）'
    )
    
    parser.add_argument(
        '--threshold',
        type=int,
        default=100,
        help='亮度阈值，大于此值认为是白天图像（默认: 100）'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='预览模式，只显示将要执行的操作，不实际重命名文件'
    )
    
    parser.add_argument(
        '--folders',
        nargs='+',
        default=['test', 'train'],
        help='要处理的文件夹列表（默认: test train）'
    )
    
    args = parser.parse_args()
    
    base_path = Path(args.dataset_path)
    
    if not base_path.exists():
        print(f"错误: 数据集路径 {base_path} 不存在")
        return
    
    print("=" * 60)
    print("DN-FMB数据集图像重命名脚本")
    print("=" * 60)
    print(f"数据集路径: {base_path}")
    print(f"亮度阈值: {args.threshold}")
    print(f"处理文件夹: {', '.join(args.folders)}")
    print(f"模式: {'预览模式（不会实际重命名）' if args.dry_run else '执行模式（将实际重命名文件）'}")
    print("=" * 60)
    
    if not args.dry_run:
        response = input("\n确认要执行重命名操作吗？(yes/no): ")
        if response.lower() != 'yes':
            print("操作已取消")
            return
    
    # 遍历指定的文件夹
    for folder_name in args.folders:
        folder_path = base_path / folder_name
        if not folder_path.exists():
            print(f"警告: 文件夹 {folder_path} 不存在，跳过")
            continue
        
        # 查找vi文件夹
        vi_folder = folder_path / 'vi'
        if vi_folder.exists():
            process_vi_folder(vi_folder, args.threshold, args.dry_run)
        else:
            print(f"警告: {vi_folder} 不存在，跳过")
    
    print("\n" + "=" * 60)
    print("所有处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

