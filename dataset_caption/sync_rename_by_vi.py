#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本功能：根据vi文件夹下的图片命名，同步重命名ir和Segmentation_labels文件夹中的文件
遍历 /data/dataset/DN-FMB_dataset 下的 test 和 train 文件夹
根据vi文件夹中文件的后缀（D或N），为ir和Segmentation_labels文件夹中相同前缀的文件添加相同后缀
"""

import os
from pathlib import Path
from tqdm import tqdm
import re


def extract_prefix_and_suffix(filename):
    """
    从文件名中提取前缀和后缀（D或N）
    
    参数:
        filename: 文件名（不含路径）
    
    返回:
        (prefix, suffix) 元组，如果没有后缀则suffix为None
        例如: "image001D.jpg" -> ("image001", "D")
             "image001N.png" -> ("image001", "N")
             "image001.jpg" -> ("image001", None)
    """
    path = Path(filename)
    stem = path.stem  # 不含扩展名的文件名
    
    # 检查是否以D或N结尾
    if stem.endswith('D'):
        prefix = stem[:-1]  # 去掉最后的D
        suffix = 'D'
        return prefix, suffix
    elif stem.endswith('N'):
        prefix = stem[:-1]  # 去掉最后的N
        suffix = 'N'
        return prefix, suffix
    else:
        return stem, None


def find_matching_files(target_folder, prefix, exclude_suffix=None):
    """
    在目标文件夹中查找与给定前缀匹配的文件
    
    参数:
        target_folder: 目标文件夹路径
        prefix: 文件名前缀
        exclude_suffix: 要排除的后缀（如果文件已经有这个后缀，则跳过）
    
    返回:
        匹配的文件列表（Path对象）
    """
    target_path = Path(target_folder)
    if not target_path.exists():
        return []
    
    matching_files = []
    for file_path in target_path.iterdir():
        if not file_path.is_file():
            continue
        
        file_stem = file_path.stem
        # 如果文件已经有后缀，跳过
        if exclude_suffix and (file_stem.endswith('D') or file_stem.endswith('N')):
            continue
        
        # 检查前缀是否匹配
        if file_stem == prefix or file_stem.startswith(prefix):
            # 更精确的匹配：确保前缀后面是扩展名或下划线等分隔符
            # 或者完全匹配
            if file_stem == prefix:
                matching_files.append(file_path)
            elif file_stem.startswith(prefix + '_') or file_stem.startswith(prefix + '-'):
                # 如果前缀后面有分隔符，也认为是匹配的
                matching_files.append(file_path)
    
    return matching_files


def rename_file_with_suffix(file_path, suffix):
    """
    为文件添加后缀（在扩展名之前）
    
    参数:
        file_path: 原始文件路径
        suffix: 要添加的后缀 ('D' 或 'N')
    
    返回:
        新的文件路径，如果重命名失败则返回None
    """
    path = Path(file_path)
    new_stem = f"{path.stem}{suffix}"
    new_path = path.parent / f"{new_stem}{path.suffix}"
    
    # 如果新文件名已存在，跳过
    if new_path.exists():
        return None
    
    return new_path


def process_dataset_folder(dataset_folder_path, dry_run=False):
    """
    处理单个数据集文件夹（test或train）
    
    参数:
        dataset_folder_path: 数据集文件夹路径（包含vi, ir, Segmentation_labels子文件夹）
        dry_run: 如果为True，只显示将要执行的操作，不实际重命名
    """
    dataset_path = Path(dataset_folder_path)
    
    # 检查必要的文件夹
    vi_folder = dataset_path / 'vi'
    ir_folder = dataset_path / 'ir'
    labels_folder = dataset_path / 'Segmentation_labels'
    
    if not vi_folder.exists():
        print(f"警告: {vi_folder} 不存在，跳过")
        return
    
    # 获取vi文件夹中的所有文件
    vi_files = [f for f in vi_folder.iterdir() if f.is_file()]
    
    if not vi_files:
        print(f"警告: {vi_folder} 中没有文件，跳过")
        return
    
    print(f"\n处理文件夹: {dataset_folder_path}")
    print(f"VI文件夹中找到 {len(vi_files)} 个文件")
    
    # 统计信息
    ir_renamed = 0
    labels_renamed = 0
    ir_skipped = 0
    labels_skipped = 0
    no_match = 0
    
    # 处理每个vi文件
    for vi_file in tqdm(vi_files, desc="处理文件"):
        prefix, suffix = extract_prefix_and_suffix(vi_file.name)
        
        # 如果vi文件没有后缀，跳过
        if suffix is None:
            continue
        
        # 在ir文件夹中查找匹配的文件
        if ir_folder.exists():
            ir_matches = find_matching_files(ir_folder, prefix, exclude_suffix=suffix)
            for ir_file in ir_matches:
                new_ir_path = rename_file_with_suffix(ir_file, suffix)
                if new_ir_path is None:
                    ir_skipped += 1
                    if not dry_run:
                        print(f"  跳过IR: {ir_file.name} (目标文件已存在)")
                else:
                    if dry_run:
                        print(f"  [预览] IR: {ir_file.name} -> {new_ir_path.name}")
                    else:
                        try:
                            ir_file.rename(new_ir_path)
                            ir_renamed += 1
                        except Exception as e:
                            print(f"  错误: 重命名 {ir_file.name} 失败: {e}")
                            ir_skipped += 1
        
        # 在Segmentation_labels文件夹中查找匹配的文件
        if labels_folder.exists():
            labels_matches = find_matching_files(labels_folder, prefix, exclude_suffix=suffix)
            for label_file in labels_matches:
                new_label_path = rename_file_with_suffix(label_file, suffix)
                if new_label_path is None:
                    labels_skipped += 1
                    if not dry_run:
                        print(f"  跳过标签: {label_file.name} (目标文件已存在)")
                else:
                    if dry_run:
                        print(f"  [预览] 标签: {label_file.name} -> {new_label_path.name}")
                    else:
                        try:
                            label_file.rename(new_label_path)
                            labels_renamed += 1
                        except Exception as e:
                            print(f"  错误: 重命名 {label_file.name} 失败: {e}")
                            labels_skipped += 1
        
        # 检查是否找到了匹配的文件
        ir_found = ir_folder.exists() and len(find_matching_files(ir_folder, prefix, exclude_suffix=suffix)) > 0
        labels_found = labels_folder.exists() and len(find_matching_files(labels_folder, prefix, exclude_suffix=suffix)) > 0
        
        if not ir_found and not labels_found:
            no_match += 1
    
    print(f"\n完成统计:")
    print(f"  IR文件重命名: {ir_renamed}, 跳过: {ir_skipped}")
    print(f"  标签文件重命名: {labels_renamed}, 跳过: {labels_skipped}")
    if no_match > 0:
        print(f"  未找到匹配的文件: {no_match}")


def main():
    """
    主函数：遍历数据集文件夹并处理所有test和train文件夹
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='根据vi文件夹下的图片命名，同步重命名ir和Segmentation_labels文件夹中的文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 预览模式（不实际重命名）
  python sync_rename_by_vi.py --dry-run
  
  # 执行重命名
  python sync_rename_by_vi.py
  
  # 自定义数据集路径
  python sync_rename_by_vi.py --dataset-path /path/to/dataset
  
  # 只处理特定文件夹
  python sync_rename_by_vi.py --folders test
        """
    )
    
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='/data/dataset/DN-FMB_dataset',
        help='数据集根目录路径（默认: /data/dataset/DN-FMB_dataset）'
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
    print("根据VI文件夹同步重命名IR和标签文件脚本")
    print("=" * 60)
    print(f"数据集路径: {base_path}")
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
        
        process_dataset_folder(folder_path, args.dry_run)
    
    print("\n" + "=" * 60)
    print("所有处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

