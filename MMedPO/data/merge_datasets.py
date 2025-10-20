#!/usr/bin/env python3
"""
合并三个TIE DPO数据集并重新编号
"""

import json
import os
from typing import List, Dict, Any

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """加载JSON数据集文件"""
    print(f"正在加载: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"加载完成: {len(data)} 条记录")
    return data

def merge_and_reindex_datasets(dataset_files: List[str], output_file: str) -> None:
    """合并多个数据集并重新编号"""
    
    # 加载所有数据集
    all_data = []
    dataset_info = []
    
    for i, file_path in enumerate(dataset_files, 1):
        if not os.path.exists(file_path):
            print(f"警告: 文件不存在 {file_path}")
            continue
            
        dataset = load_dataset(file_path)
        dataset_name = f"method{i}"
        
        # 记录数据集信息
        dataset_info.append({
            'name': dataset_name,
            'file': os.path.basename(file_path),
            'count': len(dataset),
            'start_index': len(all_data),
            'end_index': len(all_data) + len(dataset) - 1
        })
        
        # 添加数据集来源标识并合并
        for entry in dataset:
            # 为每个条目添加数据集来源信息
            entry['dataset_source'] = dataset_name
            all_data.append(entry)
    
    # 重新编号所有条目
    print(f"\n开始重新编号 {len(all_data)} 条记录...")
    for i, entry in enumerate(all_data):
        # 生成新的ID格式: 普通数字编号
        entry['id'] = i + 1
    
    # 保存合并后的数据集
    print(f"正在保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    # 打印合并统计信息
    print("\n=== 合并统计信息 ===")
    total_count = 0
    for info in dataset_info:
        print(f"{info['name']}: {info['count']} 条记录 (索引 {info['start_index']+1}-{info['end_index']+1})")
        total_count += info['count']
    
    print(f"\n总计: {total_count} 条记录")
    print(f"输出文件: {output_file}")
    print(f"文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

def main():
    # 定义输入文件路径
    dataset_files = [
        "./tie_dpo_dataset_method1_vqa_rad.json",
        "./tie_dpo_dataset_method2_vqa_rad.json", 
    ]
    
    # 定义输出文件路径
    output_file = "./tie_dpo_dataset_combined_1+2.json"
    
    print("=== TIE DPO数据集合并工具 ===")
    print(f"输入文件:")
    for i, file_path in enumerate(dataset_files, 1):
        print(f"  Method {i}: {file_path}")
    print(f"输出文件: {output_file}")
    print()
    
    # 执行合并
    merge_and_reindex_datasets(dataset_files, output_file)
    
    print("\n合并完成!")

if __name__ == "__main__":
    main()