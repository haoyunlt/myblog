#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用模块文档合并脚本
自动发现并合并所有项目的模块文档（概览、时序图、数据结构、API）
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Tuple, Dict, List

def extract_front_matter(content: str) -> Tuple[str, str]:
    """提取 front matter 和正文"""
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            return parts[1], parts[2].strip()
    return "", content

def merge_module_docs(posts_dir: Path, module_prefix: str, output_file: str, delete_originals: bool = True):
    """
    合并一个模块的四个文档
    
    Args:
        posts_dir: posts 目录路径
        module_prefix: 模块前缀
        output_file: 输出文件名
        delete_originals: 是否删除原始文件
    """
    # 四个文档的后缀（按推荐顺序）
    suffixes = ["-概览", "-时序图", "-数据结构", "-API", "-调用链分析"]
    
    contents = []
    files_to_delete = []
    found_count = 0
    
    front_matter = None
    
    for suffix in suffixes:
        file_path = posts_dir / f"{module_prefix}{suffix}.md"
        if not file_path.exists():
            continue
        
        found_count += 1
        print(f"  📖 读取: {file_path.name}")
        files_to_delete.append(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取 front matter 和正文
        fm, body = extract_front_matter(content)
        
        # 只保留第一个文件的 front matter
        if front_matter is None and fm:
            front_matter = fm
            # 移除 front matter 中的后缀标识
            front_matter = re.sub(r'title: "(.+?)(-概览|-时序图|-数据结构|-API|-调用链分析)"', 
                                 r'title: "\1"', front_matter)
        
        # 移除正文中的重复一级标题
        lines = body.split('\n')
        filtered_lines = []
        for line in lines:
            # 跳过与文件名相同的一级标题
            if line.startswith('# ') and any(suffix.replace('-', '') in line for suffix in suffixes):
                continue
            filtered_lines.append(line)
        
        body = '\n'.join(filtered_lines).strip()
        contents.append(body)
    
    if found_count < 2:
        print(f"  ⚠️  只找到 {found_count} 个文档，跳过合并")
        return False
    
    # 合并内容
    if front_matter:
        merged_content = f"---{front_matter}---\n\n" + "\n\n---\n\n".join(contents)
    else:
        merged_content = "\n\n---\n\n".join(contents)
    
    # 写入输出文件
    output_path = posts_dir / output_file
    print(f"  ✍️  写入: {output_path.name}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(merged_content)
    
    # 删除原始文件
    if delete_originals:
        for file_path in files_to_delete:
            print(f"  🗑️  删除: {file_path.name}")
            file_path.unlink()
    
    print(f"  ✅ 合并完成: {output_file} (合并了 {found_count} 个文档)")
    return True

def discover_modules(posts_dir: Path) -> Dict[str, List[Tuple[str, str]]]:
    """
    自动发现需要合并的模块
    
    Returns:
        字典 {项目名: [(模块前缀, 输出文件名), ...]}
    """
    modules = defaultdict(list)
    
    # 查找所有拆分的模块文档
    for file_path in posts_dir.glob("*.md"):
        filename = file_path.stem
        
        # 匹配模式：项目名-模块号-模块名-后缀
        # 例如：AutoGen-01-PythonCore-API
        pattern = r'^(.+?-\d+-[^-]+?)(-概览|-时序图|-数据结构|-API|-调用链分析)$'
        match = re.match(pattern, filename)
        
        if match:
            module_prefix = match.group(1)
            suffix = match.group(2)
            
            # 提取项目名
            project_match = re.match(r'^([^-]+)', module_prefix)
            if project_match:
                project_name = project_match.group(1)
                
                # 检查是否已添加
                output_file = f"{module_prefix}.md"
                if (module_prefix, output_file) not in modules[project_name]:
                    modules[project_name].append((module_prefix, output_file))
    
    return dict(modules)

def main():
    """主函数"""
    script_dir = Path(__file__).parent
    posts_dir = script_dir.parent / "content" / "posts"
    
    if not posts_dir.exists():
        print(f"❌ posts 目录不存在: {posts_dir}")
        return 1
    
    print("🔍 自动发现需要合并的模块...\n")
    
    modules = discover_modules(posts_dir)
    
    if not modules:
        print("✅ 没有发现需要合并的模块文档")
        return 0
    
    print(f"📚 发现 {len(modules)} 个项目需要合并\n")
    
    total_merged = 0
    
    for project_name, module_list in sorted(modules.items()):
        print(f"\n{'='*70}")
        print(f"📦 项目: {project_name}")
        print(f"{'='*70}\n")
        
        for module_prefix, output_file in sorted(module_list):
            print(f"🔧 处理模块: {module_prefix}")
            
            try:
                if merge_module_docs(posts_dir, module_prefix, output_file):
                    total_merged += 1
            except Exception as e:
                print(f"  ❌ 合并失败: {e}")
                continue
            
            print()
    
    print(f"\n{'='*70}")
    print(f"🎉 合并完成！")
    print(f"{'='*70}")
    print(f"✅ 成功合并: {total_merged} 个模块")
    print()
    
    return 0

if __name__ == "__main__":
    exit(main())

