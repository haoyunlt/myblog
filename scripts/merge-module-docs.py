#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并 Kubernetes 模块文档（概览、时序图、数据结构、API）
"""

import os
import sys
from pathlib import Path

def merge_module_docs(posts_dir: Path, module_prefix: str, output_file: str):
    """
    合并一个模块的四个文档
    
    Args:
        posts_dir: posts 目录路径
        module_prefix: 模块前缀，例如 "Kubernetes-01-API Server"
        output_file: 输出文件名
    """
    # 四个文档的后缀
    suffixes = ["-概览", "-时序图", "-数据结构", "-API"]
    
    # 读取所有文档内容
    contents = []
    for suffix in suffixes:
        file_path = posts_dir / f"{module_prefix}{suffix}.md"
        if not file_path.exists():
            print(f"⚠️  文件不存在: {file_path}")
            continue
        
        print(f"📖 读取: {file_path.name}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 移除开头的 front matter（如果有）
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                # 保留第一个文件的 front matter，其他文件只保留正文
                if len(contents) == 0:
                    # 第一个文件，保留 front matter
                    front_matter = parts[1]
                    body = parts[2].strip()
                    contents.append(f"---{front_matter}---\n\n{body}")
                else:
                    # 后续文件，只保留正文
                    body = parts[2].strip()
                    contents.append(body)
            else:
                contents.append(content.strip())
        else:
            contents.append(content.strip())
    
    # 合并内容
    merged_content = "\n\n---\n\n".join(contents)
    
    # 写入输出文件
    output_path = posts_dir / output_file
    print(f"✍️  写入: {output_path.name}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(merged_content)
    
    print(f"✅ 合并完成: {output_file}")
    return output_path

def main():
    """主函数"""
    # 获取 posts 目录
    script_dir = Path(__file__).parent
    posts_dir = script_dir.parent / "content" / "posts"
    
    if not posts_dir.exists():
        print(f"❌ posts 目录不存在: {posts_dir}")
        sys.exit(1)
    
    # 定义要合并的模块
    modules = [
        ("Kubernetes-01-API Server", "Kubernetes-01-API Server.md"),
        ("Kubernetes-02-Controller Manager", "Kubernetes-02-Controller Manager.md"),
        ("Kubernetes-03-Scheduler", "Kubernetes-03-Scheduler.md"),
        ("Kubernetes-04-Kubelet", "Kubernetes-04-Kubelet.md"),
        ("Kubernetes-05-Kube Proxy", "Kubernetes-05-Kube Proxy.md"),
        ("Kubernetes-06-Client Go", "Kubernetes-06-Client Go.md"),
    ]
    
    print("🚀 开始合并 Kubernetes 模块文档\n")
    
    for module_prefix, output_file in modules:
        print(f"\n{'='*60}")
        print(f"模块: {module_prefix}")
        print(f"{'='*60}")
        
        try:
            merge_module_docs(posts_dir, module_prefix, output_file)
        except Exception as e:
            print(f"❌ 合并失败: {e}")
            continue
    
    print("\n" + "="*60)
    print("🎉 所有模块合并完成！")
    print("="*60)

if __name__ == "__main__":
    main()

