#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
博客文章处理脚本
功能：
1. 为没有Hugo front matter的文章添加front matter
2. 删除文章中的目录部分
"""

import os
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse

class BlogPostProcessor:
    def __init__(self, posts_dir: str):
        self.posts_dir = Path(posts_dir)
        self.processed_count = 0
        self.error_count = 0
        
        # 目录模式匹配正则表达式
        self.toc_patterns = [
            # 标准目录格式
            (r'^## 目录\s*\n((?:\s*\d+\.\s*\[.*?\]\(#.*?\)\s*\n)*)', re.MULTILINE),
            # 文档结构概览
            (r'^## 📚 文档结构概览\s*\n(.*?)(?=^##)', re.MULTILINE | re.DOTALL),
            # 文档目录
            (r'^## 📖 文档目录\s*\n(.*?)(?=^##)', re.MULTILINE | re.DOTALL),
            # 表格形式的目录
            (r'^### 📖 文档目录\s*\n(.*?)(?=^##)', re.MULTILINE | re.DOTALL),
        ]
        
        # 分类映射
        self.category_mapping = {
            'python': ['Python'],
            'fastapi': ['FastAPI', 'Python框架'],
            'chatwoot': ['聊天助手'],
            'homeassistant': ['语音助手'],
            'livekit': ['语音助手'],
            'envoy': ['代理服务器'],
            'ray': ['分布式计算'],
            'mongodb': ['数据库'],
            'ceph': ['分布式存储'],
            'pulsar': ['消息队列'],
            'kubernetes': ['容器编排'],
            'kube': ['容器编排'],
            'vllm': ['AI推理'],
            'docker': ['容器化'],
            'eino': ['AI框架'],
        }
        
        # 标签映射
        self.tag_mapping = {
            'python': ['Python', '源码分析'],
            'fastapi': ['FastAPI', 'Python', 'Web框架', 'API'],
            'chatwoot': ['Chatwoot', 'Ruby', 'Rails', '客服系统'],
            'homeassistant': ['Home Assistant', 'Python', '智能家居', '自动化'],
            'livekit': ['LiveKit', 'WebRTC', '实时通信', '语音处理'],
            'envoy': ['Envoy', 'C++', '代理', '微服务', '负载均衡'],
            'ray': ['Ray', 'Python', '分布式计算', '机器学习'],
            'mongodb': ['MongoDB', 'C++', '数据库', 'NoSQL'],
            'ceph': ['Ceph', 'C++', '分布式存储', '对象存储'],
            'pulsar': ['Apache Pulsar', 'Java', '消息队列', '流处理'],
            'kubernetes': ['Kubernetes', 'Go', '容器编排', 'DevOps'],
            'vllm': ['vLLM', 'Python', 'AI推理', 'LLM'],
            'docker': ['Docker', 'Go', '容器化', 'DevOps'],
            'eino': ['Eino', 'Python', 'AI框架', 'LLM应用'],
        }

    def detect_category_and_tags(self, filename: str, title: str) -> Tuple[List[str], List[str]]:
        """根据文件名和标题检测分类和标签"""
        filename_lower = filename.lower()
        title_lower = title.lower()
        
        # 检测分类
        categories = []
        tags = []
        
        for key, category in self.category_mapping.items():
            if key in filename_lower or key in title_lower:
                categories.extend(category)
                if key in self.tag_mapping:
                    tags.extend(self.tag_mapping[key])
                break
        
        # 默认分类
        if not categories:
            categories = ['技术分析']
            tags = ['源码分析', '技术文档']
        
        # 根据特定关键词添加额外标签
        if '源码' in title or 'source' in title_lower:
            tags.append('源码分析')
        if '实战' in title or '最佳实践' in title:
            tags.append('最佳实践')
        if 'api' in title_lower:
            tags.append('API')
        if 'uml' in title_lower or '数据结构' in title:
            tags.append('架构设计')
        
        return list(set(categories)), list(set(tags))

    def extract_title_from_content(self, content: str) -> str:
        """从内容中提取标题"""
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        return "技术文档"

    def has_front_matter(self, content: str) -> bool:
        """检查文件是否已有front matter"""
        return content.strip().startswith('---')

    def remove_table_of_contents(self, content: str) -> str:
        """删除文章中的目录部分"""
        original_content = content
        
        for pattern, flags in self.toc_patterns:
            # 尝试匹配并删除目录
            match = re.search(pattern, content, flags)
            if match:
                # 删除整个匹配的部分
                content = re.sub(pattern, '', content, flags=flags)
                print(f"    ✓ 删除目录部分: {match.group(0)[:100]}...")
                break
        
        # 清理多余的空行
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        return content

    def generate_front_matter(self, title: str, filename: str) -> str:
        """生成Hugo front matter"""
        categories, tags = self.detect_category_and_tags(filename, title)
        
        # 生成描述
        description = f"{title}的深入技术分析文档"
        
        # 当前日期
        current_date = datetime.now().strftime('%Y-%m-%dT%H:%M:%S+08:00')
        
        front_matter = f"""---
title: "{title}"
date: {current_date}
draft: false
tags: {tags}
categories: {categories}
description: "{description}"
keywords: {tags}
author: "技术分析师"
weight: 1
---

"""
        return front_matter

    def process_file(self, file_path: Path) -> bool:
        """处理单个文件"""
        try:
            print(f"\n处理文件: {file_path.name}")
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            modified = False
            
            # 检查是否已有front matter
            if not self.has_front_matter(content):
                print("    ✓ 添加Hugo front matter")
                title = self.extract_title_from_content(content)
                front_matter = self.generate_front_matter(title, file_path.name)
                
                # 如果内容以#开头，移除第一个标题（避免重复）
                lines = content.split('\n')
                if lines and lines[0].strip().startswith('# '):
                    lines = lines[1:]
                    while lines and not lines[0].strip():  # 移除标题后的空行
                        lines = lines[1:]
                    content = '\n'.join(lines)
                
                content = front_matter + content
                modified = True
            else:
                print("    → 已有front matter，跳过添加")
            
            # 删除目录
            new_content = self.remove_table_of_contents(content)
            if new_content != content:
                content = new_content
                modified = True
            else:
                print("    → 未发现目录，跳过删除")
            
            # 保存修改后的内容
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"    ✅ 文件已更新")
                self.processed_count += 1
            else:
                print(f"    → 无需修改")
            
            return True
            
        except Exception as e:
            print(f"    ❌ 处理失败: {e}")
            self.error_count += 1
            return False

    def process_all_files(self, dry_run: bool = False):
        """处理所有markdown文件"""
        if not self.posts_dir.exists():
            print(f"错误: 目录不存在: {self.posts_dir}")
            return
        
        # 获取所有markdown文件
        md_files = list(self.posts_dir.glob("*.md"))
        
        if not md_files:
            print(f"在 {self.posts_dir} 中未找到markdown文件")
            return
        
        print(f"找到 {len(md_files)} 个markdown文件")
        
        if dry_run:
            print("\n=== 试运行模式 ===")
            for file_path in md_files:
                print(f"将处理: {file_path.name}")
            return
        
        print("\n开始处理文件...")
        print("=" * 60)
        
        for file_path in md_files:
            self.process_file(file_path)
        
        print("\n" + "=" * 60)
        print(f"处理完成!")
        print(f"成功处理: {self.processed_count} 个文件")
        if self.error_count > 0:
            print(f"处理失败: {self.error_count} 个文件")

def main():
    parser = argparse.ArgumentParser(description='博客文章处理脚本')
    parser.add_argument('posts_dir', help='文章目录路径', default='content/posts', nargs='?')
    parser.add_argument('--dry-run', action='store_true', help='试运行，不实际修改文件')
    parser.add_argument('--file', help='处理指定文件')
    
    args = parser.parse_args()
    
    # 获取脚本所在目录的父目录作为项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    posts_dir = project_root / args.posts_dir
    
    processor = BlogPostProcessor(posts_dir)
    
    if args.file:
        # 处理指定文件
        file_path = posts_dir / args.file
        if file_path.exists():
            processor.process_file(file_path)
        else:
            print(f"错误: 文件不存在: {file_path}")
    else:
        # 处理所有文件
        processor.process_all_files(dry_run=args.dry_run)

if __name__ == "__main__":
    main()
