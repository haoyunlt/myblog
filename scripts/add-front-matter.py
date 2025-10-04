#!/usr/bin/env python3
"""
为缺少 Front Matter 的 Markdown 文件添加 YAML Front Matter
自动从标题提取信息并生成合适的元数据
"""

import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


def extract_info_from_content(content: str, filename: str) -> Dict:
    """从内容和文件名中提取信息"""
    
    # 提取第一个标题作为 title
    title_match = re.search(r'^#\s+(.+?)$', content, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()
    else:
        # 从文件名生成 title
        title = filename.replace('.md', '').replace('-', ' ')
    
    # 从文件名提取项目和模块信息
    filename_lower = filename.lower()
    
    # 项目映射
    project_map = {
        'autogpt': 'AutoGPT',
        'dify': 'Dify',
        'docker': 'Docker',
        'eino': 'Eino',
        'elasticsearch': 'Elasticsearch',
        'etcd': 'etcd',
        'fastapi': 'FastAPI',
        'go': 'Go',
        'grpc-go': 'gRPC-Go',
        'kafka': 'Apache Kafka',
        'kitex': 'Kitex',
        'kubernetes': 'Kubernetes',
        'langchain': 'LangChain',
        'langgraph': 'LangGraph',
        'milvus': 'Milvus',
        'nginx': 'Nginx',
        'openaiagent': 'OpenAI Agent',
        'pgvector': 'pgvector',
        'rocksdb': 'RocksDB',
    }
    
    # 分类映射
    category_map = {
        'autogpt': ['AutoGPT', 'AI应用开发'],
        'dify': ['Dify', 'AI应用开发'],
        'docker': ['Docker', '容器技术'],
        'eino': ['Eino', 'AI框架', 'Go'],
        'elasticsearch': ['Elasticsearch', '搜索引擎', '分布式系统'],
        'etcd': ['etcd', '分布式系统', '键值存储'],
        'fastapi': ['FastAPI', 'Python', 'Web框架'],
        'go': ['Go', '编程语言', '运行时'],
        'grpc-go': ['gRPC', 'Go', 'RPC框架'],
        'kafka': ['Kafka', '消息队列', '分布式系统'],
        'kitex': ['Kitex', 'Go', 'RPC框架'],
        'kubernetes': ['Kubernetes', '容器编排', '云原生'],
        'langchain': ['LangChain', 'AI框架', 'Python'],
        'langgraph': ['LangGraph', 'AI框架', 'Python'],
        'milvus': ['Milvus', '向量数据库', '分布式系统'],
        'nginx': ['Nginx', 'Web服务器', 'C'],
        'openaiagent': ['OpenAI', 'AI Agent', 'Python'],
        'pgvector': ['PostgreSQL', '向量检索', '数据库'],
        'rocksdb': ['RocksDB', '存储引擎', 'C++'],
    }
    
    # 标签生成
    tags = []
    categories = []
    project_name = ''
    
    for key, name in project_map.items():
        if key in filename_lower:
            project_name = name
            tags.append(name)
            categories = category_map.get(key, [name])
            break
    
    # 根据文件名添加更多标签
    if 'api' in filename_lower:
        tags.extend(['API设计', '接口文档'])
    if '数据结构' in filename or 'datastructure' in filename_lower:
        tags.extend(['数据结构', 'UML'])
    if '时序图' in filename or 'sequence' in filename_lower:
        tags.extend(['时序图', '流程分析'])
    if '概览' in filename or 'overview' in filename_lower:
        tags.extend(['架构设计', '概览'])
    if '总览' in filename:
        tags.extend(['源码剖析', '架构分析'])
    if '最佳实践' in filename or 'best-practice' in filename_lower:
        tags.extend(['最佳实践', '实战经验'])
    
    # 添加"源码分析"标签
    tags.append('源码分析')
    
    # 去重
    tags = list(dict.fromkeys(tags))  # 保持顺序去重
    
    # 生成描述
    description = f"{project_name} 源码剖析 - {title}" if project_name else f"源码剖析 - {title}"
    
    # 确定 series
    series = None
    if project_name:
        series = f"{project_name.lower()}-source-analysis"
    
    return {
        'title': title,
        'date': datetime.now().strftime('%Y-%m-%dT%H:%M:%S+08:00'),
        'draft': False,
        'tags': tags[:8],  # 最多 8 个标签
        'categories': categories[:3] if categories else [project_name] if project_name else ['技术文档'],
        'series': series,
        'description': description,
        'author': '源码分析',
        'weight': 500,
        'ShowToc': True,
        'TocOpen': True,
    }


def generate_front_matter(info: Dict) -> str:
    """生成 YAML Front Matter"""
    fm_lines = ['---']
    
    # 基本字段
    fm_lines.append(f'title: "{info["title"]}"')
    fm_lines.append(f'date: {info["date"]}')
    fm_lines.append(f'draft: {str(info["draft"]).lower()}')
    
    # 标签
    if info['tags']:
        fm_lines.append('tags:')
        for tag in info['tags']:
            fm_lines.append(f'  - {tag}')
    
    # 分类
    if info['categories']:
        fm_lines.append('categories:')
        for cat in info['categories']:
            fm_lines.append(f'  - {cat}')
    
    # 系列
    if info.get('series'):
        fm_lines.append(f'series: "{info["series"]}"')
    
    # 其他字段
    fm_lines.append(f'description: "{info["description"]}"')
    fm_lines.append(f'author: "{info["author"]}"')
    fm_lines.append(f'weight: {info["weight"]}')
    fm_lines.append(f'ShowToc: {str(info["ShowToc"]).lower()}')
    fm_lines.append(f'TocOpen: {str(info["TocOpen"]).lower()}')
    
    fm_lines.append('---')
    return '\n'.join(fm_lines)


def add_front_matter(file_path: Path, dry_run: bool = False) -> bool:
    """为文件添加 Front Matter"""
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # 检查是否已有 Front Matter
        if content.startswith('---'):
            return False
        
        # 提取信息
        info = extract_info_from_content(content, file_path.name)
        
        # 生成 Front Matter
        front_matter = generate_front_matter(info)
        
        # 组合新内容
        new_content = front_matter + '\n\n' + content
        
        if not dry_run:
            file_path.write_text(new_content, encoding='utf-8')
        
        return True
    
    except Exception as e:
        print(f"❌ 处理失败 {file_path.name}: {e}", file=sys.stderr)
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='为 Markdown 文件添加 Front Matter')
    parser.add_argument('path', type=str, help='要处理的目录路径')
    parser.add_argument('--dry-run', action='store_true', help='试运行，不实际修改文件')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细信息')
    args = parser.parse_args()
    
    path = Path(args.path)
    if not path.exists():
        print(f"❌ 路径不存在: {path}", file=sys.stderr)
        sys.exit(1)
    
    # 收集需要处理的文件
    md_files = sorted(path.rglob('*.md'))
    
    print(f"🔍 检查 {len(md_files)} 个 Markdown 文件")
    
    if args.dry_run:
        print("🔧 试运行模式（不会修改文件）\n")
    else:
        print("🔧 开始添加 Front Matter\n")
    
    modified_count = 0
    skipped_count = 0
    
    for file_path in md_files:
        # 读取前几行检查
        try:
            content = file_path.read_text(encoding='utf-8')
            if content.startswith('---'):
                skipped_count += 1
                if args.verbose:
                    print(f"⏭️  跳过（已有 Front Matter）: {file_path.name}")
                continue
        except:
            continue
        
        if add_front_matter(file_path, args.dry_run):
            modified_count += 1
            print(f"✅ {file_path.name}")
            
            if args.verbose:
                # 显示生成的信息
                info = extract_info_from_content(content, file_path.name)
                print(f"   标题: {info['title']}")
                print(f"   分类: {', '.join(info['categories'])}")
                print(f"   标签: {', '.join(info['tags'][:5])}...")
    
    # 打印摘要
    print("\n" + "="*80)
    print("📊 处理摘要")
    print("="*80)
    print(f"检查文件数:    {len(md_files)}")
    print(f"✅ 已添加:     {modified_count}")
    print(f"⏭️  已跳过:     {skipped_count}")
    print("="*80)
    
    if args.dry_run:
        print("\n💡 使用不带 --dry-run 参数运行以实际修改文件")
    else:
        print(f"\n✅ 成功为 {modified_count} 个文件添加 Front Matter")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

