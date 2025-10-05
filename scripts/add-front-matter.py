#!/usr/bin/env python3
"""
批量为Markdown文档添加front matter的脚本
"""

import os
import re
from datetime import datetime, timedelta
from pathlib import Path

# 定义各个系列的元数据
SERIES_METADATA = {
    'ClickHouse': {
        'series': 'ClickHouse源码剖析',
        'categories': ['ClickHouse'],
        'base_tags': ['ClickHouse', '源码剖析', '列式数据库'],
        'base_date': datetime(2024, 12, 28, 10, 0, 0)
    },
    'Ray': {
        'series': 'Ray源码剖析',
        'categories': ['Ray'],
        'base_tags': ['Ray', '源码剖析', '分布式计算', '机器学习'],
        'base_date': datetime(2024, 12, 28, 11, 0, 0)
    },
    'Flink': {
        'series': 'Apache Flink源码剖析',
        'categories': ['Flink'],
        'base_tags': ['Flink', '源码剖析', '流计算', '实时计算'],
        'base_date': datetime(2024, 12, 28, 12, 0, 0)
    },
    'MetaGPT': {
        'series': 'MetaGPT源码剖析',
        'categories': ['MetaGPT'],
        'base_tags': ['MetaGPT', '源码剖析', '多智能体', 'AI代码生成'],
        'base_date': datetime(2024, 12, 28, 13, 0, 0)
    },
    'Pulsar': {
        'series': 'Apache Pulsar源码剖析',
        'categories': ['Pulsar'],
        'base_tags': ['Pulsar', '源码剖析', '消息队列', '发布订阅'],
        'base_date': datetime(2024, 12, 28, 14, 0, 0)
    }
}

# 模块特定的标签映射
MODULE_TAGS = {
    'Server': ['网络通信', '协议处理', '多协议支持'],
    'Core': ['数据结构', 'Block', 'Field', 'Settings'],
    'Storages': ['存储引擎', 'MergeTree', '数据持久化'],
    'Processors': ['执行引擎', '流式处理', '向量化执行', '状态机'],
    'Parsers': ['SQL解析', '语法分析', 'AST'],
    'Interpreters': ['查询解释', '查询执行', '计划生成'],
    'Functions': ['函数系统', '表达式计算', '标量函数'],
    'DataTypes': ['类型系统', '数据类型', '序列化'],
    'IO': ['输入输出', '文件系统', '网络IO'],
    'Formats': ['数据格式', '序列化', '导入导出'],
    'Columns': ['列存储', '内存表示', '压缩'],
    'AggregateFunctions': ['聚合函数', 'GROUP BY', '状态管理'],
    'Databases': ['数据库层', '元数据管理', '表管理'],
    'Access': ['权限控制', '用户认证', '安全'],
    'QueryPipeline': ['查询管道', '执行计划', '并行执行'],
    'Coordination': ['分布式协调', 'Keeper', '一致性'],
    'Client': ['客户端', '连接管理', 'CLI工具'],
    'Backups': ['备份恢复', '数据迁移', '容灾'],
    'Actor': ['Actor模型', '有状态计算', '远程调用'],
    'GCS': ['全局控制服务', '元数据管理', '资源调度'],
    'Raylet': ['本地调度器', '任务执行', '对象存储'],
    'Data': ['数据处理', 'DataFrame', '数据集'],
    'Serve': ['模型服务', '在线推理', 'API服务'],
    'Train': ['分布式训练', '机器学习', '模型训练'],
    'Tune': ['超参数调优', 'AutoML', '实验管理'],
    'Autoscaler': ['自动扩缩容', '资源管理', '弹性计算'],
    'RuntimeEnv': ['运行时环境', '依赖管理', '环境隔离'],
    'Dashboard': ['监控面板', '可视化', '系统监控'],
}

def parse_filename(filename):
    """解析文件名，提取项目名、序号和模块名"""
    # 匹配格式：ProjectName-NN-ModuleName.md
    match = re.match(r'([A-Za-z]+)-(\d+)-(.+)\.md$', filename)
    if match:
        project = match.group(1)
        number = int(match.group(2))
        module = match.group(3)
        return project, number, module
    
    # 匹配格式：ProjectName-NN-总览.md
    match = re.match(r'([A-Za-z]+)-(\d+)-总览\.md$', filename)
    if match:
        project = match.group(1)
        number = int(match.group(2))
        return project, number, '总览'
    
    return None, None, None

def get_module_specific_tags(module_name):
    """获取模块特定的标签"""
    for key, tags in MODULE_TAGS.items():
        if key in module_name:
            return tags
    return []

def generate_front_matter(project, number, module_name):
    """生成front matter"""
    if project not in SERIES_METADATA:
        return None
    
    metadata = SERIES_METADATA[project]
    
    # 生成标题
    if module_name == '总览':
        title = f"{project}-{number:02d}-总览"
    else:
        title = f"{project}-{number:02d}-{module_name}"
    
    # 生成日期（基于序号递增）
    date = metadata['base_date'] + timedelta(minutes=number)
    date_str = date.strftime('%Y-%m-%dT%H:%M:%S+08:00')
    
    # 生成标签
    tags = metadata['base_tags'].copy()
    
    # 添加模块特定标签
    module_tags = get_module_specific_tags(module_name)
    tags.extend(module_tags)
    
    # 生成描述
    if module_name == '总览':
        description = f"{project}源码剖析系列总览 - 深入分析{project}的整体架构、核心组件及设计理念"
    else:
        description = f"{project} {module_name}模块源码剖析 - 详细分析{module_name}模块的架构设计、核心功能和实现机制"
    
    # 生成front matter
    front_matter = f"""---
title: "{title}"
date: {date_str}
series: ["{metadata['series']}"]
categories: {metadata['categories']}
tags: {tags}
description: "{description}"
---

"""
    
    return front_matter

def has_front_matter(content):
    """检查文件是否已有front matter"""
    return content.strip().startswith('---')

def remove_existing_front_matter(content):
    """移除现有的front matter"""
    if not has_front_matter(content):
        return content
    
    lines = content.split('\n')
    if lines[0].strip() == '---':
        # 找到第二个 ---
        for i in range(1, len(lines)):
            if lines[i].strip() == '---':
                # 返回去掉front matter的内容
                return '\n'.join(lines[i+1:])
    
    return content

def process_file(file_path):
    """处理单个文件"""
    filename = os.path.basename(file_path)
    project, number, module_name = parse_filename(filename)
    
    if not project or project not in SERIES_METADATA:
        print(f"跳过文件: {filename} (无法识别项目或不在支持列表中)")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 移除现有的front matter（如果有的话）
        content_without_fm = remove_existing_front_matter(content)
        
        front_matter = generate_front_matter(project, number, module_name)
        if not front_matter:
            print(f"跳过文件: {filename} (无法生成front matter)")
            return False
        
        new_content = front_matter + content_without_fm
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        action = "更新" if has_front_matter(content) else "添加"
        print(f"✅ 已{action}front matter: {filename}")
        return True
        
    except Exception as e:
        print(f"❌ 处理文件失败: {filename}, 错误: {e}")
        return False

def main():
    """主函数"""
    posts_dir = Path('content/posts')
    
    if not posts_dir.exists():
        print("错误: content/posts 目录不存在")
        return
    
    processed_count = 0
    total_count = 0
    
    # 遍历所有Markdown文件
    for file_path in posts_dir.glob('*.md'):
        total_count += 1
        if process_file(file_path):
            processed_count += 1
    
    print(f"\n处理完成: 共处理 {total_count} 个文件，成功更新front matter {processed_count} 个")

if __name__ == '__main__':
    main()