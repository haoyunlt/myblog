#!/usr/bin/env python3
"""
合并项目文档脚本
将每个项目-模块的概览、API、数据结构、时序图合并成一个文档
"""
import os
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# 配置
POSTS_DIR = Path(__file__).parent.parent / "content" / "posts"
BACKUP_DIR = Path(__file__).parent.parent / "content" / "posts_backup"

# 文档类型映射
DOC_TYPES = {
    "概览": {"order": 1, "section_title": "## 模块概览"},
    "API": {"order": 2, "section_title": "## API接口"},
    "数据结构": {"order": 3, "section_title": "## 数据结构"},
    "时序图": {"order": 4, "section_title": "## 时序图"},
}


def parse_filename(filename):
    """
    解析文件名，识别项目名、模块名、文档类型
    
    支持的模式：
    - 项目名-序号-模块名-文档类型.md
    - 项目名-模块名-文档类型.md
    
    返回: (project, module, doc_type) 或 None
    """
    # 移除.md后缀
    name = filename.replace(".md", "")
    
    # 跳过特殊文档
    if name.endswith("-00-总览") or name.endswith("-总览") or name == "00-总览":
        return None
    
    # 模式1: 项目名-序号-模块名-文档类型
    pattern1 = r"^(.+?)-(\d{2})-(.+?)-(概览|API|数据结构|时序图)$"
    match = re.match(pattern1, name)
    if match:
        project, seq, module, doc_type = match.groups()
        return (project, f"{seq}-{module}", doc_type)
    
    # 模式2: 项目名-模块名-文档类型 (无序号)
    pattern2 = r"^(.+?)-(.+?)-(概览|API|数据结构|时序图)$"
    match = re.match(pattern2, name)
    if match:
        project, module, doc_type = match.groups()
        # 跳过已知的总览文档
        if module in ["00-总览", "总览"]:
            return None
        return (project, module, doc_type)
    
    return None


def read_markdown_file(filepath):
    """读取markdown文件，返回front matter和content"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 分离front matter和正文
    parts = content.split("---\n", 2)
    if len(parts) >= 3 and parts[0] == "":
        front_matter = parts[1]
        body = parts[2]
    else:
        front_matter = ""
        body = content
    
    return front_matter, body


def extract_body_content(body):
    """提取正文内容，跳过第一个标题（因为会重新生成）"""
    lines = body.split("\n")
    
    # 跳过第一个 # 标题
    result_lines = []
    found_first_title = False
    
    for line in lines:
        if not found_first_title and line.startswith("# "):
            found_first_title = True
            continue
        if found_first_title:
            result_lines.append(line)
    
    return "\n".join(result_lines).strip()


def merge_documents(project, module, docs):
    """合并单个模块的所有文档"""
    
    # 按文档类型排序
    sorted_docs = sorted(
        docs.items(),
        key=lambda x: DOC_TYPES.get(x[0], {"order": 99})["order"]
    )
    
    # 获取第一个文档的front matter作为基础
    first_filepath = sorted_docs[0][1]
    front_matter, _ = read_markdown_file(first_filepath)
    
    # 修改front matter
    front_matter_lines = front_matter.split("\n")
    new_front_matter = []
    
    for line in front_matter_lines:
        if line.startswith("title:"):
            # 移除原标题中的文档类型后缀
            new_front_matter.append(f'title: "{project}-{module}"')
        elif line.startswith("date:"):
            new_front_matter.append(f"date: {datetime.now().strftime('%Y-%m-%dT%H:%M:%S+08:00')}")
        elif line.startswith("description:"):
            new_front_matter.append(f'description: "{project} 源码剖析 - {module}"')
        elif line.startswith("author:"):
            new_front_matter.append(f'author: "源码分析"')
        else:
            new_front_matter.append(line)
    
    # 构建合并后的内容
    merged_content = []
    merged_content.append("---")
    merged_content.append("\n".join(new_front_matter))
    merged_content.append("---")
    merged_content.append("")
    merged_content.append(f"# {project}-{module}")
    merged_content.append("")
    
    # 添加各个章节
    for doc_type, filepath in sorted_docs:
        section_title = DOC_TYPES.get(doc_type, {}).get("section_title", f"## {doc_type}")
        _, body = read_markdown_file(filepath)
        content = extract_body_content(body)
        
        merged_content.append(section_title)
        merged_content.append("")
        merged_content.append(content)
        merged_content.append("")
        merged_content.append("---")
        merged_content.append("")
    
    return "\n".join(merged_content)


def main():
    print("📚 开始扫描文档目录...")
    
    # 扫描所有文档
    docs_map = defaultdict(lambda: defaultdict(dict))
    
    for filepath in POSTS_DIR.glob("*.md"):
        parsed = parse_filename(filepath.name)
        if parsed:
            project, module, doc_type = parsed
            docs_map[project][module][doc_type] = filepath
    
    print(f"✅ 找到 {len(docs_map)} 个项目")
    
    # 统计需要合并的模块
    merge_count = 0
    for project, modules in docs_map.items():
        for module, docs in modules.items():
            if len(docs) > 1:
                merge_count += 1
    
    print(f"📝 需要合并 {merge_count} 个模块")
    
    # 创建备份目录
    if not BACKUP_DIR.exists():
        BACKUP_DIR.mkdir(parents=True)
        print(f"📁 创建备份目录: {BACKUP_DIR}")
    
    # 合并文档
    merged_files = []
    deleted_files = []
    
    for project, modules in sorted(docs_map.items()):
        for module, docs in sorted(modules.items()):
            if len(docs) <= 1:
                continue
            
            print(f"\n🔄 合并 {project}-{module}...")
            print(f"   包含文档类型: {', '.join(docs.keys())}")
            
            # 生成合并后的内容
            merged_content = merge_documents(project, module, docs)
            
            # 写入新文件
            output_filename = f"{project}-{module}.md"
            output_filepath = POSTS_DIR / output_filename
            
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(merged_content)
            
            merged_files.append(output_filename)
            print(f"   ✅ 生成合并文档: {output_filename}")
            
            # 备份并删除原文件
            for doc_type, filepath in docs.items():
                backup_path = BACKUP_DIR / filepath.name
                filepath.rename(backup_path)
                deleted_files.append(filepath.name)
                print(f"   🗑️  备份原文件: {filepath.name} -> {backup_path.name}")
    
    # 总结
    print("\n" + "="*70)
    print(f"✨ 合并完成！")
    print(f"   - 生成合并文档: {len(merged_files)} 个")
    print(f"   - 备份原文件: {len(deleted_files)} 个")
    print(f"   - 备份位置: {BACKUP_DIR}")
    print("="*70)
    
    if merged_files:
        print("\n生成的合并文档列表：")
        for filename in sorted(merged_files):
            print(f"  - {filename}")


if __name__ == "__main__":
    main()

