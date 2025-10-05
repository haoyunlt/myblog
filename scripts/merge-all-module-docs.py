#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用模块文档合并脚本
合并所有项目的模块文档（概览、时序图、数据结构、API）
"""

import os
import sys
from pathlib import Path
import re

def merge_module_docs(posts_dir: Path, module_prefix: str, output_file: str, delete_originals: bool = True):
    """
    合并一个模块的四个文档
    
    Args:
        posts_dir: posts 目录路径
        module_prefix: 模块前缀，例如 "Kubernetes-01-API Server"
        output_file: 输出文件名
        delete_originals: 是否删除原始文件
    """
    # 四个文档的后缀
    suffixes = ["-概览", "-时序图", "-数据结构", "-API"]
    
    # 读取所有文档内容
    contents = []
    files_to_delete = []
    found_files = []
    
    for suffix in suffixes:
        file_path = posts_dir / f"{module_prefix}{suffix}.md"
        if not file_path.exists():
            # 尝试不带连字符的后缀（例如 "概览" 而不是 "-概览"）
            file_path = posts_dir / f"{module_prefix}{suffix[1:]}.md"
            if not file_path.exists():
                continue
        
        print(f"  📖 读取: {file_path.name}")
        files_to_delete.append(file_path)
        found_files.append(suffix)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 移除开头的 front matter（如果有）
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                # 保留第一个文件的 front matter，其他文件只保留正文
                if len(contents) == 0:
                    # 第一个文件，保留 front matter 但修改 title
                    front_matter = parts[1]
                    body = parts[2].strip()
                    
                    # 移除 body 中的第一个标题（因为它会和 title 重复）
                    lines = body.split('\n')
                    if lines and lines[0].startswith('# '):
                        body = '\n'.join(lines[1:]).strip()
                    
                    # 修改 front matter 中的 title，移除后缀
                    new_front_matter_lines = []
                    for line in front_matter.split('\n'):
                        if line.startswith('title:'):
                            # 移除 -概览 等后缀
                            title = line.split(':', 1)[1].strip().strip('"').strip("'")
                            for s in ['-概览', '-时序图', '-数据结构', '-API', ' 概览', ' 时序图', ' 数据结构', ' API']:
                                title = title.replace(s, '')
                            new_front_matter_lines.append(f'title: "{title}"')
                        else:
                            new_front_matter_lines.append(line)
                    
                    new_front_matter = '\n'.join(new_front_matter_lines)
                    
                    # 添加总标题
                    title_parts = module_prefix.split('-', 2)
                    if len(title_parts) >= 3:
                        module_title = f"{title_parts[0]}-{title_parts[1]} {title_parts[2]}"
                    else:
                        module_title = module_prefix
                    
                    contents.append(f"---{new_front_matter}---\n\n# {module_title}\n\n{body}")
                else:
                    # 后续文件，只保留正文，并移除第一个标题
                    body = parts[2].strip()
                    lines = body.split('\n')
                    if lines and lines[0].startswith('# '):
                        body = '\n'.join(lines[1:]).strip()
                    contents.append(body)
            else:
                contents.append(content.strip())
        else:
            contents.append(content.strip())
    
    if not contents:
        print(f"  ⚠️  没有找到任何文档内容")
        return None
    
    print(f"  ℹ️  找到 {len(found_files)} 个文档: {', '.join(found_files)}")
    
    # 合并内容
    merged_content = "\n\n---\n\n".join(contents)
    
    # 写入输出文件
    output_path = posts_dir / output_file
    print(f"  ✍️  写入: {output_path.name}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(merged_content)
    
    print(f"  ✅ 合并完成: {output_file}")
    
    # 删除原始文件（如果需要）
    if delete_originals and len(files_to_delete) > 0:
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                print(f"  🗑️  删除: {file_path.name}")
            except Exception as e:
                print(f"  ⚠️  删除失败 {file_path.name}: {e}")
    
    return output_path

def find_modules_to_merge(posts_dir: Path):
    """
    自动发现需要合并的模块
    """
    # 查找所有 -概览.md 文件
    overview_files = list(posts_dir.glob("*-概览.md"))
    
    modules = {}
    
    for file in overview_files:
        # 提取模块前缀
        name = file.stem  # 去掉 .md
        if name.endswith("-概览"):
            prefix = name[:-3]  # 去掉 -概览
            
            # 检查是否有其他相关文件
            has_sequence = (posts_dir / f"{prefix}-时序图.md").exists()
            has_datastructure = (posts_dir / f"{prefix}-数据结构.md").exists()
            has_api = (posts_dir / f"{prefix}-API.md").exists()
            
            # 只有至少有2个文档才考虑合并
            count = 1 + (1 if has_sequence else 0) + (1 if has_datastructure else 0) + (1 if has_api else 0)
            if count >= 2:
                modules[prefix] = {
                    'overview': True,
                    'sequence': has_sequence,
                    'datastructure': has_datastructure,
                    'api': has_api,
                    'count': count
                }
    
    return modules

def main():
    """主函数"""
    # 获取 posts 目录
    script_dir = Path(__file__).parent
    posts_dir = script_dir.parent / "content" / "posts"
    
    if not posts_dir.exists():
        print(f"❌ posts 目录不存在: {posts_dir}")
        sys.exit(1)
    
    print("🔍 扫描需要合并的模块...")
    modules = find_modules_to_merge(posts_dir)
    
    if not modules:
        print("✨ 没有找到需要合并的模块文档")
        return
    
    print(f"\n📋 找到 {len(modules)} 个模块需要合并：")
    for prefix, info in sorted(modules.items()):
        docs = []
        if info['overview']: docs.append('概览')
        if info['sequence']: docs.append('时序图')
        if info['datastructure']: docs.append('数据结构')
        if info['api']: docs.append('API')
        print(f"  • {prefix} ({info['count']} 个文档: {', '.join(docs)})")
    
    print("\n" + "="*70)
    response = input("是否继续合并？(y/n): ")
    if response.lower() != 'y':
        print("❌ 取消合并")
        return
    
    print("\n🚀 开始合并模块文档\n")
    
    success_count = 0
    failed_count = 0
    
    for prefix in sorted(modules.keys()):
        print(f"\n{'='*70}")
        print(f"📦 模块: {prefix}")
        print(f"{'='*70}")
        
        output_file = f"{prefix}.md"
        
        try:
            result = merge_module_docs(posts_dir, prefix, output_file, delete_originals=True)
            if result:
                success_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"  ❌ 合并失败: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            continue
    
    print("\n" + "="*70)
    print(f"🎉 合并完成！")
    print(f"   ✅ 成功: {success_count} 个模块")
    print(f"   ❌ 失败: {failed_count} 个模块")
    print("="*70)

if __name__ == "__main__":
    main()

