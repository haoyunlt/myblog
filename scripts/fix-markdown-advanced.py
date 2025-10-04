#!/usr/bin/env python3
"""
修复 Markdown 文件的高级格式问题
- 修复有序列表编号（MD029）
- 为裸代码块添加语言标识（MD040）
- 删除文件末尾多余空行（MD012）
"""

import re
import sys
from pathlib import Path
from typing import Tuple, List


def fix_ordered_list_numbers(content: str) -> Tuple[str, int]:
    """修复有序列表编号，确保从1开始连续"""
    lines = content.split('\n')
    fixes = 0
    i = 0
    
    while i < len(lines):
        line = lines[i]
        match = re.match(r'^(\s*)(\d+)\.\s+', line)
        
        if match:
            indent = match.group(1)
            # 找到列表的开始
            list_start = i
            
            # 收集整个同级列表
            list_items = []
            while i < len(lines):
                current_line = lines[i]
                current_match = re.match(r'^(\s*)(\d+)\.\s+', current_line)
                
                if current_match:
                    current_indent = current_match.group(1)
                    if len(current_indent) == len(indent):
                        list_items.append(i)
                        i += 1
                    elif len(current_indent) > len(indent):
                        # 更深层级的列表，跳过
                        i += 1
                    else:
                        # 回到上层级
                        break
                elif current_line.strip() == '' or current_line.startswith(indent + '  '):
                    # 空行或缩进内容，继续
                    i += 1
                else:
                    # 列表结束
                    break
            
            # 重新编号
            for idx, line_idx in enumerate(list_items, 1):
                old_line = lines[line_idx]
                new_line = re.sub(r'^(\s*)\d+\.', f'{indent}{idx}.', old_line)
                if new_line != old_line:
                    lines[line_idx] = new_line
                    fixes += 1
        else:
            i += 1
    
    return '\n'.join(lines), fixes


def add_code_language(content: str) -> Tuple[str, int]:
    """为没有语言标识的代码块添加通用语言标识"""
    lines = content.split('\n')
    fixes = 0
    
    for i, line in enumerate(lines):
        # 只匹配开始的 ``` 且后面没有语言标识
        if line.strip() == '```':
            # 尝试推断语言
            language = 'text'  # 默认使用 text
            
            # 查看下一行内容推断语言
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith('import ') or next_line.startswith('from '):
                    language = 'python'
                elif next_line.startswith('func ') or next_line.startswith('package '):
                    language = 'go'
                elif next_line.startswith('class ') or next_line.startswith('public '):
                    language = 'java'
                elif next_line.startswith('{') or next_line.startswith('const ') or next_line.startswith('let '):
                    language = 'javascript'
            
            lines[i] = f'```{language}'
            fixes += 1
    
    return '\n'.join(lines), fixes


def remove_trailing_blank_lines(content: str) -> Tuple[str, int]:
    """删除文件末尾多余的空行，保留一个"""
    original_len = len(content)
    content = content.rstrip('\n') + '\n'
    
    if len(content) < original_len:
        return content, 1
    return content, 0


def fix_advanced_issues(file_path: Path, dry_run: bool = False) -> bool:
    """修复高级格式问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        fixes = []
        
        # 1. 修复有序列表编号
        content, count = fix_ordered_list_numbers(content)
        if count > 0:
            fixes.append(f"修复了 {count} 处有序列表编号")
        
        # 2. 为代码块添加语言标识
        content, count = add_code_language(content)
        if count > 0:
            fixes.append(f"为 {count} 个代码块添加了语言标识")
        
        # 3. 删除末尾多余空行
        content, count = remove_trailing_blank_lines(content)
        if count > 0:
            fixes.append("删除了文件末尾多余空行")
        
        if content != original:
            if not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            print(f"✅ {file_path.name}")
            for fix in fixes:
                print(f"   - {fix}")
            return True
        
        return False
    
    except Exception as e:
        print(f"❌ 处理失败 {file_path}: {e}", file=sys.stderr)
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='修复 Markdown 高级格式问题')
    parser.add_argument('path', type=str, help='要处理的文件或目录路径')
    parser.add_argument('--dry-run', action='store_true', help='试运行，不实际修改文件')
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if not path.exists():
        print(f"❌ 路径不存在: {path}", file=sys.stderr)
        sys.exit(1)
    
    # 收集要处理的文件
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob('*.md'))
    
    print(f"🔍 找到 {len(files)} 个 Markdown 文件")
    
    if args.dry_run:
        print("🔧 试运行模式（不会修改文件）\n")
    else:
        print("🔧 开始修复高级格式问题\n")
    
    modified_count = 0
    for file_path in files:
        if fix_advanced_issues(file_path, args.dry_run):
            modified_count += 1
    
    print(f"\n📊 完成！共修改 {modified_count}/{len(files)} 个文件")
    
    if args.dry_run:
        print("💡 使用不带 --dry-run 参数运行以实际修改文件")


if __name__ == '__main__':
    main()

