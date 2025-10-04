#!/usr/bin/env python3
"""
修复 Markdown 文件的常见格式问题
- 列表前后添加空行（MD032）
- 代码块前后添加空行（MD031）
- 硬制表符替换为空格（MD010）
- 删除行尾空格（MD009）
- 删除多余的连续空行（MD012）
"""

import re
import os
import sys
from pathlib import Path
from typing import Tuple, List


def fix_markdown_format(content: str) -> Tuple[str, List[str]]:
    """修复 Markdown 格式问题，返回修复后的内容和修复日志"""
    fixed = content
    fixes = []
    
    # 1. 替换硬制表符为 4 个空格（MD010）
    if '\t' in fixed:
        tab_count = fixed.count('\t')
        fixed = fixed.replace('\t', '    ')
        fixes.append(f"替换了 {tab_count} 个硬制表符为空格")
    
    # 2. 删除行尾空格（MD009）
    lines = fixed.split('\n')
    trailing_spaces = 0
    for i, line in enumerate(lines):
        if line.endswith(' ') and not line.endswith('  '):  # 保留 Markdown 换行符（两个空格）
            trailing_spaces += 1
            lines[i] = line.rstrip()
    if trailing_spaces > 0:
        fixed = '\n'.join(lines)
        fixes.append(f"删除了 {trailing_spaces} 行的行尾空格")
    
    # 3. 修复列表前后的空行（MD032）
    # 匹配列表项：以 "- " 或 "* " 或 数字. 开头
    lines = fixed.split('\n')
    result = []
    i = 0
    list_fixes = 0
    
    while i < len(lines):
        line = lines[i]
        
        # 检测列表开始
        if re.match(r'^(\s*)([-*]|\d+\.)\s+', line):
            # 检查前一行是否需要空行
            if result and result[-1].strip() != '' and not result[-1].startswith('#'):
                # 前一行不是空行且不是标题，添加空行
                if not re.match(r'^(\s*)([-*]|\d+\.)\s+', result[-1]):
                    result.append('')
                    list_fixes += 1
            
            # 收集整个列表
            list_lines = [line]
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if re.match(r'^(\s*)([-*]|\d+\.)\s+', next_line) or next_line.strip() == '' or next_line.startswith('  '):
                    list_lines.append(next_line)
                    i += 1
                else:
                    break
            
            # 添加列表
            result.extend(list_lines)
            
            # 检查后一行是否需要空行
            if i < len(lines) and lines[i].strip() != '':
                if not re.match(r'^(\s*)([-*]|\d+\.)\s+', lines[i]) and not lines[i].startswith('#'):
                    result.append('')
                    list_fixes += 1
        else:
            result.append(line)
            i += 1
    
    if list_fixes > 0:
        fixed = '\n'.join(result)
        fixes.append(f"修复了列表前后空行 {list_fixes} 处")
    
    # 4. 修复代码块前后的空行（MD031）
    lines = fixed.split('\n')
    result = []
    fence_fixes = 0
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # 检测代码块开始
        if line.strip().startswith('```'):
            # 检查前一行是否需要空行
            if result and result[-1].strip() != '':
                result.append('')
                fence_fixes += 1
            
            # 添加代码块开始标记
            result.append(line)
            i += 1
            
            # 收集代码块内容
            while i < len(lines) and not lines[i].strip().startswith('```'):
                result.append(lines[i])
                i += 1
            
            # 添加代码块结束标记
            if i < len(lines):
                result.append(lines[i])
                i += 1
            
            # 检查后一行是否需要空行
            if i < len(lines) and lines[i].strip() != '':
                result.append('')
                fence_fixes += 1
        else:
            result.append(line)
            i += 1
    
    if fence_fixes > 0:
        fixed = '\n'.join(result)
        fixes.append(f"修复了代码块前后空行 {fence_fixes} 处")
    
    # 5. 删除多余的连续空行（MD012），保留最多 1 个空行
    multiple_blank_fixes = 0
    while '\n\n\n' in fixed:
        fixed = fixed.replace('\n\n\n', '\n\n')
        multiple_blank_fixes += 1
    
    if multiple_blank_fixes > 0:
        fixes.append(f"删除了多余的连续空行")
    
    # 6. 确保文件以单个换行符结尾
    if not fixed.endswith('\n'):
        fixed += '\n'
        fixes.append("添加了文件末尾换行符")
    
    return fixed, fixes


def process_file(file_path: Path, dry_run: bool = False) -> bool:
    """处理单个文件，返回是否有修改"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original = f.read()
        
        fixed, fixes = fix_markdown_format(original)
        
        if fixed != original:
            if not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed)
            
            print(f"✅ {file_path.relative_to(file_path.parents[2])}")
            for fix in fixes:
                print(f"   - {fix}")
            return True
        else:
            return False
    
    except Exception as e:
        print(f"❌ 处理失败 {file_path}: {e}", file=sys.stderr)
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='修复 Markdown 文件格式问题')
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
        print("🔧 开始修复格式问题\n")
    
    modified_count = 0
    for file_path in files:
        if process_file(file_path, args.dry_run):
            modified_count += 1
    
    print(f"\n📊 完成！共修改 {modified_count}/{len(files)} 个文件")
    
    if args.dry_run:
        print("💡 使用不带 --dry-run 参数运行以实际修改文件")


if __name__ == '__main__':
    main()

