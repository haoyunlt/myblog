#!/usr/bin/env python3
"""
修复 OpenAI Agent 文档的分类
将 ['OpenAI', 'AI Agent', 'Python'] 改为 ['OpenAIAgent', 'Python']
"""

import re
import sys
from pathlib import Path


def fix_categories(file_path: Path) -> bool:
    """修复单个文件的分类"""
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # 检查是否是 OpenAI Agent 文档
        if 'OpenAI Agent' not in content[:500]:
            return False
        
        # 替换分类部分
        # 查找 categories 部分
        pattern = r'categories:\n(  - .+\n)+'
        
        def replace_categories(match):
            return 'categories:\n  - OpenAIAgent\n  - Python\n'
        
        new_content = re.sub(pattern, replace_categories, content)
        
        if new_content != content:
            file_path.write_text(new_content, encoding='utf-8')
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False


def main():
    posts_dir = Path('content/posts')
    
    # 查找所有 OpenAIAgent 文件
    files = list(posts_dir.glob('OpenAIAgent-*.md'))
    
    print(f"找到 {len(files)} 个 OpenAI Agent 文档")
    
    fixed_count = 0
    for file_path in files:
        if fix_categories(file_path):
            fixed_count += 1
            print(f"✅ {file_path.name}")
    
    print(f"\n修复完成: {fixed_count}/{len(files)} 个文件")


if __name__ == '__main__':
    main()

