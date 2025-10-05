#!/usr/bin/env python3
"""
修复MetaGPT时序图文章中的Mermaid HTML格式
将HTML格式的Mermaid容器转换为标准的Markdown代码块
"""

import re
import sys

def fix_mermaid_html(content):
    """
    将HTML格式的Mermaid容器转换为标准的Markdown代码块
    """
    
    # 匹配HTML格式的Mermaid容器
    pattern = r'<div class="mermaid-image-container"[^>]*>.*?<pre class="mermaid">(.*?)</pre>.*?</div>'
    
    def replace_mermaid_container(match):
        # 提取Mermaid源码
        mermaid_code = match.group(1).strip()
        
        # 转换为标准的Markdown代码块
        return f"```mermaid\n{mermaid_code}\n```"
    
    # 执行替换
    fixed_content = re.sub(pattern, replace_mermaid_container, content, flags=re.DOTALL)
    
    return fixed_content

def main():
    if len(sys.argv) != 2:
        print("Usage: python fix-mermaid-html.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 修复内容
        fixed_content = fix_mermaid_html(content)
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"✅ 已修复文件: {file_path}")
        
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
