#!/usr/bin/env python3
"""
Hugo博客链接维护脚本
用于日常维护博客中的内部链接，确保所有链接都是正确的格式
"""

import os
import re
import glob
import sys
from pathlib import Path

def fix_relative_links(content, existing_posts):
    """修复相对路径链接"""
    patterns = [
        # ./filename.md 格式
        (r'\]\(\./([^)]+)\.md\)', r'](/posts/\1/)'),
        # filename.md 格式（不带./）
        (r'\]\(([^/)\s]+)\.md\)', r'](/posts/\1/)'),
    ]
    
    fixed_content = content
    changes = 0
    
    for pattern, replacement in patterns:
        matches = list(re.finditer(pattern, fixed_content))
        for match in matches:
            filename = match.group(1)
            if filename in existing_posts:
                old_text = match.group(0)
                new_text = re.sub(pattern, replacement, old_text)
                fixed_content = fixed_content.replace(old_text, new_text, 1)
                changes += 1
                print(f"  ✅ 修复: {filename}.md -> /posts/{filename}/")
    
    return fixed_content, changes

def get_existing_posts(posts_dir="content/posts"):
    """获取所有存在的文章"""
    posts = set()
    for md_file in glob.glob(f"{posts_dir}/*.md"):
        basename = os.path.splitext(os.path.basename(md_file))[0]
        posts.add(basename)
    return posts

def check_and_fix_links(posts_dir="content/posts", dry_run=False):
    """检查并修复链接"""
    existing_posts = get_existing_posts(posts_dir)
    print(f"📚 发现 {len(existing_posts)} 篇文章")
    
    md_files = glob.glob(f"{posts_dir}/*.md")
    total_files = len(md_files)
    fixed_files = 0
    total_changes = 0
    
    print(f"📝 检查 {total_files} 个文件...")
    
    for md_file in sorted(md_files):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            print(f"❌ 编码错误: {md_file}")
            continue
        
        # 检查是否有需要修复的链接
        has_relative_links = bool(re.search(r'\]\(\./[^)]+\.md\)', content)) or \
                           bool(re.search(r'\]\([^/)\s]+\.md\)', content))
        
        if has_relative_links:
            print(f"\n📄 处理: {os.path.basename(md_file)}")
            fixed_content, changes = fix_relative_links(content, existing_posts)
            
            if changes > 0 and not dry_run:
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                fixed_files += 1
                total_changes += changes
            elif changes > 0:
                print(f"  🔍 [预览模式] 将修复 {changes} 个链接")
                fixed_files += 1
                total_changes += changes
    
    return {
        'total_files': total_files,
        'fixed_files': fixed_files,
        'total_changes': total_changes
    }

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hugo博客链接维护脚本')
    parser.add_argument('--dry-run', action='store_true', 
                       help='预览模式，不实际修改文件')
    parser.add_argument('--posts-dir', default='content/posts',
                       help='文章目录路径 (默认: content/posts)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.posts_dir):
        print(f"❌ 错误: 目录不存在 {args.posts_dir}")
        sys.exit(1)
    
    print("🚀 Hugo博客链接维护工具")
    if args.dry_run:
        print("🔍 运行模式: 预览模式 (不会修改文件)")
    else:
        print("✏️  运行模式: 修复模式")
    
    print("=" * 50)
    
    stats = check_and_fix_links(args.posts_dir, args.dry_run)
    
    print("\n" + "=" * 50)
    print("📊 处理结果:")
    print(f"  检查文件数: {stats['total_files']}")
    print(f"  修复文件数: {stats['fixed_files']}")
    print(f"  修复链接数: {stats['total_changes']}")
    
    if stats['total_changes'] > 0:
        if args.dry_run:
            print("\n💡 提示: 使用 --dry-run 参数运行以实际修改文件")
        else:
            print(f"\n✅ 修复完成!")
    else:
        print(f"\n✨ 所有链接都是正确的!")

if __name__ == "__main__":
    main()
