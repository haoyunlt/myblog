#!/usr/bin/env python3
"""
验证 Hugo 内容文件是否符合规范
检查：
- Front Matter 完整性
- 日期格式
- 必需字段
- 文件命名
- 内容结构
"""

import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import yaml


class HugoValidator:
    """Hugo 内容验证器"""
    
    # 必需的 Front Matter 字段
    REQUIRED_FIELDS = ['title', 'date']
    
    # 推荐的 Front Matter 字段
    RECOMMENDED_FIELDS = ['draft', 'tags', 'categories', 'description']
    
    # Hugo 日期格式（支持多种）
    DATE_FORMATS = [
        '%Y-%m-%dT%H:%M:%S%z',  # 2025-09-28T00:47:16+08:00
        '%Y-%m-%d %H:%M:%S',    # 2025-09-28 00:47:16
        '%Y-%m-%d',             # 2025-09-28
    ]
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.stats = {
            'total': 0,
            'valid': 0,
            'has_issues': 0,
            'has_warnings': 0,
        }
    
    def validate_file(self, file_path: Path) -> Tuple[List[str], List[str]]:
        """验证单个文件"""
        issues = []
        warnings = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # 1. 检查 Front Matter 存在
            if not content.startswith('---'):
                issues.append("缺少 Front Matter（文件必须以 --- 开头）")
                return issues, warnings
            
            # 2. 提取 Front Matter
            parts = content.split('---', 2)
            if len(parts) < 3:
                issues.append("Front Matter 格式错误（需要两个 --- 分隔符）")
                return issues, warnings
            
            front_matter_str = parts[1].strip()
            body = parts[2].strip()
            
            # 3. 解析 YAML Front Matter
            try:
                front_matter = yaml.safe_load(front_matter_str)
                if not isinstance(front_matter, dict):
                    issues.append("Front Matter 必须是 YAML 对象")
                    return issues, warnings
            except yaml.YAMLError as e:
                issues.append(f"Front Matter YAML 解析错误: {e}")
                return issues, warnings
            
            # 4. 检查必需字段
            for field in self.REQUIRED_FIELDS:
                if field not in front_matter:
                    issues.append(f"缺少必需字段: {field}")
            
            # 5. 检查推荐字段
            for field in self.RECOMMENDED_FIELDS:
                if field not in front_matter:
                    warnings.append(f"缺少推荐字段: {field}")
            
            # 6. 验证日期格式
            if 'date' in front_matter:
                date_str = str(front_matter['date'])
                valid_date = False
                for fmt in self.DATE_FORMATS:
                    try:
                        datetime.strptime(date_str.replace('+08:00', '+0800'), fmt.replace('%z', '+0800'))
                        valid_date = True
                        break
                    except (ValueError, TypeError):
                        continue
                
                if not valid_date:
                    issues.append(f"日期格式不正确: {date_str}（应为 YYYY-MM-DDTHH:MM:SS+TZ 格式）")
            
            # 7. 检查 draft 状态
            if 'draft' in front_matter and front_matter['draft']:
                warnings.append("文档标记为草稿状态")
            
            # 8. 检查标签和分类
            if 'tags' in front_matter:
                tags = front_matter['tags']
                if not isinstance(tags, list):
                    issues.append("tags 应该是列表格式")
                elif len(tags) == 0:
                    warnings.append("tags 为空")
            
            if 'categories' in front_matter:
                categories = front_matter['categories']
                if not isinstance(categories, list):
                    issues.append("categories 应该是列表格式")
                elif len(categories) == 0:
                    warnings.append("categories 为空")
            
            # 9. 检查内容长度
            if len(body) < 100:
                warnings.append(f"内容过短（{len(body)} 字符）")
            
            # 10. 检查标题层级
            if body:
                # 检查是否有标题
                if not re.search(r'^#{1,6}\s+', body, re.MULTILINE):
                    warnings.append("文档内容没有标题")
                
                # 检查标题层级跳跃
                headings = re.findall(r'^(#{1,6})\s+', body, re.MULTILINE)
                if headings:
                    levels = [len(h) for h in headings]
                    for i in range(1, len(levels)):
                        if levels[i] > levels[i-1] + 1:
                            warnings.append(f"标题层级跳跃：从 h{levels[i-1]} 直接跳到 h{levels[i]}")
                            break
            
            # 11. 检查 Mermaid 图表语法
            mermaid_blocks = re.findall(r'```mermaid\n(.*?)```', body, re.DOTALL)
            for idx, block in enumerate(mermaid_blocks):
                if not block.strip():
                    warnings.append(f"第 {idx+1} 个 Mermaid 图表为空")
            
            # 12. 检查代码块语言标识
            code_blocks = re.findall(r'```(\w*)\n', body)
            unnamed_blocks = sum(1 for lang in code_blocks if not lang)
            if unnamed_blocks > 0:
                warnings.append(f"有 {unnamed_blocks} 个代码块缺少语言标识")
            
        except Exception as e:
            issues.append(f"文件读取或处理错误: {e}")
        
        return issues, warnings
    
    def validate_directory(self, directory: Path) -> Dict:
        """验证目录中的所有文件"""
        md_files = sorted(directory.rglob('*.md'))
        
        print(f"🔍 检查 {len(md_files)} 个 Markdown 文件...\n")
        
        self.stats['total'] = len(md_files)
        
        for file_path in md_files:
            issues, warnings = self.validate_file(file_path)
            
            if issues:
                self.stats['has_issues'] += 1
                print(f"❌ {file_path.relative_to(directory.parent)}")
                for issue in issues:
                    print(f"   错误: {issue}")
                self.issues.append((file_path, issues))
            elif warnings:
                self.stats['has_warnings'] += 1
                if '--verbose' in sys.argv:
                    print(f"⚠️  {file_path.relative_to(directory.parent)}")
                    for warning in warnings:
                        print(f"   警告: {warning}")
                self.warnings.append((file_path, warnings))
            else:
                self.stats['valid'] += 1
                if '--verbose' in sys.argv:
                    print(f"✅ {file_path.relative_to(directory.parent)}")
        
        return self.stats
    
    def print_summary(self):
        """打印验证摘要"""
        print("\n" + "="*80)
        print("📊 验证摘要")
        print("="*80)
        print(f"总文件数:      {self.stats['total']}")
        print(f"✅ 完全符合:   {self.stats['valid']}")
        print(f"⚠️  有警告:     {self.stats['has_warnings']}")
        print(f"❌ 有错误:     {self.stats['has_issues']}")
        print("="*80)
        
        # 计算通过率
        if self.stats['total'] > 0:
            pass_rate = (self.stats['valid'] + self.stats['has_warnings']) / self.stats['total'] * 100
            print(f"\n通过率: {pass_rate:.1f}% (无致命错误)")
        
        # Hugo 兼容性评估
        print("\n🎯 Hugo 兼容性评估:")
        if self.stats['has_issues'] == 0:
            print("   ✅ 所有文件都符合 Hugo 要求，可以正常构建")
        else:
            print(f"   ⚠️  {self.stats['has_issues']} 个文件存在问题，可能影响 Hugo 构建")
        
        # 常见问题统计
        if self.issues or self.warnings:
            print("\n📋 常见问题统计:")
            
            all_messages = []
            for _, msgs in self.issues:
                all_messages.extend(msgs)
            for _, msgs in self.warnings:
                all_messages.extend(msgs)
            
            issue_counts = {}
            for msg in all_messages:
                # 提取问题类型
                issue_type = msg.split(':')[0] if ':' in msg else msg.split('（')[0]
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
            
            for issue_type, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"   - {issue_type}: {count} 次")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='验证 Hugo 内容文件')
    parser.add_argument('path', type=str, help='要验证的目录路径')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细信息')
    args = parser.parse_args()
    
    path = Path(args.path)
    if not path.exists():
        print(f"❌ 路径不存在: {path}", file=sys.stderr)
        sys.exit(1)
    
    if not path.is_dir():
        print(f"❌ 不是目录: {path}", file=sys.stderr)
        sys.exit(1)
    
    validator = HugoValidator()
    validator.validate_directory(path)
    validator.print_summary()
    
    # 如果有错误，返回非零退出码
    sys.exit(1 if validator.stats['has_issues'] > 0 else 0)


if __name__ == '__main__':
    main()

