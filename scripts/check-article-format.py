#!/usr/bin/env python3
"""
文章格式综合检查脚本
整合多个检查脚本的功能，专门用于检查Hugo博客文章格式问题

功能包括：
- YAML Front Matter 完整性检查
- 必需字段和推荐字段验证
- 文件名格式检查
- 内容结构分析
- Mermaid图表语法检查
- Markdown语法验证
- 链接和图片检查
- 项目分类匹配验证
"""

import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse

# 尝试导入yaml，如果失败则使用简单的解析器
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def simple_yaml_parse(yaml_text: str) -> Dict:
    """简单的YAML解析器，处理基本的key: value和列表"""
    result = {}
    current_key = None
    current_list = []
    
    lines = yaml_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # 检测键值对
        if ':' in line and not line.startswith('-'):
            # 先处理之前的列表
            if current_key and current_list:
                result[current_key] = current_list
                current_list = []
            
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # 处理不同的值格式
            if value.startswith('"') and value.endswith('"'):
                # 带引号的字符串
                result[key] = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                # 单引号字符串
                result[key] = value[1:-1]
            elif value.startswith('[') and value.endswith(']'):
                # 内联列表格式 [item1, item2]
                list_content = value[1:-1].strip()
                if list_content:
                    items = [item.strip().strip('"').strip("'") for item in list_content.split(',')]
                    result[key] = items
                else:
                    result[key] = []
            elif value.lower() in ['true', 'false']:
                # 布尔值
                result[key] = value.lower() == 'true'
            elif value.isdigit():
                # 数字
                result[key] = int(value)
            elif value:
                # 普通字符串
                result[key] = value
            else:
                # 空值，可能后面跟列表
                current_key = key
        elif line.startswith('-') and current_key:
            # 列表项
            item = line[1:].strip()
            if item.startswith('"') and item.endswith('"'):
                item = item[1:-1]
            elif item.startswith("'") and item.endswith("'"):
                item = item[1:-1]
            current_list.append(item)
    
    # 处理最后的列表
    if current_key and current_list:
        result[current_key] = current_list
    
    return result


class ArticleFormatChecker:
    """文章格式检查器"""
    
    # 必需的 Front Matter 字段
    REQUIRED_FIELDS = ['title', 'date']
    
    # 推荐的 Front Matter 字段
    RECOMMENDED_FIELDS = ['draft', 'tags', 'categories', 'description']
    
    # Hugo 支持的日期格式
    DATE_FORMATS = [
        '%Y-%m-%dT%H:%M:%S%z',      # 2025-09-28T00:47:16+08:00
        '%Y-%m-%dT%H:%M:%SZ',       # 2025-09-28T00:47:16Z
        '%Y-%m-%dT%H:%M:%S.%f%z',   # 2025-09-28T00:47:16.123+08:00
        '%Y-%m-%dT%H:%M:%S.%fZ',    # 2025-09-28T00:47:16.123Z
        '%Y-%m-%d %H:%M:%S',        # 2025-09-28 00:47:16
        '%Y-%m-%d',                 # 2025-09-28
    ]
    
    # 常见的 Mermaid 图表类型关键词
    MERMAID_KEYWORDS = [
        'graph', 'flowchart', 'sequenceDiagram', 'classDiagram',
        'stateDiagram', 'pie', 'gantt', 'gitgraph', 'erDiagram',
        'journey', 'mindmap', 'timeline'
    ]
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.stats = {
            'total': 0,
            'valid': 0,
            'has_issues': 0,
            'has_warnings': 0,
            'files_with_issues': [],
            'files_with_warnings': []
        }
        self.issue_summary = {}
    
    def extract_project_name(self, filename: str) -> str:
        """从文件名提取项目名"""
        basename = filename.replace('.md', '')
        
        # 特殊处理AI应用前缀
        if basename.startswith('AI应用-'):
            temp = basename[4:]  # 移除 'AI应用-'
            if temp.startswith('Open-Assistant'):
                return 'Open-Assistant'
            elif temp:
                parts = temp.split('-')
                if len(parts) > 0:
                    return parts[0]
                else:
                    return temp
        elif basename.startswith('grpc-go-'):
            return 'grpc-go'
        else:
            # 标准格式: ProjectName-xx-xx -> ProjectName
            parts = basename.split('-')
            if len(parts) > 0:
                return parts[0]
        
        return basename
    
    def validate_date_format(self, date_str: str) -> bool:
        """验证日期格式"""
        date_str = str(date_str).strip()
        
        for fmt in self.DATE_FORMATS:
            try:
                if fmt.endswith('Z'):
                    # 处理 Z 时区格式
                    datetime.strptime(date_str, fmt)
                elif '%z' in fmt:
                    # 处理 +08:00 时区格式
                    test_date = date_str.replace('+08:00', '+0800').replace('-08:00', '-0800')
                    test_fmt = fmt.replace('%z', '+0800')
                    datetime.strptime(test_date, test_fmt)
                else:
                    # 处理无时区格式
                    datetime.strptime(date_str, fmt)
                return True
            except (ValueError, TypeError):
                continue
        return False
    
    def check_filename(self, file_path: Path) -> Tuple[List[str], List[str]]:
        """检查文件名格式"""
        issues = []
        warnings = []
        
        filename = file_path.name
        
        # 检查空格
        if ' ' in filename:
            warnings.append("文件名包含空格，建议使用连字符替代")
        
        # 检查扩展名
        if not filename.endswith('.md'):
            issues.append("文件扩展名必须是.md")
        
        # 检查特殊字符
        if re.search(r'[<>:"/\\|?*]', filename):
            issues.append("文件名包含不允许的特殊字符")
        
        # 检查长度
        if len(filename) > 255:
            warnings.append("文件名过长，可能在某些系统上有问题")
        
        return issues, warnings
    
    def parse_front_matter(self, content: str) -> Tuple[Optional[Dict], str, List[str]]:
        """解析 Front Matter"""
        issues = []
        
        if not content.startswith('---'):
            issues.append("缺少 Front Matter 开始标记 (---)")
            return None, content, issues
        
        parts = content.split('---', 2)
        if len(parts) < 3:
            issues.append("Front Matter 格式错误（需要两个 --- 分隔符）")
            return None, content, issues
        
        front_matter_str = parts[1].strip()
        body = parts[2].strip()
        
        try:
            if HAS_YAML:
                # 使用标准yaml库
                front_matter = yaml.safe_load(front_matter_str)
            else:
                # 使用简单解析器
                front_matter = simple_yaml_parse(front_matter_str)
            
            if not isinstance(front_matter, dict):
                issues.append("Front Matter 必须是 YAML 对象")
                return None, body, issues
        except Exception as e:
            issues.append(f"Front Matter 解析错误: {e}")
            return None, body, issues
        
        return front_matter, body, issues
    
    def check_front_matter_fields(self, front_matter: Dict, filename: str) -> Tuple[List[str], List[str]]:
        """检查 Front Matter 字段"""
        issues = []
        warnings = []
        
        # 跳过索引文件的date字段检查
        is_index_file = filename.lower() in ['_index.md', 'index.md']
        
        # 检查必需字段
        for field in self.REQUIRED_FIELDS:
            if field not in front_matter:
                # 索引文件不需要date字段
                if is_index_file and field == 'date':
                    continue
                issues.append(f"缺少必需字段: {field}")
        
        # 检查推荐字段
        for field in self.RECOMMENDED_FIELDS:
            if field not in front_matter:
                warnings.append(f"缺少推荐字段: {field}")
        
        # 验证日期格式（如果存在）
        if 'date' in front_matter:
            if not self.validate_date_format(front_matter['date']):
                issues.append(f"日期格式不正确: {front_matter['date']}")
        
        # 检查 draft 状态
        if 'draft' in front_matter and front_matter['draft']:
            warnings.append("文档标记为草稿状态")
        
        # 检查 tags 和 categories 格式
        for field_name in ['tags', 'categories']:
            if field_name in front_matter:
                field_value = front_matter[field_name]
                if not isinstance(field_value, list):
                    issues.append(f"{field_name} 应该是列表格式")
                elif len(field_value) == 0:
                    warnings.append(f"{field_name} 为空")
        
        # 检查 categories 是否包含项目名（跳过索引文件）
        if not is_index_file and 'categories' in front_matter:
            expected_project = self.extract_project_name(filename)
            if expected_project not in ['_index', 'index']:
                categories = front_matter['categories']
                if isinstance(categories, list):
                    if expected_project not in categories:
                        warnings.append(f"categories 应包含项目名 '{expected_project}'")
        
        # 检查 title 是否过短或过长
        if 'title' in front_matter:
            title = str(front_matter['title'])
            if len(title) < 5:
                warnings.append("title 过短")
            elif len(title) > 100:
                warnings.append("title 过长")
        
        return issues, warnings
    
    def check_content_structure(self, body: str) -> Tuple[List[str], List[str]]:
        """检查内容结构"""
        issues = []
        warnings = []
        
        # 检查内容长度
        if len(body) < 100:
            warnings.append(f"内容过短（{len(body)} 字符）")
        
        # 检查是否有标题
        if not re.search(r'^#{1,6}\s+', body, re.MULTILINE):
            warnings.append("内容没有 Markdown 标题")
        
        # 检查标题层级跳跃
        headings = re.findall(r'^(#{1,6})\s+', body, re.MULTILINE)
        if headings:
            levels = [len(h) for h in headings]
            for i in range(1, len(levels)):
                if levels[i] > levels[i-1] + 1:
                    warnings.append(f"标题层级跳跃：从 h{levels[i-1]} 直接跳到 h{levels[i]}")
                    break
        
        return issues, warnings
    
    def check_mermaid_diagrams(self, body: str) -> Tuple[List[str], List[str]]:
        """检查 Mermaid 图表"""
        issues = []
        warnings = []
        
        # 查找所有 mermaid 代码块
        mermaid_blocks = re.findall(r'```mermaid\n(.*?)```', body, re.DOTALL)
        
        if not mermaid_blocks:
            return issues, warnings
        
        # 检查代码块配对
        mermaid_start = len(re.findall(r'```mermaid', body))
        code_end = len(re.findall(r'^```$', body, re.MULTILINE))
        
        if mermaid_start > code_end:
            issues.append("存在未闭合的 mermaid 代码块")
        
        # 检查每个 mermaid 块
        for idx, block in enumerate(mermaid_blocks):
            block = block.strip()
            
            if not block:
                warnings.append(f"第 {idx+1} 个 Mermaid 图表为空")
                continue
            
            # 检查是否包含 Mermaid 关键词
            has_keyword = any(keyword in block for keyword in self.MERMAID_KEYWORDS)
            if not has_keyword:
                warnings.append(f"第 {idx+1} 个 Mermaid 图表可能缺少图表类型声明")
            
            # 检查流程图箭头
            if any(keyword in block for keyword in ['graph', 'flowchart']):
                if '-->' not in block and '->' not in block:
                    warnings.append(f"第 {idx+1} 个流程图缺少连接箭头")
            
            # 检查时序图语法
            if 'sequenceDiagram' in block:
                if not re.search(r'->>|-->>|\+\+|\-\-', block):
                    warnings.append(f"第 {idx+1} 个时序图可能缺少消息箭头")
        
        return issues, warnings
    
    def check_markdown_syntax(self, body: str) -> Tuple[List[str], List[str]]:
        """检查 Markdown 语法"""
        issues = []
        warnings = []
        
        # 检查代码块配对
        code_blocks = re.findall(r'^```', body, re.MULTILINE)
        if len(code_blocks) % 2 != 0:
            warnings.append("可能存在未闭合的代码块")
        
        # 检查代码块语言标识
        code_block_langs = re.findall(r'```(\w*)', body)
        unnamed_blocks = sum(1 for lang in code_block_langs if not lang and lang != 'mermaid')
        if unnamed_blocks > 0:
            warnings.append(f"有 {unnamed_blocks} 个代码块缺少语言标识")
        
        # 检查链接格式
        if re.search(r'\[\]\(', body):
            warnings.append("存在空的链接文本")
        
        # 检查图片链接
        if re.search(r'!\[\]\(', body):
            warnings.append("存在空的图片 alt 文本")
        
        # 检查无效链接格式
        invalid_links = re.findall(r'\[([^\]]*)\]\(([^)]*)\)', body)
        for link_text, link_url in invalid_links:
            if not link_url.strip():
                warnings.append("存在空的链接 URL")
            elif link_url.startswith('http') and ' ' in link_url:
                warnings.append("链接 URL 包含空格")
        
        return issues, warnings
    
    def record_issue(self, issue_type: str):
        """记录问题类型统计"""
        self.issue_summary[issue_type] = self.issue_summary.get(issue_type, 0) + 1
    
    def check_file(self, file_path: Path) -> Tuple[List[str], List[str]]:
        """检查单个文件"""
        all_issues = []
        all_warnings = []
        
        try:
            # 1. 检查文件名
            issues, warnings = self.check_filename(file_path)
            all_issues.extend(issues)
            all_warnings.extend(warnings)
            
            # 记录问题类型
            for issue in issues:
                self.record_issue("文件名格式")
            
            # 2. 读取文件内容
            content = file_path.read_text(encoding='utf-8')
            
            # 3. 解析 Front Matter
            front_matter, body, fm_issues = self.parse_front_matter(content)
            all_issues.extend(fm_issues)
            
            for issue in fm_issues:
                self.record_issue("Front Matter 格式")
            
            if front_matter is not None:
                # 4. 检查 Front Matter 字段
                issues, warnings = self.check_front_matter_fields(front_matter, file_path.name)
                all_issues.extend(issues)
                all_warnings.extend(warnings)
                
                for issue in issues:
                    if "缺少必需字段" in issue:
                        self.record_issue("缺少必需字段")
                    elif "日期格式" in issue:
                        self.record_issue("日期格式错误")
                
                for warning in warnings:
                    if "categories应包含项目名" in warning:
                        self.record_issue("分类匹配问题")
            
            # 5. 检查内容结构
            issues, warnings = self.check_content_structure(body)
            all_issues.extend(issues)
            all_warnings.extend(warnings)
            
            for warning in warnings:
                if "内容过短" in warning:
                    self.record_issue("内容长度")
                elif "标题" in warning:
                    self.record_issue("标题结构")
            
            # 6. 检查 Mermaid 图表
            issues, warnings = self.check_mermaid_diagrams(body)
            all_issues.extend(issues)
            all_warnings.extend(warnings)
            
            for issue in issues:
                self.record_issue("Mermaid 语法")
            
            # 7. 检查 Markdown 语法
            issues, warnings = self.check_markdown_syntax(body)
            all_issues.extend(issues)
            all_warnings.extend(warnings)
            
            for warning in warnings:
                if "链接" in warning:
                    self.record_issue("链接格式")
                elif "代码块" in warning:
                    self.record_issue("代码块格式")
            
        except Exception as e:
            all_issues.append(f"文件处理错误: {e}")
            self.record_issue("文件处理错误")
        
        return all_issues, all_warnings
    
    def check_directory(self, directory: Path) -> Dict:
        """检查目录中的所有 Markdown 文件"""
        md_files = sorted(directory.rglob('*.md'))
        
        print(f"🔍 检查 {len(md_files)} 个 Markdown 文件...\n")
        
        self.stats['total'] = len(md_files)
        
        for file_path in md_files:
            issues, warnings = self.check_file(file_path)
            
            rel_path = file_path.relative_to(directory.parent if directory.name == 'posts' else directory)
            
            if issues:
                self.stats['has_issues'] += 1
                self.stats['files_with_issues'].append((file_path, issues))
                print(f"❌ {rel_path}")
                for issue in issues:
                    print(f"   错误: {issue}")
            elif warnings:
                self.stats['has_warnings'] += 1
                self.stats['files_with_warnings'].append((file_path, warnings))
                if self.verbose:
                    print(f"⚠️  {rel_path}")
                    for warning in warnings:
                        print(f"   警告: {warning}")
            else:
                self.stats['valid'] += 1
                if self.verbose:
                    print(f"✅ {rel_path}")
        
        return self.stats
    
    def print_summary(self):
        """打印检查摘要"""
        print("\n" + "="*80)
        print("📊 文章格式检查摘要")
        print("="*80)
        print(f"总文件数:      {self.stats['total']}")
        print(f"✅ 完全符合:   {self.stats['valid']}")
        print(f"⚠️  有警告:     {self.stats['has_warnings']}")
        print(f"❌ 有错误:     {self.stats['has_issues']}")
        print("="*80)
        
        # 计算通过率
        if self.stats['total'] > 0:
            pass_rate = (self.stats['valid'] + self.stats['has_warnings']) / self.stats['total'] * 100
            print(f"\n📈 通过率: {pass_rate:.1f}% (无致命错误)")
        
        # Hugo 兼容性评估
        print("\n🎯 Hugo 兼容性评估:")
        if self.stats['has_issues'] == 0:
            print("   ✅ 所有文件都符合 Hugo 基本要求，可以正常构建")
        else:
            print(f"   ⚠️  {self.stats['has_issues']} 个文件存在问题，可能影响 Hugo 构建")
        
        # 问题类型统计
        if self.issue_summary:
            print("\n📋 问题类型统计（前10项）:")
            sorted_issues = sorted(self.issue_summary.items(), key=lambda x: x[1], reverse=True)
            for issue_type, count in sorted_issues[:10]:
                print(f"   - {issue_type}: {count} 次")
        
        # 修复建议
        print("\n🔧 修复建议:")
        if self.stats['has_issues'] > 0:
            print("   1. 优先修复标记为 '错误' 的问题，这些会影响 Hugo 构建")
        if self.stats['has_warnings'] > 0:
            print("   2. 处理 '警告' 问题以提升文章质量")
        if "分类匹配问题" in self.issue_summary:
            print("   3. 更新 categories 字段以包含正确的项目名")
        if "文件名格式" in self.issue_summary:
            print("   4. 重命名文件，移除空格和特殊字符")
        if "Mermaid 语法" in self.issue_summary:
            print("   5. 检查和修复 Mermaid 图表语法")


def main():
    parser = argparse.ArgumentParser(
        description='检查Hugo博客文章格式问题',
        epilog='示例: python check-article-format.py content/posts --verbose'
    )
    parser.add_argument('path', type=str, help='要检查的目录路径')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细信息')
    parser.add_argument('--no-warnings', action='store_true', help='不显示警告，仅显示错误')
    
    args = parser.parse_args()
    
    path = Path(args.path)
    if not path.exists():
        print(f"❌ 路径不存在: {path}", file=sys.stderr)
        sys.exit(1)
    
    if not path.is_dir():
        print(f"❌ 不是目录: {path}", file=sys.stderr)
        sys.exit(1)
    
    checker = ArticleFormatChecker(verbose=args.verbose)
    checker.check_directory(path)
    checker.print_summary()
    
    # 返回合适的退出码
    if checker.stats['has_issues'] > 0:
        sys.exit(1)  # 有错误
    elif not args.no_warnings and checker.stats['has_warnings'] > 0:
        sys.exit(2)  # 仅有警告
    else:
        sys.exit(0)  # 成功


if __name__ == '__main__':
    main()
