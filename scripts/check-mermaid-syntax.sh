#!/bin/bash
# Mermaid语法检查脚本
# 用于检测可能导致渲染失败的Mermaid代码

set -e

BLOG_DIR="/Users/lintao/important/ai-customer/myblog"
CONTENT_DIR="$BLOG_DIR/content/posts"

echo "🔍 开始扫描Mermaid语法问题..."
echo ""

# 颜色定义
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

issue_count=0

# 检查1: 类图中的复杂JSON对象
echo "${BLUE}检查1: 类图中的复杂对象定义${NC}"
if grep -rn "^\s*+.*:.*{.*:.*}" "$CONTENT_DIR" --include="*.md" 2>/dev/null; then
    echo "${RED}❌ 发现复杂对象定义（可能导致解析错误）${NC}"
    echo ""
    ((issue_count++))
else
    echo "${GREEN}✅ 未发现复杂对象定义${NC}"
    echo ""
fi

# 检查2: 未转义的特殊字符
echo "${BLUE}检查2: 类成员中的花括号${NC}"
if grep -rn "^\s*[+\-#~].*:.*{" "$CONTENT_DIR" --include="*.md" | grep -v "^\s*%%" 2>/dev/null; then
    echo "${YELLOW}⚠️  发现花括号（可能需要简化或转义）${NC}"
    echo ""
    ((issue_count++))
else
    echo "${GREEN}✅ 未发现可疑花括号${NC}"
    echo ""
fi

# 检查3: Python类型注解（Mermaid不支持）
echo "${BLUE}检查3: Python类型注解${NC}"
if grep -rn "^\s*[+\-#~].*:.*\[.*,.*\]" "$CONTENT_DIR" --include="*.md" 2>/dev/null; then
    echo "${YELLOW}⚠️  发现Python风格类型注解（Mermaid不支持）${NC}"
    echo ""
    ((issue_count++))
else
    echo "${GREEN}✅ 未发现Python类型注解${NC}"
    echo ""
fi

# 检查4: 超长Mermaid代码块（移动端限制）
echo "${BLUE}检查4: 超长Mermaid代码块（移动端限制5000字符）${NC}"
found_long=false

find "$CONTENT_DIR" -name "*.md" -type f | while read -r file; do
    in_mermaid=false
    mermaid_content=""
    start_line=0
    
    line_num=0
    while IFS= read -r line; do
        ((line_num++))
        
        if [[ "$line" =~ ^\`\`\`mermaid ]]; then
            in_mermaid=true
            mermaid_content=""
            start_line=$line_num
        elif [[ "$line" =~ ^\`\`\`$ ]] && [ "$in_mermaid" = true ]; then
            in_mermaid=false
            length=${#mermaid_content}
            
            if [ $length -gt 5000 ]; then
                echo "${YELLOW}⚠️  ${file}:${start_line} - Mermaid块过长 (${length}字符 > 5000)${NC}"
                found_long=true
            fi
        elif [ "$in_mermaid" = true ]; then
            mermaid_content="${mermaid_content}${line}"
        fi
    done < "$file"
done

if [ "$found_long" = false ]; then
    echo "${GREEN}✅ 所有Mermaid块大小正常${NC}"
fi
echo ""

# 检查5: 查找langchain-01-runnables文件
echo "${BLUE}检查5: 定位问题文件${NC}"
problem_files=$(find "$CONTENT_DIR" -name "*langchain*runnable*.md" -o -name "*langchain-01*.md" 2>/dev/null)

if [ -n "$problem_files" ]; then
    echo "${BLUE}找到可能的问题文件：${NC}"
    echo "$problem_files"
    echo ""
    
    for file in $problem_files; do
        echo "${BLUE}分析文件: $(basename "$file")${NC}"
        
        # 查找第14行附近的classDiagram
        awk '
        /```mermaid/ { in_mermaid=1; mermaid_start=NR; line_in_block=0; next }
        /```/ && in_mermaid { 
            in_mermaid=0; 
            if (found_issue) {
                print "  📍 Mermaid块: 第" mermaid_start "行开始"
                print "  ❌ 问题行: " issue_line
                print "  💡 建议: 将复杂类型简化为 Object 或 Dict~K,V~"
                print ""
                found_issue=0
            }
            next 
        }
        in_mermaid { 
            line_in_block++;
            if ($0 ~ /+.*:.*\{.*:/) {
                found_issue=1
                issue_line="第" line_in_block "行: " $0
            }
        }
        ' "$file"
    done
else
    echo "${YELLOW}⚠️  未找到langchain-01-runnables相关文件${NC}"
    echo ""
fi

# 统计结果
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "${BLUE}扫描完成${NC}"
echo ""

if [ $issue_count -eq 0 ]; then
    echo "${GREEN}✅ 未发现明显问题${NC}"
else
    echo "${YELLOW}⚠️  发现 $issue_count 类潜在问题${NC}"
    echo ""
    echo "${BLUE}修复建议：${NC}"
    echo "1. 简化类型定义：+data: {\"input\": Any} → +data: Object"
    echo "2. 使用泛型语法：+data: Dict~str,Any~"
    echo "3. 使用注释：%% data: {\"input\": Any}"
    echo "4. 参考: MERMAID-ERROR-FIX-GUIDE.md"
fi

echo ""
echo "${BLUE}快速修复命令：${NC}"
echo "  查看具体文件: find content/posts -name '*langchain*01*.md'"
echo "  编辑文件: vim \$(find content/posts -name '*langchain*01*.md')"
echo "  搜索问题: grep -n 'data.*{.*:' \$(find content/posts -name '*langchain*01*.md')"

exit 0

