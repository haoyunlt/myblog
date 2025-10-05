#!/bin/bash

# 问题修复脚本 - 自动修复今天发现的常见问题
# 基于comprehensive-check.sh的检查结果进行自动修复

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔧 开始自动修复问题...${NC}"
echo ""

# 统计变量
FIXED_COUNT=0
SKIPPED_COUNT=0

# 函数：记录修复操作
log_fix() {
    local action=$1
    local target=$2
    local result=$3
    
    if [ "$result" = "success" ]; then
        echo -e "${GREEN}[FIXED]${NC} $action: $target"
        FIXED_COUNT=$((FIXED_COUNT + 1))
    else
        echo -e "${YELLOW}[SKIPPED]${NC} $action: $target ($result)"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    fi
}

echo "=== 1. 修复文件名空格问题 ==="

# 修复文件名中的空格
fix_filename_spaces() {
    local fixed=0
    
    find content/posts -name "*.md" | while read -r file; do
        local basename=$(basename "$file")
        local dirname=$(dirname "$file")
        
        if echo "$basename" | grep -q " "; then
            local new_name=$(echo "$basename" | sed 's/ /-/g')
            local new_path="$dirname/$new_name"
            
            if [ ! -f "$new_path" ]; then
                mv "$file" "$new_path"
                log_fix "重命名文件" "$basename → $new_name" "success"
                fixed=$((fixed + 1))
            else
                log_fix "重命名文件" "$basename" "目标文件已存在"
            fi
        fi
    done
    
    echo "修复了文件名空格问题"
}

echo ""
echo "=== 2. 修复YAML Front Matter问题 ==="

# 为缺少title的文件添加title
fix_missing_titles() {
    find content/posts -name "*.md" | while read -r file; do
        local basename=$(basename "$file" .md)
        
        if [ "$basename" = "_index" ]; then
            continue
        fi
        
        # 检查是否缺少title
        if ! grep -q "^title:" "$file"; then
            # 检查是否有YAML front matter
            if head -1 "$file" | grep -q "^---$"; then
                # 在第二行插入title
                local title_line="title: \"$basename\""
                sed -i.bak "2i\\
$title_line" "$file"
                rm -f "$file.bak"
                log_fix "添加title字段" "$basename" "success"
            else
                log_fix "添加title字段" "$basename" "缺少YAML front matter"
            fi
        fi
    done
}

# 为缺少categories的文件添加categories
fix_missing_categories() {
    find content/posts -name "*.md" | while read -r file; do
        local basename=$(basename "$file" .md)
        
        if [ "$basename" = "_index" ]; then
            continue
        fi
        
        # 提取项目名
        local project_name=""
        if echo "$basename" | grep -q "^AI应用-"; then
            local temp=$(echo "$basename" | sed 's/^AI应用-//')
            if echo "$temp" | grep -q "^Open-Assistant"; then
                project_name="Open-Assistant"
            else
                project_name=$(echo "$temp" | cut -d'-' -f1)
            fi
        elif echo "$basename" | grep -q "^grpc-go-"; then
            project_name="grpc-go"
        else
            project_name=$(echo "$basename" | cut -d'-' -f1)
        fi
        
        # 检查是否缺少categories
        if ! grep -q "^categories:" "$file" && [ -n "$project_name" ]; then
            # 找到适当的位置插入categories
            local line_num=$(grep -n "^date:" "$file" | cut -d: -f1)
            if [ -n "$line_num" ]; then
                local insert_line=$((line_num + 1))
                sed -i.bak "${insert_line}i\\
categories: ['$project_name']" "$file"
                rm -f "$file.bak"
                log_fix "添加categories字段" "$basename ($project_name)" "success"
            else
                log_fix "添加categories字段" "$basename" "找不到插入位置"
            fi
        fi
    done
}

echo ""
echo "=== 3. 修复Mermaid图表问题 ==="

# 修复未闭合的mermaid代码块
fix_mermaid_blocks() {
    find content/posts -name "*.md" -exec grep -l 'mermaid' {} \; | while read -r file; do
        local basename=$(basename "$file")
        
        # 检查mermaid代码块配对
        local mermaid_start=$(grep -n '```mermaid' "$file" | wc -l)
        local code_end=$(grep -n '^```$' "$file" | wc -l)
        
        if [ "$mermaid_start" -gt "$code_end" ]; then
            # 在文件末尾添加闭合标记
            echo '```' >> "$file"
            log_fix "修复mermaid代码块" "$basename" "success"
        fi
    done
}

echo ""
echo "=== 4. 修复链接问题 ==="

# 标记空链接（需要手动处理）
mark_empty_links() {
    find content/posts -name "*.md" | while read -r file; do
        local basename=$(basename "$file")
        
        if grep -q '\[\](' "$file"; then
            # 在空链接前添加注释
            sed -i.bak 's/\[\]((/<!-- TODO: 修复空链接 -->[]((/g' "$file"
            rm -f "$file.bak"
            log_fix "标记空链接" "$basename" "success"
        fi
    done
}

echo ""
echo "=== 5. 修复脚本权限问题 ==="

# 确保脚本有执行权限
fix_script_permissions() {
    find scripts -name "*.sh" | while read -r script; do
        if [ ! -x "$script" ]; then
            chmod +x "$script"
            log_fix "添加执行权限" "$script" "success"
        fi
    done
}

echo ""
echo "=== 6. 清理和优化 ==="

# 清理Hugo缓存和临时文件
cleanup_hugo_cache() {
    if [ -d "resources/_gen" ]; then
        rm -rf resources/_gen
        log_fix "清理Hugo缓存" "resources/_gen" "success"
    fi
    
    if [ -d "public" ]; then
        rm -rf public
        log_fix "清理构建输出" "public" "success"
    fi
}

# 检查并修复Node.js问题（如果需要）
check_nodejs_deps() {
    if [ -f "package.json" ]; then
        if [ ! -d "node_modules" ]; then
            npm install
            log_fix "安装Node.js依赖" "package.json" "success"
        fi
    fi
}

echo ""
echo "=== 执行修复操作 ==="

# 执行修复操作
fix_filename_spaces
fix_missing_titles
fix_missing_categories
fix_mermaid_blocks
mark_empty_links
fix_script_permissions
cleanup_hugo_cache
check_nodejs_deps

echo ""
echo "========================================"
echo -e "${BLUE}📊 修复结果统计${NC}"
echo "========================================"
echo -e "${GREEN}成功修复: $FIXED_COUNT 个问题${NC}"
echo -e "${YELLOW}跳过处理: $SKIPPED_COUNT 个问题${NC}"

echo ""
echo "🎯 后续建议:"
echo "1. 运行 ./scripts/comprehensive-check.sh 重新检查"
echo "2. 手动处理标记为TODO的空链接"
echo "3. 检查修复后的categories是否正确"
echo "4. 测试Hugo构建: hugo --minify"

echo ""
if [ $FIXED_COUNT -gt 0 ]; then
    echo -e "${GREEN}✅ 修复完成！请检查修改内容并提交更改${NC}"
else
    echo -e "${CYAN}ℹ️  没有发现需要自动修复的问题${NC}"
fi
