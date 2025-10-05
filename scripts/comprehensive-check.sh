#!/bin/bash

# 综合检查脚本 - 整合今天遇到的所有问题
# 包括：文章格式、Mermaid图表、部署环境、Git状态等检查

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 统计变量
TOTAL_ISSUES=0
CRITICAL_ISSUES=0
WARNING_ISSUES=0
INFO_ISSUES=0

# 日志文件
LOG_FILE="comprehensive-check-$(date +%Y%m%d_%H%M%S).log"

echo -e "${BLUE}🔍 综合环境检查开始...${NC}"
echo "详细日志: $LOG_FILE"
echo ""

# 函数：记录问题
log_issue() {
    local level=$1
    local category=$2
    local message=$3
    
    case $level in
        "CRITICAL")
            echo -e "${RED}[CRITICAL]${NC} [$category] $message" | tee -a "$LOG_FILE"
            CRITICAL_ISSUES=$((CRITICAL_ISSUES + 1))
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} [$category] $message" | tee -a "$LOG_FILE"
            WARNING_ISSUES=$((WARNING_ISSUES + 1))
            ;;
        "INFO")
            echo -e "${CYAN}[INFO]${NC} [$category] $message" | tee -a "$LOG_FILE"
            INFO_ISSUES=$((INFO_ISSUES + 1))
            ;;
    esac
    TOTAL_ISSUES=$((TOTAL_ISSUES + 1))
}

echo "=== 1. 环境基础检查 ===" | tee -a "$LOG_FILE"

# 检查Hugo版本
check_hugo_environment() {
    echo -n "检查Hugo环境... "
    
    if ! command -v hugo &> /dev/null; then
        log_issue "CRITICAL" "环境" "Hugo未安装"
        return 1
    fi
    
    local hugo_version=$(hugo version)
    echo "$hugo_version" >> "$LOG_FILE"
    
    # 检查是否是extended版本
    if echo "$hugo_version" | grep -q "extended"; then
        echo -e "${GREEN}OK${NC} ($(echo $hugo_version | grep -o 'v[0-9.]*'))"
    else
        log_issue "WARNING" "环境" "建议使用Hugo Extended版本以支持SCSS"
        echo -e "${YELLOW}基础版本${NC}"
    fi
}

# 检查Git状态
check_git_status() {
    echo -n "检查Git状态... "
    
    if ! git rev-parse --git-dir &> /dev/null; then
        log_issue "CRITICAL" "Git" "当前目录不是Git仓库"
        return 1
    fi
    
    local current_branch=$(git branch --show-current)
    local has_unstaged=$(git diff --name-only | wc -l)
    local has_staged=$(git diff --cached --name-only | wc -l)
    local has_untracked=$(git ls-files --others --exclude-standard | wc -l)
    
    echo "当前分支: $current_branch" >> "$LOG_FILE"
    
    if [ "$has_unstaged" -gt 0 ] || [ "$has_staged" -gt 0 ] || [ "$has_untracked" -gt 0 ]; then
        log_issue "WARNING" "Git" "有未提交的更改 (未暂存:$has_unstaged, 已暂存:$has_staged, 未跟踪:$has_untracked)"
        echo -e "${YELLOW}有未提交更改${NC}"
    else
        echo -e "${GREEN}OK${NC} ($current_branch)"
    fi
}

# 检查项目结构
check_project_structure() {
    echo -n "检查项目结构... "
    
    local missing_dirs=()
    local required_dirs=("content/posts" "layouts" "static" "assets" "config")
    
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            missing_dirs+=("$dir")
        fi
    done
    
    if [ ${#missing_dirs[@]} -gt 0 ]; then
        log_issue "CRITICAL" "结构" "缺少关键目录: ${missing_dirs[*]}"
        echo -e "${RED}缺少目录${NC}"
        return 1
    else
        echo -e "${GREEN}OK${NC}"
    fi
}

echo ""
echo "=== 2. 文章格式检查 ===" | tee -a "$LOG_FILE"

# 检查文章格式问题
check_posts_format() {
    local posts_dir="content/posts"
    local total_posts=0
    local format_issues=0
    
    echo "正在检查文章格式..."
    
    if [ ! -d "$posts_dir" ]; then
        log_issue "CRITICAL" "文章" "posts目录不存在"
        return 1
    fi
    
    while IFS= read -r -d '' file; do
        total_posts=$((total_posts + 1))
        local basename=$(basename "$file")
        local issues_before=$TOTAL_ISSUES
        
        # 跳过索引文件
        if [ "$basename" = "_index.md" ]; then
            continue
        fi
        
        # 检查文件名格式
        if echo "$basename" | grep -q " "; then
            log_issue "WARNING" "文件名" "$basename 包含空格"
        fi
        
        # 检查YAML front matter
        if ! head -1 "$file" | grep -q "^---$"; then
            log_issue "CRITICAL" "格式" "$basename 缺少YAML front matter"
        else
            # 检查必需字段
            if ! grep -q "^title:" "$file"; then
                log_issue "CRITICAL" "格式" "$basename 缺少title字段"
            fi
            
            if ! grep -q "^categories:" "$file"; then
                log_issue "WARNING" "格式" "$basename 缺少categories字段"
            else
                # 检查categories是否包含项目名
                local project_name=$(echo "$basename" | cut -d'-' -f1)
                if ! grep -A 3 "^categories:" "$file" | grep -q "$project_name"; then
                    log_issue "WARNING" "分类" "$basename categories应包含项目名 '$project_name'"
                fi
            fi
        fi
        
        # 检查内容结构
        if ! grep -q "^# " "$file"; then
            log_issue "WARNING" "结构" "$basename 缺少一级标题"
        fi
        
        # 检查空链接
        if grep -q '\[\](' "$file"; then
            log_issue "WARNING" "链接" "$basename 存在空链接文本"
        fi
        
        # 统计此文件的问题
        if [ $TOTAL_ISSUES -gt $issues_before ]; then
            format_issues=$((format_issues + 1))
        fi
        
    done < <(find "$posts_dir" -name "*.md" -type f -print0)
    
    echo "文章检查完成: $total_posts 个文件, $format_issues 个有问题" >> "$LOG_FILE"
    echo "  - 检查了 $total_posts 个文章"
    echo "  - 发现 $format_issues 个文件有格式问题"
}

echo ""
echo "=== 3. Mermaid图表检查 ===" | tee -a "$LOG_FILE"

# 检查Mermaid图表
check_mermaid_diagrams() {
    echo "正在检查Mermaid图表..."
    
    local mermaid_files=0
    local mermaid_issues=0
    
    while IFS= read -r file; do
        if [ -z "$file" ]; then continue; fi
        
        mermaid_files=$((mermaid_files + 1))
        local basename=$(basename "$file")
        local issues_before=$TOTAL_ISSUES
        
        # 检查mermaid代码块配对
        local mermaid_start=$(grep -c '```mermaid' "$file")
        local code_end=$(grep -c '^```$' "$file")
        
        if [ "$mermaid_start" -gt "$code_end" ]; then
            log_issue "CRITICAL" "Mermaid" "$basename 有未闭合的mermaid代码块"
        fi
        
        # 检查mermaid内容
        if grep -A 5 '```mermaid' "$file" | grep -q '^```$'; then
            log_issue "WARNING" "Mermaid" "$basename 包含空的mermaid代码块"
        fi
        
        # 检查常见的mermaid语法错误
        local mermaid_content=$(sed -n '/```mermaid/,/^```$/p' "$file")
        if echo "$mermaid_content" | grep -q 'flowchart\|graph'; then
            if ! echo "$mermaid_content" | grep -q '-->'; then
                log_issue "WARNING" "Mermaid" "$basename 流程图缺少连接箭头"
            fi
        fi
        
        if [ $TOTAL_ISSUES -gt $issues_before ]; then
            mermaid_issues=$((mermaid_issues + 1))
        fi
        
    done < <(find content/posts -name "*.md" -exec grep -l 'mermaid' {} \; 2>/dev/null || true)
    
    echo "Mermaid检查完成: $mermaid_files 个文件包含图表, $mermaid_issues 个有问题" >> "$LOG_FILE"
    echo "  - 检查了 $mermaid_files 个包含Mermaid的文件"
    echo "  - 发现 $mermaid_issues 个文件有图表问题"
}

echo ""
echo "=== 4. 构建和部署检查 ===" | tee -a "$LOG_FILE"

# 检查构建状态
check_build_status() {
    echo -n "检查Hugo构建... "
    
    # 尝试构建（不输出文件）
    if hugo --quiet --logLevel error --destination /tmp/hugo-check 2>/dev/null; then
        echo -e "${GREEN}OK${NC}"
        rm -rf /tmp/hugo-check
    else
        log_issue "CRITICAL" "构建" "Hugo构建失败"
        echo -e "${RED}构建失败${NC}"
        return 1
    fi
}

# 检查部署脚本
check_deployment_scripts() {
    echo -n "检查部署脚本... "
    
    local scripts_found=0
    local script_files=("scripts/deploy-enhanced.sh" "scripts/enhanced-build.sh")
    
    for script in "${script_files[@]}"; do
        if [ -f "$script" ]; then
            scripts_found=$((scripts_found + 1))
            
            # 检查脚本权限
            if [ ! -x "$script" ]; then
                log_issue "WARNING" "部署" "$script 不可执行"
            fi
            
            # 检查脚本语法
            if ! bash -n "$script" 2>/dev/null; then
                log_issue "CRITICAL" "部署" "$script 语法错误"
            fi
        else
            log_issue "INFO" "部署" "$script 不存在"
        fi
    done
    
    if [ $scripts_found -gt 0 ]; then
        echo -e "${GREEN}找到 $scripts_found 个部署脚本${NC}"
    else
        echo -e "${YELLOW}无部署脚本${NC}"
    fi
}

# 检查Service Worker问题
check_service_worker() {
    echo -n "检查Service Worker... "
    
    if [ -f "static/sw.js" ]; then
        # 检查Service Worker语法
        if node -c "static/sw.js" 2>/dev/null; then
            echo -e "${GREEN}OK${NC}"
        else
            log_issue "WARNING" "SW" "Service Worker语法可能有问题"
            echo -e "${YELLOW}语法警告${NC}"
        fi
    else
        echo -e "${CYAN}未使用SW${NC}"
    fi
}

echo ""
echo "=== 5. 分支和版本检查 ===" | tee -a "$LOG_FILE"

# 检查分支状态
check_branch_status() {
    echo "检查分支状态..."
    
    local current_branch=$(git branch --show-current)
    local available_branches=($(git branch | grep -E "(v7|v8|main)" | sed 's/[* ]//g'))
    
    echo "当前分支: $current_branch" >> "$LOG_FILE"
    echo "可用分支: ${available_branches[*]}" >> "$LOG_FILE"
    
    # 检查是否在预期的分支上
    if [[ "$current_branch" =~ ^(v7|v8|main)$ ]]; then
        echo "  - 当前分支: $current_branch ✓"
    else
        log_issue "WARNING" "分支" "当前分支 '$current_branch' 不是标准发布分支"
    fi
    
    # 检查分支差异
    if git branch | grep -q "v8" && git branch | grep -q "v7"; then
        local v7_v8_diff=$(git log v7..v8 --oneline | wc -l)
        echo "  - v7到v8差异: $v7_v8_diff 个提交"
        echo "v7到v8差异: $v7_v8_diff 个提交" >> "$LOG_FILE"
    fi
}

# 检查stash状态
check_stash_status() {
    echo -n "检查Git stash... "
    
    local stash_count=$(git stash list | wc -l)
    if [ "$stash_count" -gt 0 ]; then
        log_issue "INFO" "Git" "有 $stash_count 个stash条目"
        echo -e "${CYAN}$stash_count 个stash${NC}"
    else
        echo -e "${GREEN}无stash${NC}"
    fi
}

echo ""
echo "=== 6. 执行检查 ===" | tee -a "$LOG_FILE"

# 执行所有检查
check_hugo_environment
check_git_status
check_project_structure
echo ""
check_posts_format
echo ""
check_mermaid_diagrams
echo ""
check_build_status
check_deployment_scripts
check_service_worker
echo ""
check_branch_status
check_stash_status

echo ""
echo "========================================" | tee -a "$LOG_FILE"
echo -e "${BLUE}📊 综合检查结果${NC}" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo -e "${RED}严重问题: $CRITICAL_ISSUES${NC}" | tee -a "$LOG_FILE"
echo -e "${YELLOW}警告问题: $WARNING_ISSUES${NC}" | tee -a "$LOG_FILE"
echo -e "${CYAN}信息提示: $INFO_ISSUES${NC}" | tee -a "$LOG_FILE"
echo "总问题数: $TOTAL_ISSUES" | tee -a "$LOG_FILE"

# 生成建议
echo "" | tee -a "$LOG_FILE"
echo "🔧 修复建议:" | tee -a "$LOG_FILE"

if [ $CRITICAL_ISSUES -gt 0 ]; then
    echo "1. 优先解决 $CRITICAL_ISSUES 个严重问题" | tee -a "$LOG_FILE"
fi

if [ $WARNING_ISSUES -gt 0 ]; then
    echo "2. 处理 $WARNING_ISSUES 个警告问题以提升质量" | tee -a "$LOG_FILE"
fi

if grep -q "文件名.*空格" "$LOG_FILE"; then
    echo "3. 重命名包含空格的文件: 将空格替换为连字符" | tee -a "$LOG_FILE"
fi

if grep -q "categories应包含项目名" "$LOG_FILE"; then
    echo "4. 更新文章的categories字段以匹配项目名" | tee -a "$LOG_FILE"
fi

if grep -q "mermaid" "$LOG_FILE"; then
    echo "5. 检查和修复Mermaid图表语法问题" | tee -a "$LOG_FILE"
fi

# 返回状态码
if [ $CRITICAL_ISSUES -gt 0 ]; then
    echo -e "\n${RED}❌ 检查发现严重问题，需要立即修复${NC}"
    exit 1
elif [ $WARNING_ISSUES -gt 0 ]; then
    echo -e "\n${YELLOW}⚠️  检查发现警告问题，建议修复${NC}"
    exit 2
else
    echo -e "\n${GREEN}✅ 所有检查通过！${NC}"
    exit 0
fi
