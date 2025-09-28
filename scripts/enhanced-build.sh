#!/bin/bash

# 🔨 增强版Hugo博客构建脚本
# 支持完整的构建流程、错误检查和性能优化

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 构建配置
BUILD_CONFIG="hugo.toml"
BUILD_DIR="public"
THEME_DIR="themes/PaperMod"
CONTENT_DIR="content"
BASE_URL="https://www.tommienotes.com"

# 检查Hugo环境
check_hugo_env() {
    log_info "检查Hugo环境..."
    
    # 检查Hugo是否安装
    if ! command -v hugo &> /dev/null; then
        log_error "Hugo未安装，请安装Hugo"
        log_info "macOS: brew install hugo"
        log_info "Ubuntu: sudo apt install hugo"
        exit 1
    fi
    
    # 获取Hugo版本信息
    local hugo_version=$(hugo version)
    log_success "Hugo环境正常: $hugo_version"
    
    # 检查是否为extended版本（支持SCSS/SASS）
    if echo "$hugo_version" | grep -q "extended"; then
        log_success "Hugo Extended版本，支持SCSS/SASS"
    else
        log_warning "使用的是标准Hugo版本，可能无法处理SCSS/SASS文件"
    fi
}

# 检查项目结构
check_project_structure() {
    log_info "检查项目结构..."
    
    # 必要文件和目录检查
    local required_items=(
        "$BUILD_CONFIG"
        "$CONTENT_DIR"
        "$THEME_DIR"
        "layouts"
        "static"
    )
    
    for item in "${required_items[@]}"; do
        if [[ ! -e "$item" ]]; then
            log_error "缺少必要的文件或目录: $item"
            exit 1
        fi
    done
    
    log_success "项目结构检查通过"
}

# 更新Git子模块（主题）
update_theme() {
    log_info "更新主题..."
    
    if [[ -d ".git" ]]; then
        # 检查是否有子模块
        if [[ -f ".gitmodules" ]]; then
            log_info "更新Git子模块..."
            git submodule init
            git submodule update --remote --merge
            log_success "主题更新完成"
        else
            log_warning "未找到Git子模块配置"
        fi
    else
        log_warning "不在Git仓库中，跳过主题更新"
    fi
}

# 验证Hugo配置
validate_hugo_config() {
    log_info "验证Hugo配置..."
    
    # 检查配置文件语法
    if hugo config --source=. --config="$BUILD_CONFIG" > /dev/null 2>&1; then
        log_success "Hugo配置语法正确"
    else
        log_error "Hugo配置文件有语法错误"
        hugo config --source=. --config="$BUILD_CONFIG"
        exit 1
    fi
    
    # 检查关键配置项
    if ! grep -q "baseURL" "$BUILD_CONFIG"; then
        log_warning "未设置baseURL"
    fi
    
    if ! grep -q "title" "$BUILD_CONFIG"; then
        log_warning "未设置网站标题"
    fi
}

# 优化Hugo配置（修复已知问题）
optimize_hugo_config() {
    log_info "优化Hugo配置..."
    
    local config_changed=false
    
    # 备份配置文件
    cp "$BUILD_CONFIG" "${BUILD_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
    
    # 检查并修复代码高亮配置
    if grep -q "provider.*shiki" "$BUILD_CONFIG" 2>/dev/null; then
        log_warning "检测到Shiki代码高亮配置，移除以避免问题"
        sed -i.tmp '/provider = "shiki"/d' "$BUILD_CONFIG"
        config_changed=true
    fi
    
    # 确保summaryLength设置为0
    if ! grep -q "summaryLength = 0" "$BUILD_CONFIG" 2>/dev/null; then
        log_info "设置summaryLength = 0以避免内容截断"
        if grep -q "summaryLength" "$BUILD_CONFIG"; then
            sed -i.tmp 's/summaryLength = .*/summaryLength = 0/' "$BUILD_CONFIG"
        else
            echo "summaryLength = 0  # 禁用摘要长度限制" >> "$BUILD_CONFIG"
        fi
        config_changed=true
    fi
    
    # 确保unsafe HTML渲染
    if ! grep -q "unsafe = true" "$BUILD_CONFIG" 2>/dev/null; then
        log_info "确保unsafe HTML渲染已启用"
        if grep -q "\[markup.goldmark.renderer\]" "$BUILD_CONFIG"; then
            sed -i.tmp '/\[markup.goldmark.renderer\]/a\
      unsafe = true  # 允许HTML' "$BUILD_CONFIG"
        fi
        config_changed=true
    fi
    
    # 清理临时文件
    rm -f "${BUILD_CONFIG}.tmp"
    
    if [ "$config_changed" = true ]; then
        log_success "Hugo配置已优化"
    else
        log_success "Hugo配置检查通过，无需优化"
    fi
}

# 清理构建文件
clean_build() {
    log_info "清理构建文件..."
    
    if [[ -d "$BUILD_DIR" ]]; then
        local file_count=$(find "$BUILD_DIR" -type f | wc -l)
        log_info "删除 $file_count 个旧构建文件"
        rm -rf "$BUILD_DIR"
    fi
    
    # 清理Hugo缓存
    if [[ -d "resources" ]]; then
        log_info "清理Hugo资源缓存"
        rm -rf "resources/_gen"
    fi
    
    log_success "构建文件清理完成"
}

# 执行Hugo构建
run_hugo_build() {
    log_info "开始Hugo构建..."
    
    local build_start=$(date +%s)
    local hugo_flags=(
        --gc                    # 启用垃圾收集
        --minify               # 压缩输出文件
        --cleanDestinationDir  # 清理目标目录
        --baseURL "$BASE_URL"  # 设置基础URL
    )
    
    # 执行构建
    if hugo "${hugo_flags[@]}"; then
        local build_end=$(date +%s)
        local build_time=$((build_end - build_start))
        log_success "Hugo构建完成 (耗时: ${build_time}秒)"
    else
        log_error "Hugo构建失败"
        exit 1
    fi
}

# 验证构建结果
verify_build() {
    log_info "验证构建结果..."
    
    if [[ ! -d "$BUILD_DIR" ]]; then
        log_error "构建目录不存在: $BUILD_DIR"
        exit 1
    fi
    
    # 检查关键文件
    local required_files=(
        "$BUILD_DIR/index.html"
        "$BUILD_DIR/index.xml"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "缺少关键文件: $file"
            exit 1
        fi
    done
    
    # 统计构建结果
    local file_count=$(find "$BUILD_DIR" -type f | wc -l)
    local dir_size=$(du -sh "$BUILD_DIR" | cut -f1)
    local html_count=$(find "$BUILD_DIR" -name "*.html" | wc -l)
    local css_count=$(find "$BUILD_DIR" -name "*.css" | wc -l)
    local js_count=$(find "$BUILD_DIR" -name "*.js" | wc -l)
    
    log_success "构建统计信息:"
    echo "   📁 总文件数: $file_count"
    echo "   📊 总大小: $dir_size"
    echo "   📄 HTML文件: $html_count"
    echo "   🎨 CSS文件: $css_count"
    echo "   ⚡ JS文件: $js_count"
}

# 内容质量检查
check_content_quality() {
    log_info "检查内容质量..."
    
    # 检查代码块渲染
    local code_blocks_count=$(find "$BUILD_DIR" -name "*.html" -exec grep -l "class=\"highlight\"" {} \; | wc -l)
    if [ $code_blocks_count -gt 0 ]; then
        log_success "检测到 $code_blocks_count 个文件包含代码块"
    else
        log_warning "未检测到代码块，可能存在渲染问题"
    fi
    
    # 检查Mermaid图表
    local mermaid_count=$(find "$BUILD_DIR" -name "*.html" -exec grep -l "mermaid" {} \; | wc -l)
    if [ $mermaid_count -gt 0 ]; then
        log_success "检测到 $mermaid_count 个文件包含Mermaid图表"
    fi
    
    # 检查图片文件
    local image_count=$(find "$BUILD_DIR" -name "*.jpg" -o -name "*.png" -o -name "*.gif" -o -name "*.svg" | wc -l)
    if [ $image_count -gt 0 ]; then
        log_success "检测到 $image_count 个图片文件"
    fi
    
    # 检查RSS feed
    if [[ -f "$BUILD_DIR/index.xml" ]]; then
        local rss_items=$(grep -c "<item>" "$BUILD_DIR/index.xml" 2>/dev/null || echo 0)
        log_success "RSS feed包含 $rss_items 个条目"
    fi
}

# 性能优化检查
check_performance() {
    log_info "性能优化检查..."
    
    # 检查CSS压缩
    local css_files=$(find "$BUILD_DIR" -name "*.css" | head -5)
    for css_file in $css_files; do
        if [[ -f "$css_file" ]]; then
            local is_minified=$(head -1 "$css_file" | grep -c "^[^[:space:]]" || echo 0)
            if [ $is_minified -gt 0 ]; then
                log_success "CSS文件已压缩: $(basename "$css_file")"
            else
                log_warning "CSS文件可能未压缩: $(basename "$css_file")"
            fi
        fi
    done
    
    # 检查HTML压缩
    local html_file="$BUILD_DIR/index.html"
    if [[ -f "$html_file" ]]; then
        local lines=$(wc -l < "$html_file")
        local size=$(du -h "$html_file" | cut -f1)
        log_info "首页HTML: $lines 行, $size"
    fi
}

# 生成构建报告
generate_build_report() {
    local report_file="build-report-$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "Hugo博客构建报告"
        echo "=================="
        echo "构建时间: $(date)"
        echo "Hugo版本: $(hugo version)"
        echo "构建配置: $BUILD_CONFIG"
        echo "基础URL: $BASE_URL"
        echo ""
        
        echo "构建统计:"
        if [[ -d "$BUILD_DIR" ]]; then
            echo "- 总文件数: $(find "$BUILD_DIR" -type f | wc -l)"
            echo "- 总大小: $(du -sh "$BUILD_DIR" | cut -f1)"
            echo "- HTML文件: $(find "$BUILD_DIR" -name "*.html" | wc -l)"
            echo "- CSS文件: $(find "$BUILD_DIR" -name "*.css" | wc -l)"
            echo "- JS文件: $(find "$BUILD_DIR" -name "*.js" | wc -l)"
            echo "- 图片文件: $(find "$BUILD_DIR" -name "*.jpg" -o -name "*.png" -o -name "*.gif" -o -name "*.svg" | wc -l)"
        fi
        echo ""
        
        echo "内容统计:"
        echo "- Markdown文件: $(find "$CONTENT_DIR" -name "*.md" | wc -l)"
        echo "- 文章数量: $(find "$CONTENT_DIR/posts" -name "*.md" | wc -l 2>/dev/null || echo 0)"
        echo ""
        
        echo "配置文件状态:"
        if [[ -f "$BUILD_CONFIG" ]]; then
            echo "- baseURL: $(grep 'baseURL' "$BUILD_CONFIG" | cut -d'=' -f2 | tr -d '"' | xargs)"
            echo "- title: $(grep 'title' "$BUILD_CONFIG" | cut -d'=' -f2 | tr -d '"' | xargs)"
            echo "- 主题: $(grep 'theme' "$BUILD_CONFIG" | cut -d'=' -f2 | tr -d '"' | xargs)"
        fi
    } > "$report_file"
    
    log_success "构建报告已生成: $report_file"
}

# 主函数
main() {
    echo "🔨 增强版Hugo博客构建"
    echo "======================"
    echo ""
    
    local start_time=$(date +%s)
    
    # 第一阶段：环境检查
    check_hugo_env
    check_project_structure
    validate_hugo_config
    
    # 第二阶段：优化和准备
    optimize_hugo_config
    update_theme
    
    # 第三阶段：构建
    clean_build
    run_hugo_build
    
    # 第四阶段：验证和报告
    verify_build
    check_content_quality
    check_performance
    generate_build_report
    
    local end_time=$(date +%s)
    local total_time=$((end_time - start_time))
    
    echo ""
    log_success "🎉 构建完成！(总耗时: ${total_time}秒)"
    log_info "构建文件位于: $BUILD_DIR"
    log_info "准备部署到阿里云..."
}

# 参数处理
case "${1:-}" in
    "clean")
        clean_build
        ;;
    "config")
        check_hugo_env
        validate_hugo_config
        optimize_hugo_config
        ;;
    "theme")
        update_theme
        ;;
    "build-only")
        run_hugo_build
        verify_build
        ;;
    "verify")
        verify_build
        check_content_quality
        check_performance
        ;;
    "report")
        generate_build_report
        ;;
    "help"|"-h"|"--help")
        echo "用法: $0 [选项]"
        echo ""
        echo "选项:"
        echo "  (无参数)    完整构建流程"
        echo "  clean       只清理构建文件"
        echo "  config      检查和优化配置"
        echo "  theme       只更新主题"
        echo "  build-only  只执行构建（跳过检查）"
        echo "  verify      验证构建结果"
        echo "  report      生成构建报告"
        echo "  help        显示此帮助"
        ;;
    *)
        main
        ;;
esac
