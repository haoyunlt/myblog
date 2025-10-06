#!/bin/bash

# 移动端性能优化构建脚本
# 自动化构建、优化和部署流程

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# 配置
SITE_DIR="./public"
ASSETS_DIR="./assets"
STATIC_DIR="./static"
BUILD_TIME=$(date +%Y%m%d_%H%M%S)
PERF_REPORT="./performance-report-${BUILD_TIME}.json"

# 检查依赖
check_dependencies() {
    log_step "检查构建依赖..."
    
    local missing_deps=()
    
    if ! command -v hugo &> /dev/null; then
        missing_deps+=("hugo")
    fi
    
    if ! command -v terser &> /dev/null; then
        missing_deps+=("terser")
    fi
    
    if ! command -v cleancss &> /dev/null; then
        missing_deps+=("clean-css-cli")
    fi
    
    if ! command -v convert &> /dev/null; then
        missing_deps+=("imagemagick")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "缺少依赖：${missing_deps[*]}"
        log_info "安装命令："
        log_info "  Hugo: brew install hugo (macOS) 或 sudo apt-get install hugo (Ubuntu)"
        log_info "  Terser: npm install -g terser"
        log_info "  Clean-CSS: npm install -g clean-css-cli"
        log_info "  ImageMagick: brew install imagemagick (macOS) 或 sudo apt-get install imagemagick (Ubuntu)"
        return 1
    fi
    
    log_success "所有依赖已满足"
    return 0
}

# 清理构建目录
clean_build_dir() {
    log_step "清理构建目录..."
    
    if [ -d "$SITE_DIR" ]; then
        rm -rf "$SITE_DIR"
        log_success "已清理 $SITE_DIR"
    fi
    
    # 清理临时文件
    find . -name "*.tmp" -delete 2>/dev/null || true
    find . -name ".DS_Store" -delete 2>/dev/null || true
    
    log_success "构建目录清理完成"
}

# Hugo构建（移动端优化）
build_hugo() {
    log_step "开始Hugo构建（移动端优化模式）..."
    
    # 设置环境变量
    export HUGO_ENV="production"
    export NODE_ENV="production"
    
    # 构建命令
    hugo --gc --minify \
         --enableGitInfo \
         --cleanDestinationDir \
         --logLevel info
    
    if [ $? -eq 0 ]; then
        log_success "Hugo构建完成"
    else
        log_error "Hugo构建失败"
        return 1
    fi
}

# CSS优化
optimize_css() {
    log_step "优化CSS文件..."
    
    find "$SITE_DIR" -name "*.css" -type f | while read -r css_file; do
        local original_size=$(stat -f%z "$css_file" 2>/dev/null || stat --format=%s "$css_file" 2>/dev/null)
        
        # 使用clean-css优化（简化版本）
        if cleancss "$css_file" > "${css_file}.tmp" 2>/dev/null; then
            if [ -s "${css_file}.tmp" ]; then
                mv "${css_file}.tmp" "$css_file"
                local optimized_size=$(stat -f%z "$css_file" 2>/dev/null || stat --format=%s "$css_file" 2>/dev/null)
                local savings=$((original_size - optimized_size))
                log_success "CSS优化: $(basename "$css_file") - 节省 ${savings} 字节"
            else
                rm -f "${css_file}.tmp"
                log_warning "CSS优化失败: $(basename "$css_file")"
            fi
        else
            rm -f "${css_file}.tmp"
            log_warning "CSS优化跳过: $(basename "$css_file")"
        fi
    done
}

# JavaScript优化
optimize_js() {
    log_step "优化JavaScript文件..."
    
    find "$SITE_DIR" -name "*.js" -type f | while read -r js_file; do
        # 跳过已经压缩的文件
        if [[ "$js_file" == *".min.js" ]]; then
            continue
        fi
        
        local original_size=$(stat -f%z "$js_file" 2>/dev/null || stat --format=%s "$js_file" 2>/dev/null)
        
        # 使用terser压缩
        terser "$js_file" \
               --compress sequences=true,dead_code=true,conditionals=true,booleans=true,unused=true,if_return=true,join_vars=true,drop_console=true \
               --mangle \
               --output "${js_file}.tmp"
        
        if [ -s "${js_file}.tmp" ]; then
            mv "${js_file}.tmp" "$js_file"
            local optimized_size=$(stat -f%z "$js_file" 2>/dev/null || stat --format=%s "$js_file" 2>/dev/null)
            local savings=$((original_size - optimized_size))
            log_success "JS优化: $(basename "$js_file") - 节省 ${savings} 字节"
        else
            rm -f "${js_file}.tmp"
            log_warning "JS优化失败: $(basename "$js_file")"
        fi
    done
}

# 图片优化
optimize_images() {
    log_step "优化图片文件..."
    
    # 运行图片优化脚本
    if [ -f "./scripts/optimize-images.sh" ]; then
        chmod +x "./scripts/optimize-images.sh"
        "./scripts/optimize-images.sh"
    else
        log_warning "图片优化脚本不存在，跳过图片优化"
    fi
}

# 生成压缩文件
generate_compressed_files() {
    log_step "生成预压缩文件..."
    
    # Gzip压缩
    find "$SITE_DIR" \( -name "*.html" -o -name "*.css" -o -name "*.js" -o -name "*.xml" -o -name "*.json" \) -type f | while read -r file; do
        if [[ ! "$file" == *".gz" ]] && [[ ! "$file" == *".br" ]]; then
            # 生成gzip文件
            gzip -9 -c "$file" > "${file}.gz"
            
            # 生成brotli文件（如果支持）
            if command -v brotli &> /dev/null; then
                brotli -q 11 -c "$file" > "${file}.br"
            fi
        fi
    done
    
    log_success "预压缩文件生成完成"
}

# 生成资源清单
generate_asset_manifest() {
    log_step "生成资源清单..."
    
    local manifest_file="$SITE_DIR/asset-manifest.json"
    
    cat > "$manifest_file" << EOF
{
  "build_time": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "build_version": "${BUILD_TIME}",
  "assets": {
EOF

    local first=true
    find "$SITE_DIR" \( -name "*.css" -o -name "*.js" -o -name "*.png" -o -name "*.jpg" -o -name "*.webp" -o -name "*.woff2" \) -type f | while read -r file; do
        local rel_path=$(echo "$file" | sed "s|^$SITE_DIR||")
        local file_size=$(stat -f%z "$file" 2>/dev/null || stat --format=%s "$file" 2>/dev/null)
        local file_hash=$(sha256sum "$file" 2>/dev/null | cut -d' ' -f1 || shasum -a 256 "$file" | cut -d' ' -f1)
        
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "$manifest_file"
        fi
        
        echo -n "    \"$rel_path\": {\"size\": $file_size, \"hash\": \"$file_hash\"}" >> "$manifest_file"
    done
    
    cat >> "$manifest_file" << EOF

  }
}
EOF

    log_success "资源清单生成完成: $manifest_file"
}

# 性能分析
analyze_performance() {
    log_step "分析构建性能..."
    
    local total_size=0
    local file_count=0
    local css_size=0
    local js_size=0
    local image_size=0
    local html_size=0
    
    # 统计文件大小
    find "$SITE_DIR" -type f | while read -r file; do
        local size=$(stat -f%z "$file" 2>/dev/null || stat --format=%s "$file" 2>/dev/null)
        total_size=$((total_size + size))
        file_count=$((file_count + 1))
        
        case "$file" in
            *.css) css_size=$((css_size + size)) ;;
            *.js) js_size=$((js_size + size)) ;;
            *.png|*.jpg|*.jpeg|*.gif|*.webp|*.svg) image_size=$((image_size + size)) ;;
            *.html) html_size=$((html_size + size)) ;;
        esac
    done
    
    # 生成性能报告
    cat > "$PERF_REPORT" << EOF
{
  "build_time": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "total_size": $total_size,
  "total_files": $file_count,
  "size_breakdown": {
    "css": $css_size,
    "javascript": $js_size,
    "images": $image_size,
    "html": $html_size
  },
  "size_formatted": {
    "total": "$(numfmt --to=iec-i --suffix=B $total_size)",
    "css": "$(numfmt --to=iec-i --suffix=B $css_size)",
    "javascript": "$(numfmt --to=iec-i --suffix=B $js_size)",
    "images": "$(numfmt --to=iec-i --suffix=B $image_size)",
    "html": "$(numfmt --to=iec-i --suffix=B $html_size)"
  }
}
EOF

    log_success "性能报告生成: $PERF_REPORT"
    
    # 显示摘要
    echo
    log_info "📊 构建性能摘要:"
    log_info "  总文件数: $file_count"
    log_info "  总大小: $(numfmt --to=iec-i --suffix=B $total_size)"
    log_info "  CSS: $(numfmt --to=iec-i --suffix=B $css_size)"
    log_info "  JavaScript: $(numfmt --to=iec-i --suffix=B $js_size)"
    log_info "  图片: $(numfmt --to=iec-i --suffix=B $image_size)"
    log_info "  HTML: $(numfmt --to=iec-i --suffix=B $html_size)"
}

# 验证输出
validate_output() {
    log_step "验证构建输出..."
    
    # 检查关键文件
    local critical_files=(
        "$SITE_DIR/index.html"
        "$SITE_DIR/manifest.json"
        "$SITE_DIR/sw.js"
        "$SITE_DIR/offline/index.html"
    )
    
    for file in "${critical_files[@]}"; do
        if [ -f "$file" ]; then
            log_success "✓ $(basename "$file")"
        else
            log_error "✗ $(basename "$file") 不存在"
            return 1
        fi
    done
    
    # 检查Service Worker
    if grep -q "CACHE_NAME" "$SITE_DIR/sw.js"; then
        log_success "✓ Service Worker配置正确"
    else
        log_warning "⚠ Service Worker可能配置不正确"
    fi
    
    # 检查压缩文件
    local gzip_count=$(find "$SITE_DIR" -name "*.gz" | wc -l)
    local brotli_count=$(find "$SITE_DIR" -name "*.br" | wc -l)
    
    log_info "  Gzip文件: $gzip_count"
    log_info "  Brotli文件: $brotli_count"
    
    log_success "构建输出验证完成"
}

# 性能测试建议
suggest_performance_tests() {
    log_step "性能测试建议..."
    
    echo
    log_info "🚀 建议进行以下性能测试:"
    log_info "  1. Lighthouse审计:"
    log_info "     npx lighthouse http://localhost:1313 --output=html --output-path=lighthouse-report.html"
    log_info "  2. WebPageTest测试:"
    log_info "     https://www.webpagetest.org/"
    log_info "  3. Core Web Vitals测试:"
    log_info "     https://web.dev/measure/"
    log_info "  4. 移动端性能测试:"
    log_info "     Chrome DevTools > Network > Slow 3G 模式"
    log_info "  5. Service Worker测试:"
    log_info "     Application > Service Workers > 离线模式测试"
    echo
}

# 主函数
main() {
    log_info "🚀 开始移动端性能优化构建..."
    echo
    
    # 记录开始时间
    local start_time=$(date +%s)
    
    # 执行构建步骤
    if ! check_dependencies; then
        exit 1
    fi
    
    clean_build_dir
    build_hugo
    optimize_images
    optimize_css
    optimize_js
    generate_compressed_files
    generate_asset_manifest
    analyze_performance
    validate_output
    
    # 计算构建时间
    local end_time=$(date +%s)
    local build_duration=$((end_time - start_time))
    
    echo
    log_success "✅ 移动端性能优化构建完成！"
    log_info "⏱ 构建耗时: ${build_duration}秒"
    log_info "📦 输出目录: $SITE_DIR"
    log_info "📊 性能报告: $PERF_REPORT"
    
    suggest_performance_tests
    
    echo
    log_info "🎯 下一步操作:"
    log_info "  1. 本地测试: hugo server -s $SITE_DIR --port 8080"
    log_info "  2. 部署到服务器: ./scripts/deploy-enhanced.sh"
    log_info "  3. 性能监控: 查看 $PERF_REPORT"
    echo
}

# 运行主函数
main "$@"
