#!/bin/bash

# 移动端图片优化脚本
# 自动压缩和转换图片格式以提升移动端性能

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[图片优化]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[成功]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[警告]${NC} $1"
}

# 配置
STATIC_DIR="./static/images"
PUBLIC_DIR="./public/images"
CONTENT_DIR="./content/posts"

# 创建目录
mkdir -p "$STATIC_DIR"
mkdir -p "$PUBLIC_DIR"

# 检查依赖
check_dependencies() {
    local missing_deps=()
    
    if ! command -v convert &> /dev/null; then
        missing_deps+=("imagemagick")
    fi
    
    if ! command -v cwebp &> /dev/null; then
        missing_deps+=("webp")
    fi
    
    if ! command -v jpegoptim &> /dev/null; then
        missing_deps+=("jpegoptim")
    fi
    
    if ! command -v optipng &> /dev/null; then
        missing_deps+=("optipng")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_warning "缺少依赖：${missing_deps[*]}"
        log_info "macOS安装命令：brew install imagemagick webp jpegoptim optipng"
        log_info "Ubuntu安装命令：sudo apt-get install imagemagick webp jpegoptim optipng"
        return 1
    fi
    
    return 0
}

# 优化JPEG图片
optimize_jpeg() {
    local input_file="$1"
    local output_file="$2"
    
    # 移动端优化：质量85%，渐进式JPEG
    jpegoptim --max=85 --strip-all --preserve --totals "$input_file"
    
    # 转换为渐进式JPEG
    convert "$input_file" -quality 85 -interlace Plane "$output_file"
    
    log_success "JPEG优化完成：$(basename "$output_file")"
}

# 优化PNG图片
optimize_png() {
    local input_file="$1"
    local output_file="$2"
    
    # PNG压缩优化
    optipng -o2 -strip all "$input_file"
    cp "$input_file" "$output_file"
    
    log_success "PNG优化完成：$(basename "$output_file")"
}

# 生成WebP格式
generate_webp() {
    local input_file="$1"
    local webp_file="${input_file%.*}.webp"
    
    # 移动端WebP优化：质量80%
    cwebp -q 80 -method 6 -alpha_q 95 "$input_file" -o "$webp_file"
    
    log_success "WebP生成完成：$(basename "$webp_file")"
    echo "$webp_file"
}

# 生成响应式图片尺寸
generate_responsive_sizes() {
    local input_file="$1"
    local base_name="${input_file%.*}"
    local extension="${input_file##*.}"
    
    # 移动端常用尺寸
    local sizes=(320 640 768 1024 1200)
    
    for size in "${sizes[@]}"; do
        local output_file="${base_name}-${size}w.${extension}"
        
        # 只在原图更大时才缩放
        local original_width=$(identify -format "%w" "$input_file")
        if [ "$original_width" -gt "$size" ]; then
            convert "$input_file" -resize "${size}>" -quality 85 "$output_file"
            log_success "生成响应式图片：$(basename "$output_file")"
            
            # 同时生成WebP版本
            generate_webp "$output_file" > /dev/null
        fi
    done
}

# 处理单个图片文件
process_image() {
    local input_file="$1"
    local filename=$(basename "$input_file")
    local extension="${filename##*.}"
    local base_name="${filename%.*}"
    
    log_info "处理图片：$filename"
    
    case "${extension,,}" in
        jpg|jpeg)
            optimize_jpeg "$input_file" "$PUBLIC_DIR/$filename"
            generate_webp "$PUBLIC_DIR/$filename"
            generate_responsive_sizes "$PUBLIC_DIR/$filename"
            ;;
        png)
            optimize_png "$input_file" "$PUBLIC_DIR/$filename"
            generate_webp "$PUBLIC_DIR/$filename"
            generate_responsive_sizes "$PUBLIC_DIR/$filename"
            ;;
        svg)
            # SVG压缩
            if command -v svgo &> /dev/null; then
                svgo "$input_file" -o "$PUBLIC_DIR/$filename"
            else
                cp "$input_file" "$PUBLIC_DIR/$filename"
            fi
            log_success "SVG处理完成：$filename"
            ;;
        *)
            log_warning "不支持的图片格式：$extension"
            ;;
    esac
}

# 批量处理图片
process_all_images() {
    log_info "开始批量处理图片..."
    
    # 处理static目录中的图片
    if [ -d "$STATIC_DIR" ]; then
        find "$STATIC_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.svg" \) | while read -r image_file; do
            process_image "$image_file"
        done
    fi
    
    # 统计优化结果
    local original_size=$(du -sh "$STATIC_DIR" 2>/dev/null | cut -f1 || echo "0K")
    local optimized_size=$(du -sh "$PUBLIC_DIR" 2>/dev/null | cut -f1 || echo "0K")
    
    log_success "图片优化完成！"
    log_info "原始大小：$original_size"
    log_info "优化后大小：$optimized_size"
}

# 生成图片优化报告
generate_report() {
    local report_file="./image-optimization-report.md"
    
    cat > "$report_file" << EOF
# 图片优化报告

生成时间：$(date)

## 优化策略

### JPEG优化
- 质量设置：85%
- 启用渐进式加载
- 移除EXIF数据

### PNG优化
- 使用optipng优化
- 移除元数据

### WebP转换
- 质量设置：80%
- 支持透明度

### 响应式图片
- 生成尺寸：320w, 640w, 768w, 1024w, 1200w
- 同时提供WebP和原格式

## 使用建议

### HTML中使用响应式图片：
\`\`\`html
<picture>
  <source srcset="image-320w.webp 320w, image-640w.webp 640w, image-768w.webp 768w" type="image/webp">
  <source srcset="image-320w.jpg 320w, image-640w.jpg 640w, image-768w.jpg 768w" type="image/jpeg">
  <img src="image-640w.jpg" alt="描述" loading="lazy">
</picture>
\`\`\`

### Hugo shortcode使用：
\`\`\`
{{< responsive-image src="image.jpg" alt="描述" >}}
\`\`\`

EOF

    log_success "优化报告已生成：$report_file"
}

# 主函数
main() {
    log_info "🚀 开始移动端图片优化..."
    
    if ! check_dependencies; then
        exit 1
    fi
    
    process_all_images
    generate_report
    
    log_success "✅ 移动端图片优化完成！"
}

# 运行主函数
main "$@"
