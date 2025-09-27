#!/bin/bash

# 🚀 阿里云服务器部署脚本
# 支持Hugo博客的全量部署到阿里云ECS

set -e

# 颜色输出
RED='\033[0;31m'想
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

# 配置变量
REMOTE_HOST="blog-svr"
REMOTE_USER="root"
REMOTE_WEB_ROOT="/var/www/html"
REMOTE_NGINX_CONF="/etc/nginx/sites-available/blog"
REMOTE_NGINX_ENABLED="/etc/nginx/sites-enabled/blog"
LOCAL_PUBLIC="./public"

# 检查本地环境
check_local_env() {
    log_info "检查本地环境..."

    # 检查Hugo
    if ! command -v hugo >/dev/null 2>&1; then
        log_error "Hugo 未安装，请先运行: brew install hugo"
        exit 1
    fi

    # 检查rsync
    if ! command -v rsync >/dev/null 2>&1; then
        log_error "rsync 未安装"
        exit 1
    fi

    # 检查public目录
    if [[ ! -d "$LOCAL_PUBLIC" ]]; then
        log_warning "public目录不存在，正在构建..."
        hugo --minify --baseURL "https://www.tommienotes.com"
    fi

    log_success "本地环境检查通过"
}

# 检查和修复Hugo配置
check_and_fix_hugo_config() {
    log_info "检查Hugo配置..."
    
    local config_file="hugo.toml"
    local config_changed=false
    
    # 检查代码高亮配置
    if grep -q "provider.*shiki" "$config_file" 2>/dev/null; then
        log_warning "检测到Shiki代码高亮配置，这可能导致代码块内容丢失"
        log_info "修复代码高亮配置..."
        
        # 备份原配置
        cp "$config_file" "${config_file}.backup.$(date +%Y%m%d_%H%M%S)"
        
        # 修复代码高亮配置 - 移除Shiki相关配置
        sed -i.tmp '
        /# 代码高亮配置 - 使用Shiki替代/,/provider = "shiki"/ {
            /provider = "shiki"/d
            /enabled = true/d
        }
        ' "$config_file"
        
        # 确保使用Hugo内置高亮
        if grep -q "lineNos = false" "$config_file"; then
            sed -i.tmp 's/lineNos = false/lineNos = true/' "$config_file"
        fi
        if grep -q "noClasses = true" "$config_file"; then
            sed -i.tmp 's/noClasses = true/noClasses = false/' "$config_file"
        fi
        if grep -q "guessSyntax = false" "$config_file"; then
            sed -i.tmp 's/guessSyntax = false/guessSyntax = true/' "$config_file"
        fi
        
        # 删除临时文件
        rm -f "${config_file}.tmp"
        
        config_changed=true
        log_success "代码高亮配置已修复"
    fi
    
    # 检查summaryLength配置
    if ! grep -q "summaryLength = 0" "$config_file" 2>/dev/null; then
        log_info "设置summaryLength = 0以避免内容截断..."
        if grep -q "summaryLength" "$config_file"; then
            sed -i.tmp 's/summaryLength = .*/summaryLength = 0/' "$config_file"
        else
            sed -i.tmp '1a\
summaryLength = 0  # 禁用摘要长度限制' "$config_file"
        fi
        rm -f "${config_file}.tmp"
        config_changed=true
    fi
    
    # 检查unsafe配置
    if ! grep -q "unsafe = true" "$config_file" 2>/dev/null; then
        log_info "启用unsafe HTML渲染..."
        if grep -q "\[markup.goldmark.renderer\]" "$config_file"; then
            sed -i.tmp '/\[markup.goldmark.renderer\]/a\
      unsafe = true  # 允许HTML' "$config_file"
        fi
        rm -f "${config_file}.tmp"
        config_changed=true
    fi
    
    if [ "$config_changed" = true ]; then
        log_success "Hugo配置已优化"
        log_info "配置备份保存在: ${config_file}.backup.*"
        
        # 重新生成静态文件
        regenerate_static_files
    else
        log_success "Hugo配置检查通过"
    fi
}

# 重新生成静态文件
regenerate_static_files() {
    log_info "重新生成静态文件..."
    
    # 清理旧文件
    if [ -d "$LOCAL_PUBLIC" ]; then
        rm -rf "$LOCAL_PUBLIC"
        log_info "已清理旧的public目录"
    fi
    
    # 生成新文件
    if hugo --minify --baseURL "https://www.tommienotes.com"; then
        log_success "静态文件生成完成"
        
        # 验证关键文件
        local test_file="$LOCAL_PUBLIC/posts/2025/go-语言运行时初始化流程深度剖析从-rt0_go-到-main.main/index.html"
        if [ -f "$test_file" ]; then
            local line_count=$(wc -l < "$test_file")
            log_info "测试文件行数: $line_count"
            
            # 检查是否包含代码块
            if grep -q "class=\"highlight\"" "$test_file" || grep -q "<code" "$test_file"; then
                log_success "代码块渲染正常"
            else
                log_warning "代码块可能未正确渲染"
            fi
        fi
    else
        log_error "静态文件生成失败"
        exit 1
    fi
}

# 验证部署内容
validate_deployment_content() {
    log_info "验证部署内容..."
    
    # 检查关键页面
    local test_urls=(
        "https://www.tommienotes.com/"
        "https://www.tommienotes.com/posts/2025/go-语言运行时初始化流程深度剖析从-rt0_go-到-main.main/"
        "https://www.tommienotes.com/archives/"
    )
    
    for url in "${test_urls[@]}"; do
        log_info "检查: $url"
        
        # 检查HTTP状态码
        local status_code=$(curl -s -o /dev/null -w "%{http_code}" "$url")
        if [ "$status_code" = "200" ]; then
            log_success "✓ $url (状态码: $status_code)"
            
            # 对于Go文章，检查是否包含代码内容
            if [[ "$url" == *"go-语言运行时初始化流程"* ]]; then
                if curl -s "$url" | grep -q "MOVQ.*AX"; then
                    log_success "✓ 代码块内容正常显示"
                else
                    log_warning "⚠ 代码块内容可能缺失"
                fi
            fi
        else
            log_warning "✗ $url (状态码: $status_code)"
        fi
    done
    
    # 检查HTTPS重定向
    log_info "检查HTTP到HTTPS重定向..."
    local redirect_status=$(curl -s -o /dev/null -w "%{http_code}" "http://www.tommienotes.com/")
    if [ "$redirect_status" = "301" ]; then
        log_success "✓ HTTP到HTTPS重定向正常"
    else
        log_warning "✗ HTTP重定向异常 (状态码: $redirect_status)"
    fi
}

# 测试SSH连接
test_ssh_connection() {
    log_info "测试SSH连接..."

    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$REMOTE_HOST" "echo 'SSH连接正常'" >/dev/null 2>&1; then
        log_error "SSH连接失败，请检查配置"
        exit 1
    fi

    log_success "SSH连接正常"
}

# 检查远程服务器环境
check_remote_env() {
    log_info "检查远程服务器环境..."

    # 检查nginx
    if ! ssh "$REMOTE_HOST" "command -v nginx" >/dev/null 2>&1; then
        log_warning "nginx未安装，正在安装..."
        ssh "$REMOTE_HOST" "apt update && apt install -y nginx"
    fi

    # 检查网站目录
    ssh "$REMOTE_HOST" "mkdir -p $REMOTE_WEB_ROOT"

    # 检查nginx配置目录
    ssh "$REMOTE_HOST" "mkdir -p /etc/nginx/sites-available /etc/nginx/sites-enabled"

    log_success "远程环境检查完成"
}

# 生成nginx配置
generate_nginx_config() {
    log_info "生成nginx配置..."
    
    local nginx_config_file="deploy/nginx.conf"
    
    if [[ ! -f "$nginx_config_file" ]]; then
        log_error "nginx配置文件不存在: $nginx_config_file"
        return 1
    fi
    
    # 检查SSL证书状态并调整配置
    if ssh $REMOTE_HOST "test -f /etc/letsencrypt/live/www.tommienotes.com/fullchain.pem"; then
        log_info "使用Let's Encrypt证书配置"
        # 直接使用配置文件（Let's Encrypt证书路径已是默认）
        cp "$nginx_config_file" /tmp/nginx-blog.conf
    else
        log_warning "Let's Encrypt证书不存在，切换到自签名证书"
        # 切换到自签名证书路径
        sed 's|ssl_certificate /etc/letsencrypt/live/www.tommienotes.com/fullchain.pem;|# ssl_certificate /etc/letsencrypt/live/www.tommienotes.com/fullchain.pem;|g; s|ssl_certificate_key /etc/letsencrypt/live/www.tommienotes.com/privkey.pem;|# ssl_certificate_key /etc/letsencrypt/live/www.tommienotes.com/privkey.pem;|g; s|# ssl_certificate /etc/ssl/certs/nginx-selfsigned.crt;|ssl_certificate /etc/ssl/certs/nginx-selfsigned.crt;|g; s|# ssl_certificate_key /etc/ssl/private/nginx-selfsigned.key;|ssl_certificate_key /etc/ssl/private/nginx-selfsigned.key;|g' "$nginx_config_file" > /tmp/nginx-blog.conf
    fi
    
    log_success "nginx配置生成完成"
}

# 部署nginx配置
deploy_nginx_config() {
    log_info "部署nginx配置..."

    # 上传配置
    scp /tmp/nginx-blog.conf "$REMOTE_HOST:/tmp/nginx-blog.conf"

    # 安装配置
    ssh "$REMOTE_HOST" "cp /tmp/nginx-blog.conf $REMOTE_NGINX_CONF"

    # 启用站点
    ssh "$REMOTE_HOST" "ln -sf $REMOTE_NGINX_CONF $REMOTE_NGINX_ENABLED"

    # 测试配置
    if ssh "$REMOTE_HOST" "nginx -t"; then
        log_success "nginx配置测试通过"
    else
        log_error "nginx配置有误"
        exit 1
    fi

    log_success "nginx配置部署完成"
}

# 同步网站文件
sync_website_files() {
    log_info "同步网站文件到阿里云服务器..."

    # 使用rsync同步文件，排除索引相关文件，启用压缩和性能优化
    rsync -avz --delete \
        --exclude='.git' \
        --exclude='ai-tools/index/' \
        --exclude='ai-tools/output/' \
        --exclude='ai-tools/cache/' \
        --exclude='ai-tools/*.log' \
        --exclude='*.tmp' \
        --exclude='.DS_Store' \
        --exclude='Thumbs.db' \
        --compress-level=6 \
        --partial \
        --progress \
        "$LOCAL_PUBLIC/" "$REMOTE_HOST:$REMOTE_WEB_ROOT/"

    # 设置正确的文件权限
    ssh "$REMOTE_HOST" "chown -R www-data:www-data $REMOTE_WEB_ROOT"
    ssh "$REMOTE_HOST" "find $REMOTE_WEB_ROOT -type f -name '*.html' -exec chmod 644 {} \;"
    ssh "$REMOTE_HOST" "find $REMOTE_WEB_ROOT -type f -name '*.css' -exec chmod 644 {} \;"
    ssh "$REMOTE_HOST" "find $REMOTE_WEB_ROOT -type f -name '*.js' -exec chmod 644 {} \;"
    ssh "$REMOTE_HOST" "find $REMOTE_WEB_ROOT -type d -exec chmod 755 {} \;"

    log_success "网站文件同步完成"
}

# 构建远程AI索引（已禁用Python依赖）
build_remote_ai_index() {
    log_info "跳过AI索引构建（已禁用Python环境安装）..."
    log_warning "如需AI功能，请手动运行: deploy/build-remote-index.sh"
    log_success "AI索引构建已跳过"
}

# 重启服务
restart_services() {
    log_info "重启nginx服务..."

    ssh "$REMOTE_HOST" "systemctl reload nginx"

    log_success "nginx服务重启完成"
}

# 验证部署
verify_deployment() {
    log_info "验证部署结果..."

    # 获取服务器公网IP
    SERVER_IP=$(ssh "$REMOTE_HOST" "curl -s ifconfig.me" 2>/dev/null || echo "unknown")

    log_info "部署验证:"
    log_info "服务器IP: $SERVER_IP"
    log_info "网站URL: http://$SERVER_IP"
    log_info "nginx状态: $(ssh "$REMOTE_HOST" "systemctl is-active nginx")"

    # 检查关键文件
    ssh "$REMOTE_HOST" "ls -la $REMOTE_WEB_ROOT/index.html"

    log_success "部署验证完成"
}

# 显示部署信息
show_deployment_info() {
    echo ""
    log_success "🎉 阿里云部署完成！"
    echo ""
    echo "📊 部署信息:"
    echo "服务器: $REMOTE_HOST"
    echo "网站目录: $REMOTE_WEB_ROOT"
    echo "nginx配置: $REMOTE_NGINX_CONF"
    echo ""
    
    # 检查SSL证书状态
    if ssh $REMOTE_HOST "test -f /etc/letsencrypt/live/www.tommienotes.com/fullchain.pem"; then
        echo "🔒 HTTPS状态: 已启用"
        echo "📄 SSL证书: Let's Encrypt"
        CERT_EXPIRY=$(ssh $REMOTE_HOST "openssl x509 -enddate -noout -in /etc/letsencrypt/live/www.tommienotes.com/fullchain.pem | cut -d= -f2")
        echo "📅 证书有效期: $CERT_EXPIRY"
        echo ""
        echo "🌐 网站访问:"
        echo "HTTPS主页: https://www.tommienotes.com/"
        echo "HTTP重定向: http://www.tommienotes.com/ → https://www.tommienotes.com/"
        echo ""
        echo "🔍 HTTPS验证命令:"
        echo "curl -I https://www.tommienotes.com"
        echo "curl -I http://www.tommienotes.com  # 应该返回301重定向"
        echo ""
        echo "🔒 SSL证书管理:"
        echo "ssh $REMOTE_HOST 'certbot certificates'     # 查看证书"
        echo "ssh $REMOTE_HOST 'certbot renew'           # 手动续期"
        echo "ssh $REMOTE_HOST 'certbot renew --dry-run' # 测试续期"
    else
        echo "🔓 HTTPS状态: 未启用"
        echo "🌐 网站访问:"
        echo "HTTP主页: http://www.tommienotes.com/"
        echo ""
        echo "🔍 HTTP验证命令:"
        echo "curl -I http://www.tommienotes.com"
        echo ""
        echo "💡 启用HTTPS:"
        echo "./deploy/setup-ssl.sh www.tommienotes.com"
    fi
    
    echo ""
    echo "📝 管理命令:"
    echo "ssh $REMOTE_HOST 'systemctl status nginx'"
    echo "ssh $REMOTE_HOST 'systemctl reload nginx'"
    echo "ssh $REMOTE_HOST 'systemctl restart nginx'"
    echo ""
    echo "🔧 故障排除:"
    echo "ssh $REMOTE_HOST 'nginx -t'                   # 测试配置"
    echo "ssh $REMOTE_HOST 'tail -f /var/log/nginx/error.log'  # 查看错误日志"
}

# 检查和配置SSL证书
setup_ssl_certificates() {
    log_info "检查SSL证书配置..."
    
    # 检查SSL证书是否存在
    if ssh $REMOTE_HOST "test -f /etc/letsencrypt/live/www.tommienotes.com/fullchain.pem"; then
        log_success "SSL证书已存在"
        
        # 检查证书有效期
        CERT_EXPIRY=$(ssh $REMOTE_HOST "openssl x509 -enddate -noout -in /etc/letsencrypt/live/www.tommienotes.com/fullchain.pem | cut -d= -f2")
        log_info "证书有效期至: $CERT_EXPIRY"
        
        # 检查证书是否即将过期（30天内）
        if ssh $REMOTE_HOST "openssl x509 -checkend 2592000 -noout -in /etc/letsencrypt/live/www.tommienotes.com/fullchain.pem"; then
            log_success "SSL证书有效期正常"
        else
            log_warning "SSL证书即将过期，尝试续期..."
            ssh $REMOTE_HOST "certbot renew --quiet"
        fi
    else
        log_warning "SSL证书不存在，尝试自动获取..."
        
        # 检查Certbot是否安装
        if ! ssh $REMOTE_HOST "command -v certbot >/dev/null 2>&1"; then
            log_info "安装Certbot..."
            ssh $REMOTE_HOST "apt-get update && apt-get install -y certbot python3-certbot-nginx"
        fi
        
        # 获取SSL证书
        log_info "获取Let's Encrypt SSL证书..."
        if ssh $REMOTE_HOST "certbot --nginx -d www.tommienotes.com --non-interactive --agree-tos --email admin@tommienotes.com"; then
            log_success "SSL证书获取成功"
        else
            log_error "SSL证书获取失败，将使用自签名证书配置"
            # SSL证书获取失败时，generate_nginx_config会自动使用自签名证书
            return 1
        fi
    fi
    
    return 0
}

# 主函数
main() {
    echo "🚀 阿里云服务器部署脚本"
    echo "========================="

    # 第一阶段：本地环境检查和配置优化
    check_local_env
    check_and_fix_hugo_config
    
    # 第二阶段：远程环境准备
    test_ssh_connection
    check_remote_env
    
    # 第三阶段：SSL证书和HTTPS配置
    if setup_ssl_certificates; then
        log_info "使用HTTPS配置"
        generate_nginx_config
    else
        log_warning "使用HTTP配置"
    fi
    
    # 第四阶段：部署和服务配置
    deploy_nginx_config
    sync_website_files
    build_remote_ai_index
    restart_services
    
    # 第五阶段：验证和测试
    verify_deployment
    validate_deployment_content
    show_deployment_info

    log_success "🎉 部署完成！"
}

# 参数处理
case "${1:-}" in
    "config")
        generate_nginx_config
        deploy_nginx_config
        ;;
    "ssl")
        setup_ssl_certificates
        generate_nginx_config
        deploy_nginx_config
        restart_services
        ;;
    "sync")
        sync_website_files
        ;;
    "restart")
        restart_services
        ;;
    "verify")
        verify_deployment
        ;;
    "index")
        build_remote_ai_index
        ;;
    "fix")
        check_and_fix_hugo_config
        ;;
    "validate")
        validate_deployment_content
        ;;
    "help"|"-h"|"--help")
        echo "用法: $0 [选项]"
        echo ""
        echo "选项:"
        echo "  (无参数)    完整部署（包含HTTPS配置和问题修复）"
        echo "  config      只配置nginx"
        echo "  ssl         配置SSL证书和HTTPS"
        echo "  sync        只同步文件"
        echo "  restart     只重启服务"
        echo "  verify      只验证部署"
        echo "  validate    验证部署内容和功能"
        echo "  index       只构建远程AI索引"
        echo "  fix         检查和修复Hugo配置问题"
        echo "  help        显示此帮助"
        echo ""
        echo "HTTPS相关:"
        echo "  ./deploy/setup-ssl.sh www.tommienotes.com  # 单独配置SSL"
        echo "  $0 ssl                                     # 通过部署脚本配置SSL"
        echo ""
        echo "故障排除:"
        echo "  $0 fix          # 修复Hugo配置问题（如代码块不显示）"
        echo "  $0 validate     # 验证网站内容和功能"
        echo ""
        echo "示例:"
        echo "  $0              # 完整部署（自动检查和配置HTTPS）"
        echo "  $0 ssl          # 只配置SSL和HTTPS"
        echo "  $0 config       # 只更新nginx配置"
        echo "  $0 fix          # 修复代码块渲染等问题"
        ;;
    *)
        main
        ;;
esac
