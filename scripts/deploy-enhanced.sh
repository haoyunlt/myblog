#!/bin/bash

# 🚀 增强版阿里云部署脚本
# 支持智能构建、快速部署和全面验证

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

# 部署配置
REMOTE_HOST="blog-svr"
REMOTE_USER="root"
REMOTE_WEB_ROOT="/var/www/html"
REMOTE_NGINX_CONF="/etc/nginx/sites-available/blog"
REMOTE_NGINX_ENABLED="/etc/nginx/sites-enabled/blog"
LOCAL_PUBLIC="./public"
BUILD_SCRIPT="./scripts/build-mobile-optimized.sh"
DEPLOY_LOG="deploy-$(date +%Y%m%d_%H%M%S).log"

# SSH配置
SSH_OPTS="-o ConnectTimeout=10 -o ServerAliveInterval=60 -o ServerAliveCountMax=3"

# 显示部署横幅
show_banner() {
    echo -e "${CYAN}"
    echo "🚀 增强版阿里云部署系统"
    echo "=========================="
    echo "目标服务器: $REMOTE_HOST"
    echo "部署目录: $REMOTE_WEB_ROOT"  
    echo "日志文件: $DEPLOY_LOG"
    echo "开始时间: $(date)"
    echo -e "${NC}"
}

# 预检查本地环境
pre_check_local() {
    log_step "预检查本地环境"
    
    # 检查必要命令
    local required_commands=("hugo" "rsync" "ssh" "scp")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            log_error "缺少必要命令: $cmd"
            exit 1
        fi
    done
    
    # 检查SSH配置
    if [[ ! -f ~/.ssh/config ]] || ! grep -q "$REMOTE_HOST" ~/.ssh/config; then
        log_warning "SSH配置可能不完整，请确保 $REMOTE_HOST 已在 ~/.ssh/config 中配置"
    fi
    
    # 检查构建脚本
    if [[ ! -x "$BUILD_SCRIPT" ]]; then
        if [[ -f "$BUILD_SCRIPT" ]]; then
            chmod +x "$BUILD_SCRIPT"
            log_info "已为构建脚本添加执行权限"
        else
            log_error "构建脚本不存在: $BUILD_SCRIPT"
            exit 1
        fi
    fi
    
    log_success "本地环境检查通过"
}

# 测试SSH连接
test_ssh_connection() {
    log_step "测试SSH连接"
    
    if ssh $SSH_OPTS "$REMOTE_HOST" "echo 'SSH连接正常'" >/dev/null 2>&1; then
        log_success "SSH连接测试通过"
        
        # 获取服务器基本信息
        local server_info=$(ssh $SSH_OPTS "$REMOTE_HOST" "uname -a && uptime" 2>/dev/null)
        log_info "服务器信息: $(echo "$server_info" | head -1)"
        log_info "服务器负载: $(echo "$server_info" | tail -1 | awk '{print $3,$4,$5}')"
    else
        log_error "SSH连接失败，请检查："
        echo "1. 服务器是否正常运行"
        echo "2. SSH配置是否正确"
        echo "3. 网络连接是否正常"
        exit 1
    fi
}

# 智能构建检查
smart_build_check() {
    log_step "智能构建检查"
    
    local need_rebuild=false
    
    # 检查public目录是否存在
    if [[ ! -d "$LOCAL_PUBLIC" ]]; then
        log_info "public目录不存在，需要构建"
        need_rebuild=true
    else
        # 检查内容是否有更新
        local content_newer=$(find content -newer "$LOCAL_PUBLIC" -type f | wc -l)
        local config_newer=$(find . -maxdepth 1 -name "*.toml" -newer "$LOCAL_PUBLIC" -type f | wc -l)
        
        if [ "$content_newer" -gt 0 ] || [ "$config_newer" -gt 0 ]; then
            log_info "检测到内容或配置更新，需要重新构建"
            need_rebuild=true
        else
            log_success "构建文件是最新的"
        fi
    fi
    
    return $need_rebuild
}

# 执行智能构建
execute_build() {
    log_step "执行构建"
    
    if smart_build_check; then
        log_info "开始执行增强构建..."
        if "$BUILD_SCRIPT"; then
            log_success "构建完成"
        else
            log_error "构建失败"
            exit 1
        fi
    else
        log_success "跳过构建，使用现有文件"
    fi
}

# 验证构建产物
verify_build_artifacts() {
    log_step "验证构建产物"
    
    if [[ ! -d "$LOCAL_PUBLIC" ]]; then
        log_error "构建目录不存在: $LOCAL_PUBLIC"
        exit 1
    fi
    
    # 关键文件检查
    local required_files=("index.html" "index.xml" "404.html")
    for file in "${required_files[@]}"; do
        if [[ ! -f "$LOCAL_PUBLIC/$file" ]]; then
            log_error "缺少关键文件: $file"
            exit 1
        fi
    done
    
    # 统计信息
    local file_count=$(find "$LOCAL_PUBLIC" -type f | wc -l)
    local dir_size=$(du -sh "$LOCAL_PUBLIC" | cut -f1)
    
    log_success "构建产物验证通过"
    log_info "文件数量: $file_count, 总大小: $dir_size"
}

# 检查远程环境
check_remote_environment() {
    log_step "检查远程环境"
    
    # 检查并安装nginx
    if ! ssh $SSH_OPTS "$REMOTE_HOST" "command -v nginx" >/dev/null 2>&1; then
        log_warning "Nginx未安装，正在安装..."
        ssh $SSH_OPTS "$REMOTE_HOST" "
            apt update >/dev/null 2>&1 && 
            apt install -y nginx >/dev/null 2>&1
        " || {
            log_error "Nginx安装失败"
            exit 1
        }
        log_success "Nginx安装完成"
    else
        log_success "Nginx已安装"
    fi
    
    # 创建必要目录
    ssh $SSH_OPTS "$REMOTE_HOST" "
        mkdir -p $REMOTE_WEB_ROOT &&
        mkdir -p /etc/nginx/sites-available &&
        mkdir -p /etc/nginx/sites-enabled &&
        mkdir -p /var/log/nginx
    " || {
        log_error "目录创建失败"
        exit 1
    }
    
    log_success "远程环境检查完成"
}

# 智能同步文件
smart_sync_files() {
    log_step "同步网站文件"
    
    local sync_start=$(date +%s)
    
    # 使用rsync进行高效同步
    log_info "正在同步文件到 $REMOTE_HOST:$REMOTE_WEB_ROOT ..."
    
    rsync -avz --delete \
        --progress \
        --exclude='.git' \
        --exclude='ai-tools/index/' \
        --exclude='ai-tools/output/' \
        --exclude='ai-tools/cache/' \
        --exclude='*.log' \
        --exclude='.DS_Store' \
        --exclude='Thumbs.db' \
        --compress-level=6 \
        --stats \
        "$LOCAL_PUBLIC/" "$REMOTE_HOST:$REMOTE_WEB_ROOT/" || {
        log_error "文件同步失败"
        exit 1
    }
    
    local sync_end=$(date +%s)
    local sync_time=$((sync_end - sync_start))
    
    log_success "文件同步完成 (耗时: ${sync_time}秒)"
}

# 设置文件权限
set_file_permissions() {
    log_step "设置文件权限"
    
    ssh $SSH_OPTS "$REMOTE_HOST" "
        chown -R www-data:www-data $REMOTE_WEB_ROOT &&
        find $REMOTE_WEB_ROOT -type f -exec chmod 644 {} \; &&
        find $REMOTE_WEB_ROOT -type d -exec chmod 755 {} \;
    " || {
        log_error "权限设置失败"
        exit 1
    }
    
    log_success "文件权限设置完成"
}

# 部署Nginx配置
deploy_nginx_config() {
    log_step "部署Nginx配置"
    
    local nginx_config="deploy/nginx.conf"
    
    if [[ ! -f "$nginx_config" ]]; then
        log_error "Nginx配置文件不存在: $nginx_config"
        exit 1
    fi
    
    # 检查SSL证书并选择合适的配置
    if ssh $SSH_OPTS "$REMOTE_HOST" "test -f /etc/letsencrypt/live/www.tommienotes.com/fullchain.pem"; then
        log_info "使用HTTPS配置"
        scp "$nginx_config" "$REMOTE_HOST:/tmp/nginx-blog.conf"
    else
        log_warning "SSL证书不存在，使用HTTP配置"
        # 修改配置以使用自签名证书或HTTP
        sed 's|ssl_certificate /etc/letsencrypt/live/www.tommienotes.com/fullchain.pem;|# ssl_certificate /etc/letsencrypt/live/www.tommienotes.com/fullchain.pem;|g; s|ssl_certificate_key /etc/letsencrypt/live/www.tommienotes.com/privkey.pem;|# ssl_certificate_key /etc/letsencrypt/live/www.tommienotes.com/privkey.pem;|g' "$nginx_config" > /tmp/nginx-blog-http.conf
        scp /tmp/nginx-blog-http.conf "$REMOTE_HOST:/tmp/nginx-blog.conf"
        rm -f /tmp/nginx-blog-http.conf
    fi
    
    # 部署配置
    ssh $SSH_OPTS "$REMOTE_HOST" "
        cp /tmp/nginx-blog.conf $REMOTE_NGINX_CONF &&
        ln -sf $REMOTE_NGINX_CONF $REMOTE_NGINX_ENABLED &&
        nginx -t
    " || {
        log_error "Nginx配置部署失败"
        exit 1
    }
    
    log_success "Nginx配置部署完成"
}

# 重启服务
restart_nginx() {
    log_step "重启Nginx服务"
    
    ssh $SSH_OPTS "$REMOTE_HOST" "systemctl reload nginx" || {
        log_error "Nginx重启失败"
        exit 1
    }
    
    log_success "Nginx服务重启完成"
}

# 部署验证
verify_deployment() {
    log_step "验证部署"
    
    # 检查服务状态
    local nginx_status=$(ssh $SSH_OPTS "$REMOTE_HOST" "systemctl is-active nginx")
    if [[ "$nginx_status" == "active" ]]; then
        log_success "Nginx服务运行正常"
    else
        log_error "Nginx服务状态异常: $nginx_status"
        return 1
    fi
    
    # 检查关键文件
    local key_files=("$REMOTE_WEB_ROOT/index.html" "$REMOTE_WEB_ROOT/index.xml")
    for file in "${key_files[@]}"; do
        if ssh $SSH_OPTS "$REMOTE_HOST" "test -f $file"; then
            log_success "关键文件存在: $(basename "$file")"
        else
            log_error "关键文件缺失: $(basename "$file")"
            return 1
        fi
    done
    
    # 获取服务器IP
    local server_ip=$(ssh $SSH_OPTS "$REMOTE_HOST" "curl -s ifconfig.me" 2>/dev/null || echo "unknown")
    log_info "服务器IP: $server_ip"
    
    return 0
}

# 在线验证
online_verification() {
    log_step "在线验证"
    
    local base_url="https://www.tommienotes.com"
    local test_urls=("$base_url/" "$base_url/posts/" "$base_url/archives/")
    
    log_info "测试网站可访问性..."
    
    for url in "${test_urls[@]}"; do
        local status_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 "$url" 2>/dev/null)
        if [[ "$status_code" == "200" ]]; then
            log_success "✓ $url (状态码: $status_code)"
        else
            log_warning "✗ $url (状态码: $status_code)"
        fi
        sleep 1
    done
    
    # 检查HTTP重定向到HTTPS
    local redirect_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 "http://www.tommienotes.com/" 2>/dev/null)
    if [[ "$redirect_code" == "301" ]]; then
        log_success "✓ HTTP到HTTPS重定向正常"
    else
        log_info "HTTP重定向状态: $redirect_code"
    fi
}

# 生成部署报告
generate_deploy_report() {
    log_step "生成部署报告"
    
    {
        echo "阿里云部署报告"
        echo "================"
        echo "部署时间: $(date)"
        echo "目标服务器: $REMOTE_HOST"
        echo "部署目录: $REMOTE_WEB_ROOT"
        echo ""
        
        echo "构建信息:"
        if [[ -d "$LOCAL_PUBLIC" ]]; then
            echo "- 本地文件数: $(find "$LOCAL_PUBLIC" -type f | wc -l)"
            echo "- 本地大小: $(du -sh "$LOCAL_PUBLIC" | cut -f1)"
        fi
        echo ""
        
        echo "服务器状态:"
        local server_info=$(ssh $SSH_OPTS "$REMOTE_HOST" "
            echo 'Nginx状态: '$(systemctl is-active nginx)
            echo '磁盘使用: '$(df -h $REMOTE_WEB_ROOT | tail -1 | awk '{print \$5}')
            echo '内存使用: '$(free -h | grep Mem | awk '{print \$3\"/\"\$2}')
        " 2>/dev/null)
        echo "$server_info"
        echo ""
        
        echo "访问地址:"
        echo "- 主页: https://www.tommienotes.com/"
        echo "- 文章: https://www.tommienotes.com/posts/"
        echo "- 归档: https://www.tommienotes.com/archives/"
        
    } > "$DEPLOY_LOG"
    
    log_success "部署报告已生成: $DEPLOY_LOG"
}

# 显示部署完成信息
show_completion_info() {
    echo ""
    echo -e "${GREEN}🎉 阿里云部署完成！${NC}"
    echo ""
    echo -e "${CYAN}📊 部署摘要:${NC}"
    echo "服务器: $REMOTE_HOST"
    echo "网站目录: $REMOTE_WEB_ROOT"
    echo "日志文件: $DEPLOY_LOG"
    echo ""
    
    echo -e "${CYAN}🌐 访问地址:${NC}"
    echo "主页: https://www.tommienotes.com/"
    echo "管理: ssh $REMOTE_HOST"
    echo ""
    
    echo -e "${CYAN}🔧 常用命令:${NC}"
    echo "检查状态: ssh $REMOTE_HOST 'systemctl status nginx'"
    echo "重启服务: ssh $REMOTE_HOST 'systemctl reload nginx'"
    echo "查看日志: ssh $REMOTE_HOST 'tail -f /var/log/nginx/error.log'"
}

# 主函数
main() {
    # 开始记录日志
    exec > >(tee -a "$DEPLOY_LOG")
    exec 2>&1
    
    local deploy_start=$(date +%s)
    
    show_banner
    
    # 第一阶段：预检查
    pre_check_local
    test_ssh_connection
    
    # 第二阶段：构建
    execute_build
    verify_build_artifacts
    
    # 第三阶段：部署
    check_remote_environment
    deploy_nginx_config
    smart_sync_files
    set_file_permissions
    restart_nginx
    
    # 第四阶段：验证
    if verify_deployment; then
        online_verification
        generate_deploy_report
        
        local deploy_end=$(date +%s)
        local total_time=$((deploy_end - deploy_start))
        
        show_completion_info
        echo "总部署时间: ${total_time}秒"
    else
        log_error "部署验证失败"
        exit 1
    fi
}

# 快捷功能函数
quick_sync() {
    log_info "快速同步模式"
    test_ssh_connection
    smart_sync_files
    set_file_permissions
    restart_nginx
    verify_deployment
}

quick_build() {
    log_info "快速构建模式"
    execute_build
    verify_build_artifacts
}

# 参数处理
case "${1:-}" in
    "build")
        quick_build
        ;;
    "sync")
        quick_sync
        ;;
    "nginx")
        test_ssh_connection
        check_remote_environment
        deploy_nginx_config
        restart_nginx
        ;;
    "verify")
        verify_deployment
        online_verification
        ;;
    "help"|"-h"|"--help")
        echo "用法: $0 [选项]"
        echo ""
        echo "选项:"
        echo "  (无参数)    完整部署流程"
        echo "  build       只执行构建"
        echo "  sync        快速同步文件"
        echo "  nginx       只更新nginx配置"
        echo "  verify      只验证部署"
        echo "  help        显示此帮助"
        echo ""
        echo "示例:"
        echo "  $0              # 完整部署"
        echo "  $0 build        # 只构建"
        echo "  $0 sync         # 快速同步"
        ;;
    *)
        main
        ;;
esac
