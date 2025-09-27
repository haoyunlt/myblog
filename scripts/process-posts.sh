#!/bin/bash

# 博客文章处理脚本包装器
# 提供更方便的命令行接口

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
POSTS_DIR="$PROJECT_ROOT/content/posts"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示帮助信息
show_help() {
    echo -e "${BLUE}博客文章处理脚本${NC}"
    echo "功能："
    echo "  1. 为没有Hugo front matter的文章添加front matter"
    echo "  2. 删除文章中的目录部分"
    echo ""
    echo "用法："
    echo "  $0 [选项] [文件名]"
    echo ""
    echo "选项："
    echo "  -h, --help     显示此帮助信息"
    echo "  -d, --dry-run  试运行，不实际修改文件"
    echo "  -a, --all      处理所有文件"
    echo "  -f, --file     处理指定文件"
    echo ""
    echo "示例："
    echo "  $0 --dry-run                    # 试运行，查看将处理哪些文件"
    echo "  $0 --all                        # 处理所有文件"
    echo "  $0 --file example.md            # 处理指定文件"
    echo "  $0 example.md                   # 处理指定文件（简化语法）"
}

# 确认操作
confirm_action() {
    local message="$1"
    echo -e "${YELLOW}$message${NC}"
    read -p "是否继续？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}操作已取消${NC}"
        exit 1
    fi
}

# 处理所有文件
process_all() {
    local dry_run="$1"
    
    if [[ "$dry_run" == "true" ]]; then
        echo -e "${BLUE}=== 试运行模式 ===${NC}"
        python3 "$SCRIPT_DIR/process-blog-posts.py" --dry-run
    else
        local file_count=$(find "$POSTS_DIR" -name "*.md" | wc -l)
        confirm_action "即将处理 $file_count 个markdown文件"
        
        echo -e "${GREEN}开始处理所有文件...${NC}"
        python3 "$SCRIPT_DIR/process-blog-posts.py"
        echo -e "${GREEN}处理完成！${NC}"
    fi
}

# 处理单个文件
process_file() {
    local filename="$1"
    local filepath="$POSTS_DIR/$filename"
    
    if [[ ! -f "$filepath" ]]; then
        echo -e "${RED}错误: 文件不存在: $filename${NC}"
        echo "请检查文件名是否正确，或使用以下命令查看可用文件："
        echo "  ls $POSTS_DIR/*.md"
        exit 1
    fi
    
    echo -e "${GREEN}处理文件: $filename${NC}"
    python3 "$SCRIPT_DIR/process-blog-posts.py" --file "$filename"
}

# 主逻辑
main() {
    # 检查Python脚本是否存在
    if [[ ! -f "$SCRIPT_DIR/process-blog-posts.py" ]]; then
        echo -e "${RED}错误: 找不到处理脚本 process-blog-posts.py${NC}"
        exit 1
    fi
    
    # 检查posts目录是否存在
    if [[ ! -d "$POSTS_DIR" ]]; then
        echo -e "${RED}错误: 找不到posts目录: $POSTS_DIR${NC}"
        exit 1
    fi
    
    # 解析命令行参数
    case "${1:-}" in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--dry-run)
            process_all "true"
            ;;
        -a|--all)
            process_all "false"
            ;;
        -f|--file)
            if [[ -z "${2:-}" ]]; then
                echo -e "${RED}错误: 请指定文件名${NC}"
                echo "用法: $0 --file <filename.md>"
                exit 1
            fi
            process_file "$2"
            ;;
        "")
            echo -e "${YELLOW}请指定操作选项${NC}"
            show_help
            exit 1
            ;;
        *)
            # 如果参数不以-开头，假设是文件名
            if [[ ! "$1" =~ ^- ]]; then
                process_file "$1"
            else
                echo -e "${RED}错误: 未知选项 $1${NC}"
                show_help
                exit 1
            fi
            ;;
    esac
}

# 运行主函数
main "$@"
