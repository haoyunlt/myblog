#!/bin/bash

# 类目功能测试脚本
echo "🏷️ 开始测试文章类目功能..."

BASE_URL="http://localhost:1313"

# 测试函数
test_url() {
    local url="$1"
    local name="$2"
    local status_code=$(curl -s -o /dev/null -w "%{http_code}" "$url")
    
    if [ "$status_code" = "200" ]; then
        echo "✅ $name - 测试通过 ($status_code)"
        return 0
    else
        echo "❌ $name - 测试失败 ($status_code)"
        return 1
    fi
}

# 检查首页是否包含类目信息
echo ""
echo "📄 测试首页类目展示..."
test_url "$BASE_URL/" "首页"

# 检查首页内容是否包含类目
echo ""
echo "🔍 检查首页类目内容..."
homepage_content=$(curl -s "$BASE_URL/")

if echo "$homepage_content" | grep -q "文章类目"; then
    echo "✅ 首页包含类目标题"
else
    echo "❌ 首页缺少类目标题"
fi

if echo "$homepage_content" | grep -q "categories-grid"; then
    echo "✅ 首页包含类目网格"
else
    echo "❌ 首页缺少类目网格"
fi

# 检查主要类目
echo ""
echo "📚 检查主要类目..."
categories=("autogen" "langchain" "mysql" "kubernetes" "golang" "pytorch" "kafka" "grpc")

for category in "${categories[@]}"; do
    if echo "$homepage_content" | grep -q "$category"; then
        echo "✅ 找到类目: $category"
    else
        echo "❌ 缺少类目: $category"
    fi
done

# 统计文章数量
echo ""
echo "📊 文章统计信息..."
echo "总文章数量: $(ls content/posts/*.md | wc -l)"
echo "类目数量: $(ls content/posts/*.md | cut -d'/' -f3 | cut -d'-' -f1 | sort | uniq | wc -l)"

echo ""
echo "🔝 文章数量最多的类目:"
ls content/posts/*.md | cut -d'/' -f3 | cut -d'-' -f1 | sort | uniq -c | sort -nr | head -5

echo ""
echo "🎉 类目功能测试完成！"
echo ""
echo "📋 访问地址："
echo "   🏠 首页（含类目）: $BASE_URL/"
echo "   📚 归档页面: $BASE_URL/archives/"
echo "   🏷️ 分类页面: $BASE_URL/categories/"
echo ""
echo "💡 提示："
echo "   - 首页现在显示按文件名第一个字符串分组的类目"
echo "   - 每个类目显示文章数量和最新3篇文章"
echo "   - 点击类目名称可查看该类目下的所有文章"
