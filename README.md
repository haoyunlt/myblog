# 我的个人博客

基于Hugo和PaperMod主题搭建的简洁美观的个人博客系统。

## ✨ 特性

- 🚀 **极快的构建速度** - Hugo静态网站生成器
- 📱 **响应式设计** - 完美适配各种设备
- 🎨 **简洁美观** - PaperMod主题，专注内容
- 🔍 **全文搜索** - 内置搜索功能
- 🏷️ **标签分类** - 完善的内容组织
- 📊 **SEO优化** - 搜索引擎友好
- 🌙 **暗色模式** - 支持明暗主题切换
- 💬 **社交分享** - 内置分享按钮

## 🛠️ 技术栈

- **静态网站生成器**: [Hugo](https://gohugo.io/)
- **主题**: [PaperMod](https://github.com/adityatelange/hugo-PaperMod)
- **样式**: CSS3 + 自定义样式
- **部署**: GitHub Pages / Netlify / Vercel
- **版本控制**: Git

## 📦 安装与使用

### 环境要求

- Hugo Extended >= 0.120.0
- Git
- Node.js (可选，用于额外的构建工具)

### 快速开始

1. **克隆项目**
   ```bash
   git clone <your-repo-url>
   cd myblog
   ```

2. **初始化主题**
   ```bash
   git submodule update --init --recursive
   ```

3. **启动开发服务器**
   ```bash
   ./scripts/dev.sh
   # 或者
   hugo server -D
   ```

4. **访问博客**
   打开浏览器访问 `http://localhost:1313`

### 📝 写作流程

1. **创建新文章**
   ```bash
   ./scripts/new-post.sh "文章标题"
   ```

2. **编辑文章**
   - 文章位于 `content/posts/` 目录
   - 使用Markdown格式编写
   - 完成后将 `draft: false`

3. **预览文章**
   ```bash
   hugo server -D  # 包含草稿
   hugo server     # 仅发布的文章
   ```

4. **发布文章**
   - 提交代码到Git仓库
   - 自动触发部署流程

## 📁 项目结构

```
myblog/
├── archetypes/          # 文章模板
├── assets/             # 资源文件
│   └── css/extended/   # 自定义样式
├── content/            # 内容目录
│   ├── posts/         # 博客文章
│   ├── about.md       # 关于页面
│   └── archives.md    # 归档页面
├── data/              # 数据文件
├── layouts/           # 自定义布局
├── static/            # 静态文件
├── themes/            # 主题目录
│   └── PaperMod/     # PaperMod主题
├── scripts/           # 脚本工具
│   ├── dev.sh        # 开发服务器
│   ├── build.sh      # 构建脚本
│   └── new-post.sh   # 新建文章
├── hugo.toml         # Hugo配置
├── netlify.toml      # Netlify配置
├── vercel.json       # Vercel配置
└── .github/workflows/ # GitHub Actions
```

## 🚀 部署

### GitHub Pages

1. 推送代码到GitHub仓库
2. 启用GitHub Pages
3. GitHub Actions自动构建和部署

### Netlify

1. 连接GitHub仓库到Netlify
2. 构建设置已在 `netlify.toml` 中配置
3. 自动部署

### Vercel

1. 导入GitHub仓库到Vercel
2. 配置已在 `vercel.json` 中设置
3. 自动部署

## ⚙️ 配置

### 基本配置

编辑 `hugo.toml` 文件：

```toml
baseURL = "https://yourdomain.com"
title = "我的个人博客"
[params]
  author = "Your Name"
  description = "博客描述"
```

### 菜单配置

```toml
[menu]
  [[menu.main]]
    name = "首页"
    url = "/"
    weight = 10
```

### 社交链接

```toml
[[params.socialIcons]]
  name = "github"
  url = "https://github.com/yourusername"
```

## 🎨 自定义

### 样式定制

- 编辑 `assets/css/extended/custom.css`
- 支持CSS变量和响应式设计
- 自动支持暗色模式

### 布局定制

- 在 `layouts/` 目录添加自定义布局
- 覆盖主题默认模板

## 📊 SEO优化

- ✅ 语义化HTML结构
- ✅ Meta标签优化
- ✅ Open Graph支持
- ✅ 站点地图自动生成
- ✅ RSS订阅支持
- ✅ 结构化数据

## 🔧 开发工具

### 有用的脚本

```bash
# 开发服务器
./scripts/dev.sh

# 构建网站
./scripts/build.sh

# 创建新文章
./scripts/new-post.sh "文章标题"

# 更新主题
git submodule update --remote --merge
```

### Hugo命令

```bash
# 创建新内容
hugo new content posts/my-post.md

# 启动服务器
hugo server -D --bind 0.0.0.0

# 构建网站
hugo --gc --minify

# 检查配置
hugo config
```

## 📝 写作指南

### Front Matter

```yaml
---
title: "文章标题"
date: 2024-01-01T10:00:00+08:00
draft: false
tags: ["标签1", "标签2"]
categories: ["分类"]
series: ["系列名称"]
description: "文章描述"
cover:
    image: "images/cover.jpg"
    alt: "封面图描述"
---
```

### Markdown扩展

- 代码高亮
- 数学公式（KaTeX）
- 图表（Mermaid）
- 表格
- 任务列表

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个博客系统！

## 📄 许可证

MIT License

## 🙏 致谢

- [Hugo](https://gohugo.io/) - 优秀的静态网站生成器
- [PaperMod](https://github.com/adityatelange/hugo-PaperMod) - 简洁美观的主题
- 所有开源贡献者

---

**Happy Blogging! 📝✨**
