---
title: "GitHub MCP Server - 框架使用示例"
date: 2025-09-28T00:47:16+08:00
draft: false
tags: ['源码分析', '技术文档']
categories:
  - GithubMcpServer
description: "GitHub MCP Server - 框架使用示例的深入技术分析文档"
keywords: ['源码分析', '技术文档']
author: "技术分析师"
weight: 1
---

## 1. 快速开始示例

### 1.1 本地部署示例

```bash
# 1. 创建GitHub Personal Access Token
# 访问 https://github.com/settings/personal-access-tokens/new
# 选择所需权限：repo, read:packages, read:org

# 2. 设置环境变量
export GITHUB_PERSONAL_ACCESS_TOKEN="your_token_here"

# 3. 使用Docker运行
docker run -i --rm \
  -e GITHUB_PERSONAL_ACCESS_TOKEN \
  ghcr.io/github/github-mcp-server

# 4. 或者从源码构建运行
go build -o github-mcp-server cmd/github-mcp-server/main.go
./github-mcp-server stdio
```

### 1.2 VS Code 集成示例

**使用一键安装按钮**：点击README中的"Install in VS Code"按钮

**手动配置** - 在VS Code MCP配置中添加：

```json
{
  "mcp": {
    "servers": {
      "github": {
        "command": "docker",
        "args": [
          "run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
          "ghcr.io/github/github-mcp-server"
        ],
        "env": {
          "GITHUB_PERSONAL_ACCESS_TOKEN": "${input:github_token}"
        }
      }
    },
    "inputs": [
      {
        "type": "promptString",
        "id": "github_token",
        "description": "GitHub Personal Access Token",
        "password": true
      }
    ]
  }
}
```

### 1.3 Claude Desktop 集成示例

**配置文件位置**：

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "github": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "GITHUB_PERSONAL_ACCESS_TOKEN=your_token_here",
        "ghcr.io/github/github-mcp-server"
      ]
    }
  }
}
```

## 2. 工具集配置示例

### 2.1 基本工具集配置

```bash
# 启用特定工具集
./github-mcp-server --toolsets repos,issues,pull_requests,actions stdio

# 使用环境变量
GITHUB_TOOLSETS="repos,issues,pull_requests,actions" ./github-mcp-server stdio

# 启用所有工具集
./github-mcp-server --toolsets all stdio

# 只读模式
./github-mcp-server --read-only stdio
```

### 2.2 Docker 工具集配置

```bash
# 启用特定工具集
docker run -i --rm \
  -e GITHUB_PERSONAL_ACCESS_TOKEN=<your-token> \
  -e GITHUB_TOOLSETS="repos,issues,pull_requests,actions,code_security" \
  ghcr.io/github/github-mcp-server

# 启用动态工具集
docker run -i --rm \
  -e GITHUB_PERSONAL_ACCESS_TOKEN=<your-token> \
  -e GITHUB_DYNAMIC_TOOLSETS=1 \
  ghcr.io/github/github-mcp-server

# 只读模式
docker run -i --rm \
  -e GITHUB_PERSONAL_ACCESS_TOKEN=<your-token> \
  -e GITHUB_READ_ONLY=1 \
  ghcr.io/github/github-mcp-server
```

### 2.3 企业版配置示例

```bash
# GitHub Enterprise Server
docker run -i --rm \
  -e GITHUB_PERSONAL_ACCESS_TOKEN=<your-token> \
  -e GITHUB_HOST="https://github.enterprise.com" \
  ghcr.io/github/github-mcp-server

# GitHub Enterprise Cloud (data residency)
docker run -i --rm \
  -e GITHUB_PERSONAL_ACCESS_TOKEN=<your-token> \
  -e GITHUB_HOST="https://yourcompany.ghe.com" \
  ghcr.io/github/github-mcp-server
```

## 3. 常见使用场景示例

### 3.1 仓库管理场景

```javascript
// AI工具可以执行的自然语言命令示例

"帮我在 microsoft/vscode 仓库中搜索包含 'extension' 的文件"
→ 调用 search_code 工具

"创建一个新的仓库叫 my-awesome-project"
→ 调用 create_repository 工具

"获取 react 仓库的 README.md 文件内容"
→ 调用 get_file_contents 工具

"在 main 分支上创建一个新文件 docs/api.md"
→ 调用 create_or_update_file 工具
```

### 3.2 Issue 管理场景

```javascript
"列出我的仓库中所有开放的 bug 标签的 issues"
→ 调用 list_issues 工具，参数: labels: ["bug"], state: "OPEN"

"创建一个新的 issue 标题为 'Fix login bug'"
→ 调用 create_issue 工具

"给 issue #123 添加评论说明问题已经修复"
→ 调用 add_issue_comment 工具

"将 issue #456 分配给 Copilot 来处理"
→ 调用 assign_copilot_to_issue 工具
```

### 3.3 CI/CD 自动化场景

```javascript
"运行仓库中的 CI 工作流"
→ 调用 run_workflow 工具

"检查最新的构建状态"
→ 调用 list_workflow_runs 工具

"获取失败作业的日志来诊断问题"
→ 调用 get_job_logs 工具，参数: failed_only: true

"取消正在运行的工作流"
→ 调用 cancel_workflow_run 工具
```

### 3.4 Pull Request 管理场景

```javascript
"创建一个从 feature-branch 到 main 的 PR"
→ 调用 create_pull_request 工具

"获取 PR #789 的文件变更列表"
→ 调用 get_pull_request_files 工具

"合并已通过审查的 PR"
→ 调用 merge_pull_request 工具

"请求 Copilot 审查这个 PR"
→ 调用 request_copilot_review 工具
```

## 4. 高级配置示例

### 4.1 自定义翻译配置

```json
// github-mcp-server-config.json
{
  "TOOL_GET_ISSUE_DESCRIPTION": "获取GitHub仓库中特定issue的详细信息",
  "TOOL_CREATE_ISSUE_DESCRIPTION": "在GitHub仓库中创建新的issue",
  "TOOL_LIST_WORKFLOWS_DESCRIPTION": "列出仓库中的所有GitHub Actions工作流"
}
```

```bash
# 导出当前翻译配置
./github-mcp-server --export-translations

# 使用环境变量覆盖描述
export GITHUB_MCP_TOOL_GET_ISSUE_DESCRIPTION="获取issue的详细信息和状态"
./github-mcp-server stdio
```

### 4.2 内容窗口大小配置

```bash
# 设置较大的内容窗口以处理大文件
./github-mcp-server --content-window-size 10000 stdio

# 使用环境变量
GITHUB_CONTENT_WINDOW_SIZE=10000 ./github-mcp-server stdio
```

### 4.3 日志和调试配置

```bash
# 启用命令日志记录
./github-mcp-server --enable-command-logging --log-file debug.log stdio

# 使用环境变量
GITHUB_ENABLE_COMMAND_LOGGING=1 GITHUB_LOG_FILE=debug.log ./github-mcp-server stdio
```

## 5. 编程接口使用示例

### 5.1 Go 库使用示例

```go
package main

import (
    "context"
    "log"
    
    "github.com/github/github-mcp-server/internal/ghmcp"
    "github.com/github/github-mcp-server/pkg/github"
)

func main() {
    // 创建服务器配置
    config := ghmcp.MCPServerConfig{
        Version:           "1.0.0",
        Host:              "", // 使用默认的 GitHub.com
        Token:             "your-github-token",
        EnabledToolsets:   []string{"repos", "issues"},
        DynamicToolsets:   false,
        ReadOnly:          false,
        ContentWindowSize: 5000,
    }
    
    // 创建 MCP 服务器实例
    server, err := ghmcp.NewMCPServer(config)
    if err != nil {
        log.Fatalf("Failed to create MCP server: %v", err)
    }
    
    // 这里可以添加自定义工具
    // server.AddTool(customTool, customHandler)
    
    log.Println("MCP Server created successfully")
}
```

### 5.2 自定义工具实现示例

```go
// 自定义工具实现
func CustomRepositoryTool(getClient github.GetClientFn, t translations.TranslationHelperFunc) (tool mcp.Tool, handler server.ToolHandlerFunc) {
    return mcp.NewTool("custom_repo_info",
            // 工具描述和参数定义
            mcp.WithDescription(t("CUSTOM_REPO_INFO_DESCRIPTION", "获取自定义仓库信息")),
            mcp.WithString("owner",
                mcp.Required(),
                mcp.Description("仓库所有者"),
            ),
            mcp.WithString("repo",
                mcp.Required(),
                mcp.Description("仓库名称"),
            ),
        ),
        // 工具处理函数
        func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
            // 获取参数
            owner, err := github.RequiredParam[string](request, "owner")
            if err != nil {
                return mcp.NewToolResultError(err.Error()), nil
            }
            
            repo, err := github.RequiredParam[string](request, "repo")
            if err != nil {
                return mcp.NewToolResultError(err.Error()), nil
            }
            
            // 获取 GitHub 客户端
            client, err := getClient(ctx)
            if err != nil {
                return nil, fmt.Errorf("failed to get GitHub client: %w", err)
            }
            
            // 调用 GitHub API
            repository, resp, err := client.Repositories.Get(ctx, owner, repo)
            if err != nil {
                return nil, fmt.Errorf("failed to get repository: %w", err)
            }
            defer resp.Body.Close()
            
            // 处理结果
            result := map[string]interface{}{
                "name":        repository.GetName(),
                "description": repository.GetDescription(),
                "stars":       repository.GetStargazersCount(),
                "forks":       repository.GetForksCount(),
                "language":    repository.GetLanguage(),
            }
            
            // 序列化并返回结果
            r, err := json.Marshal(result)
            if err != nil {
                return nil, fmt.Errorf("failed to marshal result: %w", err)
            }
            
            return mcp.NewToolResultText(string(r)), nil
        }
}
```

### 5.3 工具集注册示例

```go
// 创建自定义工具集
func createCustomToolset(getClient github.GetClientFn, t translations.TranslationHelperFunc) *toolsets.Toolset {
    customToolset := toolsets.NewToolset("custom", "自定义工具集")
        
    // 添加只读工具
    customToolset.AddReadTools(
        toolsets.NewServerTool(CustomRepositoryTool(getClient, t)),
        toolsets.NewServerTool(AnotherReadOnlyTool(getClient, t)),
    )
    
    // 添加写操作工具
    customToolset.AddWriteTools(
        toolsets.NewServerTool(CustomCreateTool(getClient, t)),
    )
    
    return customToolset
}

// 注册到工具集组
func setupCustomToolsets() *toolsets.ToolsetGroup {
    tsg := toolsets.NewToolsetGroup(false) // false = 允许写操作
    
    // 注册默认工具集
    defaultToolset := github.DefaultToolsetGroup(false, getClient, getGQLClient, getRawClient, translator, 5000)
    
    // 添加自定义工具集
    customToolset := createCustomToolset(getClient, translator)
    tsg.AddToolset(customToolset)
    
    return tsg
}
```

## 6. 故障排除和调试示例

### 6.1 常见问题诊断

```bash
# 1. 检查token权限
curl -H "Authorization: Bearer YOUR_TOKEN" https://api.github.com/user

# 2. 启用详细日志
./github-mcp-server --enable-command-logging --log-file debug.log stdio

# 3. 测试特定工具集
./github-mcp-server --toolsets context stdio

# 4. 检查企业版连接
./github-mcp-server --gh-host "https://github.enterprise.com" stdio
```

### 6.2 性能优化配置

```bash
# 减少内容窗口大小以处理大量数据
./github-mcp-server --content-window-size 1000 stdio

# 只启用必需的工具集
./github-mcp-server --toolsets "repos,issues" stdio

# 启用只读模式提高性能
./github-mcp-server --read-only stdio
```

### 6.3 错误处理示例

```go
// 在工具实现中的错误处理
func ExampleToolWithErrorHandling(getClient github.GetClientFn, t translations.TranslationHelperFunc) (tool mcp.Tool, handler server.ToolHandlerFunc) {
    return mcp.NewTool("example_tool",
            mcp.WithDescription("示例工具"),
        ),
        func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
            client, err := getClient(ctx)
            if err != nil {
                return nil, fmt.Errorf("failed to get GitHub client: %w", err)
            }
            
            // API调用
            result, resp, err := client.Repositories.Get(ctx, "owner", "repo")
            if err != nil {
                // 使用错误处理器创建详细的错误响应
                return ghErrors.NewGitHubAPIErrorResponse(ctx,
                    "failed to get repository",
                    resp,
                    err,
                ), nil
            }
            defer resp.Body.Close()
            
            // 检查HTTP状态码
            if resp.StatusCode != 200 {
                body, _ := io.ReadAll(resp.Body)
                return mcp.NewToolResultError(fmt.Sprintf("API error: %s", string(body))), nil
            }
            
            return github.MarshalledTextResult(result), nil
        }
}
```

## 7. 最佳实践示例

### 7.1 安全配置最佳实践

```bash
# 1. 使用环境变量而不是硬编码token
export GITHUB_PAT="your_token_here"

# 2. 创建.env文件（记得添加到.gitignore）
echo "GITHUB_PAT=your_token_here" > .env
echo ".env" >> .gitignore

# 3. 使用最小权限原则
# 只授予必要的scope：repo, read:packages, read:org

# 4. 定期轮换token
# 建议每90天更新一次PAT

# 5. 限制配置文件权限
chmod 600 ~/.config/mcp-server.json
```

### 7.2 生产环境部署示例

```yaml
# docker-compose.yml
version: '3.8'
services:
  github-mcp-server:
    image: ghcr.io/github/github-mcp-server:latest
    environment:

      - GITHUB_PERSONAL_ACCESS_TOKEN=${GITHUB_PAT}
      - GITHUB_TOOLSETS=repos,issues,pull_requests,actions
      - GITHUB_READ_ONLY=false
      - GITHUB_CONTENT_WINDOW_SIZE=5000
      - GITHUB_ENABLE_COMMAND_LOGGING=true
      - GITHUB_LOG_FILE=/logs/github-mcp.log
    volumes:
      - ./logs:/logs
    restart: unless-stopped
    stdin_open: true
    tty: true

```

### 7.3 监控和日志分析

```bash
# 分析日志中的API调用模式
grep "API Request" debug.log | awk '{print $3}' | sort | uniq -c

# 监控错误率
grep "ERROR" debug.log | wc -l

# 查看最频繁使用的工具
grep "Tool Called" debug.log | awk '{print $4}' | sort | uniq -c | sort -nr
```

这些示例展示了 GitHub MCP Server 的各种使用场景，从简单的本地部署到复杂的生产环境配置，为用户提供了全面的参考指南。
