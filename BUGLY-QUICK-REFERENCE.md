# Bugly 日志上报 - 快速参考卡片

## 🚀 快速开始

### 通过界面上报（最简单）
1. 访问：https://www.tommienotes.com/bugly-dashboard.html
2. 点击：**📝 上报日志**
3. 填写并提交

### 通过代码上报
```javascript
// 复制这段代码到你的项目
function logToBugly(level, message, details = '', tags = []) {
    const log = {
        type: `log_${level}`,
        message, details, tags, level,
        timestamp: Date.now(),
        url: location.href,
        userAgent: navigator.userAgent,
        sessionId: 'manual_' + Date.now(),
        source: 'manual_report'
    };
    const reports = JSON.parse(localStorage.getItem('bugly_reports') || '[]');
    reports.push(log);
    localStorage.setItem('bugly_reports', JSON.stringify(reports.slice(-1000)));
}

// 使用示例
logToBugly('info', '用户登录', '用户ID: 123', ['用户', '认证']);
logToBugly('warn', 'API响应慢', '耗时: 3.5秒', ['性能', 'API']);
logToBugly('error', '支付失败', '错误码: 500', ['支付', '错误']);
```

## 📊 日志级别速查

| 级别 | 图标 | 颜色 | 用途 | 示例 |
|------|------|------|------|------|
| **debug** | 🔍 | 灰色 | 调试信息 | `logToBugly('debug', '变量值', 'x=10, y=20')` |
| **info** | ℹ️ | 蓝色 | 一般信息 | `logToBugly('info', '操作完成', '用户点击按钮')` |
| **warn** | ⚠️ | 橙色 | 警告 | `logToBugly('warn', '性能问题', '加载时间>3秒')` |
| **error** | ❌ | 红色 | 错误 | `logToBugly('error', 'API失败', '状态码: 500')` |

## 🎯 常用场景

### 场景 1：用户操作
```javascript
logToBugly('info', '用户点击购买', '商品: SKU123', ['用户', '购买']);
```

### 场景 2：性能监控
```javascript
const time = performance.now() - startTime;
logToBugly('warn', '操作耗时过长', `${time}ms`, ['性能']);
```

### 场景 3：API 调用
```javascript
logToBugly('info', 'API调用', 'GET /api/user/123', ['API', '成功']);
```

### 场景 4：错误捕获
```javascript
try {
    // 代码
} catch (e) {
    logToBugly('error', '操作失败', e.message, ['错误']);
}
```

## 🛠️ 工具函数库

```javascript
// 一次性复制所有工具函数
const Logger = {
    // 基础日志
    debug: (msg, details = '', tags = []) => logToBugly('debug', msg, details, tags),
    info: (msg, details = '', tags = []) => logToBugly('info', msg, details, tags),
    warn: (msg, details = '', tags = []) => logToBugly('warn', msg, details, tags),
    error: (msg, details = '', tags = []) => logToBugly('error', msg, details, tags),
    
    // 性能日志
    perf: (name, duration) => {
        const level = duration > 1000 ? 'warn' : 'info';
        logToBugly(level, `${name} 完成`, `耗时: ${duration}ms`, ['性能']);
    },
    
    // API 日志
    api: (method, url, status, duration) => {
        const level = status >= 400 ? 'error' : status >= 300 ? 'warn' : 'info';
        logToBugly(level, `API ${method}`, `${url} - ${status} (${duration}ms)`, ['API']);
    },
    
    // 用户行为
    action: (action, details = '') => {
        logToBugly('info', `用户${action}`, details, ['用户行为']);
    }
};

// 使用
Logger.info('页面加载完成');
Logger.perf('数据加载', 250);
Logger.api('GET', '/api/users', 200, 150);
Logger.action('点击按钮', '按钮ID: btn-submit');
```

## 🔍 查询和筛选

### 按类型筛选
仪表板 → 类型过滤 → 选择日志级别

### 按关键词搜索
仪表板 → 搜索框 → 输入关键词

### 组合查询
同时使用类型筛选 + 关键词搜索

## 📥 数据管理

### 查看所有日志
```javascript
JSON.parse(localStorage.getItem('bugly_reports') || '[]')
```

### 查看日志数量
```javascript
JSON.parse(localStorage.getItem('bugly_reports') || '[]').length
```

### 清除所有日志
```javascript
localStorage.removeItem('bugly_reports')
```

### 导出日志
仪表板 → **📥 导出JSON** 按钮

## ⌨️ 快捷键

| 操作 | 快捷键 |
|------|--------|
| 关闭模态框 | `ESC` |
| 刷新页面 | `F5` |

## 🔗 快速链接

| 页面 | 链接 |
|------|------|
| 仪表板 | https://www.tommienotes.com/bugly-dashboard.html |
| 诊断工具 | https://www.tommienotes.com/bugly-test.html |

## 💡 提示

### ✅ 推荐做法
- 使用清晰的消息描述
- 添加有意义的标签
- 合理使用日志级别
- 定期清理旧数据

### ❌ 避免做法
- 记录敏感信息（密码、密钥）
- 过度记录（每秒数十条）
- 使用模糊的消息（"错误"、"完成"）

## 📱 移动端支持

完全支持移动设备，响应式设计，触摸优化。

---

**快速帮助：** 打开浏览器控制台输入 `getBuglyReports()` 查看所有日志

