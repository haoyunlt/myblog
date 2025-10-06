# 腾讯 Bugly 崩溃上报工具集成指南

**版本**: v1.0  
**日期**: 2025-10-06  
**目的**: 实时监控和分析移动端 Chrome 浏览器崩溃问题

---

## 📋 目录

1. [功能概览](#功能概览)
2. [快速开始](#快速开始)
3. [配置说明](#配置说明)
4. [API 文档](#api-文档)
5. [仪表板使用](#仪表板使用)
6. [常见问题](#常见问题)
7. [最佳实践](#最佳实践)

---

## 功能概览

### 核心功能

- ✅ **JavaScript 错误捕获** - 自动捕获所有未捕获的 JavaScript 错误
- ✅ **资源加载错误** - 监控图片、CSS、JS 等资源加载失败
- ✅ **Promise 拒绝** - 捕获未处理的 Promise rejection
- ✅ **性能监控** - 检测长任务和内存警告
- ✅ **设备信息** - 自动收集设备、浏览器、屏幕等信息
- ✅ **本地存储** - 错误报告本地缓存，离线也能收集
- ✅ **批量上报** - 智能合并上报，减少网络请求
- ✅ **防重复** - 相同错误去重，避免重复上报
- ✅ **可视化仪表板** - Web 端查看和分析错误报告

### 技术特点

| 特性 | 说明 |
|------|------|
| 零依赖 | 纯 JavaScript 实现，无需任何第三方库 |
| 轻量级 | 压缩后 < 10KB |
| 高性能 | 异步上报，不影响页面性能 |
| 兼容性 | 支持所有现代浏览器 |
| 可配置 | 灵活的配置选项 |

---

## 快速开始

### 1. 注册 Bugly 账号（可选）

访问 [腾讯 Bugly 官网](https://bugly.qq.com/)，注册账号并创建产品，获取 `App ID`。

**注意**: 本实现也支持本地存储模式，即使不注册 Bugly 也能使用。

### 2. 配置 App ID

编辑 `static/js/bugly-report.js`，修改配置：

```javascript
const BUGLY_CONFIG = {
    appId: 'YOUR_APP_ID',  // 替换为你的 Bugly App ID
    appVersion: '1.0.0',
    enableDebug: false,    // 生产环境设为 false
    // ... 其他配置
};
```

### 3. 引入脚本

脚本已自动引入到 `layouts/partials/mobile-head.html`：

```html
{{- /* Bugly 崩溃上报 - 最先加载 */ -}}
<script src="{{ "js/bugly-report.js" | relURL }}"></script>
```

### 4. 构建和部署

```bash
# 构建网站
hugo --cleanDestinationDir --minify

# 部署到服务器
./deploy/deploy-aliyun.sh
```

### 5. 访问仪表板

打开浏览器访问：
```
https://www.tommienotes.com/bugly-dashboard.html
```

---

## 配置说明

### 完整配置选项

```javascript
const BUGLY_CONFIG = {
    // ===== 必填配置 =====
    appId: 'YOUR_APP_ID',          // Bugly App ID
    
    // ===== 基础配置 =====
    appVersion: '1.0.0',            // 应用版本号
    userId: '',                     // 用户标识（可选）
    enableDebug: false,             // 是否启用调试模式
    
    // ===== 上报配置 =====
    delay: 1000,                    // 延迟上报时间（毫秒）
    random: 1,                      // 上报采样率 (0-1)
    repeat: 5,                      // 同一错误重复上报次数限制
    reportUrl: 'https://bugly.qq.com/api/report',  // 上报地址
    
    // ===== 自定义字段 =====
    customFields: {
        website: 'tommienotes.com',
        platform: 'mobile-web',
        // 可以添加更多自定义字段
    }
};
```

### 配置项详解

#### 1. `appId` (必填)
- **类型**: String
- **说明**: Bugly 应用唯一标识
- **获取方式**: 在 Bugly 控制台创建产品后获得
- **示例**: `'2a1b3c4d5e6f'`

#### 2. `enableDebug`
- **类型**: Boolean
- **默认**: `false`
- **说明**: 是否在控制台输出详细日志
- **推荐**: 开发环境 `true`，生产环境 `false`

#### 3. `random` (采样率)
- **类型**: Number (0-1)
- **默认**: `1`
- **说明**: 错误上报采样率，用于控制上报量
- **示例**: 
  - `1` - 上报所有错误 (100%)
  - `0.5` - 上报 50% 的错误
  - `0.1` - 上报 10% 的错误

#### 4. `repeat` (重复限制)
- **类型**: Number
- **默认**: `5`
- **说明**: 同一错误最多上报次数
- **推荐**: 3-10 之间

#### 5. `delay` (延迟上报)
- **类型**: Number (毫秒)
- **默认**: `1000`
- **说明**: 错误发生后延迟多久上报
- **推荐**: 1000-3000 毫秒

---

## API 文档

### 全局对象

#### `window.BuglyReporter`

主要的错误上报实例。

#### `window.reportToBugly(errorData)`

手动上报自定义错误。

**参数**:
```javascript
{
    message: string,      // 错误消息（必填）
    stack: string,        // 堆栈信息（可选）
    level: string,        // 错误级别: 'error' | 'warning' | 'info'（可选）
    extra: object         // 额外数据（可选）
}
```

**示例**:
```javascript
// 基础用法
reportToBugly({
    message: '用户支付失败',
    level: 'error'
});

// 完整用法
reportToBugly({
    message: '用户支付失败',
    stack: new Error().stack,
    level: 'error',
    extra: {
        orderId: '12345',
        amount: 99.99,
        userId: 'user_001'
    }
});
```

#### `window.getBuglyReports()`

获取本地存储的所有错误报告。

**返回**: `Array<ErrorReport>`

**示例**:
```javascript
const reports = getBuglyReports();
console.log(`共有 ${reports.length} 个错误报告`);
console.table(reports);
```

#### `window.clearBuglyReports()`

清除本地存储的所有错误报告。

**示例**:
```javascript
clearBuglyReports();
console.log('错误报告已清除');
```

### 错误类型

#### JavaScript 错误 (`javascript_error`)

**触发条件**: 
- 未捕获的 JavaScript 异常
- `throw new Error()` 未被捕获
- 语法错误、引用错误等

**捕获信息**:
```javascript
{
    type: 'javascript_error',
    message: '错误消息',
    stack: '堆栈跟踪',
    filename: '文件路径',
    lineno: 行号,
    colno: 列号,
    errorType: 'TypeError'
}
```

#### 资源加载错误 (`resource_error`)

**触发条件**:
- 图片加载失败
- CSS 文件加载失败
- JavaScript 文件加载失败
- 字体文件加载失败

**捕获信息**:
```javascript
{
    type: 'resource_error',
    message: 'Resource load failed: IMG',
    resourceType: 'img',
    resourceUrl: 'https://example.com/image.jpg'
}
```

#### Promise 拒绝 (`promise_rejection`)

**触发条件**:
- 未处理的 Promise rejection
- `Promise.reject()` 未被 `.catch()` 捕获

**捕获信息**:
```javascript
{
    type: 'promise_rejection',
    message: 'Promise rejected',
    stack: '堆栈跟踪'
}
```

#### 性能长任务 (`performance_long_task`)

**触发条件**:
- JavaScript 执行时间超过 50ms

**捕获信息**:
```javascript
{
    type: 'performance_long_task',
    message: 'Long task detected: 125.50ms',
    duration: 125.50,
    startTime: 1234.56
}
```

#### 内存警告 (`performance_memory_warning`)

**触发条件**:
- JavaScript 堆内存使用超过 90%

**捕获信息**:
```javascript
{
    type: 'performance_memory_warning',
    message: 'High memory usage: 92.5%',
    usedMB: 185,
    limitMB: 200
}
```

---

## 仪表板使用

### 访问仪表板

```
https://www.tommienotes.com/bugly-dashboard.html
```

### 功能说明

#### 1. 统计卡片

显示各类错误的统计数量：
- JavaScript 错误数量
- 资源加载错误数量
- Promise 拒绝数量
- 总报告数

#### 2. 控制按钮

| 按钮 | 功能 |
|------|------|
| 🔄 刷新 | 重新加载错误报告 |
| 📥 导出JSON | 导出所有报告为 JSON 文件 |
| 🗑️ 清除所有 | 清空所有错误报告（不可恢复） |

#### 3. 过滤器

- **类型过滤**: 按错误类型筛选
- **搜索框**: 按错误消息关键词搜索

#### 4. 报告列表

点击报告条目可展开查看详细信息：
- 错误堆栈
- 设备信息
- 浏览器信息
- 内存使用情况
- URL 和文件位置

### 导出数据

1. 点击"导出JSON"按钮
2. 浏览器自动下载 `bugly-reports-{timestamp}.json` 文件
3. 可用于进一步分析或上传到 Bugly 控制台

**导出格式**:
```json
[
  {
    "type": "javascript_error",
    "message": "Cannot read property 'x' of undefined",
    "timestamp": 1696588800000,
    "device": { /* 设备信息 */ },
    "stack": "Error: ...",
    /* ... 更多字段 */
  }
]
```

---

## 常见问题

### Q1: 为什么控制台看不到 Bugly 初始化消息？

**A**: 检查以下几点：
1. 确认 `bugly-report.js` 文件存在且路径正确
2. 检查浏览器是否为移动设备或窗口宽度 ≤ 768px
3. 打开开发者工具的 Console 标签
4. 刷新页面（`Ctrl+Shift+R` 硬刷新）

### Q2: 错误没有上报到 Bugly 服务器？

**A**: 可能原因：
1. **App ID 未配置**: 检查 `BUGLY_CONFIG.appId` 是否正确
2. **采样率设置**: 检查 `random` 配置是否 < 1
3. **网络问题**: 检查 `reportUrl` 是否可访问
4. **本地存储模式**: 默认使用本地存储，需要手动导出

**解决方案**: 启用调试模式
```javascript
enableDebug: true  // 在 BUGLY_CONFIG 中设置
```

### Q3: 相同错误被重复上报？

**A**: 这是正常行为，受 `repeat` 配置控制：
```javascript
repeat: 5  // 同一错误最多上报 5 次
```

如果希望只上报一次，设置为 `1`。

### Q4: 如何测试 Bugly 是否正常工作？

**A**: 在控制台执行：
```javascript
// 触发 JavaScript 错误
setTimeout(() => {
    throw new Error('测试错误 - Bugly 集成测试');
}, 1000);

// 手动上报
reportToBugly({
    message: '手动测试错误',
    level: 'error'
});

// 查看报告
setTimeout(() => {
    console.log('错误报告:', getBuglyReports());
}, 2000);
```

### Q5: 仪表板显示为空？

**A**: 检查步骤：
1. 确认是否有错误发生（可以手动触发）
2. 检查本地存储：`localStorage.getItem('bugly_reports')`
3. 清除浏览器缓存后重试
4. 查看控制台是否有错误信息

### Q6: 如何在生产环境中使用？

**A**: 生产环境配置：
```javascript
const BUGLY_CONFIG = {
    appId: 'YOUR_REAL_APP_ID',    // 使用真实 App ID
    appVersion: '1.0.0',
    enableDebug: false,            // 关闭调试
    random: 0.5,                   // 50% 采样率（可选）
    repeat: 3,                     // 减少重复上报
};
```

---

## 最佳实践

### 1. 采样策略

根据流量大小调整采样率：

| 日访问量 | 推荐采样率 |
|---------|----------|
| < 1,000 | 1.0 (100%) |
| 1,000 - 10,000 | 0.5 (50%) |
| 10,000 - 100,000 | 0.2 (20%) |
| > 100,000 | 0.1 (10%) |

### 2. 错误分级

为不同级别的错误设置不同处理：

```javascript
// 致命错误 - 立即上报
reportToBugly({
    message: '支付失败',
    level: 'fatal',
    extra: { orderId: '123' }
});

// 警告 - 延迟上报
reportToBugly({
    message: '图片加载慢',
    level: 'warning'
});

// 信息 - 批量上报
reportToBugly({
    message: '用户点击按钮',
    level: 'info'
});
```

### 3. 用户标识

设置用户标识便于追踪：

```javascript
// 用户登录后设置
if (window.BuglyReporter) {
    window.BuglyReporter.config.userId = 'user_12345';
}
```

### 4. 自定义字段

添加业务相关字段：

```javascript
customFields: {
    website: 'tommienotes.com',
    platform: 'mobile-web',
    environment: 'production',
    version: '2.0.1',
    region: 'cn-north'
}
```

### 5. 定期清理

定期清理本地报告防止占用过多空间：

```javascript
// 每周自动清理
setInterval(() => {
    const reports = getBuglyReports();
    if (reports.length > 1000) {
        clearBuglyReports();
        console.log('[Bugly] 自动清理本地报告');
    }
}, 7 * 24 * 60 * 60 * 1000);  // 7天
```

### 6. 关键路径监控

为关键业务流程添加自定义上报：

```javascript
// 支付流程监控
async function processPayment(orderId) {
    try {
        const result = await pay(orderId);
        
        // 成功也上报（用于统计）
        reportToBugly({
            message: '支付成功',
            level: 'info',
            extra: { orderId, amount: result.amount }
        });
        
    } catch (error) {
        // 失败上报
        reportToBugly({
            message: `支付失败: ${error.message}`,
            level: 'error',
            extra: { orderId, error: error.toString() }
        });
        throw error;
    }
}
```

### 7. 性能监控

监控关键性能指标：

```javascript
// 页面加载性能
window.addEventListener('load', () => {
    const timing = performance.timing;
    const loadTime = timing.loadEventEnd - timing.navigationStart;
    
    if (loadTime > 3000) {  // 超过3秒
        reportToBugly({
            message: `页面加载较慢: ${loadTime}ms`,
            level: 'warning',
            extra: {
                loadTime,
                domReady: timing.domContentLoadedEventEnd - timing.navigationStart
            }
        });
    }
});
```

---

## 🎯 快速检查清单

部署前请确认：

- [ ] `bugly-report.js` 已添加到 `static/js/`
- [ ] `mobile-head.html` 已引入 Bugly 脚本
- [ ] `bugly-dashboard.html` 已添加到 `static/`
- [ ] `BUGLY_CONFIG.appId` 已配置
- [ ] `enableDebug` 在生产环境设为 `false`
- [ ] 采样率 `random` 已根据流量调整
- [ ] 本地测试通过（手动触发错误并查看仪表板）
- [ ] 移动端设备测试通过

---

## 📚 相关文档

- [腾讯 Bugly 官网](https://bugly.qq.com/)
- [MOBILE-PARAM-VALIDATION-ENHANCEMENT.md](./MOBILE-PARAM-VALIDATION-ENHANCEMENT.md) - 参数验证增强
- [DEPLOYMENT-SUCCESS-REPORT.md](./DEPLOYMENT-SUCCESS-REPORT.md) - 部署报告
- [NGINX-CSS-FIX-COMPLETE.md](./NGINX-CSS-FIX-COMPLETE.md) - Nginx 修复

---

## 🔄 版本历史

### v1.0 (2025-10-06)
- ✅ 首次发布
- ✅ JavaScript 错误捕获
- ✅ 资源加载错误监控
- ✅ Promise rejection 处理
- ✅ 性能监控（长任务、内存警告）
- ✅ 设备信息收集
- ✅ 本地存储支持
- ✅ 可视化仪表板
- ✅ 批量上报和去重

---

## 📞 技术支持

如有问题或建议，请：

1. 查看 [常见问题](#常见问题) 章节
2. 检查控制台错误日志
3. 访问 [Bugly 帮助中心](https://bugly.qq.com/docs/)
4. 联系技术支持

---

**集成完成！开始监控移动端错误，提升用户体验！** 🎉

