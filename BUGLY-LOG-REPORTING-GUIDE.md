# Bugly 仪表板 - 日志上报功能使用指南

## 📝 功能概述

Bugly 仪表板现已集成**日志上报功能**，允许你手动记录和追踪系统日志，不仅仅是错误报告。这对于调试、监控和分析用户行为非常有用。

## ✨ 新增功能

### 1. 日志统计卡片
- 仪表板顶部新增"**日志记录**"统计卡片
- 实时显示所有日志类型的总数（debug, info, warn, error）
- 独立于错误统计，便于区分

### 2. 日志上报按钮
- 工具栏新增 "**📝 上报日志**" 按钮
- 点击打开优雅的模态框表单
- 支持快捷键：`ESC` 关闭模态框

### 3. 四种日志级别
| 级别 | 图标 | 颜色 | 用途 |
|------|------|------|------|
| **Debug** | 🔍 | 灰色 | 调试信息、开发时使用 |
| **Info** | ℹ️ | 蓝色 | 一般信息、用户操作记录 |
| **Warn** | ⚠️ | 橙色 | 警告信息、潜在问题 |
| **Error** | ❌ | 红色 | 错误信息、需要关注的问题 |

### 4. 日志字段支持
- **日志消息***（必填）：简短描述，最多200字符
- **详细信息**（可选）：多行文本，详细说明
- **标签**（可选）：用逗号分隔的标签，便于分类筛选
- **自动字段**：时间戳、URL、用户代理、会话ID

### 5. 类型过滤增强
筛选下拉菜单新增日志类型选项：
- 日志-调试
- 日志-信息
- 日志-警告
- 日志-错误

### 6. 增强的报告详情
日志报告展开后显示：
- 日志级别（大写显示）
- 来源标识（手动上报）
- 标签（彩色徽章展示）
- 详细信息（格式化显示）

## 🚀 使用方法

### 方式 1：通过仪表板界面

1. **打开仪表板**
   ```
   https://www.tommienotes.com/bugly-dashboard.html
   ```

2. **点击"上报日志"按钮**
   - 位于工具栏，绿色按钮，图标 📝

3. **填写日志表单**
   ```
   日志级别: Info ✓
   日志消息: 用户完成支付流程 ✓
   详细信息: 订单号: 12345, 金额: 99.99元
   标签: 支付流程, 用户操作
   ```

4. **提交**
   - 点击"✅ 提交日志"按钮
   - 看到成功提示："✅ 日志已成功上报！"
   - 页面自动刷新显示新日志

5. **查看日志**
   - 日志出现在报告列表顶部
   - 点击展开查看详细信息
   - 使用筛选器过滤日志类型

### 方式 2：通过控制台（编程方式）

在浏览器控制台或你的代码中：

```javascript
// 方式 A：使用 reportToBugly（如果 Bugly 已加载）
if (window.reportToBugly) {
    window.reportToBugly({
        message: '用户登录成功',
        level: 'info',
        extra: {
            userId: '12345',
            loginMethod: 'email'
        }
    });
}

// 方式 B：直接写入 localStorage
function manualLog(level, message, details, tags) {
    const logReport = {
        type: `log_${level}`,
        message: message,
        details: details,
        tags: tags,
        level: level,
        timestamp: Date.now(),
        url: window.location.href,
        userAgent: navigator.userAgent,
        sessionId: 'manual_' + Date.now(),
        source: 'manual_report'
    };
    
    const existing = JSON.parse(localStorage.getItem('bugly_reports') || '[]');
    existing.push(logReport);
    localStorage.setItem('bugly_reports', JSON.stringify(existing.slice(-1000)));
}

// 使用示例
manualLog('info', '页面加载完成', '耗时: 2.5秒', ['性能', '加载']);
manualLog('warn', '内存使用较高', '当前使用: 85%', ['性能', '内存']);
manualLog('error', '支付接口调用失败', '错误代码: 500', ['支付', '接口']);
```

### 方式 3：集成到代码中

在你的应用代码中：

```javascript
// 工具函数
const logger = {
    debug: (msg, details = '', tags = []) => {
        manualLog('debug', msg, details, tags);
        console.debug('[Debug]', msg);
    },
    
    info: (msg, details = '', tags = []) => {
        manualLog('info', msg, details, tags);
        console.info('[Info]', msg);
    },
    
    warn: (msg, details = '', tags = []) => {
        manualLog('warn', msg, details, tags);
        console.warn('[Warn]', msg);
    },
    
    error: (msg, details = '', tags = []) => {
        manualLog('error', msg, details, tags);
        console.error('[Error]', msg);
    }
};

// 使用示例
logger.info('用户点击购买按钮', '商品ID: SKU-001', ['用户操作', '购买流程']);
logger.warn('API 响应时间过长', '耗时: 3500ms', ['性能', 'API']);
logger.error('支付失败', '错误: 余额不足', ['支付', '错误']);
```

## 📊 实际应用场景

### 场景 1：用户行为追踪
```javascript
// 追踪关键用户操作
logger.info('用户查看商品详情', 'SKU: ABC123', ['用户行为', '商品']);
logger.info('添加到购物车', '数量: 2', ['用户行为', '购物车']);
logger.info('开始结账流程', '', ['用户行为', '支付']);
```

### 场景 2：性能监控
```javascript
// 记录性能指标
const loadTime = performance.timing.loadEventEnd - performance.timing.navigationStart;
if (loadTime > 3000) {
    logger.warn(
        '页面加载较慢',
        `耗时: ${loadTime}ms`,
        ['性能', '加载时间']
    );
}
```

### 场景 3：API 调用日志
```javascript
// 记录 API 调用
async function fetchUserData(userId) {
    try {
        logger.info('开始获取用户数据', `用户ID: ${userId}`, ['API', '用户']);
        
        const response = await fetch(`/api/user/${userId}`);
        
        if (!response.ok) {
            logger.error(
                'API 调用失败',
                `状态码: ${response.status}, 用户ID: ${userId}`,
                ['API', '错误']
            );
        } else {
            logger.info('用户数据获取成功', `用户ID: ${userId}`, ['API', '成功']);
        }
        
        return await response.json();
    } catch (error) {
        logger.error(
            'API 调用异常',
            `错误: ${error.message}, 用户ID: ${userId}`,
            ['API', '异常']
        );
        throw error;
    }
}
```

### 场景 4：调试信息
```javascript
// 开发阶段调试
if (process.env.NODE_ENV === 'development') {
    logger.debug('Redux 状态更新', JSON.stringify(newState), ['Redux', '调试']);
    logger.debug('路由变化', `从 ${oldPath} 到 ${newPath}`, ['路由', '调试']);
}
```

## 🎨 界面特性

### 1. 模态框设计
- **优雅动画**：淡入淡出 + 滑动效果
- **自动聚焦**：打开后自动聚焦到消息输入框
- **点击外部关闭**：点击模态框外部区域自动关闭
- **ESC 关闭**：按 ESC 键快速关闭
- **表单重置**：关闭时自动重置表单内容

### 2. Toast 提示
- **成功提示**：绿色背景，3秒后自动消失
- **错误提示**：红色背景，显示错误信息
- **滑入滑出**：从右侧滑入，动画流畅

### 3. 标签展示
- **彩色徽章**：蓝色背景，圆角设计
- **多标签支持**：自动分隔显示
- **点击可见**：在报告详情中展开查看

### 4. 日志分类显示
不同级别的日志使用不同颜色标识：
- Debug: 灰色背景 (#f4f4f5)
- Info: 蓝色背景 (#ecf5ff)
- Warn: 橙色背景 (#fdf6ec)
- Error: 红色背景 (#fef0f0)

## 📈 数据管理

### 存储限制
- **最大容量**：1000 条报告（包括错误和日志）
- **超出处理**：自动保留最新的 1000 条
- **存储位置**：localStorage `bugly_reports`

### 数据结构
```json
{
  "type": "log_info",
  "message": "用户完成支付",
  "details": "订单号: 12345, 金额: 99.99元",
  "tags": ["支付", "用户操作"],
  "level": "info",
  "timestamp": 1696588800000,
  "url": "https://www.tommienotes.com/checkout",
  "userAgent": "Mozilla/5.0...",
  "sessionId": "manual_1696588800123",
  "source": "manual_report"
}
```

### 导出数据
1. 点击 "📥 导出JSON" 按钮
2. 下载包含所有报告的 JSON 文件
3. 可以导入到其他工具进行分析

### 清除数据
1. 点击 "🗑️ 清除所有" 按钮
2. 确认操作
3. 所有错误和日志将被清除

## 🔍 查询和筛选

### 按类型筛选
使用类型过滤下拉菜单：
- 选择 "日志-信息" 只显示 info 级别日志
- 选择 "日志-错误" 只显示 error 级别日志
- 选择 "全部" 显示所有报告

### 按消息搜索
使用搜索框：
- 输入关键词，实时过滤
- 搜索范围：日志消息字段
- 不区分大小写

### 组合筛选
同时使用类型过滤和搜索：
```
类型: 日志-警告
搜索: 性能
结果: 只显示包含"性能"关键词的警告日志
```

## 💡 最佳实践

### 1. 合理使用日志级别
- **Debug**：仅在开发环境使用，不要在生产环境滥用
- **Info**：记录重要的业务操作和里程碑事件
- **Warn**：记录潜在问题，但不影响功能
- **Error**：记录实际错误和失败操作

### 2. 编写清晰的日志消息
❌ **不好的例子：**
```javascript
logger.info('操作完成', '', []);
```

✅ **好的例子：**
```javascript
logger.info(
    '用户注册成功',
    '用户ID: 12345, 注册方式: 邮箱',
    ['用户管理', '注册']
);
```

### 3. 使用标签进行分类
建议的标签分类：
- **功能模块**：用户管理、订单处理、支付、商品
- **操作类型**：创建、更新、删除、查询
- **性能**：加载时间、API 响应、内存使用
- **安全**：认证、授权、敏感操作

### 4. 避免记录敏感信息
❌ **不要记录：**
- 密码、密钥、令牌
- 完整的信用卡号
- 身份证号、手机号等个人敏感信息

✅ **可以记录：**
- 用户 ID（非敏感标识符）
- 操作类型和时间
- 错误代码和状态
- 脱敏后的部分信息（如：手机号 138****1234）

### 5. 定期清理旧日志
在生产环境建议：
- 每周清理一次旧日志
- 或者在数据导出后清理
- 保持 localStorage 容量合理

## 🛠️ 高级用法

### 批量日志上报
```javascript
const logs = [
    { level: 'info', message: '步骤1完成', details: '', tags: ['流程'] },
    { level: 'info', message: '步骤2完成', details: '', tags: ['流程'] },
    { level: 'info', message: '步骤3完成', details: '', tags: ['流程'] }
];

logs.forEach(log => {
    manualLog(log.level, log.message, log.details, log.tags);
});
```

### 带时间戳的日志
```javascript
function timestampLog(level, message, details, tags) {
    const timestamp = new Date().toLocaleString('zh-CN');
    manualLog(
        level,
        `[${timestamp}] ${message}`,
        details,
        tags
    );
}
```

### 性能日志包装器
```javascript
function measurePerformance(fn, name) {
    return async function(...args) {
        const start = performance.now();
        
        try {
            const result = await fn.apply(this, args);
            const duration = performance.now() - start;
            
            logger.info(
                `${name} 执行成功`,
                `耗时: ${duration.toFixed(2)}ms`,
                ['性能', name]
            );
            
            return result;
        } catch (error) {
            const duration = performance.now() - start;
            
            logger.error(
                `${name} 执行失败`,
                `耗时: ${duration.toFixed(2)}ms, 错误: ${error.message}`,
                ['性能', name, '错误']
            );
            
            throw error;
        }
    };
}

// 使用
const fetchData = measurePerformance(
    async () => await fetch('/api/data').then(r => r.json()),
    'fetchData'
);
```

## 🎯 与错误报告的区别

| 特性 | 错误报告 | 日志报告 |
|------|---------|---------|
| **触发方式** | 自动捕获 | 手动记录 |
| **用途** | 追踪错误和异常 | 记录操作和状态 |
| **级别** | 错误/警告 | Debug/Info/Warn/Error |
| **字段** | 堆栈跟踪、文件位置 | 详细信息、标签 |
| **适用场景** | 调试 bug | 监控业务流程 |

## 📱 移动端支持

日志上报功能在移动设备上完全可用：
- ✅ 响应式模态框设计
- ✅ 触摸优化的表单控件
- ✅ 移动端友好的 Toast 提示
- ✅ 支持移动浏览器的 localStorage

## 🔗 相关文件

- `/static/bugly-dashboard.html` - 主仪表板（已增强）
- `/static/bugly-test.html` - 诊断工具
- `/static/js/bugly-report.js` - 错误收集器
- `BUGLY-NO-DATA-FIX.md` - 问题诊断文档

## 📝 更新日志

### v1.1.0 (2025-10-06)
- ✨ 新增日志上报功能
- ✨ 新增四种日志级别（Debug, Info, Warn, Error）
- ✨ 新增标签支持
- ✨ 新增详细信息字段
- 🎨 优化模态框设计和动画
- 🎨 新增 Toast 提示
- 📊 新增日志统计卡片
- 🔍 增强类型筛选功能

### v1.0.0
- 基础错误报告功能
- 统计和筛选
- 导出和清除

---

**提示：** 如有任何问题或建议，欢迎反馈！

