# 移动端 Try-Catch-Finally 错误处理指南

**版本**: v1.0  
**日期**: 2025-10-06  
**目的**: 统一移动端错误处理机制，提升代码健壮性

---

## 📋 概述

本文档介绍如何在移动端代码中正确使用 try-catch-finally 错误处理机制。

### 核心工具

- **`mobile-error-handler.js`**: 统一错误处理工具
- **`safeCall()`**: 同步函数包装器
- **`safeCallAsync()`**: 异步函数包装器
- **`MobileErrorHandler`**: 错误处理器类

---

## 🚀 快速开始

### 1. 基础用法

#### 包装同步函数

```javascript
// 原始代码（不安全）
function loadImage(img) {
    img.src = img.dataset.src;
    img.classList.add('loaded');
}

// 使用 try-catch（手动）
function loadImage(img) {
    try {
        img.src = img.dataset.src;
        img.classList.add('loaded');
    } catch (e) {
        console.error('加载图片失败', e);
    }
}

// 使用错误处理器（推荐）
const loadImage = safeCall(function(img) {
    img.src = img.dataset.src;
    img.classList.add('loaded');
}, {
    name: 'loadImage',
    context: 'ImageLoader',
    onError: (error, args) => {
        console.log('图片加载失败，使用占位符');
    }
});
```

#### 包装异步函数

```javascript
// 原始异步代码
async function fetchData(url) {
    const response = await fetch(url);
    return await response.json();
}

// 使用错误处理器
const fetchData = safeCallAsync(async function(url) {
    const response = await fetch(url);
    return await response.json();
}, {
    name: 'fetchData',
    context: 'DataFetcher',
    timeout: 5000,  // 5秒超时
    onError: (error) => {
        console.error('获取数据失败', error);
    },
    defaultReturn: null
});
```

### 2. 立即执行

```javascript
// 同步立即执行
window.MobileErrorHandler.safeExecute(() => {
    // 你的代码
    initializeApp();
}, {
    name: 'initializeApp',
    context: 'AppInit'
});

// 异步立即执行
await window.MobileErrorHandler.safeExecuteAsync(async () => {
    // 你的异步代码
    await loadConfig();
}, {
    name: 'loadConfig',
    context: 'ConfigLoader'
});
```

---

## 📚 详细功能

### 1. `safeCall()` - 同步函数包装器

**完整选项**:

```javascript
const wrappedFn = safeCall(originalFn, {
    // 基础选项
    name: 'functionName',           // 函数名（调试用）
    context: 'ModuleName',          // 上下文/模块名
    
    // 错误处理
    onError: (error, args) => {     // 错误回调
        console.log('处理错误', error);
    },
    rethrow: false,                 // 是否重新抛出错误
    defaultReturn: undefined,       // 错误时的默认返回值
    
    // 清理操作
    onFinally: (error, result, duration) => {  // 清理回调
        console.log('清理资源');
    },
    
    // 参数验证
    validateArgs: (args) => {       // 参数验证函数
        if (!args[0]) {
            return { valid: false, message: '参数不能为空' };
        }
        return { valid: true };
    }
});
```

**示例 1: 图片加载**

```javascript
const loadImage = safeCall(function(img) {
    // 验证参数
    if (!(img instanceof HTMLImageElement)) {
        throw new Error('参数必须是 HTMLImageElement');
    }
    
    // 检查 DOM
    if (!document.contains(img)) {
        throw new Error('图片不在文档中');
    }
    
    // 加载图片
    const src = img.dataset.src || img.src;
    if (!src) {
        throw new Error('图片src为空');
    }
    
    img.src = src;
    img.removeAttribute('data-src');
    img.classList.add('loaded');
    
    return true;
}, {
    name: 'loadImage',
    context: 'LazyLoader',
    
    // 参数验证
    validateArgs: (args) => {
        const img = args[0];
        if (!img || !(img instanceof HTMLImageElement)) {
            return {
                valid: false,
                message: '第一个参数必须是 HTMLImageElement'
            };
        }
        return { valid: true };
    },
    
    // 错误处理
    onError: (error, args) => {
        const img = args[0];
        console.warn(`图片加载失败: ${error.message}`, img);
        
        // 设置错误占位符
        try {
            img.style.background = '#f0f0f0';
            img.alt = '图片加载失败';
        } catch (e) {
            // 静默失败
        }
    },
    
    // 清理
    onFinally: (error, result, duration) => {
        if (duration > 100) {
            console.warn(`loadImage 执行较慢: ${duration.toFixed(2)}ms`);
        }
    },
    
    // 默认返回值
    defaultReturn: false,
    
    // 不重新抛出错误
    rethrow: false
});

// 使用
loadImage(document.querySelector('img'));
```

**示例 2: DOM 操作**

```javascript
const updateElement = safeCall(function(element, content) {
    element.textContent = content;
    element.classList.add('updated');
    element.setAttribute('data-updated', Date.now());
}, {
    name: 'updateElement',
    context: 'DOMUpdater',
    
    onError: (error) => {
        console.error('DOM更新失败', error);
    },
    
    onFinally: (error, result, duration) => {
        // 清理临时类
        if (!error) {
            setTimeout(() => {
                try {
                    element.classList.remove('updating');
                } catch (e) {
                    // 静默失败
                }
            }, 1000);
        }
    }
});
```

### 2. `safeCallAsync()` - 异步函数包装器

**完整选项**:

```javascript
const wrappedAsyncFn = safeCallAsync(async originalFn, {
    // 基础选项
    name: 'asyncFunctionName',
    context: 'ModuleName',
    
    // 超时控制
    timeout: 30000,                 // 超时时间（毫秒），默认30秒
    
    // 错误处理
    onError: async (error, args) => {
        await handleError(error);
    },
    rethrow: false,
    defaultReturn: null,
    
    // 清理操作（支持异步）
    onFinally: async (error, result, duration) => {
        await cleanup();
    }
});
```

**示例 1: 数据获取**

```javascript
const fetchUserData = safeCallAsync(async function(userId) {
    const response = await fetch(`/api/users/${userId}`);
    
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    return data;
}, {
    name: 'fetchUserData',
    context: 'UserAPI',
    
    // 5秒超时
    timeout: 5000,
    
    // 错误处理
    onError: async (error, args) => {
        const userId = args[0];
        console.error(`获取用户${userId}失败:`, error);
        
        // 上报到分析系统
        await reportError({
            type: 'api_error',
            userId: userId,
            error: error.message
        });
    },
    
    // 清理
    onFinally: async (error, result, duration) => {
        console.log(`API调用完成: ${duration.toFixed(2)}ms`);
        
        // 记录性能指标
        if (duration > 2000) {
            await reportSlowAPI({
                endpoint: '/api/users',
                duration: duration
            });
        }
    },
    
    // 错误时返回缓存数据
    defaultReturn: null,
    
    // 不重新抛出错误
    rethrow: false
});

// 使用
const userData = await fetchUserData('user123');
if (userData) {
    console.log('用户数据:', userData);
} else {
    console.log('使用默认数据');
}
```

**示例 2: 批量操作**

```javascript
const batchLoadImages = safeCallAsync(async function(imageElements) {
    const promises = Array.from(imageElements).map(async (img) => {
        return new Promise((resolve, reject) => {
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error(`Failed to load: ${img.src}`));
            img.src = img.dataset.src;
        });
    });
    
    // 等待所有图片加载（允许部分失败）
    const results = await Promise.allSettled(promises);
    
    return {
        success: results.filter(r => r.status === 'fulfilled').length,
        failed: results.filter(r => r.status === 'rejected').length,
        total: imageElements.length
    };
}, {
    name: 'batchLoadImages',
    context: 'ImageBatcher',
    
    // 30秒超时
    timeout: 30000,
    
    onError: (error) => {
        console.error('批量加载图片失败', error);
    },
    
    onFinally: (error, result, duration) => {
        if (result) {
            console.log(`图片加载完成: ${result.success}/${result.total} (${duration.toFixed(2)}ms)`);
        }
    },
    
    defaultReturn: { success: 0, failed: 0, total: 0 }
});
```

---

## 🎯 最佳实践

### 1. 函数命名规范

```javascript
// ✅ 好的命名
const loadImage = safeCall(fn, { name: 'loadImage', context: 'ImageLoader' });
const fetchData = safeCallAsync(fn, { name: 'fetchData', context: 'DataAPI' });

// ❌ 不好的命名
const fn1 = safeCall(fn, { name: 'fn1', context: 'unknown' });
const temp = safeCallAsync(fn, { name: 'temp' });
```

### 2. 错误回调处理

```javascript
// ✅ 好的错误处理
onError: (error, args) => {
    // 1. 记录日志
    console.error('操作失败', error);
    
    // 2. 用户提示
    showToast('操作失败，请重试');
    
    // 3. 上报错误
    reportToBugly({
        message: error.message,
        level: 'error'
    });
    
    // 4. 降级处理
    useFallbackMethod();
},

// ❌ 不好的错误处理
onError: (error) => {
    // 什么都不做
},
```

### 3. 清理操作

```javascript
// ✅ 正确的清理
onFinally: (error, result, duration) => {
    // 清理资源
    if (tempElement) {
        tempElement.remove();
    }
    
    // 取消定时器
    if (timerId) {
        clearTimeout(timerId);
    }
    
    // 移除事件监听
    if (listener) {
        window.removeEventListener('scroll', listener);
    }
    
    // 记录性能
    if (duration > 100) {
        console.warn(`性能警告: ${duration}ms`);
    }
},

// ❌ 不完整的清理
onFinally: () => {
    // 忘记清理资源
},
```

### 4. 参数验证

```javascript
// ✅ 完整的参数验证
validateArgs: (args) => {
    const [element, content, options] = args;
    
    // 检查必需参数
    if (!element) {
        return { valid: false, message: 'element 参数缺失' };
    }
    
    // 检查类型
    if (!(element instanceof Element)) {
        return { valid: false, message: 'element 必须是 DOM 元素' };
    }
    
    // 检查内容
    if (content === undefined || content === null) {
        return { valid: false, message: 'content 参数缺失' };
    }
    
    // 检查选项
    if (options && typeof options !== 'object') {
        return { valid: false, message: 'options 必须是对象' };
    }
    
    return { valid: true };
},

// ❌ 不充分的验证
validateArgs: (args) => {
    return { valid: !!args[0] };
},
```

### 5. 超时设置

```javascript
// ✅ 合理的超时设置
const fetchData = safeCallAsync(fn, {
    timeout: 5000,     // API调用：5秒
});

const uploadFile = safeCallAsync(fn, {
    timeout: 60000,    // 文件上传：60秒
});

const processImage = safeCallAsync(fn, {
    timeout: 10000,    // 图片处理：10秒
});

// ❌ 不合理的超时
const quickOperation = safeCallAsync(fn, {
    timeout: 100,      // 太短，容易超时
});

const longOperation = safeCallAsync(fn, {
    timeout: 300000,   // 太长（5分钟），移动端不合适
});
```

---

## 📊 错误统计

### 查看错误统计

```javascript
// 在控制台执行
const stats = getErrorStats();

// 输出示例:
// 📊 错误统计
// 总错误数: 15
// 按上下文: {
//   ImageLoader: 8,
//   DataAPI: 5,
//   DOMUpdater: 2
// }
// 按类型: {
//   sync_error: 10,
//   async_error: 3,
//   promise_rejection: 2
// }
// 按函数: {
//   loadImage: 8,
//   fetchData: 5,
//   updateElement: 2
// }
```

### 获取本地错误记录

```javascript
const errors = window.MobileErrorHandler.getLocalErrors();
console.table(errors);
```

### 清除错误记录

```javascript
window.MobileErrorHandler.clearLocalErrors();
```

---

## 🔍 调试技巧

### 1. 启用详细日志

```javascript
// 在 mobile-error-handler.js 中设置
const ERROR_HANDLER_CONFIG = {
    enableConsoleLog: true,    // 启用控制台日志
    enableBuglyReport: true,   // 启用 Bugly 上报
    enableLocalStorage: true,  // 启用本地存储
    maxLocalErrors: 100        // 最多保存100个错误
};
```

### 2. 测试错误处理

```javascript
// 测试同步错误
const testFn = safeCall(() => {
    throw new Error('测试同步错误');
}, {
    name: 'testSync',
    context: 'Test'
});
testFn();

// 测试异步错误
const testAsyncFn = safeCallAsync(async () => {
    await new Promise(r => setTimeout(r, 100));
    throw new Error('测试异步错误');
}, {
    name: 'testAsync',
    context: 'Test'
});
testAsyncFn();

// 测试超时
const testTimeout = safeCallAsync(async () => {
    await new Promise(r => setTimeout(r, 6000));  // 6秒
    return 'complete';
}, {
    name: 'testTimeout',
    context: 'Test',
    timeout: 5000  // 5秒超时
});
testTimeout();
```

### 3. 监控性能

```javascript
// 包装后自动记录执行时间
const slowFn = safeCall(() => {
    // 模拟慢操作
    const start = Date.now();
    while (Date.now() - start < 200) {}  // 200ms
}, {
    name: 'slowFn',
    context: 'Performance',
    onFinally: (error, result, duration) => {
        console.log(`执行时间: ${duration.toFixed(2)}ms`);
        
        if (duration > 100) {
            console.warn('⚠️ 性能警告: 执行时间超过100ms');
        }
    }
});
```

---

## 🎨 实战示例

### 示例 1: 完整的图片懒加载

```javascript
// 懒加载类
class LazyLoader {
    constructor() {
        this.observer = null;
        this.loadedCount = 0;
        
        // 使用错误处理器包装初始化
        this.init = safeCall(this.init.bind(this), {
            name: 'LazyLoader.init',
            context: 'LazyLoader',
            onError: () => {
                console.error('懒加载初始化失败，使用降级方案');
                this.fallbackLoad();
            }
        });
        
        // 包装加载方法
        this.loadImage = safeCall(this.loadImage.bind(this), {
            name: 'LazyLoader.loadImage',
            context: 'LazyLoader',
            validateArgs: (args) => {
                const img = args[0];
                if (!img || !(img instanceof HTMLImageElement)) {
                    return { valid: false, message: '参数必须是 HTMLImageElement' };
                }
                return { valid: true };
            },
            onError: (error, args) => {
                const img = args[0];
                img.style.background = '#f0f0f0';
                img.alt = '图片加载失败';
            }
        });
        
        this.init();
    }
    
    init() {
        if (!('IntersectionObserver' in window)) {
            throw new Error('IntersectionObserver not supported');
        }
        
        this.observer = new IntersectionObserver(
            safeCall(this.handleIntersection.bind(this), {
                name: 'LazyLoader.handleIntersection',
                context: 'LazyLoader'
            }),
            { rootMargin: '100px' }
        );
        
        this.observeImages();
    }
    
    handleIntersection(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                this.loadImage(entry.target);
                this.observer.unobserve(entry.target);
            }
        });
    }
    
    loadImage(img) {
        const src = img.dataset.src || img.src;
        if (!src) {
            throw new Error('图片src为空');
        }
        
        img.src = src;
        img.removeAttribute('data-src');
        img.classList.add('loaded');
        this.loadedCount++;
    }
    
    observeImages() {
        const images = document.querySelectorAll('img[loading="lazy"]');
        images.forEach(img => this.observer.observe(img));
    }
    
    fallbackLoad() {
        // 降级方案：立即加载所有图片
        const images = document.querySelectorAll('img[data-src]');
        images.forEach(img => {
            try {
                this.loadImage(img);
            } catch (e) {
                console.warn('降级加载失败', e);
            }
        });
    }
}

// 使用
const lazyLoader = new LazyLoader();
```

### 示例 2: 表单提交

```javascript
const submitForm = safeCallAsync(async function(formData) {
    // 验证表单
    const validation = validateForm(formData);
    if (!validation.valid) {
        throw new Error(`表单验证失败: ${validation.message}`);
    }
    
    // 显示加载状态
    showLoading();
    
    // 提交数据
    const response = await fetch('/api/submit', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    });
    
    if (!response.ok) {
        throw new Error(`提交失败: HTTP ${response.status}`);
    }
    
    const result = await response.json();
    return result;
}, {
    name: 'submitForm',
    context: 'FormSubmission',
    timeout: 10000,  // 10秒超时
    
    onError: async (error) => {
        hideLoading();
        showError(`提交失败: ${error.message}`);
        
        // 上报错误
        await reportToBugly({
            message: `表单提交失败: ${error.message}`,
            level: 'error'
        });
    },
    
    onFinally: async (error, result, duration) => {
        hideLoading();
        
        if (!error && result) {
            showSuccess('提交成功');
        }
        
        // 记录性能
        console.log(`表单提交耗时: ${duration.toFixed(2)}ms`);
    },
    
    defaultReturn: null
});

// 使用
const result = await submitForm({
    name: '张三',
    email: 'zhangsan@example.com'
});
```

---

## ✅ 检查清单

部署前确认：

- [ ] `mobile-error-handler.js` 已添加到 `static/js/`
- [ ] `mobile-head.html` 已引入错误处理器
- [ ] 所有关键函数都使用 `safeCall()` 或 `safeCallAsync()` 包装
- [ ] 错误回调正确处理（日志、提示、降级）
- [ ] 清理操作在 `onFinally` 中正确执行
- [ ] 参数验证已添加到所有公共函数
- [ ] 超时设置合理（API: 5-10秒，上传: 30-60秒）
- [ ] 测试通过（手动触发错误并验证处理）

---

## 📚 相关文档

- [BUGLY-INTEGRATION-GUIDE.md](./BUGLY-INTEGRATION-GUIDE.md) - Bugly 集成
- [MOBILE-PARAM-VALIDATION-ENHANCEMENT.md](./MOBILE-PARAM-VALIDATION-ENHANCEMENT.md) - 参数验证
- [DEPLOYMENT-SUCCESS-REPORT.md](./DEPLOYMENT-SUCCESS-REPORT.md) - 部署报告

---

**错误处理已完善！代码更健壮，用户体验更好！** 🎉

