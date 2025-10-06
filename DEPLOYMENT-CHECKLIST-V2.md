# 移动端优化部署检查清单 v2.0

**部署日期**: 2025-10-06  
**版本**: v2.0 (Try-Catch-Finally + Bugly集成)

---

## ✅ 文件检查

### 新增文件

- [ ] `static/js/mobile-param-validator.js` - 参数验证工具
- [ ] `static/js/bugly-report.js` - Bugly崩溃上报
- [ ] `static/js/mobile-error-handler.js` - 统一错误处理器
- [ ] `static/bugly-dashboard.html` - 错误报告仪表板

### 文档文件

- [ ] `MOBILE-PARAM-VALIDATION-ENHANCEMENT.md` - 参数验证文档
- [ ] `BUGLY-INTEGRATION-GUIDE.md` - Bugly集成指南
- [ ] `MOBILE-ERROR-HANDLING-GUIDE.md` - 错误处理指南
- [ ] `MOBILE-OPTIMIZATION-SUMMARY.md` - 优化总结
- [ ] `DEPLOYMENT-CHECKLIST-V2.md` - 本检查清单

### 修改文件

- [ ] `static/js/mobile-performance.js` - 添加参数验证
- [ ] `layouts/partials/mobile-head.html` - 集成新工具
- [ ] `layouts/partials/image-fix.html` - 完善错误处理

---

## 🔧 配置检查

### Bugly 配置

编辑 `static/js/bugly-report.js`：

```javascript
const BUGLY_CONFIG = {
    appId: 'YOUR_APP_ID',  // ⚠️ 替换为真实 App ID
    appVersion: '1.0.0',
    enableDebug: false,    // 生产环境设为 false
    random: 1,             // 根据流量调整采样率
    repeat: 5
};
```

### 错误处理器配置

编辑 `static/js/mobile-error-handler.js`：

```javascript
const ERROR_HANDLER_CONFIG = {
    enableConsoleLog: true,     // ⚠️ 生产环境可设为 false
    enableBuglyReport: true,
    enableLocalStorage: true,
    maxLocalErrors: 100
};
```

---

## 🏗️ 构建步骤

### 1. 清理旧构建

```bash
cd /Users/lintao/important/ai-customer/myblog
rm -rf public/ resources/
```

### 2. 运行 Hugo 构建

```bash
hugo --cleanDestinationDir --minify --baseURL "https://www.tommienotes.com" --gc
```

### 3. 验证构建输出

```bash
# 检查新增文件
ls -lh public/js/mobile-param-validator.js
ls -lh public/js/bugly-report.js
ls -lh public/js/mobile-error-handler.js
ls -lh public/bugly-dashboard.html

# 检查文件大小（应该合理）
du -sh public/js/*.js
```

预期输出：
```
10K public/js/mobile-param-validator.js
15K public/js/bugly-report.js
12K public/js/mobile-error-handler.js
```

---

## 🚀 部署步骤

### 1. 部署到阿里云

```bash
./deploy/deploy-aliyun.sh
```

### 2. 验证 Nginx 配置

```bash
ssh blog-svr 'sudo nginx -t'
```

预期输出：
```
nginx: configuration file test is successful
```

### 3. 检查文件上传

```bash
ssh blog-svr 'ls -lh /var/www/html/js/mobile-*.js'
ssh blog-svr 'ls -lh /var/www/html/js/bugly-*.js'
ssh blog-svr 'ls -lh /var/www/html/bugly-dashboard.html'
```

---

## 🔍 部署验证

### 1. 文件可访问性

```bash
# 检查参数验证器
curl -I https://www.tommienotes.com/js/mobile-param-validator.js
# 预期: HTTP/2 200

# 检查 Bugly
curl -I https://www.tommienotes.com/js/bugly-report.js
# 预期: HTTP/2 200

# 检查错误处理器
curl -I https://www.tommienotes.com/js/mobile-error-handler.js
# 预期: HTTP/2 200

# 检查仪表板
curl -I https://www.tommienotes.com/bugly-dashboard.html
# 预期: HTTP/2 200
```

### 2. CSS 路径验证

```bash
# 检查正确路径
curl -I https://www.tommienotes.com/css/extended/mobile-performance.css
# 预期: HTTP/2 200

# 检查错误路径（应该404）
curl -I https://www.tommienotes.com/assets/css/mobile-performance.css
# 预期: HTTP/2 404
```

---

## 🌐 浏览器验证

### 1. 桌面浏览器测试

打开 https://www.tommienotes.com/ 并按 F12：

#### Console 检查

应该看到以下初始化消息：

```
✅ 预期输出：
[ParamValidator] ✅ 参数验证工具已加载
[ErrorHandler] ✅ 移动端错误处理器已启动
[Bugly] ✅ 崩溃上报系统已启动
[Mobile-Perf] 移动端轻量级优化系统 v3.1 启动（参数验证增强）
```

❌ 不应该看到：
```
- MutationObserver 错误
- querySelector 错误
- CSS 404 错误
- Mermaid 渲染错误
```

#### Network 检查

```
✅ 应该看到（状态 200）：
- mobile-param-validator.js
- bugly-report.js
- mobile-error-handler.js
- mobile-performance.css

❌ 不应该看到（状态 404）：
- /assets/css/mobile-performance.css
```

#### Application 检查

Service Worker 标签：
```
✅ 状态: activated and is running
✅ Source: /sw.js
✅ Version: v2.1.0
```

### 2. 移动设备测试

#### 测试设备

- [ ] iPhone (Safari)
- [ ] Android (Chrome)
- [ ] iPad (Safari)
- [ ] Android 平板

#### 测试项目

- [ ] 页面正常加载
- [ ] 图片懒加载工作
- [ ] 无崩溃或白屏
- [ ] Mermaid图表正常显示
- [ ] 滚动流畅
- [ ] 内存占用正常（< 150MB）

#### Chrome DevTools 移动模拟

1. F12 → Toggle Device Toolbar
2. 选择设备: iPhone 12 Pro
3. 刷新页面
4. 检查 Console 无错误
5. 检查 Network 无 404
6. 检查 Performance (Lighthouse 移动端评分)

预期分数：
- Performance: > 80
- Accessibility: > 90
- Best Practices: > 90

---

## 🧪 功能测试

### 1. 参数验证测试

在 Console 执行：

```javascript
// 测试 notNull
window.mobileValidator.notNull(null, 'test', 'testFunc');
// 预期: 返回 false，输出错误日志

// 测试 isImage
const img = document.querySelector('img');
window.mobileValidator.isImage(img, 'img', 'testFunc');
// 预期: 返回 true

// 测试 inRange
window.mobileValidator.inRange(150, 0, 100, 'value', 'testFunc');
// 预期: 返回 false，输出错误日志
```

### 2. 错误处理测试

在 Console 执行：

```javascript
// 测试同步错误
const testFn = safeCall(() => {
    throw new Error('测试同步错误');
}, {
    name: 'testSync',
    context: 'Test'
});
testFn();
// 预期: 捕获错误，输出日志，不崩溃

// 测试异步错误
const testAsync = safeCallAsync(async () => {
    throw new Error('测试异步错误');
}, {
    name: 'testAsync',
    context: 'Test'
});
testAsync();
// 预期: 捕获错误，输出日志，不崩溃

// 查看错误统计
getErrorStats();
// 预期: 显示错误统计信息
```

### 3. Bugly 上报测试

在 Console 执行：

```javascript
// 手动上报
reportToBugly({
    message: '测试错误上报',
    level: 'error'
});

// 查看本地报告
const reports = getBuglyReports();
console.log('报告数量:', reports.length);
console.table(reports);
// 预期: 显示已上报的错误
```

### 4. 仪表板测试

访问 https://www.tommienotes.com/bugly-dashboard.html

- [ ] 页面正常加载
- [ ] 统计卡片显示正确
- [ ] 错误列表显示（如果有错误）
- [ ] 过滤功能正常
- [ ] 搜索功能正常
- [ ] 导出功能正常
- [ ] 清除功能正常

---

## 📊 性能验证

### 1. Lighthouse 测试

运行 Lighthouse 移动端测试：

```bash
# 或在 Chrome DevTools → Lighthouse
```

预期分数：
- Performance: > 80
- Accessibility: > 90
- Best Practices: > 90
- SEO: > 90

### 2. 内存使用

打开 Chrome DevTools → Performance Monitor：

监控指标：
- JavaScript heap size: < 150MB
- DOM Nodes: < 2500
- Event Listeners: < 50

### 3. 错误率

监控 24 小时后：

通过 Bugly Dashboard 查看：
- 错误率: < 5%
- 崩溃率: < 2%
- 用户影响: < 1%

---

## 🔐 安全检查

### 1. Content Security Policy

检查 CSP 头：

```bash
curl -I https://www.tommienotes.com/ | grep -i "content-security-policy"
```

### 2. HTTPS

```bash
# 检查 SSL 证书
openssl s_client -connect www.tommienotes.com:443 -servername www.tommienotes.com

# 检查有效期
echo | openssl s_client -servername www.tommienotes.com -connect www.tommienotes.com:443 2>/dev/null | openssl x509 -noout -dates
```

### 3. 敏感信息

确认没有泄露：
- [ ] API Keys
- [ ] App IDs（应该是占位符或实际配置）
- [ ] 调试信息（生产环境应禁用）

---

## 📝 用户通知

### 清除缓存通知

建议用户：

#### 方法1: 硬刷新
- Windows/Linux: `Ctrl + Shift + R`
- Mac: `Cmd + Shift + R`

#### 方法2: 清除 Service Worker
1. F12 → Application
2. Service Workers → Unregister
3. Storage → Clear site data
4. 刷新页面

### 预期效果

- 页面加载更快
- 无崩溃或白屏
- 图片正常加载
- Mermaid图表正常显示

---

## 📈 监控设置

### 1. Bugly Dashboard

设置定期检查：
- 每日: 查看错误趋势
- 每周: 导出错误报告
- 每月: 分析错误模式

### 2. 性能监控

设置基线和告警：
- 页面加载时间 > 3秒 → 告警
- 错误率 > 5% → 告警
- 崩溃率 > 2% → 告警
- 内存使用 > 200MB → 告警

### 3. 用户反馈

收集渠道：
- 错误报告
- 用户评论
- 分析数据

---

## ✅ 最终确认

### 核心功能

- [ ] 网站可正常访问
- [ ] 移动端无崩溃
- [ ] 图片正常加载
- [ ] Mermaid图表正常
- [ ] 错误自动上报
- [ ] 仪表板正常工作

### 性能指标

- [ ] 首屏加载 < 2秒
- [ ] 内存占用 < 150MB
- [ ] Lighthouse 性能 > 80

### 监控系统

- [ ] Bugly正常上报
- [ ] 本地错误存储正常
- [ ] 错误统计正常
- [ ] 仪表板显示正常

### 文档完整性

- [ ] 所有技术文档已创建
- [ ] 使用指南已完善
- [ ] 部署步骤已记录
- [ ] 故障排查指南已准备

---

## 🎉 部署完成

### 签署确认

```
部署人员: ______________
部署时间: ______________
验证人员: ______________
验证时间: ______________
```

### 部署状态

```
[ ] 成功 - 所有检查通过
[ ] 部分成功 - 有警告但可接受
[ ] 失败 - 需要回滚
```

### 备注

```
_______________________________________
_______________________________________
_______________________________________
```

---

## 📞 应急联系

### 回滚步骤

如果出现严重问题：

```bash
# 1. SSH到服务器
ssh blog-svr

# 2. 恢复 Nginx 配置
sudo cp /etc/nginx/sites-available/blog.backup-YYYYMMDD /etc/nginx/sites-available/blog
sudo nginx -t
sudo systemctl restart nginx

# 3. 恢复网站文件
# （如果有备份）
sudo rsync -avz /var/www/html.backup/ /var/www/html/
```

### 技术支持

- Bugly 文档: https://bugly.qq.com/docs/
- Hugo 文档: https://gohugo.io/documentation/
- 项目文档: 见上述各文档链接

---

**检查清单版本**: v2.0  
**最后更新**: 2025-10-06  
**下次审查**: 2025-10-13

