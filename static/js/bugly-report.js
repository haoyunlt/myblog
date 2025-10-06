// 腾讯 Bugly 崩溃上报工具 v1.0
// 用于收集移动端 Chrome 崩溃和错误信息

(function() {
    'use strict';
    
    // Bugly 配置
    const BUGLY_CONFIG = {
        appId: 'YOUR_APP_ID',  // 从 Bugly 控制台获取
        appVersion: '1.0.0',
        userId: '',  // 可选：用户标识
        enableDebug: false,  // 生产环境设为 false
        delay: 1000,  // 延迟上报（毫秒）
        random: 1,  // 上报采样率 (0-1)，1 表示全部上报
        repeat: 5,  // 重复上报次数限制
        reportUrl: 'https://bugly.qq.com/api/report',  // 上报地址
        
        // 自定义字段
        customFields: {
            website: 'tommienotes.com',
            platform: 'mobile-web'
        }
    };
    
    // 检测是否为移动设备
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ||
                     window.innerWidth <= 768;
    
    // 非移动设备可以选择性启用
    // 修改为 true 以在桌面浏览器测试
    const ENABLE_ON_DESKTOP = true;  // 已启用桌面支持用于测试
    
    if (!isMobile && !ENABLE_ON_DESKTOP) {
        console.log('[Bugly] 非移动设备，跳过错误上报');
        return;
    }
    
    // 桌面设备提示
    if (!isMobile && ENABLE_ON_DESKTOP) {
        console.log('[Bugly] 桌面设备检测模式已启用');
    }
    
    // 错误收集器
    class BuglyReporter {
        constructor(config) {
            this.config = config;
            this.errorCache = [];
            this.reportedErrors = new Set();
            this.reportCount = {};
            this.isInitialized = false;
            this.sessionId = this.generateSessionId();
            
            this.init();
        }
        
        init() {
            try {
                console.log('[Bugly] 初始化错误上报系统');
                
                // 收集设备信息
                this.deviceInfo = this.getDeviceInfo();
                
                // 监听全局错误
                this.setupErrorListeners();
                
                // 监听资源加载错误
                this.setupResourceErrorListeners();
                
                // 监听 Promise 拒绝
                this.setupPromiseRejectionListeners();
                
                // 监听性能问题
                this.setupPerformanceMonitoring();
                
                // 定期上报
                this.startReportInterval();
                
                // 页面卸载时上报
                this.setupBeforeUnloadReport();
                
                this.isInitialized = true;
                console.log('[Bugly] 错误上报系统初始化成功', this.sessionId);
                
            } catch (e) {
                console.error('[Bugly] 初始化失败', e);
            }
        }
        
        // 生成会话ID
        generateSessionId() {
            return 'session_' + Date.now() + '_' + Math.random().toString(36).substring(2, 15);
        }
        
        // 获取设备信息
        getDeviceInfo() {
            const ua = navigator.userAgent;
            const screen = window.screen || {};
            const performance = window.performance || {};
            const timing = performance.timing || {};
            
            return {
                // 浏览器信息
                userAgent: ua,
                platform: navigator.platform,
                language: navigator.language,
                cookieEnabled: navigator.cookieEnabled,
                onLine: navigator.onLine,
                
                // 屏幕信息
                screenWidth: screen.width,
                screenHeight: screen.height,
                screenOrientation: screen.orientation ? screen.orientation.type : 'unknown',
                pixelRatio: window.devicePixelRatio || 1,
                colorDepth: screen.colorDepth,
                
                // 窗口信息
                windowWidth: window.innerWidth,
                windowHeight: window.innerHeight,
                
                // 内存信息
                memory: performance.memory ? {
                    usedJSHeapSize: performance.memory.usedJSHeapSize,
                    totalJSHeapSize: performance.memory.totalJSHeapSize,
                    jsHeapSizeLimit: performance.memory.jsHeapSizeLimit
                } : null,
                
                // 时间信息
                timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
                timestamp: Date.now(),
                
                // 页面信息
                url: window.location.href,
                referrer: document.referrer,
                title: document.title,
                
                // 性能信息
                loadTime: timing.loadEventEnd ? timing.loadEventEnd - timing.navigationStart : 0,
                domReady: timing.domContentLoadedEventEnd ? timing.domContentLoadedEventEnd - timing.navigationStart : 0,
                
                // 自定义字段
                ...this.config.customFields
            };
        }
        
        // 监听全局错误
        setupErrorListeners() {
            window.addEventListener('error', (event) => {
                try {
                    // JavaScript 错误
                    if (event.error) {
                        this.reportError({
                            type: 'javascript_error',
                            message: event.error.message || event.message,
                            stack: event.error.stack || '',
                            filename: event.filename || '',
                            lineno: event.lineno || 0,
                            colno: event.colno || 0,
                            timestamp: Date.now(),
                            url: window.location.href,
                            errorType: event.error.name || 'Error'
                        });
                    }
                } catch (e) {
                    console.error('[Bugly] 处理错误事件失败', e);
                }
            }, true);
        }
        
        // 监听资源加载错误
        setupResourceErrorListeners() {
            window.addEventListener('error', (event) => {
                try {
                    // 资源加载错误
                    if (event.target && event.target !== window) {
                        const element = event.target;
                        this.reportError({
                            type: 'resource_error',
                            message: `Resource load failed: ${element.tagName}`,
                            resourceType: element.tagName.toLowerCase(),
                            resourceUrl: element.src || element.href || '',
                            timestamp: Date.now(),
                            url: window.location.href
                        });
                    }
                } catch (e) {
                    console.error('[Bugly] 处理资源错误失败', e);
                }
            }, true);
        }
        
        // 监听 Promise 拒绝
        setupPromiseRejectionListeners() {
            window.addEventListener('unhandledrejection', (event) => {
                try {
                    this.reportError({
                        type: 'promise_rejection',
                        message: event.reason ? event.reason.message || String(event.reason) : 'Unhandled Promise Rejection',
                        stack: event.reason && event.reason.stack ? event.reason.stack : '',
                        timestamp: Date.now(),
                        url: window.location.href
                    });
                } catch (e) {
                    console.error('[Bugly] 处理 Promise 拒绝失败', e);
                }
            });
        }
        
        // 监听性能问题
        setupPerformanceMonitoring() {
            // 监听长任务
            if ('PerformanceObserver' in window) {
                try {
                    const observer = new PerformanceObserver((list) => {
                        for (const entry of list.getEntries()) {
                            if (entry.duration > 50) {  // 超过50ms的长任务
                                this.reportError({
                                    type: 'performance_long_task',
                                    message: `Long task detected: ${entry.duration.toFixed(2)}ms`,
                                    duration: entry.duration,
                                    startTime: entry.startTime,
                                    timestamp: Date.now(),
                                    url: window.location.href
                                });
                            }
                        }
                    });
                    
                    observer.observe({ entryTypes: ['longtask'] });
                } catch (e) {
                    console.warn('[Bugly] PerformanceObserver 不支持', e);
                }
            }
            
            // 监听内存警告
            if (window.performance && window.performance.memory) {
                setInterval(() => {
                    try {
                        const memory = window.performance.memory;
                        const usagePercent = memory.usedJSHeapSize / memory.jsHeapSizeLimit;
                        
                        if (usagePercent > 0.9) {  // 内存使用超过90%
                            this.reportError({
                                type: 'performance_memory_warning',
                                message: `High memory usage: ${(usagePercent * 100).toFixed(1)}%`,
                                usedMB: Math.round(memory.usedJSHeapSize / 1048576),
                                limitMB: Math.round(memory.jsHeapSizeLimit / 1048576),
                                timestamp: Date.now(),
                                url: window.location.href
                            });
                        }
                    } catch (e) {
                        // 静默失败
                    }
                }, 10000);  // 每10秒检查一次
            }
        }
        
        // 上报错误
        reportError(errorData) {
            try {
                // 检查采样率
                if (Math.random() > this.config.random) {
                    return;
                }
                
                // 生成错误唯一标识
                const errorId = this.generateErrorId(errorData);
                
                // 检查是否已上报
                if (this.reportedErrors.has(errorId)) {
                    this.reportCount[errorId] = (this.reportCount[errorId] || 0) + 1;
                    
                    // 检查重复次数限制
                    if (this.reportCount[errorId] > this.config.repeat) {
                        return;
                    }
                }
                
                // 标记为已上报
                this.reportedErrors.add(errorId);
                
                // 构建完整错误报告
                const report = {
                    ...errorData,
                    errorId: errorId,
                    sessionId: this.sessionId,
                    appId: this.config.appId,
                    appVersion: this.config.appVersion,
                    userId: this.config.userId,
                    device: this.deviceInfo,
                    repeatCount: this.reportCount[errorId] || 1
                };
                
                // 添加到缓存
                this.errorCache.push(report);
                
                // 调试模式输出
                if (this.config.enableDebug) {
                    console.group('[Bugly] 错误报告');
                    console.log('错误类型:', errorData.type);
                    console.log('错误消息:', errorData.message);
                    console.log('完整报告:', report);
                    console.groupEnd();
                }
                
                // 立即上报严重错误
                if (this.isCriticalError(errorData)) {
                    this.flushReports();
                }
                
            } catch (e) {
                console.error('[Bugly] 上报错误失败', e);
            }
        }
        
        // 生成错误唯一标识
        generateErrorId(errorData) {
            const key = `${errorData.type}_${errorData.message}_${errorData.filename || ''}_${errorData.lineno || 0}`;
            return this.hashCode(key);
        }
        
        // 简单哈希函数
        hashCode(str) {
            let hash = 0;
            for (let i = 0; i < str.length; i++) {
                const char = str.charCodeAt(i);
                hash = ((hash << 5) - hash) + char;
                hash = hash & hash;
            }
            return hash.toString(36);
        }
        
        // 判断是否为严重错误
        isCriticalError(errorData) {
            const criticalTypes = [
                'javascript_error',
                'promise_rejection'
            ];
            
            const criticalKeywords = [
                'crash',
                'fatal',
                'cannot read property',
                'undefined is not',
                'null is not an object'
            ];
            
            if (criticalTypes.includes(errorData.type)) {
                return true;
            }
            
            const message = (errorData.message || '').toLowerCase();
            return criticalKeywords.some(keyword => message.includes(keyword));
        }
        
        // 定期上报
        startReportInterval() {
            setInterval(() => {
                if (this.errorCache.length > 0) {
                    this.flushReports();
                }
            }, this.config.delay);
        }
        
        // 批量上报
        flushReports() {
            if (this.errorCache.length === 0) {
                return;
            }
            
            try {
                const reports = [...this.errorCache];
                this.errorCache = [];
                
                // 发送到服务器
                this.sendToServer(reports);
                
                // 同时存储到本地
                this.saveToLocalStorage(reports);
                
            } catch (e) {
                console.error('[Bugly] 批量上报失败', e);
            }
        }
        
        // 发送到服务器
        sendToServer(reports) {
            try {
                // 使用 sendBeacon（优先）
                if (navigator.sendBeacon) {
                    const data = JSON.stringify({
                        appId: this.config.appId,
                        reports: reports,
                        timestamp: Date.now()
                    });
                    
                    const blob = new Blob([data], { type: 'application/json' });
                    const sent = navigator.sendBeacon(this.config.reportUrl, blob);
                    
                    if (sent && this.config.enableDebug) {
                        console.log('[Bugly] 使用 sendBeacon 上报成功', reports.length);
                    }
                    
                    return sent;
                }
                
                // 降级：使用 fetch
                if (window.fetch) {
                    fetch(this.config.reportUrl, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            appId: this.config.appId,
                            reports: reports,
                            timestamp: Date.now()
                        }),
                        keepalive: true
                    }).then(response => {
                        if (this.config.enableDebug) {
                            console.log('[Bugly] 使用 fetch 上报成功', response.status);
                        }
                    }).catch(e => {
                        console.error('[Bugly] fetch 上报失败', e);
                    });
                    
                    return true;
                }
                
                // 最终降级：使用 XMLHttpRequest
                const xhr = new XMLHttpRequest();
                xhr.open('POST', this.config.reportUrl, true);
                xhr.setRequestHeader('Content-Type', 'application/json');
                xhr.send(JSON.stringify({
                    appId: this.config.appId,
                    reports: reports,
                    timestamp: Date.now()
                }));
                
                if (this.config.enableDebug) {
                    console.log('[Bugly] 使用 XMLHttpRequest 上报');
                }
                
            } catch (e) {
                console.error('[Bugly] 发送到服务器失败', e);
            }
        }
        
        // 保存到本地存储
        saveToLocalStorage(reports) {
            try {
                const key = 'bugly_reports';
                const existing = JSON.parse(localStorage.getItem(key) || '[]');
                const merged = [...existing, ...reports].slice(-100);  // 最多保存100条
                
                localStorage.setItem(key, JSON.stringify(merged));
                
                if (this.config.enableDebug) {
                    console.log('[Bugly] 保存到本地存储', merged.length);
                }
            } catch (e) {
                console.error('[Bugly] 保存到本地存储失败', e);
            }
        }
        
        // 页面卸载时上报
        setupBeforeUnloadReport() {
            window.addEventListener('beforeunload', () => {
                this.flushReports();
            });
            
            // 页面隐藏时也上报
            document.addEventListener('visibilitychange', () => {
                if (document.hidden) {
                    this.flushReports();
                }
            });
        }
        
        // 手动上报自定义错误
        report(errorData) {
            this.reportError({
                type: 'custom_error',
                ...errorData,
                timestamp: Date.now(),
                url: window.location.href
            });
        }
        
        // 获取本地报告
        getLocalReports() {
            try {
                const key = 'bugly_reports';
                return JSON.parse(localStorage.getItem(key) || '[]');
            } catch (e) {
                return [];
            }
        }
        
        // 清除本地报告
        clearLocalReports() {
            try {
                localStorage.removeItem('bugly_reports');
                console.log('[Bugly] 本地报告已清除');
            } catch (e) {
                console.error('[Bugly] 清除本地报告失败', e);
            }
        }
    }
    
    // 创建全局实例
    try {
        window.BuglyReporter = new BuglyReporter(BUGLY_CONFIG);
        
        // 提供全局方法
        window.reportToBugly = function(errorData) {
            if (window.BuglyReporter) {
                window.BuglyReporter.report(errorData);
            }
        };
        
        window.getBuglyReports = function() {
            if (window.BuglyReporter) {
                return window.BuglyReporter.getLocalReports();
            }
            return [];
        };
        
        window.clearBuglyReports = function() {
            if (window.BuglyReporter) {
                window.BuglyReporter.clearLocalReports();
            }
        };
        
        console.log('[Bugly] ✅ 崩溃上报系统已启动');
        console.log('[Bugly] 💡 使用 reportToBugly({message: "..."}) 手动上报错误');
        console.log('[Bugly] 💡 使用 getBuglyReports() 查看本地报告');
        console.log('[Bugly] 💡 使用 clearBuglyReports() 清除本地报告');
        
    } catch (e) {
        console.error('[Bugly] 初始化失败', e);
    }
    
})();

