// 移动端Chrome崩溃诊断和修复工具
// 专门解决移动端Chrome崩溃问题

(function() {
    'use strict';
    
    console.log('🔧 移动端崩溃诊断工具启动');
    
    // 崩溃检测和修复系统
    class MobileCrashDiagnostic {
        constructor() {
            this.crashIndicators = [];
            this.memoryThreshold = 0.8; // 80%内存使用率阈值
            this.init();
        }
        
        init() {
            this.detectCrashCauses();
            this.implementFixes();
            this.monitorHealth();
        }
        
        // 检测崩溃原因
        detectCrashCauses() {
            console.log('🔍 检测移动端崩溃原因...');
            
            // 1. 检查内存使用
            if (performance.memory) {
                const memoryUsage = performance.memory.usedJSHeapSize / performance.memory.jsHeapSizeLimit;
                if (memoryUsage > this.memoryThreshold) {
                    this.crashIndicators.push({
                        type: 'memory',
                        severity: 'high',
                        message: `内存使用率过高: ${(memoryUsage * 100).toFixed(1)}%`,
                        usage: memoryUsage
                    });
                }
            }
            
            // 2. 检查事件监听器数量
            const eventListeners = this.countEventListeners();
            if (eventListeners > 50) {
                this.crashIndicators.push({
                    type: 'events',
                    severity: 'medium',
                    message: `事件监听器过多: ${eventListeners}个`,
                    count: eventListeners
                });
            }
            
            // 3. 检查DOM节点数量
            const domNodes = document.querySelectorAll('*').length;
            if (domNodes > 2000) {
                this.crashIndicators.push({
                    type: 'dom',
                    severity: 'medium',
                    message: `DOM节点过多: ${domNodes}个`,
                    count: domNodes
                });
            }
            
            // 4. 检查图片数量
            const images = document.querySelectorAll('img').length;
            if (images > 100) {
                this.crashIndicators.push({
                    type: 'images',
                    severity: 'medium',
                    message: `图片数量过多: ${images}个`,
                    count: images
                });
            }
            
            // 5. 检查JavaScript错误
            const jsErrors = this.getJSErrors();
            if (jsErrors.length > 5) {
                this.crashIndicators.push({
                    type: 'errors',
                    severity: 'high',
                    message: `JavaScript错误过多: ${jsErrors.length}个`,
                    errors: jsErrors
                });
            }
            
            console.log('📊 崩溃指标检测完成:', this.crashIndicators);
        }
        
        // 实施修复措施
        implementFixes() {
            console.log('🔧 实施移动端崩溃修复措施...');
            
            // 1. 内存优化
            this.optimizeMemory();
            
            // 2. 事件监听器优化
            this.optimizeEventListeners();
            
            // 3. DOM优化
            this.optimizeDOM();
            
            // 4. 图片优化
            this.optimizeImages();
            
            // 5. JavaScript执行优化
            this.optimizeJavaScript();
            
            console.log('✅ 修复措施实施完成');
        }
        
        // 内存优化
        optimizeMemory() {
            console.log('🧠 优化内存使用...');
            
            // 清理未使用的图片
            const images = document.querySelectorAll('img');
            images.forEach(img => {
                if (!this.isInViewport(img) && img.src) {
                    // 延迟加载图片，减少内存占用
                    img.loading = 'lazy';
                }
            });
            
            // 清理隐藏元素
            const hiddenElements = document.querySelectorAll('[style*="display: none"]');
            hiddenElements.forEach(el => {
                if (!el.dataset.important) {
                    el.style.display = 'none';
                }
            });
            
            // 强制垃圾回收（如果支持）
            if (window.gc) {
                window.gc();
            }
        }
        
        // 事件监听器优化
        optimizeEventListeners() {
            console.log('🎧 优化事件监听器...');
            
            // 使用事件委托减少监听器数量
            document.addEventListener('click', this.handleClickDelegation, true);
            document.addEventListener('touchstart', this.handleTouchDelegation, { passive: true });
            document.addEventListener('scroll', this.handleScrollDelegation, { passive: true });
        }
        
        // DOM优化
        optimizeDOM() {
            console.log('🌳 优化DOM结构...');
            
            // 移除不必要的DOM节点
            const unnecessaryNodes = document.querySelectorAll('script[src*="analytics"], script[src*="tracking"]');
            unnecessaryNodes.forEach(node => {
                if (!node.dataset.important) {
                    node.remove();
                }
            });
            
            // 优化CSS选择器
            this.optimizeCSSSelectors();
        }
        
        // 图片优化
        optimizeImages() {
            console.log('🖼️ 优化图片加载...');
            
            // 实现更激进的图片懒加载
            const images = document.querySelectorAll('img');
            images.forEach((img, index) => {
                if (index > 10) { // 只保留前10张图片
                    img.style.display = 'none';
                    img.dataset.lazy = 'true';
                }
            });
            
            // 压缩图片质量
            images.forEach(img => {
                if (img.src && !img.dataset.optimized) {
                    img.dataset.optimized = 'true';
                    // 添加图片压缩参数
                    if (img.src.includes('?')) {
                        img.src += '&quality=80&format=webp';
                    } else {
                        img.src += '?quality=80&format=webp';
                    }
                }
            });
        }
        
        // JavaScript执行优化
        optimizeJavaScript() {
            console.log('⚡ 优化JavaScript执行...');
            
            // 使用requestIdleCallback优化非关键任务
            if ('requestIdleCallback' in window) {
                requestIdleCallback(() => {
                    this.runNonCriticalTasks();
                });
            } else {
                setTimeout(() => {
                    this.runNonCriticalTasks();
                }, 100);
            }
            
            // 减少定时器使用
            this.optimizeTimers();
        }
        
        // 运行非关键任务
        runNonCriticalTasks() {
            // 延迟执行非关键功能
            console.log('🔄 执行非关键任务...');
            
            // 延迟初始化非关键模块
            setTimeout(() => {
                this.initializeNonCriticalModules();
            }, 1000);
        }
        
        // 初始化非关键模块
        initializeNonCriticalModules() {
            // Service Worker 已禁用，不再需要初始化
            console.log('✓ 非关键模块初始化完成（Service Worker 已禁用）');
        }
        
        // 优化定时器
        optimizeTimers() {
            // 减少定时器频率
            const originalSetInterval = window.setInterval;
            window.setInterval = function(callback, delay) {
                // 移动端增加延迟时间
                const mobileDelay = delay * 1.5;
                return originalSetInterval(callback, mobileDelay);
            };
        }
        
        // 健康监控
        monitorHealth() {
            console.log('💓 启动健康监控...');
            
            // 每30秒检查一次健康状态
            setInterval(() => {
                this.checkHealth();
            }, 30000);
            
            // 页面可见性变化时清理
            document.addEventListener('visibilitychange', () => {
                if (document.hidden) {
                    this.cleanup();
                }
            });
        }
        
        // 检查健康状态
        checkHealth() {
            const health = {
                memory: this.getMemoryUsage(),
                performance: this.getPerformanceMetrics(),
                errors: this.getJSErrors().length,
                timestamp: Date.now()
            };
            
            // 如果健康状态不佳，触发清理
            if (health.memory > 0.9 || health.errors > 10) {
                console.warn('⚠️ 健康状态不佳，触发清理:', health);
                this.emergencyCleanup();
            }
        }
        
        // 紧急清理
        emergencyCleanup() {
            console.log('🚨 执行紧急清理...');
            
            // 清理所有非关键资源
            this.cleanupImages();
            this.cleanupDOM();
            this.cleanupEventListeners();
            
            // 强制垃圾回收
            if (window.gc) {
                window.gc();
            }
        }
        
        // 工具方法
        countEventListeners() {
            // 估算事件监听器数量（简化版本）
            return document.querySelectorAll('*').length * 0.1;
        }
        
        getJSErrors() {
            return window.MobileDebug?.errors || [];
        }
        
        getMemoryUsage() {
            if (performance.memory) {
                return performance.memory.usedJSHeapSize / performance.memory.jsHeapSizeLimit;
            }
            return 0;
        }
        
        getPerformanceMetrics() {
            const navigation = performance.getEntriesByType('navigation')[0];
            return {
                loadTime: navigation ? navigation.loadEventEnd - navigation.loadEventStart : 0,
                domContentLoaded: navigation ? navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart : 0
            };
        }
        
        isInViewport(element) {
            const rect = element.getBoundingClientRect();
            return (
                rect.top >= 0 &&
                rect.left >= 0 &&
                rect.bottom <= window.innerHeight &&
                rect.right <= window.innerWidth
            );
        }
        
        // 事件委托处理器
        handleClickDelegation = (e) => {
            // 统一处理点击事件
            const target = e.target;
            if (target.tagName === 'A') {
                // 链接点击优化
                this.optimizeLinkClick(target);
            }
        };
        
        handleTouchDelegation = (e) => {
            // 统一处理触摸事件
            const target = e.target;
            if (target.classList.contains('post-entry')) {
                // 文章条目触摸优化
                this.optimizePostTouch(target);
            }
        };
        
        handleScrollDelegation = (e) => {
            // 统一处理滚动事件
            this.throttleScroll();
        };
        
        // 滚动节流
        throttleScroll() {
            if (this.scrollTimeout) {
                clearTimeout(this.scrollTimeout);
            }
            this.scrollTimeout = setTimeout(() => {
                this.handleScroll();
            }, 100);
        }
        
        handleScroll() {
            // 滚动处理逻辑
            const scrollTop = window.pageYOffset;
            if (scrollTop > 1000) {
                // 滚动到一定位置时清理资源
                this.cleanupImages();
            }
        }
        
        // 清理方法
        cleanupImages() {
            const images = document.querySelectorAll('img');
            images.forEach(img => {
                if (!this.isInViewport(img) && img.src) {
                    img.src = '';
                    img.dataset.cleaned = 'true';
                }
            });
        }
        
        cleanupDOM() {
            // 清理隐藏的DOM节点
            const hiddenElements = document.querySelectorAll('[style*="display: none"]');
            hiddenElements.forEach(el => {
                if (!el.dataset.important) {
                    el.remove();
                }
            });
        }
        
        cleanupEventListeners() {
            // 清理不必要的事件监听器
            // 这里可以添加具体的清理逻辑
        }
        
        // 优化方法
        optimizeLinkClick(link) {
            // 链接点击优化
            if (link.href && !link.dataset.optimized) {
                link.dataset.optimized = 'true';
                // 预加载链接内容
                this.prefetchLink(link);
            }
        }
        
        optimizePostTouch(post) {
            // 文章触摸优化
            post.classList.add('touching');
            setTimeout(() => {
                post.classList.remove('touching');
            }, 150);
        }
        
        prefetchLink(link) {
            // 预取链接内容
            const prefetchLink = document.createElement('link');
            prefetchLink.rel = 'prefetch';
            prefetchLink.href = link.href;
            document.head.appendChild(prefetchLink);
        }
        
        optimizeCSSSelectors() {
            // 优化CSS选择器性能
            // 这里可以添加CSS优化逻辑
        }
        
        cleanup() {
            console.log('🧹 执行清理操作...');
            this.cleanupImages();
            this.cleanupDOM();
            this.cleanupEventListeners();
        }
    }
    
    // 启动诊断系统
    try {
        const diagnostic = new MobileCrashDiagnostic();
        
        // 添加到全局对象
        window.MobileCrashDiagnostic = diagnostic;
        
        console.log('✅ 移动端崩溃诊断系统启动成功');
        
        // 导出诊断报告
        window.getCrashReport = function() {
            return {
                indicators: diagnostic.crashIndicators,
                memory: diagnostic.getMemoryUsage(),
                performance: diagnostic.getPerformanceMetrics(),
                timestamp: Date.now()
            };
        };
        
    } catch (e) {
        console.error('❌ 移动端崩溃诊断系统启动失败:', e);
    }
    
})();
