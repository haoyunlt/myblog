// 移动端性能优化JavaScript
// 延迟加载、预加载、性能监控

(function() {
    'use strict';
    
    // 性能配置
    const PERF_CONFIG = {
        // 延迟加载阈值
        lazyLoadThreshold: '50px',
        // 预加载延迟
        preloadDelay: 1000,
        // 性能监控采样率
        monitoringSampleRate: 0.1,
        // 调试模式
        debug: false
    };
    
    // 性能指标收集器
    class PerformanceMonitor {
        constructor() {
            this.metrics = {};
            this.observers = new Map();
            this.startTime = performance.now();
            
            if (Math.random() < PERF_CONFIG.monitoringSampleRate) {
                this.init();
            }
        }
        
        init() {
            this.collectCoreWebVitals();
            this.monitorResourceLoading();
            this.trackUserInteractions();
            this.setupPerformanceObserver();
            
            // 页面卸载时发送数据
            window.addEventListener('beforeunload', () => {
                this.sendMetrics();
            });
        }
        
        // 收集Core Web Vitals
        collectCoreWebVitals() {
            // LCP (Largest Contentful Paint)
            new PerformanceObserver((entryList) => {
                const entries = entryList.getEntries();
                const lastEntry = entries[entries.length - 1];
                this.metrics.lcp = lastEntry.startTime;
                this.logMetric('LCP', lastEntry.startTime);
            }).observe({entryTypes: ['largest-contentful-paint']});
            
            // FID (First Input Delay)
            new PerformanceObserver((entryList) => {
                const firstInput = entryList.getEntries()[0];
                this.metrics.fid = firstInput.processingStart - firstInput.startTime;
                this.logMetric('FID', this.metrics.fid);
            }).observe({entryTypes: ['first-input']});
            
            // CLS (Cumulative Layout Shift)
            let clsValue = 0;
            new PerformanceObserver((entryList) => {
                for (const entry of entryList.getEntries()) {
                    if (!entry.hadRecentInput) {
                        clsValue += entry.value;
                    }
                }
                this.metrics.cls = clsValue;
                this.logMetric('CLS', clsValue);
            }).observe({entryTypes: ['layout-shift']});
        }
        
        // 监控资源加载
        monitorResourceLoading() {
            const observer = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    if (entry.initiatorType === 'img') {
                        this.trackImageLoad(entry);
                    }
                }
            });
            observer.observe({entryTypes: ['resource']});
            this.observers.set('resource', observer);
        }
        
        // 跟踪图片加载
        trackImageLoad(entry) {
            const loadTime = entry.responseEnd - entry.startTime;
            if (!this.metrics.imageLoads) {
                this.metrics.imageLoads = [];
            }
            this.metrics.imageLoads.push({
                url: entry.name,
                loadTime: loadTime,
                size: entry.transferSize
            });
            
            this.logMetric('Image Load', `${entry.name}: ${loadTime.toFixed(2)}ms`);
        }
        
        // 跟踪用户交互
        trackUserInteractions() {
            let interactionCount = 0;
            const interactionTypes = ['click', 'touch', 'scroll'];
            
            interactionTypes.forEach(type => {
                document.addEventListener(type, () => {
                    interactionCount++;
                    if (interactionCount === 1) {
                        this.metrics.firstInteraction = performance.now() - this.startTime;
                    }
                }, { passive: true, once: true });
            });
        }
        
        // 设置性能观察器
        setupPerformanceObserver() {
            if ('PerformanceObserver' in window) {
                try {
                    const observer = new PerformanceObserver((list) => {
                        for (const entry of list.getEntries()) {
                            this.handlePerformanceEntry(entry);
                        }
                    });
                    observer.observe({entryTypes: ['measure', 'navigation']});
                    this.observers.set('performance', observer);
                } catch (e) {
                    console.warn('PerformanceObserver not supported:', e);
                }
            }
        }
        
        // 处理性能条目
        handlePerformanceEntry(entry) {
            if (entry.entryType === 'navigation') {
                this.metrics.navigationTiming = {
                    domInteractive: entry.domInteractive,
                    domContentLoaded: entry.domContentLoadedEventEnd,
                    loadComplete: entry.loadEventEnd
                };
            }
        }
        
        // 记录指标
        logMetric(name, value) {
            if (PERF_CONFIG.debug) {
                console.log(`[Perf] ${name}:`, value);
            }
        }
        
        // 发送指标数据
        sendMetrics() {
            if (navigator.sendBeacon && Object.keys(this.metrics).length > 0) {
                const data = JSON.stringify({
                    url: window.location.href,
                    userAgent: navigator.userAgent,
                    timestamp: Date.now(),
                    metrics: this.metrics
                });
                
                // 发送到分析端点
                navigator.sendBeacon('/api/analytics', data);
            }
        }
        
        // 清理观察器
        cleanup() {
            this.observers.forEach(observer => observer.disconnect());
            this.observers.clear();
        }
    }
    
    // 懒加载管理器
    class LazyLoadManager {
        constructor() {
            this.observer = null;
            this.imageQueue = new Set();
            this.init();
        }
        
        init() {
            if ('IntersectionObserver' in window) {
                this.observer = new IntersectionObserver(
                    this.handleIntersection.bind(this),
                    {
                        rootMargin: PERF_CONFIG.lazyLoadThreshold,
                        threshold: 0.1
                    }
                );
                
                this.observeImages();
                this.observeIframes();
            } else {
                // 降级处理
                this.fallbackLazyLoad();
            }
        }
        
        // 观察图片
        observeImages() {
            const images = document.querySelectorAll('img[loading="lazy"], img[data-src]');
            images.forEach(img => {
                this.observer.observe(img);
                this.imageQueue.add(img);
            });
        }
        
        // 观察iframe
        observeIframes() {
            const iframes = document.querySelectorAll('iframe[data-src]');
            iframes.forEach(iframe => {
                this.observer.observe(iframe);
            });
        }
        
        // 处理交集变化
        handleIntersection(entries) {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.loadElement(entry.target);
                    this.observer.unobserve(entry.target);
                }
            });
        }
        
        // 加载元素
        loadElement(element) {
            if (element.tagName === 'IMG') {
                this.loadImage(element);
            } else if (element.tagName === 'IFRAME') {
                this.loadIframe(element);
            }
        }
        
        // 加载图片
        loadImage(img) {
            const src = img.dataset.src || img.src;
            if (src && src !== img.src) {
                // 创建新图片预加载
                const newImg = new Image();
                
                newImg.onload = () => {
                    img.src = src;
                    img.classList.add('loaded');
                    img.removeAttribute('data-src');
                    this.imageQueue.delete(img);
                };
                
                newImg.onerror = () => {
                    img.classList.add('error');
                    this.imageQueue.delete(img);
                };
                
                newImg.src = src;
            }
        }
        
        // 加载iframe
        loadIframe(iframe) {
            const src = iframe.dataset.src;
            if (src) {
                iframe.src = src;
                iframe.removeAttribute('data-src');
            }
        }
        
        // 降级处理
        fallbackLazyLoad() {
            const images = document.querySelectorAll('img[data-src]');
            let timeout;
            
            const loadImages = () => {
                clearTimeout(timeout);
                timeout = setTimeout(() => {
                    images.forEach(img => {
                        const rect = img.getBoundingClientRect();
                        if (rect.top < window.innerHeight + 100) {
                            this.loadImage(img);
                        }
                    });
                }, 100);
            };
            
            window.addEventListener('scroll', loadImages, { passive: true });
            window.addEventListener('resize', loadImages, { passive: true });
            loadImages(); // 初始加载
        }
    }
    
    // 预加载管理器
    class PreloadManager {
        constructor() {
            this.preloadQueue = [];
            this.init();
        }
        
        init() {
            // 延迟启动预加载
            setTimeout(() => {
                this.preloadCriticalResources();
                this.preloadNextPages();
            }, PERF_CONFIG.preloadDelay);
        }
        
        // 预加载关键资源
        preloadCriticalResources() {
            // 预加载字体
            this.preloadFonts();
            
            // 预加载关键图片
            this.preloadImages();
            
            // 预加载关键CSS
            this.preloadStyles();
        }
        
        // 预加载字体
        preloadFonts() {
            const fonts = [
                '/fonts/inter/inter-regular.woff2',
                '/fonts/inter/inter-bold.woff2'
            ];
            
            fonts.forEach(font => {
                const link = document.createElement('link');
                link.rel = 'preload';
                link.as = 'font';
                link.type = 'font/woff2';
                link.href = font;
                link.crossOrigin = 'anonymous';
                document.head.appendChild(link);
            });
        }
        
        // 预加载图片
        preloadImages() {
            // 预加载首屏可能用到的图片
            const criticalImages = document.querySelectorAll('.post-entry img, .category-card img');
            
            Array.from(criticalImages).slice(0, 5).forEach(img => {
                if (img.dataset.src || img.src) {
                    const src = img.dataset.src || img.src;
                    this.preloadResource(src, 'image');
                }
            });
        }
        
        // 预加载样式
        preloadStyles() {
            const criticalStyles = [
                '/assets/css/mobile-performance.css'
            ];
            
            criticalStyles.forEach(href => {
                this.preloadResource(href, 'style');
            });
        }
        
        // 预加载下一页
        preloadNextPages() {
            const nextPageLinks = document.querySelectorAll('a[href^="/posts/"], .pagi a[href]');
            
            // 预加载前3个链接
            Array.from(nextPageLinks).slice(0, 3).forEach(link => {
                if (link.href) {
                    this.preloadResource(link.href, 'document');
                }
            });
        }
        
        // 预加载资源
        preloadResource(href, as) {
            const link = document.createElement('link');
            link.rel = 'preload';
            link.as = as;
            link.href = href;
            
            if (as === 'font') {
                link.crossOrigin = 'anonymous';
            }
            
            document.head.appendChild(link);
            this.preloadQueue.push(link);
        }
    }
    
    // 交互优化器
    class InteractionOptimizer {
        constructor() {
            this.touchStartTime = 0;
            this.init();
        }
        
        init() {
            this.optimizeScrolling();
            this.optimizeTouchEvents();
            this.optimizeClickEvents();
            this.preventGhostClicks();
        }
        
        // 优化滚动
        optimizeScrolling() {
            let scrollTimer;
            let isScrolling = false;
            
            const handleScroll = () => {
                if (!isScrolling) {
                    isScrolling = true;
                    document.body.classList.add('scrolling');
                }
                
                clearTimeout(scrollTimer);
                scrollTimer = setTimeout(() => {
                    isScrolling = false;
                    document.body.classList.remove('scrolling');
                }, 150);
            };
            
            window.addEventListener('scroll', handleScroll, { passive: true });
        }
        
        // 优化触摸事件
        optimizeTouchEvents() {
            document.addEventListener('touchstart', (e) => {
                this.touchStartTime = Date.now();
                
                // 添加触摸反馈
                const target = e.target.closest('.post-entry, .category-card, button, .btn');
                if (target) {
                    target.classList.add('touching');
                }
            }, { passive: true });
            
            document.addEventListener('touchend', (e) => {
                const touchDuration = Date.now() - this.touchStartTime;
                
                // 移除触摸反馈
                const target = e.target.closest('.post-entry, .category-card, button, .btn');
                if (target) {
                    setTimeout(() => {
                        target.classList.remove('touching');
                    }, 100);
                }
                
                // 快速触摸优化
                if (touchDuration < 150) {
                    e.target.classList.add('fast-touch');
                    setTimeout(() => {
                        e.target.classList.remove('fast-touch');
                    }, 300);
                }
            }, { passive: true });
        }
        
        // 优化点击事件
        optimizeClickEvents() {
            // 使用事件委托减少事件监听器数量
            document.addEventListener('click', (e) => {
                const target = e.target;
                
                // 优化链接预取
                if (target.tagName === 'A' && target.href) {
                    this.prefetchOnHover(target);
                }
                
                // 防止重复点击
                if (target.classList.contains('clicking')) {
                    e.preventDefault();
                    return false;
                }
                
                target.classList.add('clicking');
                setTimeout(() => {
                    target.classList.remove('clicking');
                }, 300);
            });
        }
        
        // 悬停时预取
        prefetchOnHover(link) {
            let hoverTimer;
            
            link.addEventListener('mouseenter', () => {
                hoverTimer = setTimeout(() => {
                    if (link.href && !link.dataset.prefetched) {
                        const prefetchLink = document.createElement('link');
                        prefetchLink.rel = 'prefetch';
                        prefetchLink.href = link.href;
                        document.head.appendChild(prefetchLink);
                        link.dataset.prefetched = 'true';
                    }
                }, 200);
            }, { once: true });
            
            link.addEventListener('mouseleave', () => {
                clearTimeout(hoverTimer);
            }, { once: true });
        }
        
        // 防止幽灵点击
        preventGhostClicks() {
            const ghostClickBuster = (e) => {
                if (Date.now() - this.touchStartTime < 500) {
                    e.preventDefault();
                    e.stopPropagation();
                    return false;
                }
            };
            
            document.addEventListener('click', ghostClickBuster, true);
        }
    }
    
    // 内存管理器
    class MemoryManager {
        constructor() {
            this.cleanupTasks = [];
            this.init();
        }
        
        init() {
            this.monitorMemory();
            this.setupCleanup();
            this.optimizeGarbageCollection();
        }
        
        // 监控内存使用
        monitorMemory() {
            if ('memory' in performance) {
                setInterval(() => {
                    const memory = performance.memory;
                    const usage = (memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100;
                    
                    if (usage > 80) {
                        console.warn('内存使用率过高:', usage.toFixed(2) + '%');
                        this.triggerCleanup();
                    }
                }, 30000); // 每30秒检查一次
            }
        }
        
        // 设置清理任务
        setupCleanup() {
            // 页面隐藏时清理
            document.addEventListener('visibilitychange', () => {
                if (document.hidden) {
                    this.triggerCleanup();
                }
            });
            
            // 页面卸载时清理
            window.addEventListener('beforeunload', () => {
                this.cleanup();
            });
        }
        
        // 触发清理
        triggerCleanup() {
            // 清理未使用的图片
            this.cleanupImages();
            
            // 清理DOM节点
            this.cleanupDOMNodes();
            
            // 强制垃圾回收（如果支持）
            if (window.gc) {
                window.gc();
            }
        }
        
        // 清理图片
        cleanupImages() {
            const images = document.querySelectorAll('img');
            images.forEach(img => {
                if (!this.isInViewport(img) && img.src) {
                    // 清理不在视口的图片
                    img.removeAttribute('src');
                    img.dataset.cleaned = 'true';
                }
            });
        }
        
        // 清理DOM节点
        cleanupDOMNodes() {
            // 移除隐藏的元素
            const hiddenElements = document.querySelectorAll('[style*="display: none"]');
            hiddenElements.forEach(el => {
                if (!el.dataset.important) {
                    el.remove();
                }
            });
        }
        
        // 检查是否在视口内
        isInViewport(element) {
            const rect = element.getBoundingClientRect();
            return (
                rect.top >= 0 &&
                rect.left >= 0 &&
                rect.bottom <= window.innerHeight &&
                rect.right <= window.innerWidth
            );
        }
        
        // 优化垃圾回收
        optimizeGarbageCollection() {
            // 定期清理引用
            setInterval(() => {
                this.cleanupTasks = this.cleanupTasks.filter(task => task.active);
            }, 60000); // 每分钟清理一次
        }
        
        // 完全清理
        cleanup() {
            this.cleanupTasks.forEach(task => {
                if (typeof task.cleanup === 'function') {
                    task.cleanup();
                }
            });
            this.cleanupTasks = [];
        }
    }
    
    // Service Worker注册
    class ServiceWorkerManager {
        constructor() {
            this.init();
        }
        
        init() {
            if ('serviceWorker' in navigator) {
                window.addEventListener('load', () => {
                    this.register();
                });
            }
        }
        
        async register() {
            try {
                const registration = await navigator.serviceWorker.register('/sw.js');
                console.log('Service Worker注册成功:', registration.scope);
                
                // 监听更新
                registration.addEventListener('updatefound', () => {
                    const newWorker = registration.installing;
                    newWorker.addEventListener('statechange', () => {
                        if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                            // 有新版本可用
                            this.showUpdateNotification();
                        }
                    });
                });
                
            } catch (error) {
                console.error('Service Worker注册失败:', error);
            }
        }
        
        showUpdateNotification() {
            // 显示更新提示
            const notification = document.createElement('div');
            notification.innerHTML = `
                <div style="position: fixed; top: 0; left: 0; right: 0; background: #2563eb; color: white; padding: 1rem; text-align: center; z-index: 10000;">
                    <p>有新版本可用！<button onclick="window.location.reload()" style="margin-left: 1rem; padding: 0.5rem 1rem; background: white; color: #2563eb; border: none; border-radius: 4px; cursor: pointer;">更新</button></p>
                </div>
            `;
            document.body.appendChild(notification);
        }
    }
    
    // 初始化所有优化器
    function initMobileOptimization() {
        // 检查是否为移动设备
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ||
                         window.innerWidth <= 768;
        
        if (isMobile) {
            document.body.classList.add('mobile-optimized');
            
            // 初始化各种优化器
            const performanceMonitor = new PerformanceMonitor();
            const lazyLoadManager = new LazyLoadManager();
            const preloadManager = new PreloadManager();
            const interactionOptimizer = new InteractionOptimizer();
            const memoryManager = new MemoryManager();
            const serviceWorkerManager = new ServiceWorkerManager();
            
            // 添加到全局对象以便调试
            if (PERF_CONFIG.debug) {
                window.mobileOptimization = {
                    performanceMonitor,
                    lazyLoadManager,
                    preloadManager,
                    interactionOptimizer,
                    memoryManager,
                    serviceWorkerManager
                };
            }
            
            console.log('移动端性能优化已启用');
        }
    }
    
    // DOM加载完成后初始化
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initMobileOptimization);
    } else {
        initMobileOptimization();
    }
    
})();
