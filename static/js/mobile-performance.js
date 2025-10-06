// 移动端性能优化JavaScript v3.1 - 轻量级版本（增强参数验证）
// 修复崩溃问题：移除内存密集型功能，仅保留核心懒加载
// v3.1: 添加统一参数验证和错误日志

(function() {
    'use strict';
    
    // 检测是否为移动设备
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ||
                     window.innerWidth <= 768;
    
    // 非移动设备直接返回
    if (!isMobile) {
        console.log('[Mobile-Perf] 非移动设备，跳过移动端优化');
        return;
    }
    
    // 获取调试对象
    const debug = window.MobileDebug || {
        debug: () => {},
        info: () => {},
        warn: () => {},
        error: console.error.bind(console),
        log: console.log.bind(console)
    };
    
    // 获取参数验证器
    const validator = window.mobileValidator || new (window.ParamValidator || function() {
        // 降级：基础验证器
        this.notNull = (v, n, f) => {
            if (v === null || v === undefined) {
                console.error(`[${f}] 参数 ${n} 不能为 null/undefined`, v);
                return false;
            }
            return true;
        };
        this.isImage = (v, n, f) => {
            if (!(v instanceof HTMLImageElement)) {
                console.error(`[${f}] 参数 ${n} 必须是 HTMLImageElement`, v);
                return false;
            }
            return true;
        };
        this.inRange = (v, min, max, n, f) => {
            if (typeof v !== 'number' || v < min || v > max) {
                console.error(`[${f}] 参数 ${n} 必须在 ${min}-${max} 范围内，当前值: ${v}`);
                return false;
            }
            return true;
        };
        this.validateMultiple = () => true; // 降级时跳过批量验证
    })('MobilePerformance');
    
    debug.log('[Mobile-Perf] 移动端轻量级优化系统 v3.1 启动（参数验证增强）');
    
    // 简化的懒加载管理器（仅处理图片）
    class SimpleLazyLoader {
        constructor() {
            this.observer = null;
            this.loadedCount = 0;
            this.maxImages = 30; // 限制加载图片数量
            this.init();
        }
        
        init() {
            try {
                if ('IntersectionObserver' in window) {
                    this.observer = new IntersectionObserver(
                        this.handleIntersection.bind(this),
                        {
                            rootMargin: '100px',
                            threshold: 0.01
                        }
                    );
                    
                    this.observeImages();
                    debug.log('[LazyLoad] 懒加载初始化成功');
                } else {
                    // 降级：立即加载所有图片
                    this.loadAllImages();
                }
            } catch (e) {
                debug.error('[LazyLoad] 初始化失败', e);
                this.loadAllImages();
            }
        }
        
        observeImages() {
            try {
                const images = document.querySelectorAll('img[loading="lazy"], img[data-src]');
                const limitedImages = Array.from(images).slice(0, this.maxImages);
                
                limitedImages.forEach(img => {
                    this.observer.observe(img);
                });
                
                debug.log(`[LazyLoad] 观察 ${limitedImages.length} 个图片`);
            } catch (e) {
                debug.error('[LazyLoad] 观察图片失败', e);
            }
        }
        
        handleIntersection(entries) {
            // 参数验证
            if (!validator.validateMultiple([
                { value: entries, type: 'array', name: 'entries', arrayNotEmpty: true }
            ], 'handleIntersection')) {
                debug.error('[LazyLoad] handleIntersection 参数验证失败');
                return;
            }
            
            try {
                entries.forEach((entry, index) => {
                    // 验证entry对象
                    if (!entry || typeof entry !== 'object') {
                        debug.warn(`[LazyLoad] entry[${index}] 不是有效对象`, entry);
                        return;
                    }
                    
                    if (!entry.target) {
                        debug.warn(`[LazyLoad] entry[${index}].target 不存在`);
                        return;
                    }
                    
                    if (entry.isIntersecting && this.loadedCount < this.maxImages) {
                        this.loadImage(entry.target);
                        
                        // 安全地取消观察
                        try {
                            this.observer.unobserve(entry.target);
                        } catch (e) {
                            debug.warn('[LazyLoad] unobserve 失败', e);
                        }
                        
                        this.loadedCount++;
                    }
                });
            } catch (e) {
                debug.error('[LazyLoad] handleIntersection 处理失败', e);
            }
        }
        
        loadImage(img) {
            // 参数验证
            if (!validator.isImage(img, 'img', 'loadImage')) {
                debug.error('[LazyLoad] loadImage 参数验证失败，参数必须是 HTMLImageElement', {
                    received: img,
                    type: typeof img,
                    isElement: img instanceof Element
                });
                return;
            }
            
            // 验证img处于文档中
            if (!document.contains(img)) {
                debug.warn('[LazyLoad] 图片不在文档中', img);
                return;
            }
            
            try {
                const src = img.dataset.src || img.src;
                
                // 验证src
                if (!src || typeof src !== 'string') {
                    debug.warn('[LazyLoad] 图片src无效', { img, src });
                    return;
                }
                
                // 避免重复加载
                if (src === img.src) {
                    debug.debug('[LazyLoad] 图片已加载，跳过', src);
                    return;
                }
                
                // 验证src格式（基本检查）
                if (src.trim().length === 0) {
                    debug.warn('[LazyLoad] 图片src为空字符串');
                    return;
                }
                
                // 加载图片
                img.src = src;
                img.removeAttribute('data-src');
                img.classList.add('loaded');
                debug.debug('[LazyLoad] 图片加载成功', src);
                
            } catch (e) {
                debug.error('[LazyLoad] 加载图片失败', {
                    error: e,
                    img: img,
                    src: img?.src,
                    dataSrc: img?.dataset?.src
                });
            }
        }
        
        loadAllImages() {
            try {
                const images = document.querySelectorAll('img[data-src]');
                const limitedImages = Array.from(images).slice(0, this.maxImages);
                
                limitedImages.forEach(img => this.loadImage(img));
                debug.log('[LazyLoad] 降级模式：直接加载图片');
            } catch (e) {
                debug.error('[LazyLoad] 加载所有图片失败', e);
            }
        }
        
        cleanup() {
            if (this.observer) {
                this.observer.disconnect();
            }
        }
    }
    
    // 初始化移动端优化
    function initMobileOptimization() {
        try {
            debug.log('[Mobile-Perf] 初始化移动端优化');
            
            // 添加移动端标记
            document.body.classList.add('mobile-optimized');
            
            // 初始化懒加载
            const lazyLoader = new SimpleLazyLoader();
            
            // 添加到全局对象以便调试
            window.mobileOptimization = {
                lazyLoader: lazyLoader,
                cleanup: () => {
                    lazyLoader.cleanup();
                }
            };
            
            // 页面卸载时清理
            window.addEventListener('beforeunload', () => {
                try {
                    lazyLoader.cleanup();
                } catch (e) {
                    debug.error('[Mobile-Perf] 清理失败', e);
                }
            });
            
            debug.log('[Mobile-Perf] 移动端优化初始化完成');
        } catch (e) {
            debug.error('[Mobile-Perf] 初始化失败', e);
        }
    }
    
    // DOM加载完成后初始化
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initMobileOptimization);
    } else {
        initMobileOptimization();
    }
    
})();
