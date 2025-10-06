// 移动端内存监控工具 v1.0
// 用于检测和预防OOM（Out Of Memory）问题

(function() {
    'use strict';
    
    // 检测是否为移动设备
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ||
                     window.innerWidth <= 768;
    
    // 非移动设备不需要严格的内存监控
    if (!isMobile) {
        console.log('[MemoryMonitor] 桌面设备，跳过内存监控');
        return;
    }
    
    // 内存监控配置
    const CONFIG = {
        // 内存警告阈值（MB）
        warningThreshold: 50,
        // 内存危险阈值（MB）
        dangerThreshold: 80,
        // 监控间隔（毫秒）
        monitorInterval: 5000,
        // 是否启用自动清理
        autoCleanup: true
    };
    
    // 内存监控器
    class MemoryMonitor {
        constructor() {
            this.intervalId = null;
            this.stats = {
                peak: 0,
                current: 0,
                samples: [],
                warnings: 0,
                cleanups: 0
            };
            
            this.leakDetectors = [];
            this.init();
        }
        
        init() {
            console.log('[MemoryMonitor] 移动端内存监控启动');
            
            // 检查Memory API支持
            if (!performance.memory) {
                console.warn('[MemoryMonitor] Memory API不可用，使用降级模式');
                this.initFallbackMode();
                return;
            }
            
            // 启动定期监控
            this.startMonitoring();
            
            // 监听页面可见性变化
            document.addEventListener('visibilitychange', () => {
                if (document.hidden) {
                    this.onPageHidden();
                } else {
                    this.onPageVisible();
                }
            });
            
            // 页面卸载前清理
            window.addEventListener('beforeunload', () => {
                this.cleanup();
            });
            
            // 注册内存泄漏检测器
            this.registerLeakDetectors();
        }
        
        startMonitoring() {
            this.intervalId = setInterval(() => {
                this.checkMemory();
            }, CONFIG.monitorInterval);
        }
        
        stopMonitoring() {
            if (this.intervalId) {
                clearInterval(this.intervalId);
                this.intervalId = null;
            }
        }
        
        checkMemory() {
            try {
                const memory = performance.memory;
                const usedMB = Math.round(memory.usedJSHeapSize / 1048576);
                const totalMB = Math.round(memory.totalJSHeapSize / 1048576);
                const limitMB = Math.round(memory.jsHeapSizeLimit / 1048576);
                
                this.stats.current = usedMB;
                this.stats.peak = Math.max(this.stats.peak, usedMB);
                this.stats.samples.push({
                    time: Date.now(),
                    used: usedMB,
                    total: totalMB
                });
                
                // 只保留最近20个样本
                if (this.stats.samples.length > 20) {
                    this.stats.samples.shift();
                }
                
                // 检查内存使用情况
                if (usedMB > CONFIG.dangerThreshold) {
                    this.onMemoryDanger(usedMB, limitMB);
                } else if (usedMB > CONFIG.warningThreshold) {
                    this.onMemoryWarning(usedMB, limitMB);
                }
                
                // 检测内存泄漏
                this.detectLeaks();
                
            } catch (e) {
                console.error('[MemoryMonitor] 检查内存失败', e);
            }
        }
        
        onMemoryWarning(used, limit) {
            this.stats.warnings++;
            console.warn(`[MemoryMonitor] ⚠️ 内存使用较高: ${used}MB / ${limit}MB`);
            
            // 记录当前状态
            this.recordMemoryState('WARNING');
        }
        
        onMemoryDanger(used, limit) {
            console.error(`[MemoryMonitor] 🚨 内存使用危险: ${used}MB / ${limit}MB`);
            
            // 记录当前状态
            this.recordMemoryState('DANGER');
            
            // 自动清理
            if (CONFIG.autoCleanup) {
                console.log('[MemoryMonitor] 触发自动清理');
                this.performCleanup();
            }
        }
        
        performCleanup() {
            this.stats.cleanups++;
            
            try {
                // 1. 清理MobileDebug日志
                if (window.MobileDebug && typeof window.MobileDebug.clear === 'function') {
                    const beforeLogs = window.MobileDebug.logs.length;
                    window.MobileDebug.clear();
                    console.log(`[MemoryMonitor] 清理调试日志: ${beforeLogs} 条`);
                }
                
                // 2. 清理Service Worker缓存（仅限旧版本）
                if ('caches' in window) {
                    caches.keys().then(cacheNames => {
                        const oldCaches = cacheNames.filter(name => 
                            name.includes('v1.') || name.includes('v2.0')
                        );
                        
                        return Promise.all(
                            oldCaches.map(cache => caches.delete(cache))
                        );
                    }).then(() => {
                        console.log('[MemoryMonitor] 清理旧Service Worker缓存');
                    });
                }
                
                // 3. 清理localStorage中的旧数据
                this.cleanupLocalStorage();
                
                // 4. 建议垃圾回收（仅建议，实际由浏览器决定）
                if (window.gc && typeof window.gc === 'function') {
                    window.gc();
                    console.log('[MemoryMonitor] 触发垃圾回收');
                }
                
                console.log('[MemoryMonitor] ✅ 内存清理完成');
                
            } catch (e) {
                console.error('[MemoryMonitor] 清理失败', e);
            }
        }
        
        cleanupLocalStorage() {
            try {
                const keys = Object.keys(localStorage);
                let cleaned = 0;
                
                keys.forEach(key => {
                    // 清理超过7天的调试日志
                    if (key.startsWith('mobile_debug_')) {
                        try {
                            const data = JSON.parse(localStorage.getItem(key) || '[]');
                            const now = Date.now();
                            const weekAgo = now - 7 * 24 * 60 * 60 * 1000;
                            
                            const filtered = data.filter(entry => 
                                entry.timestamp && entry.timestamp > weekAgo
                            );
                            
                            if (filtered.length < data.length) {
                                localStorage.setItem(key, JSON.stringify(filtered));
                                cleaned += (data.length - filtered.length);
                            }
                        } catch (e) {
                            // 如果解析失败，删除该键
                            localStorage.removeItem(key);
                        }
                    }
                });
                
                if (cleaned > 0) {
                    console.log(`[MemoryMonitor] 清理localStorage: ${cleaned} 条记录`);
                }
            } catch (e) {
                console.error('[MemoryMonitor] 清理localStorage失败', e);
            }
        }
        
        registerLeakDetectors() {
            // 检测器1: DOM节点泄漏
            this.leakDetectors.push({
                name: 'DOM节点数量',
                check: () => {
                    const count = document.getElementsByTagName('*').length;
                    return {
                        value: count,
                        isLeak: count > 3000,
                        message: `DOM节点过多: ${count} (建议<3000)`
                    };
                }
            });
            
            // 检测器2: 事件监听器泄漏
            this.leakDetectors.push({
                name: '事件监听器',
                check: () => {
                    // 通过getEventListeners（仅Chrome DevTools可用）
                    if (typeof getEventListeners !== 'undefined') {
                        const listeners = getEventListeners(window);
                        const count = Object.values(listeners).reduce((sum, arr) => sum + arr.length, 0);
                        return {
                            value: count,
                            isLeak: count > 50,
                            message: `全局事件监听器过多: ${count}`
                        };
                    }
                    return { value: 0, isLeak: false };
                }
            });
            
            // 检测器3: MobileDebug日志累积
            this.leakDetectors.push({
                name: 'MobileDebug日志',
                check: () => {
                    if (window.MobileDebug) {
                        const total = window.MobileDebug.logs.length + 
                                    window.MobileDebug.errors.length + 
                                    window.MobileDebug.warnings.length;
                        return {
                            value: total,
                            isLeak: total > 500,
                            message: `调试日志累积过多: ${total} 条`
                        };
                    }
                    return { value: 0, isLeak: false };
                }
            });
        }
        
        detectLeaks() {
            this.leakDetectors.forEach(detector => {
                try {
                    const result = detector.check();
                    if (result.isLeak) {
                        console.warn(`[MemoryMonitor] 🔍 潜在泄漏: ${detector.name} - ${result.message}`);
                    }
                } catch (e) {
                    // 静默失败
                }
            });
        }
        
        recordMemoryState(level) {
            try {
                const state = {
                    level: level,
                    time: new Date().toISOString(),
                    memory: performance.memory ? {
                        used: Math.round(performance.memory.usedJSHeapSize / 1048576),
                        total: Math.round(performance.memory.totalJSHeapSize / 1048576),
                        limit: Math.round(performance.memory.jsHeapSizeLimit / 1048576)
                    } : null,
                    dom: {
                        nodes: document.getElementsByTagName('*').length,
                        images: document.getElementsByTagName('img').length
                    },
                    debug: window.MobileDebug ? {
                        logs: window.MobileDebug.logs.length,
                        errors: window.MobileDebug.errors.length
                    } : null
                };
                
                // 存储到localStorage（最多保留10条）
                const key = 'mobile_memory_states';
                const states = JSON.parse(localStorage.getItem(key) || '[]');
                states.push(state);
                
                if (states.length > 10) {
                    states.shift();
                }
                
                localStorage.setItem(key, JSON.stringify(states));
                
            } catch (e) {
                // 静默失败
            }
        }
        
        onPageHidden() {
            console.log('[MemoryMonitor] 页面隐藏，暂停监控');
            this.stopMonitoring();
        }
        
        onPageVisible() {
            console.log('[MemoryMonitor] 页面可见，恢复监控');
            this.startMonitoring();
        }
        
        initFallbackMode() {
            console.log('[MemoryMonitor] 使用降级模式：基于DOM节点数量');
            
            this.intervalId = setInterval(() => {
                const nodeCount = document.getElementsByTagName('*').length;
                
                if (nodeCount > 5000) {
                    console.error(`[MemoryMonitor] 🚨 DOM节点过多: ${nodeCount}`);
                    if (CONFIG.autoCleanup) {
                        this.performCleanup();
                    }
                } else if (nodeCount > 3000) {
                    console.warn(`[MemoryMonitor] ⚠️ DOM节点较多: ${nodeCount}`);
                }
            }, CONFIG.monitorInterval);
        }
        
        cleanup() {
            console.log('[MemoryMonitor] 清理内存监控器');
            this.stopMonitoring();
        }
        
        getReport() {
            return {
                stats: this.stats,
                current: performance.memory ? {
                    used: Math.round(performance.memory.usedJSHeapSize / 1048576),
                    total: Math.round(performance.memory.totalJSHeapSize / 1048576),
                    limit: Math.round(performance.memory.jsHeapSizeLimit / 1048576)
                } : null,
                dom: {
                    nodes: document.getElementsByTagName('*').length,
                    images: document.getElementsByTagName('img').length
                },
                recommendations: this.getRecommendations()
            };
        }
        
        getRecommendations() {
            const recommendations = [];
            
            if (this.stats.warnings > 5) {
                recommendations.push('频繁内存警告，建议减少页面内容或启用分页');
            }
            
            if (this.stats.cleanups > 3) {
                recommendations.push('多次触发自动清理，建议优化代码以减少内存使用');
            }
            
            const nodeCount = document.getElementsByTagName('*').length;
            if (nodeCount > 3000) {
                recommendations.push(`DOM节点过多(${nodeCount})，建议使用虚拟滚动或懒加载`);
            }
            
            return recommendations;
        }
    }
    
    // 创建全局实例
    try {
        window.MemoryMonitor = new MemoryMonitor();
        
        // 提供全局方法
        window.getMemoryReport = function() {
            const report = window.MemoryMonitor.getReport();
            console.group('📊 移动端内存报告');
            console.log(JSON.stringify(report, null, 2));
            console.groupEnd();
            return report;
        };
        
        window.forceMemoryCleanup = function() {
            console.log('[MemoryMonitor] 手动触发内存清理');
            window.MemoryMonitor.performCleanup();
        };
        
        console.log('[MemoryMonitor] ✅ 内存监控已启动');
        console.log('[MemoryMonitor] 💡 使用 getMemoryReport() 查看内存报告');
        console.log('[MemoryMonitor] 💡 使用 forceMemoryCleanup() 手动清理内存');
        
    } catch (e) {
        console.error('[MemoryMonitor] 初始化失败', e);
    }
    
})();

