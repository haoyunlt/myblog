// 内存泄漏紧急修复补丁 v1.0
// 用于立即缓解移动端Chrome崩溃问题
// 部署：在页面头部最早加载此脚本

(function() {
    'use strict';
    
    console.log('[MemoryFix] 🔧 内存泄漏修复补丁加载');
    
    // ============================================
    // 修复1: 全局清理注册表
    // ============================================
    
    window.CleanupRegistry = {
        handlers: [],
        isCleaningUp: false,
        
        register: function(name, cleanupFn) {
            if (typeof cleanupFn === 'function') {
                this.handlers.push({ name, cleanupFn });
                console.log(`[CleanupRegistry] ✓ 注册清理函数: ${name}`);
            }
        },
        
        cleanup: function() {
            if (this.isCleaningUp) return;
            this.isCleaningUp = true;
            
            console.log(`[CleanupRegistry] 🧹 开始清理 (${this.handlers.length} 个任务)`);
            
            let successCount = 0;
            this.handlers.forEach(({ name, cleanupFn }) => {
                try {
                    cleanupFn();
                    successCount++;
                    console.log(`[CleanupRegistry] ✓ ${name}`);
                } catch (e) {
                    console.error(`[CleanupRegistry] ✗ ${name}`, e);
                }
            });
            
            console.log(`[CleanupRegistry] ✅ 清理完成 (${successCount}/${this.handlers.length})`);
            this.handlers = [];
            this.isCleaningUp = false;
        }
    };
    
    // ============================================
    // 修复2: 定时器管理器
    // ============================================
    
    window.TimerManager = {
        intervals: new Set(),
        timeouts: new Set(),
        
        setInterval: function(callback, delay, ...args) {
            const id = setInterval(callback, delay, ...args);
            this.intervals.add(id);
            return id;
        },
        
        setTimeout: function(callback, delay, ...args) {
            const id = setTimeout(callback, delay, ...args);
            this.timeouts.add(id);
            return id;
        },
        
        clearInterval: function(id) {
            clearInterval(id);
            this.intervals.delete(id);
        },
        
        clearTimeout: function(id) {
            clearTimeout(id);
            this.timeouts.delete(id);
        },
        
        clearAll: function() {
            console.log(`[TimerManager] 🧹 清理 ${this.intervals.size} 个interval, ${this.timeouts.size} 个timeout`);
            
            this.intervals.forEach(id => clearInterval(id));
            this.timeouts.forEach(id => clearTimeout(id));
            
            this.intervals.clear();
            this.timeouts.clear();
        }
    };
    
    CleanupRegistry.register('TimerManager', () => TimerManager.clearAll());
    
    // ============================================
    // 修复3: 事件监听器管理器
    // ============================================
    
    window.ListenerManager = {
        listeners: [],
        abortControllers: new Map(),
        
        addEventListener: function(target, event, handler, options) {
            // 如果支持AbortController，使用signal
            if ('AbortController' in window && typeof options === 'object') {
                let controller = this.abortControllers.get(target);
                if (!controller) {
                    controller = new AbortController();
                    this.abortControllers.set(target, controller);
                }
                
                options = { ...options, signal: controller.signal };
            }
            
            target.addEventListener(event, handler, options);
            this.listeners.push({ target, event, handler, options });
        },
        
        removeAll: function() {
            console.log(`[ListenerManager] 🧹 清理 ${this.listeners.length} 个事件监听器`);
            
            // 方法1: 使用AbortController批量清理
            this.abortControllers.forEach((controller, target) => {
                try {
                    controller.abort();
                } catch (e) {
                    console.warn('[ListenerManager] AbortController失败', e);
                }
            });
            this.abortControllers.clear();
            
            // 方法2: 逐个移除（备用）
            this.listeners.forEach(({ target, event, handler }) => {
                try {
                    target.removeEventListener(event, handler);
                } catch (e) {
                    // 静默失败（target可能已被移除）
                }
            });
            
            this.listeners = [];
        }
    };
    
    CleanupRegistry.register('ListenerManager', () => ListenerManager.removeAll());
    
    // ============================================
    // 修复4: MobileDebug日志限制增强
    // ============================================
    
    function enhanceMobileDebug() {
        if (!window.MobileDebug) return;
        
        const originalLog = window.MobileDebug._log;
        const MAX_TOTAL_LOGS = 50; // 从200降到50
        
        window.MobileDebug._log = function(level, message, data, error) {
            // 调用原始方法
            const result = originalLog.call(this, level, message, data, error);
            
            // 强制限制总日志数
            const totalLogs = this.logs.length + this.errors.length + 
                             this.warnings.length + this.performance.length;
            
            if (totalLogs > MAX_TOTAL_LOGS) {
                // 删除最旧的日志
                const toRemove = totalLogs - MAX_TOTAL_LOGS;
                for (let i = 0; i < toRemove; i++) {
                    if (this.logs.length > 10) this.logs.shift();
                    else if (this.warnings.length > 10) this.warnings.shift();
                    else if (this.performance.length > 0) this.performance.shift();
                }
            }
            
            return result;
        };
        
        // 添加clear方法
        window.MobileDebug.clearAll = function() {
            this.logs = [];
            this.errors = [];
            this.warnings = [];
            this.performance = [];
            console.log('[MobileDebug] 已清空所有日志');
        };
        
        console.log('[MemoryFix] ✓ MobileDebug增强 (日志限制: 50)');
    }
    
    // DOM加载后增强
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', enhanceMobileDebug);
    } else {
        enhanceMobileDebug();
    }
    
    CleanupRegistry.register('MobileDebug', () => {
        if (window.MobileDebug && typeof window.MobileDebug.clearAll === 'function') {
            window.MobileDebug.clearAll();
        }
    });
    
    // ============================================
    // 修复5: localStorage清理
    // ============================================
    
    function cleanupLocalStorage() {
        try {
            const keysToLimit = [
                { key: 'mobile_debug_logs', limit: 20 },
                { key: 'mobile_debug_errors', limit: 30 },
                { key: 'mobile_debug_warnings', limit: 20 },
                { key: 'mobile_memory_states', limit: 5 }
            ];
            
            let totalCleaned = 0;
            
            keysToLimit.forEach(({ key, limit }) => {
                try {
                    const data = JSON.parse(localStorage.getItem(key) || '[]');
                    if (data.length > limit) {
                        const cleaned = data.length - limit;
                        const trimmed = data.slice(-limit);
                        localStorage.setItem(key, JSON.stringify(trimmed));
                        totalCleaned += cleaned;
                    }
                } catch (e) {
                    // 解析失败，删除该键
                    localStorage.removeItem(key);
                }
            });
            
            // 清理超过7天的数据
            const weekAgo = Date.now() - 7 * 24 * 60 * 60 * 1000;
            Object.keys(localStorage).forEach(key => {
                if (key.startsWith('mobile_debug_')) {
                    try {
                        const data = JSON.parse(localStorage.getItem(key) || '[]');
                        const filtered = data.filter(entry => {
                            if (!entry.timestamp) return true;
                            return entry.timestamp > weekAgo;
                        });
                        
                        if (filtered.length < data.length) {
                            localStorage.setItem(key, JSON.stringify(filtered));
                            totalCleaned += (data.length - filtered.length);
                        }
                    } catch (e) {
                        // 忽略
                    }
                }
            });
            
            if (totalCleaned > 0) {
                console.log(`[MemoryFix] ✓ localStorage清理: ${totalCleaned} 条记录`);
            }
            
        } catch (e) {
            console.error('[MemoryFix] localStorage清理失败', e);
        }
    }
    
    cleanupLocalStorage();
    CleanupRegistry.register('localStorage', cleanupLocalStorage);
    
    // ============================================
    // 修复6: 移动端Mermaid图表限制
    // ============================================
    
    function limitMermaidCharts() {
        const isMobile = /Android|webOS|iPhone|iPad|iPod/i.test(navigator.userAgent) ||
                        window.innerWidth <= 768;
        
        if (!isMobile) return;
        
        const maxCharts = 3;
        const mermaidBlocks = document.querySelectorAll('.language-mermaid');
        
        if (mermaidBlocks.length > maxCharts) {
            console.log(`[MemoryFix] ⚠️ 移动端限制Mermaid图表: ${maxCharts}/${mermaidBlocks.length}`);
            
            // 隐藏多余的图表
            Array.from(mermaidBlocks).slice(maxCharts).forEach((block, index) => {
                const placeholder = document.createElement('div');
                placeholder.className = 'mermaid-placeholder-mobile';
                placeholder.style.cssText = `
                    padding: 20px;
                    background: #f0f0f0;
                    border: 2px dashed #ccc;
                    border-radius: 8px;
                    text-align: center;
                    margin: 10px 0;
                `;
                placeholder.innerHTML = `
                    <p style="margin: 0 0 10px 0; color: #666;">
                        ⚡ 移动端性能优化：图表 ${maxCharts + index + 1} 已隐藏
                    </p>
                    <small style="color: #999;">请在桌面浏览器中查看完整图表</small>
                `;
                
                block.parentNode.replaceChild(placeholder, block);
            });
        }
    }
    
    // 延迟执行，等待Mermaid渲染前
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            setTimeout(limitMermaidCharts, 100); // 在Mermaid初始化前
        });
    } else {
        setTimeout(limitMermaidCharts, 100);
    }
    
    // ============================================
    // 修复7: Observer管理器
    // ============================================
    
    window.ObserverManager = {
        observers: [],
        
        register: function(observer, name) {
            this.observers.push({ observer, name });
            console.log(`[ObserverManager] ✓ 注册Observer: ${name}`);
        },
        
        disconnectAll: function() {
            console.log(`[ObserverManager] 🧹 断开 ${this.observers.length} 个Observer`);
            
            this.observers.forEach(({ observer, name }) => {
                try {
                    observer.disconnect();
                    console.log(`[ObserverManager] ✓ ${name}`);
                } catch (e) {
                    console.error(`[ObserverManager] ✗ ${name}`, e);
                }
            });
            
            this.observers = [];
        }
    };
    
    CleanupRegistry.register('ObserverManager', () => ObserverManager.disconnectAll());
    
    // ============================================
    // 修复8: 页面刷新次数监控
    // ============================================
    
    function monitorRefreshCount() {
        const key = 'page_refresh_count';
        const count = parseInt(sessionStorage.getItem(key) || '0') + 1;
        sessionStorage.setItem(key, count);
        
        console.log(`[MemoryFix] 📊 页面刷新次数: ${count}`);
        
        if (count >= 5) {
            console.warn(`[MemoryFix] ⚠️ 刷新次数过多 (${count}次)，建议清理缓存`);
            
            if (count >= 8) {
                const shouldClean = confirm(
                    `检测到频繁刷新 (${count}次)，这可能导致内存问题。\n\n` +
                    `建议立即清理缓存？\n` +
                    `点击"确定"访问清理工具`
                );
                
                if (shouldClean) {
                    window.location.href = '/clear-cache.html?auto=1';
                }
            }
        }
        
        return count;
    }
    
    const refreshCount = monitorRefreshCount();
    
    // ============================================
    // 修复9: 内存监控
    // ============================================
    
    function checkMemoryStatus() {
        if (!performance.memory) {
            console.log('[MemoryFix] Memory API不可用');
            return;
        }
        
        const used = Math.round(performance.memory.usedJSHeapSize / 1048576);
        const limit = Math.round(performance.memory.jsHeapSizeLimit / 1048576);
        const percentage = Math.round((used / limit) * 100);
        
        console.log(`[MemoryFix] 💾 内存使用: ${used}MB / ${limit}MB (${percentage}%)`);
        
        if (percentage > 80) {
            console.error(`[MemoryFix] 🚨 内存使用过高 (${percentage}%)！`);
            
            // 触发清理
            console.log('[MemoryFix] 🧹 触发紧急清理...');
            CleanupRegistry.cleanup();
            
            // 建议用户清理缓存
            if (percentage > 90) {
                alert(
                    `⚠️ 内存使用已达 ${percentage}%，即将崩溃！\n\n` +
                    `建议立即清理浏览器缓存或访问 /clear-cache.html`
                );
            }
        } else if (percentage > 60) {
            console.warn(`[MemoryFix] ⚠️ 内存使用较高 (${percentage}%)`);
        } else {
            console.log(`[MemoryFix] ✅ 内存状态正常 (${percentage}%)`);
        }
    }
    
    // 页面加载后检查
    window.addEventListener('load', () => {
        setTimeout(checkMemoryStatus, 2000);
    });
    
    // ============================================
    // 修复10: 注册页面卸载清理
    // ============================================
    
    window.addEventListener('beforeunload', () => {
        console.log('[MemoryFix] 🧹 页面卸载，执行清理');
        CleanupRegistry.cleanup();
    });
    
    // 页面隐藏时也清理（移动端切换标签）
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            console.log('[MemoryFix] 🌙 页面隐藏，执行清理');
            CleanupRegistry.cleanup();
        }
    });
    
    // ============================================
    // 修复11: 提供手动清理接口
    // ============================================
    
    window.forceCleanup = function() {
        console.log('[MemoryFix] 🧹 手动触发清理');
        CleanupRegistry.cleanup();
        
        if (window.gc) {
            console.log('[MemoryFix] ♻️ 触发垃圾回收');
            window.gc();
        }
        
        console.log('[MemoryFix] ✅ 清理完成');
    };
    
    window.getMemoryStatus = function() {
        checkMemoryStatus();
        
        return {
            refresh_count: refreshCount,
            memory: performance.memory ? {
                used: Math.round(performance.memory.usedJSHeapSize / 1048576) + 'MB',
                limit: Math.round(performance.memory.jsHeapSizeLimit / 1048576) + 'MB'
            } : 'N/A',
            dom_nodes: document.getElementsByTagName('*').length,
            cleanup_handlers: CleanupRegistry.handlers.length,
            timers: {
                intervals: TimerManager.intervals.size,
                timeouts: TimerManager.timeouts.size
            },
            listeners: ListenerManager.listeners.length,
            observers: ObserverManager.observers.length
        };
    };
    
    // ============================================
    // 初始化完成
    // ============================================
    
    console.log('[MemoryFix] ✅ 内存泄漏修复补丁已加载');
    console.log('[MemoryFix] 💡 使用 forceCleanup() 手动清理');
    console.log('[MemoryFix] 💡 使用 getMemoryStatus() 查看状态');
    
})();

