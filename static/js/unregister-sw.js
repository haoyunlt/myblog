// Service Worker 注销脚本
// 作用：自动注销已安装的 Service Worker 并清除所有缓存
// 适用于：禁用 Service Worker 后，确保用户浏览器中旧的 SW 被清理

(function() {
    'use strict';
    
    // 仅在支持 Service Worker 的浏览器中执行
    if (!('serviceWorker' in navigator)) {
        console.log('[SW-Unregister] 浏览器不支持 Service Worker');
        return;
    }
    
    console.log('[SW-Unregister] 开始注销 Service Worker...');
    
    // 注销所有 Service Worker
    navigator.serviceWorker.getRegistrations().then(function(registrations) {
        if (registrations.length === 0) {
            console.log('[SW-Unregister] 没有找到已注册的 Service Worker');
            return;
        }
        
        console.log(`[SW-Unregister] 找到 ${registrations.length} 个已注册的 Service Worker`);
        
        // 逐个注销
        const unregisterPromises = registrations.map(function(registration, index) {
            console.log(`[SW-Unregister] 正在注销 Service Worker #${index + 1}...`, registration.scope);
            return registration.unregister().then(function(success) {
                if (success) {
                    console.log(`[SW-Unregister] ✓ Service Worker #${index + 1} 注销成功`);
                } else {
                    console.warn(`[SW-Unregister] ✗ Service Worker #${index + 1} 注销失败`);
                }
                return success;
            }).catch(function(error) {
                console.error(`[SW-Unregister] ✗ Service Worker #${index + 1} 注销出错:`, error);
                return false;
            });
        });
        
        // 等待所有注销完成
        return Promise.all(unregisterPromises);
    }).then(function(results) {
        const successCount = results ? results.filter(Boolean).length : 0;
        console.log(`[SW-Unregister] Service Worker 注销完成：${successCount} 个成功`);
        
        // 清除所有缓存
        if ('caches' in window) {
            return caches.keys().then(function(cacheNames) {
                if (cacheNames.length === 0) {
                    console.log('[SW-Unregister] 没有找到缓存');
                    return;
                }
                
                console.log(`[SW-Unregister] 找到 ${cacheNames.length} 个缓存，正在清除...`);
                
                const deletePromises = cacheNames.map(function(cacheName, index) {
                    console.log(`[SW-Unregister] 正在删除缓存 #${index + 1}: ${cacheName}`);
                    return caches.delete(cacheName).then(function(success) {
                        if (success) {
                            console.log(`[SW-Unregister] ✓ 缓存 #${index + 1} 删除成功`);
                        } else {
                            console.warn(`[SW-Unregister] ✗ 缓存 #${index + 1} 删除失败`);
                        }
                        return success;
                    }).catch(function(error) {
                        console.error(`[SW-Unregister] ✗ 缓存 #${index + 1} 删除出错:`, error);
                        return false;
                    });
                });
                
                return Promise.all(deletePromises);
            }).then(function(results) {
                const cacheSuccessCount = results ? results.filter(Boolean).length : 0;
                console.log(`[SW-Unregister] 缓存清除完成：${cacheSuccessCount} 个成功`);
                console.log('[SW-Unregister] ✓ Service Worker 和缓存已全部清理完成');
            });
        } else {
            console.log('[SW-Unregister] 浏览器不支持 Cache API');
        }
    }).then(function() {
        // 额外清理：清除可能导致冲突的存储数据
        console.log('[SW-Unregister] 清理额外存储数据...');
        
        try {
            // 清理特定的可能引起问题的 localStorage 键
            const keysToRemove = ['bugly_reports', 'mobile_error_handler_errors'];
            keysToRemove.forEach(function(key) {
                if (localStorage.getItem(key)) {
                    localStorage.removeItem(key);
                    console.log('[SW-Unregister] ✓ 清除 localStorage: ' + key);
                }
            });
        } catch (e) {
            console.warn('[SW-Unregister] localStorage 清理警告:', e);
        }
        
        console.log('[SW-Unregister] ✓ 所有清理工作完成');
    }).catch(function(error) {
        console.error('[SW-Unregister] 注销过程出错:', error);
        
        // 即使出错，也尝试强制刷新页面（仅在非无痕模式）
        if (window.location.search.indexOf('cache_cleared') === -1) {
            console.log('[SW-Unregister] 尝试强制刷新页面...');
            setTimeout(function() {
                window.location.href = window.location.pathname + '?cache_cleared=' + Date.now();
            }, 2000);
        }
    });
    
})();

