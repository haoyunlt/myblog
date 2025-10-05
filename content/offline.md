---
title: "网络连接中断"
layout: "single"
url: "/offline/"
---

<div class="offline-page" style="text-align: center; padding: 2rem; min-height: 60vh; display: flex; flex-direction: column; justify-content: center; align-items: center;">
    <div class="offline-icon" style="font-size: 4rem; margin-bottom: 1rem; opacity: 0.6;">
        📡
    </div>
    
    <h1 style="font-size: 2rem; margin-bottom: 1rem; color: #374151;">网络连接中断</h1>
    
    <p style="font-size: 1.1rem; color: #6b7280; margin-bottom: 2rem; max-width: 400px; line-height: 1.6;">
        抱歉，您当前无法连接到互联网。不过，您依然可以浏览已缓存的页面。
    </p>
    
    <div class="offline-actions" style="display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center; margin-bottom: 2rem;">
        <button onclick="window.location.reload()" 
                style="background: #2563eb; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 6px; cursor: pointer; font-size: 1rem; transition: background 0.2s ease;">
            🔄 重试连接
        </button>
        
        <button onclick="history.back()" 
                style="background: #6b7280; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 6px; cursor: pointer; font-size: 1rem; transition: background 0.2s ease;">
            ← 返回上页
        </button>
        
        <a href="/" 
           style="background: #059669; color: white; text-decoration: none; padding: 0.75rem 1.5rem; border-radius: 6px; font-size: 1rem; transition: background 0.2s ease; display: inline-block;">
            🏠 返回首页
        </a>
    </div>
    
    <div class="cached-pages" style="max-width: 600px; width: 100%; margin-top: 2rem;">
        <h3 style="font-size: 1.2rem; margin-bottom: 1rem; color: #374151;">📚 已缓存的页面</h3>
        
        <div id="cached-pages-list" style="text-align: left;">
            <p style="color: #6b7280; font-style: italic;">正在加载缓存列表...</p>
        </div>
    </div>
    
    <div class="network-status" style="margin-top: 2rem; padding: 1rem; background: #f3f4f6; border-radius: 8px; max-width: 500px; width: 100%;">
        <h4 style="margin-bottom: 0.5rem; color: #374151;">🔗 连接状态</h4>
        <p id="connection-status" style="margin: 0; color: #6b7280;">
            检测中...
        </p>
    </div>
</div>

<script>
(function() {
    'use strict';
    
    // 检查网络状态
    function updateConnectionStatus() {
        const statusElement = document.getElementById('connection-status');
        const isOnline = navigator.onLine;
        
        if (isOnline) {
            statusElement.innerHTML = '🟢 网络已连接 - <button onclick="window.location.reload()" style="background: #059669; color: white; border: none; padding: 0.25rem 0.5rem; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">刷新页面</button>';
            statusElement.style.color = '#059669';
        } else {
            statusElement.innerHTML = '🔴 网络断开连接';
            statusElement.style.color = '#dc2626';
        }
    }
    
    // 监听网络状态变化
    window.addEventListener('online', updateConnectionStatus);
    window.addEventListener('offline', updateConnectionStatus);
    
    // 初始检查
    updateConnectionStatus();
    
    // 获取缓存页面列表
    function loadCachedPages() {
        if ('caches' in window) {
            caches.keys().then(function(cacheNames) {
                const cachedPagesList = document.getElementById('cached-pages-list');
                
                if (cacheNames.length === 0) {
                    cachedPagesList.innerHTML = '<p style="color: #6b7280; font-style: italic;">暂无缓存页面</p>';
                    return;
                }
                
                let allCachedUrls = new Set();
                
                Promise.all(
                    cacheNames.map(function(cacheName) {
                        return caches.open(cacheName).then(function(cache) {
                            return cache.keys();
                        });
                    })
                ).then(function(allRequests) {
                    allRequests.forEach(function(requests) {
                        requests.forEach(function(request) {
                            if (request.url.includes(window.location.origin)) {
                                const url = new URL(request.url);
                                if (url.pathname !== '/offline/' && 
                                    url.pathname !== '/sw.js' && 
                                    !url.pathname.includes('/assets/') &&
                                    !url.pathname.includes('/icons/')) {
                                    allCachedUrls.add(url.pathname);
                                }
                            }
                        });
                    });
                    
                    if (allCachedUrls.size === 0) {
                        cachedPagesList.innerHTML = '<p style="color: #6b7280; font-style: italic;">暂无可访问的缓存页面</p>';
                        return;
                    }
                    
                    const urlArray = Array.from(allCachedUrls).sort();
                    const listHtml = urlArray.map(function(url) {
                        const title = url === '/' ? '首页' : 
                                     url.includes('/posts/') ? '文章: ' + decodeURIComponent(url.split('/').pop().replace(/-/g, ' ')) :
                                     url.includes('/categories/') ? '分类: ' + decodeURIComponent(url.split('/').pop()) :
                                     url.replace('/', '').replace(/-/g, ' ');
                        
                        return `<div style="margin-bottom: 0.5rem;">
                                  <a href="${url}" style="color: #2563eb; text-decoration: none; display: block; padding: 0.5rem; border-radius: 4px; border: 1px solid #e5e7eb; transition: background 0.2s ease;">
                                    📄 ${title}
                                  </a>
                                </div>`;
                    }).join('');
                    
                    cachedPagesList.innerHTML = listHtml;
                });
            }).catch(function(error) {
                console.error('获取缓存页面失败:', error);
                document.getElementById('cached-pages-list').innerHTML = '<p style="color: #dc2626;">无法加载缓存列表</p>';
            });
        } else {
            document.getElementById('cached-pages-list').innerHTML = '<p style="color: #6b7280; font-style: italic;">浏览器不支持缓存功能</p>';
        }
    }
    
    // 页面加载时获取缓存列表
    loadCachedPages();
    
    // 自动重试连接
    let retryCount = 0;
    const maxRetries = 3;
    
    function autoRetry() {
        if (navigator.onLine && retryCount < maxRetries) {
            retryCount++;
            setTimeout(function() {
                fetch('/', { method: 'HEAD', cache: 'no-cache' })
                    .then(function() {
                        // 连接成功，返回首页
                        window.location.href = '/';
                    })
                    .catch(function() {
                        // 连接失败，继续重试
                        if (retryCount < maxRetries) {
                            autoRetry();
                        }
                    });
            }, 2000 * retryCount); // 递增延迟
        }
    }
    
    // 网络恢复时自动重试
    window.addEventListener('online', function() {
        retryCount = 0;
        autoRetry();
    });
    
    // 键盘快捷键
    document.addEventListener('keydown', function(e) {
        if (e.key === 'F5' || (e.ctrlKey && e.key === 'r')) {
            e.preventDefault();
            window.location.reload();
        } else if (e.key === 'Escape') {
            history.back();
        } else if (e.altKey && e.key === 'h') {
            window.location.href = '/';
        }
    });
    
    // 添加按钮悬停效果
    const buttons = document.querySelectorAll('button, a[style*="background"]');
    buttons.forEach(function(button) {
        button.addEventListener('mouseenter', function() {
            const currentBg = this.style.backgroundColor;
            if (currentBg.includes('37, 99, 235')) { // 蓝色
                this.style.backgroundColor = '#1d4ed8';
            } else if (currentBg.includes('107, 114, 128')) { // 灰色
                this.style.backgroundColor = '#4b5563';
            } else if (currentBg.includes('5, 150, 105')) { // 绿色
                this.style.backgroundColor = '#047857';
            }
        });
        
        button.addEventListener('mouseleave', function() {
            const currentBg = this.style.backgroundColor;
            if (currentBg.includes('29, 78, 216')) { // 深蓝色
                this.style.backgroundColor = '#2563eb';
            } else if (currentBg.includes('75, 85, 99')) { // 深灰色
                this.style.backgroundColor = '#6b7280';
            } else if (currentBg.includes('4, 120, 87')) { // 深绿色
                this.style.backgroundColor = '#059669';
            }
        });
    });
    
})();
</script>

<style>
/* 移动端优化 */
@media (max-width: 768px) {
    .offline-page {
        padding: 1rem !important;
    }
    
    .offline-icon {
        font-size: 3rem !important;
    }
    
    h1 {
        font-size: 1.5rem !important;
    }
    
    .offline-actions {
        flex-direction: column !important;
        align-items: center;
    }
    
    .offline-actions button,
    .offline-actions a {
        width: 200px;
        text-align: center;
    }
    
    .cached-pages {
        margin-top: 1.5rem !important;
    }
    
    #cached-pages-list a {
        font-size: 0.9rem;
        padding: 0.75rem !important;
    }
}

/* 暗色主题适配 */
@media (prefers-color-scheme: dark) {
    .offline-page {
        background: #111827;
        color: #f9fafb;
    }
    
    h1 {
        color: #f9fafb !important;
    }
    
    .network-status {
        background: #1f2937 !important;
    }
    
    #cached-pages-list a {
        border-color: #374151 !important;
        color: #60a5fa !important;
    }
    
    #cached-pages-list a:hover {
        background: #1f2937 !important;
    }
}

/* 无障碍优化 */
@media (prefers-reduced-motion: reduce) {
    * {
        transition: none !important;
        animation: none !important;
    }
}

/* 高对比度模式 */
@media (prefers-contrast: high) {
    button, a[style*="background"] {
        border: 2px solid currentColor !important;
    }
}
</style>
