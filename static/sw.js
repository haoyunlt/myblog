// Service Worker - 移动端性能优化缓存策略
// 版本: 2.0.0

const CACHE_NAME = 'tommie-blog-v2.0.0';
const RUNTIME_CACHE = 'tommie-runtime-v2.0.0';

// 静态资源预缓存列表
const STATIC_CACHE_URLS = [
  '/',
  '/offline/',
  '/manifest.json',
  '/assets/css/stylesheet.css',
  '/assets/js/search.js'
];

// 缓存策略配置
const CACHE_STRATEGIES = {
  // 静态资源：缓存优先
  static: {
    cacheName: CACHE_NAME,
    maxAge: 30 * 24 * 60 * 60, // 30天
    maxEntries: 100
  },
  
  // 图片资源：缓存优先，失败时显示占位符
  images: {
    cacheName: 'tommie-images-v2.0.0',
    maxAge: 7 * 24 * 60 * 60, // 7天
    maxEntries: 200
  },
  
  // HTML页面：网络优先，离线时显示缓存
  pages: {
    cacheName: 'tommie-pages-v2.0.0',
    maxAge: 24 * 60 * 60, // 1天
    maxEntries: 100
  },
  
  // API请求：网络优先
  api: {
    cacheName: 'tommie-api-v2.0.0',
    maxAge: 5 * 60, // 5分钟
    maxEntries: 50
  }
};

// Service Worker安装
self.addEventListener('install', (event) => {
  console.log('[SW] 安装 Service Worker');
  
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('[SW] 预缓存静态资源');
        return cache.addAll(STATIC_CACHE_URLS);
      })
      .then(() => {
        // 强制激活新的 Service Worker
        return self.skipWaiting();
      })
      .catch((error) => {
        console.error('[SW] 预缓存失败:', error);
      })
  );
});

// Service Worker激活
self.addEventListener('activate', (event) => {
  console.log('[SW] 激活 Service Worker');
  
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        const validCacheNames = Object.values(CACHE_STRATEGIES)
          .map(strategy => strategy.cacheName)
          .concat([CACHE_NAME, RUNTIME_CACHE]);
        
        return Promise.all(
          cacheNames.map((cacheName) => {
            if (!validCacheNames.includes(cacheName)) {
              console.log('[SW] 删除旧缓存:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        // 立即控制所有页面
        return self.clients.claim();
      })
  );
});

// 请求拦截和缓存策略
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // 只处理HTTP(S)请求
  if (!request.url.startsWith('http')) {
    return;
  }
  
  // 根据请求类型选择缓存策略
  if (isImageRequest(request)) {
    event.respondWith(handleImageRequest(request));
  } else if (isStaticAsset(request)) {
    event.respondWith(handleStaticAssetRequest(request));
  } else if (isPageRequest(request)) {
    event.respondWith(handlePageRequest(request));
  } else if (isApiRequest(request)) {
    event.respondWith(handleApiRequest(request));
  }
});

// 判断是否为图片请求
function isImageRequest(request) {
  return /\.(jpg|jpeg|png|gif|webp|svg|ico)$/i.test(request.url);
}

// 判断是否为静态资源
function isStaticAsset(request) {
  return /\.(css|js|woff|woff2|ttf|eot)$/i.test(request.url);
}

// 判断是否为页面请求
function isPageRequest(request) {
  return request.mode === 'navigate' || 
         (request.method === 'GET' && request.headers.get('accept').includes('text/html'));
}

// 判断是否为API请求
function isApiRequest(request) {
  return request.url.includes('/api/') || 
         request.url.includes('/search.json') ||
         request.url.includes('.json');
}

// 处理图片请求 - 缓存优先策略
async function handleImageRequest(request) {
  try {
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      const cache = await caches.open(CACHE_STRATEGIES.images.cacheName);
      cache.put(request, networkResponse.clone());
      
      // 清理过期缓存
      cleanupCache(CACHE_STRATEGIES.images.cacheName, CACHE_STRATEGIES.images.maxEntries);
      
      return networkResponse;
    }
    
    // 网络失败时返回占位符图片
    return generatePlaceholderResponse();
    
  } catch (error) {
    console.warn('[SW] 图片请求失败:', request.url, error);
    return generatePlaceholderResponse();
  }
}

// 处理静态资源请求 - 缓存优先策略
async function handleStaticAssetRequest(request) {
  try {
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      const cache = await caches.open(CACHE_STRATEGIES.static.cacheName);
      cache.put(request, networkResponse.clone());
      return networkResponse;
    }
    
    return networkResponse;
    
  } catch (error) {
    console.warn('[SW] 静态资源请求失败:', request.url, error);
    throw error;
  }
}

// 处理页面请求 - 网络优先策略
async function handlePageRequest(request) {
  try {
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      const cache = await caches.open(CACHE_STRATEGIES.pages.cacheName);
      cache.put(request, networkResponse.clone());
      
      // 清理过期缓存
      cleanupCache(CACHE_STRATEGIES.pages.cacheName, CACHE_STRATEGIES.pages.maxEntries);
      
      return networkResponse;
    }
    
    // 网络失败时返回缓存
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // 都没有时返回离线页面
    return caches.match('/offline/');
    
  } catch (error) {
    console.warn('[SW] 页面请求失败:', request.url, error);
    
    // 网络失败时返回缓存
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // 返回离线页面
    return caches.match('/offline/');
  }
}

// 处理API请求 - 网络优先策略
async function handleApiRequest(request) {
  try {
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      const cache = await caches.open(CACHE_STRATEGIES.api.cacheName);
      cache.put(request, networkResponse.clone());
      
      // 清理过期缓存
      cleanupCache(CACHE_STRATEGIES.api.cacheName, CACHE_STRATEGIES.api.maxEntries);
      
      return networkResponse;
    }
    
    return networkResponse;
    
  } catch (error) {
    console.warn('[SW] API请求失败:', request.url, error);
    
    // 网络失败时返回缓存
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    throw error;
  }
}

// 生成占位符图片响应
function generatePlaceholderResponse() {
  // 1x1透明PNG的base64编码
  const transparentPng = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==';
  
  const response = new Response(atob(transparentPng), {
    status: 200,
    statusText: 'OK',
    headers: {
      'Content-Type': 'image/png',
      'Cache-Control': 'no-cache'
    }
  });
  
  return response;
}

// 清理过期缓存
async function cleanupCache(cacheName, maxEntries) {
  try {
    const cache = await caches.open(cacheName);
    const requests = await cache.keys();
    
    if (requests.length > maxEntries) {
      const requestsToDelete = requests.slice(0, requests.length - maxEntries);
      await Promise.all(
        requestsToDelete.map(request => cache.delete(request))
      );
      console.log(`[SW] 清理缓存 ${cacheName}: 删除 ${requestsToDelete.length} 个条目`);
    }
  } catch (error) {
    console.error('[SW] 清理缓存失败:', error);
  }
}

// 消息处理
self.addEventListener('message', (event) => {
  const { data } = event;
  
  switch (data.type) {
    case 'SKIP_WAITING':
      self.skipWaiting();
      break;
      
    case 'GET_CACHE_STATS':
      getCacheStats().then(stats => {
        event.ports[0].postMessage(stats);
      });
      break;
      
    case 'CLEAR_CACHE':
      clearAllCaches().then(() => {
        event.ports[0].postMessage({ success: true });
      });
      break;
      
    case 'PREFETCH_URLS':
      prefetchUrls(data.urls).then(() => {
        event.ports[0].postMessage({ success: true });
      });
      break;
  }
});

// 获取缓存统计信息
async function getCacheStats() {
  const cacheNames = await caches.keys();
  const stats = {};
  
  for (const cacheName of cacheNames) {
    const cache = await caches.open(cacheName);
    const requests = await cache.keys();
    stats[cacheName] = requests.length;
  }
  
  return stats;
}

// 清空所有缓存
async function clearAllCaches() {
  const cacheNames = await caches.keys();
  await Promise.all(cacheNames.map(cacheName => caches.delete(cacheName)));
  console.log('[SW] 所有缓存已清空');
}

// 预取URL列表
async function prefetchUrls(urls) {
  const cache = await caches.open(RUNTIME_CACHE);
  
  for (const url of urls) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        await cache.put(url, response);
        console.log('[SW] 预取成功:', url);
      }
    } catch (error) {
      console.warn('[SW] 预取失败:', url, error);
    }
  }
}

// 后台同步
self.addEventListener('sync', (event) => {
  if (event.tag === 'background-sync') {
    console.log('[SW] 执行后台同步');
    event.waitUntil(doBackgroundSync());
  }
});

// 执行后台同步任务
async function doBackgroundSync() {
  try {
    // 这里可以添加需要后台同步的任务
    // 比如上传离线时的数据、更新缓存等
    console.log('[SW] 后台同步完成');
  } catch (error) {
    console.error('[SW] 后台同步失败:', error);
  }
}

// 推送通知
self.addEventListener('push', (event) => {
  if (event.data) {
    const data = event.data.json();
    const options = {
      body: data.body,
      icon: '/icons/icon-192x192.png',
      badge: '/icons/badge-72x72.png',
      vibrate: [100, 50, 100],
      data: {
        dateOfArrival: Date.now(),
        primaryKey: '1'
      },
      actions: [
        {
          action: 'explore',
          title: '查看详情',
          icon: '/icons/checkmark.png'
        },
        {
          action: 'close',
          title: '关闭',
          icon: '/icons/xmark.png'
        }
      ]
    };
    
    event.waitUntil(
      self.registration.showNotification(data.title, options)
    );
  }
});

// 通知点击处理
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  
  if (event.action === 'explore') {
    event.waitUntil(
      clients.openWindow('/')
    );
  }
});

console.log('[SW] Service Worker 脚本加载完成');
