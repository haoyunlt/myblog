// 移动端统一错误处理工具 v1.0
// 提供 try-catch-finally 包装器和错误上报功能

(function() {
    'use strict';
    
    // 错误处理器配置
    const ERROR_HANDLER_CONFIG = {
        enableConsoleLog: true,
        enableBuglyReport: true,
        enableLocalStorage: true,
        maxLocalErrors: 100
    };
    
    // 错误处理器类
    class MobileErrorHandler {
        constructor(config) {
            this.config = config;
            this.errorCount = 0;
            this.handledErrors = new Set();
            
            console.log('[ErrorHandler] 移动端错误处理器已初始化');
        }
        
        /**
         * 包装函数，添加 try-catch-finally
         * @param {Function} fn - 要包装的函数
         * @param {Object} options - 配置选项
         * @returns {Function} 包装后的函数
         */
        wrap(fn, options = {}) {
            const {
                name = fn.name || 'anonymous',
                context = 'unknown',
                onError = null,
                onFinally = null,
                rethrow = false
            } = options;
            
            const self = this;
            
            return function wrappedFunction(...args) {
                const startTime = performance.now();
                let result;
                let error;
                
                try {
                    // 参数验证
                    if (options.validateArgs && typeof options.validateArgs === 'function') {
                        const validation = options.validateArgs(args);
                        if (!validation.valid) {
                            throw new Error(`参数验证失败: ${validation.message}`);
                        }
                    }
                    
                    // 执行原函数
                    result = fn.apply(this, args);
                    
                    // 如果返回 Promise，添加错误处理
                    if (result && typeof result.then === 'function') {
                        return result.catch(promiseError => {
                            self.handleError(promiseError, {
                                functionName: name,
                                context: context,
                                args: args,
                                type: 'promise_rejection'
                            });
                            
                            if (onError) {
                                try {
                                    onError(promiseError, args);
                                } catch (e) {
                                    console.error('[ErrorHandler] onError 回调失败', e);
                                }
                            }
                            
                            if (rethrow) {
                                throw promiseError;
                            }
                        });
                    }
                    
                    return result;
                    
                } catch (e) {
                    error = e;
                    
                    // 记录和上报错误
                    self.handleError(e, {
                        functionName: name,
                        context: context,
                        args: args,
                        type: 'sync_error'
                    });
                    
                    // 执行错误回调
                    if (onError) {
                        try {
                            onError(e, args);
                        } catch (callbackError) {
                            console.error('[ErrorHandler] onError 回调失败', callbackError);
                        }
                    }
                    
                    // 是否重新抛出错误
                    if (rethrow) {
                        throw e;
                    }
                    
                    // 返回默认值
                    return options.defaultReturn !== undefined ? options.defaultReturn : undefined;
                    
                } finally {
                    // 记录执行时间
                    const duration = performance.now() - startTime;
                    
                    if (duration > 100) {  // 超过100ms
                        console.warn(`[ErrorHandler] 函数 ${name} 执行较慢: ${duration.toFixed(2)}ms`);
                    }
                    
                    // 执行清理回调
                    if (onFinally) {
                        try {
                            onFinally(error, result, duration);
                        } catch (finallyError) {
                            console.error('[ErrorHandler] onFinally 回调失败', finallyError);
                        }
                    }
                    
                    // 调试日志
                    if (this.config.enableConsoleLog && duration > 50) {
                        console.debug(`[ErrorHandler] ${name} 完成 (${duration.toFixed(2)}ms)`, {
                            success: !error,
                            args: args,
                            result: error ? undefined : result
                        });
                    }
                }
            };
        }
        
        /**
         * 异步函数包装器
         * @param {Function} asyncFn - 异步函数
         * @param {Object} options - 配置选项
         * @returns {Function} 包装后的异步函数
         */
        wrapAsync(asyncFn, options = {}) {
            const {
                name = asyncFn.name || 'anonymous',
                context = 'unknown',
                onError = null,
                onFinally = null,
                timeout = 30000,  // 30秒超时
                rethrow = false
            } = options;
            
            const self = this;
            
            return async function wrappedAsyncFunction(...args) {
                const startTime = performance.now();
                let result;
                let error;
                let timeoutId;
                
                try {
                    // 创建超时Promise
                    const timeoutPromise = new Promise((_, reject) => {
                        timeoutId = setTimeout(() => {
                            reject(new Error(`函数 ${name} 执行超时 (${timeout}ms)`));
                        }, timeout);
                    });
                    
                    // 执行原函数，设置超时
                    const fnPromise = asyncFn.apply(this, args);
                    result = await Promise.race([fnPromise, timeoutPromise]);
                    
                    // 清除超时
                    if (timeoutId) {
                        clearTimeout(timeoutId);
                    }
                    
                    return result;
                    
                } catch (e) {
                    error = e;
                    
                    // 清除超时
                    if (timeoutId) {
                        clearTimeout(timeoutId);
                    }
                    
                    // 记录和上报错误
                    self.handleError(e, {
                        functionName: name,
                        context: context,
                        args: args,
                        type: 'async_error',
                        isTimeout: e.message.includes('超时')
                    });
                    
                    // 执行错误回调
                    if (onError) {
                        try {
                            await onError(e, args);
                        } catch (callbackError) {
                            console.error('[ErrorHandler] onError 回调失败', callbackError);
                        }
                    }
                    
                    // 是否重新抛出错误
                    if (rethrow) {
                        throw e;
                    }
                    
                    return options.defaultReturn !== undefined ? options.defaultReturn : undefined;
                    
                } finally {
                    const duration = performance.now() - startTime;
                    
                    // 执行清理回调
                    if (onFinally) {
                        try {
                            await onFinally(error, result, duration);
                        } catch (finallyError) {
                            console.error('[ErrorHandler] onFinally 回调失败', finallyError);
                        }
                    }
                }
            };
        }
        
        /**
         * 安全执行函数（立即执行）
         * @param {Function} fn - 要执行的函数
         * @param {Object} options - 配置选项
         * @returns {*} 函数执行结果
         */
        safeExecute(fn, options = {}) {
            const wrapped = this.wrap(fn, options);
            return wrapped();
        }
        
        /**
         * 安全执行异步函数（立即执行）
         * @param {Function} asyncFn - 要执行的异步函数
         * @param {Object} options - 配置选项
         * @returns {Promise} Promise 结果
         */
        async safeExecuteAsync(asyncFn, options = {}) {
            const wrapped = this.wrapAsync(asyncFn, options);
            return await wrapped();
        }
        
        /**
         * 处理错误
         * @param {Error} error - 错误对象
         * @param {Object} metadata - 元数据
         */
        handleError(error, metadata = {}) {
            try {
                this.errorCount++;
                
                // 生成错误ID
                const errorId = this.generateErrorId(error, metadata);
                
                // 防止重复处理
                if (this.handledErrors.has(errorId)) {
                    return;
                }
                this.handledErrors.add(errorId);
                
                // 构建错误信息
                const errorInfo = {
                    message: error.message || String(error),
                    stack: error.stack || '',
                    name: error.name || 'Error',
                    timestamp: Date.now(),
                    errorId: errorId,
                    count: this.errorCount,
                    ...metadata
                };
                
                // 控制台日志
                if (this.config.enableConsoleLog) {
                    console.error(
                        `[ErrorHandler] 错误 #${this.errorCount}`,
                        `\n函数: ${metadata.functionName || 'unknown'}`,
                        `\n上下文: ${metadata.context || 'unknown'}`,
                        `\n消息: ${errorInfo.message}`,
                        `\n类型: ${metadata.type || 'unknown'}`,
                        error
                    );
                }
                
                // 上报到 Bugly
                if (this.config.enableBuglyReport && window.reportToBugly) {
                    window.reportToBugly({
                        message: `[${metadata.context}] ${errorInfo.message}`,
                        stack: errorInfo.stack,
                        level: 'error',
                        extra: {
                            functionName: metadata.functionName,
                            type: metadata.type,
                            errorId: errorId,
                            args: this.serializeArgs(metadata.args)
                        }
                    });
                }
                
                // 保存到本地存储
                if (this.config.enableLocalStorage) {
                    this.saveToLocalStorage(errorInfo);
                }
                
            } catch (e) {
                // 错误处理器本身出错，直接输出到控制台
                console.error('[ErrorHandler] 处理错误时发生异常', e);
                console.error('[ErrorHandler] 原始错误', error);
            }
        }
        
        /**
         * 生成错误ID
         * @param {Error} error - 错误对象
         * @param {Object} metadata - 元数据
         * @returns {string} 错误ID
         */
        generateErrorId(error, metadata) {
            const key = `${metadata.functionName}_${error.message}_${error.name}`;
            return this.hashCode(key);
        }
        
        /**
         * 简单哈希函数
         * @param {string} str - 字符串
         * @returns {string} 哈希值
         */
        hashCode(str) {
            let hash = 0;
            for (let i = 0; i < str.length; i++) {
                const char = str.charCodeAt(i);
                hash = ((hash << 5) - hash) + char;
                hash = hash & hash;
            }
            return hash.toString(36);
        }
        
        /**
         * 序列化参数（避免循环引用）
         * @param {*} args - 参数
         * @returns {*} 序列化后的参数
         */
        serializeArgs(args) {
            try {
                if (!args) return null;
                
                // 使用 JSON.stringify 的 replacer 参数处理循环引用
                const seen = new WeakSet();
                return JSON.parse(JSON.stringify(args, (key, value) => {
                    // 处理对象
                    if (typeof value === 'object' && value !== null) {
                        if (seen.has(value)) {
                            return '[Circular]';
                        }
                        seen.add(value);
                        
                        // 处理 DOM 元素
                        if (value instanceof Element) {
                            return `[Element: ${value.tagName}]`;
                        }
                        
                        // 处理 HTMLCollection/NodeList
                        if (value instanceof HTMLCollection || value instanceof NodeList) {
                            return `[${value.constructor.name}: ${value.length} items]`;
                        }
                    }
                    
                    // 处理函数
                    if (typeof value === 'function') {
                        return `[Function: ${value.name || 'anonymous'}]`;
                    }
                    
                    return value;
                }));
            } catch (e) {
                return '[Serialize Error]';
            }
        }
        
        /**
         * 保存到本地存储
         * @param {Object} errorInfo - 错误信息
         */
        saveToLocalStorage(errorInfo) {
            try {
                const key = 'mobile_error_handler_errors';
                const existing = JSON.parse(localStorage.getItem(key) || '[]');
                existing.push(errorInfo);
                
                // 限制数量
                const limited = existing.slice(-this.config.maxLocalErrors);
                localStorage.setItem(key, JSON.stringify(limited));
                
            } catch (e) {
                console.warn('[ErrorHandler] 保存到本地存储失败', e);
            }
        }
        
        /**
         * 获取本地错误记录
         * @returns {Array} 错误记录
         */
        getLocalErrors() {
            try {
                const key = 'mobile_error_handler_errors';
                return JSON.parse(localStorage.getItem(key) || '[]');
            } catch (e) {
                return [];
            }
        }
        
        /**
         * 清除本地错误记录
         */
        clearLocalErrors() {
            try {
                localStorage.removeItem('mobile_error_handler_errors');
                this.handledErrors.clear();
                this.errorCount = 0;
                console.log('[ErrorHandler] 本地错误记录已清除');
            } catch (e) {
                console.error('[ErrorHandler] 清除本地错误记录失败', e);
            }
        }
        
        /**
         * 获取错误统计
         * @returns {Object} 统计信息
         */
        getStats() {
            const errors = this.getLocalErrors();
            const stats = {
                total: errors.length,
                byContext: {},
                byType: {},
                byFunction: {},
                recent: errors.slice(-10)
            };
            
            errors.forEach(error => {
                // 按上下文统计
                const context = error.context || 'unknown';
                stats.byContext[context] = (stats.byContext[context] || 0) + 1;
                
                // 按类型统计
                const type = error.type || 'unknown';
                stats.byType[type] = (stats.byType[type] || 0) + 1;
                
                // 按函数统计
                const fn = error.functionName || 'unknown';
                stats.byFunction[fn] = (stats.byFunction[fn] || 0) + 1;
            });
            
            return stats;
        }
    }
    
    // 创建全局实例
    window.MobileErrorHandler = new MobileErrorHandler(ERROR_HANDLER_CONFIG);
    
    // 简化的全局方法
    window.safeCall = function(fn, options) {
        return window.MobileErrorHandler.wrap(fn, options);
    };
    
    window.safeCallAsync = function(asyncFn, options) {
        return window.MobileErrorHandler.wrapAsync(asyncFn, options);
    };
    
    window.getErrorStats = function() {
        const stats = window.MobileErrorHandler.getStats();
        console.group('📊 错误统计');
        console.log('总错误数:', stats.total);
        console.log('按上下文:', stats.byContext);
        console.log('按类型:', stats.byType);
        console.log('按函数:', stats.byFunction);
        console.log('最近10个错误:', stats.recent);
        console.groupEnd();
        return stats;
    };
    
    console.log('[ErrorHandler] ✅ 移动端错误处理器已启动');
    console.log('[ErrorHandler] 💡 使用 safeCall(fn, options) 包装同步函数');
    console.log('[ErrorHandler] 💡 使用 safeCallAsync(fn, options) 包装异步函数');
    console.log('[ErrorHandler] 💡 使用 getErrorStats() 查看错误统计');
    
})();

