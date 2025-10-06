/**
 * Bugly Dashboard - 关键问题修复
 * 根据 CODE-REVIEW-2025-10-06.md 的建议
 * 
 * 使用方法：将这些修复应用到 bugly-dashboard.html
 */

// ===========================================
// 修复 1: localStorage 异常处理和容量管理
// ===========================================

function saveToLocalStorageSafe(key, data) {
    try {
        const jsonString = JSON.stringify(data);
        
        // 检查大小（localStorage 限制通常为 5-10MB）
        const sizeInMB = new Blob([jsonString]).size / 1048576;
        
        if (sizeInMB > 4) {
            console.warn('[Dashboard] 数据量过大，进行压缩');
            // 如果数据太大，只保留最近的数据
            const reduced = Array.isArray(data) ? data.slice(-500) : data;
            localStorage.setItem(key, JSON.stringify(reduced));
            return { success: true, reduced: true, count: 500 };
        }
        
        localStorage.setItem(key, jsonString);
        return { success: true, reduced: false };
        
    } catch (e) {
        if (e.name === 'QuotaExceededError') {
            console.error('[Dashboard] 存储空间已满，清理旧数据');
            
            try {
                // 尝试只保留最近 100 条
                const reduced = Array.isArray(data) ? data.slice(-100) : data;
                localStorage.setItem(key, JSON.stringify(reduced));
                return { 
                    success: true, 
                    reduced: true, 
                    count: 100,
                    warning: '存储空间不足，已自动清理旧数据'
                };
            } catch (e2) {
                console.error('[Dashboard] 无法保存数据');
                return { 
                    success: false, 
                    error: '存储空间严重不足，无法保存数据'
                };
            }
        }
        
        console.error('[Dashboard] 保存数据失败:', e);
        return { success: false, error: e.message };
    }
}

// ===========================================
// 修复 2: 安全的 HTML 渲染（防止 XSS）
// ===========================================

const SafeRenderer = {
    /**
     * 转义 HTML 特殊字符
     */
    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },
    
    /**
     * 创建标签元素（安全）
     */
    createTagBadge(tag) {
        const span = document.createElement('span');
        span.className = 'tag-badge';
        span.textContent = tag;
        span.style.cssText = 'background:#ecf5ff;color:#409eff;padding:2px 8px;border-radius:4px;margin-right:5px;display:inline-block;';
        return span;
    },
    
    /**
     * 渲染标签列表
     */
    renderTags(tags) {
        if (!tags || !Array.isArray(tags) || tags.length === 0) {
            return '';
        }
        
        const container = document.createElement('div');
        const label = document.createElement('strong');
        label.textContent = '标签: ';
        container.appendChild(label);
        
        tags.forEach(tag => {
            container.appendChild(this.createTagBadge(tag));
        });
        
        return container.outerHTML;
    }
};

// ===========================================
// 修复 3: 事件监听器管理
// ===========================================

const EventManager = {
    handlers: {},
    
    /**
     * 添加事件监听器（带清理功能）
     */
    addListener(target, event, handler, name) {
        target.addEventListener(event, handler);
        
        // 保存引用以便后续清理
        if (!this.handlers[name]) {
            this.handlers[name] = [];
        }
        this.handlers[name].push({ target, event, handler });
    },
    
    /**
     * 移除特定名称的所有监听器
     */
    removeListeners(name) {
        if (!this.handlers[name]) return;
        
        this.handlers[name].forEach(({ target, event, handler }) => {
            target.removeEventListener(event, handler);
        });
        
        delete this.handlers[name];
    },
    
    /**
     * 清理所有监听器
     */
    cleanup() {
        Object.keys(this.handlers).forEach(name => {
            this.removeListeners(name);
        });
    }
};

// ===========================================
// 修复 4: Toast 管理优化
// ===========================================

const ToastManager = {
    maxToasts: 3,
    toasts: [],
    
    /**
     * 显示 Toast 提示
     */
    show(message, type = 'info', duration = 3000) {
        // 限制同时显示的 toast 数量
        if (this.toasts.length >= this.maxToasts) {
            this.remove(this.toasts[0]);
        }
        
        const toast = this.create(message, type);
        this.toasts.push(toast);
        document.body.appendChild(toast);
        
        // 自动移除
        const timer = setTimeout(() => {
            this.remove(toast);
        }, duration);
        
        // 点击移除
        toast.addEventListener('click', () => {
            clearTimeout(timer);
            this.remove(toast);
        });
        
        return toast;
    },
    
    /**
     * 创建 Toast 元素
     */
    create(message, type) {
        const toast = document.createElement('div');
        toast.className = `toast-message toast-${type}`;
        toast.style.cssText = `
            position: fixed;
            top: ${20 + this.toasts.length * 70}px;
            right: 20px;
            padding: 15px 20px;
            background: ${this.getColor(type)};
            color: white;
            border-radius: 4px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.2);
            z-index: 10000;
            font-size: 14px;
            cursor: pointer;
            animation: slideInRight 0.3s ease-out;
            transition: all 0.3s;
        `;
        toast.textContent = message;
        return toast;
    },
    
    /**
     * 移除 Toast
     */
    remove(toast) {
        if (!toast || !toast.parentNode) return;
        
        toast.style.animation = 'slideOutRight 0.3s ease-in';
        
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
            
            // 从数组中移除
            const index = this.toasts.indexOf(toast);
            if (index > -1) {
                this.toasts.splice(index, 1);
            }
            
            // 调整其他 toast 的位置
            this.reposition();
        }, 300);
    },
    
    /**
     * 重新定位所有 Toast
     */
    reposition() {
        this.toasts.forEach((toast, index) => {
            toast.style.top = `${20 + index * 70}px`;
        });
    },
    
    /**
     * 获取类型对应的颜色
     */
    getColor(type) {
        const colors = {
            success: '#67c23a',
            error: '#f56c6c',
            warning: '#e6a23c',
            info: '#409eff'
        };
        return colors[type] || colors.info;
    },
    
    /**
     * 清除所有 Toast
     */
    clearAll() {
        this.toasts.forEach(toast => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        });
        this.toasts = [];
    }
};

// ===========================================
// 修复 5: 输入验证增强
// ===========================================

const InputValidator = {
    config: {
        maxMessageLength: 200,
        maxDetailsLength: 2000,
        maxTagCount: 10,
        maxTagLength: 20
    },
    
    /**
     * 验证日志消息
     */
    validateMessage(message) {
        if (!message || message.trim().length === 0) {
            return { valid: false, error: '消息不能为空' };
        }
        
        if (message.length > this.config.maxMessageLength) {
            return { 
                valid: false, 
                error: `消息长度不能超过 ${this.config.maxMessageLength} 字符` 
            };
        }
        
        // 检查危险字符
        if (this.containsDangerousChars(message)) {
            return { valid: false, error: '消息包含不允许的字符' };
        }
        
        return { valid: true };
    },
    
    /**
     * 验证详细信息
     */
    validateDetails(details) {
        if (details && details.length > this.config.maxDetailsLength) {
            return { 
                valid: false, 
                error: `详细信息长度不能超过 ${this.config.maxDetailsLength} 字符` 
            };
        }
        
        if (details && this.containsDangerousChars(details)) {
            return { valid: false, error: '详细信息包含不允许的字符' };
        }
        
        return { valid: true };
    },
    
    /**
     * 验证标签
     */
    validateTags(tags) {
        if (!Array.isArray(tags)) {
            return { valid: false, error: '标签必须是数组' };
        }
        
        if (tags.length > this.config.maxTagCount) {
            return { 
                valid: false, 
                error: `标签数量不能超过 ${this.config.maxTagCount} 个` 
            };
        }
        
        for (const tag of tags) {
            if (tag.length > this.config.maxTagLength) {
                return { 
                    valid: false, 
                    error: `标签 "${tag}" 过长（最多${this.config.maxTagLength}字符）` 
                };
            }
            
            if (this.containsDangerousChars(tag)) {
                return { 
                    valid: false, 
                    error: `标签 "${tag}" 包含不允许的字符` 
                };
            }
        }
        
        return { valid: true };
    },
    
    /**
     * 检查是否包含危险字符
     */
    containsDangerousChars(text) {
        // 检查常见的 XSS 字符
        const dangerousPattern = /<script|<iframe|javascript:|onerror=|onload=/i;
        return dangerousPattern.test(text);
    },
    
    /**
     * 清理输入（移除危险字符）
     */
    sanitize(text) {
        return text
            .replace(/<script[^>]*>.*?<\/script>/gi, '')
            .replace(/<iframe[^>]*>.*?<\/iframe>/gi, '')
            .replace(/javascript:/gi, '')
            .replace(/on\w+=/gi, '');
    }
};

// ===========================================
// 修复 6: DOM 查询优化
// ===========================================

const DOMCache = {
    cache: {},
    
    /**
     * 获取元素（带缓存）
     */
    get(id) {
        if (!this.cache[id]) {
            this.cache[id] = document.getElementById(id);
        }
        return this.cache[id];
    },
    
    /**
     * 批量获取元素
     */
    getAll(ids) {
        const result = {};
        ids.forEach(id => {
            result[id] = this.get(id);
        });
        return result;
    },
    
    /**
     * 清除缓存
     */
    clear() {
        this.cache = {};
    },
    
    /**
     * 移除特定缓存
     */
    remove(id) {
        delete this.cache[id];
    }
};

// ===========================================
// 修复 7: 配置管理
// ===========================================

const Config = {
    // 延迟时间
    FOCUS_DELAY: 100,
    TOAST_DURATION: 3000,
    TOAST_REMOVE_DELAY: 300,
    AUTO_REFRESH_INTERVAL: 5000,
    
    // 存储配置
    MAX_REPORTS: 1000,
    MAX_STORAGE_SIZE_MB: 4,
    REDUCED_REPORTS_COUNT: 500,
    EMERGENCY_REPORTS_COUNT: 100,
    
    // 验证配置
    MAX_MESSAGE_LENGTH: 200,
    MAX_DETAILS_LENGTH: 2000,
    MAX_TAG_COUNT: 10,
    MAX_TAG_LENGTH: 20,
    
    // Toast 配置
    MAX_TOASTS: 3,
    
    // 环境检测
    isProduction() {
        return location.hostname !== 'localhost' && 
               location.hostname !== '127.0.0.1';
    },
    
    // 调试模式
    isDebugMode() {
        return new URLSearchParams(location.search).get('debug') === '1';
    }
};

// ===========================================
// 使用示例
// ===========================================

/*
// 在 dashboard 对象的 submitLog 方法中使用：

submitLog(event) {
    event.preventDefault();
    
    try {
        const elements = DOMCache.getAll([
            'logLevel', 'logMessage', 'logDetails', 'logTags'
        ]);
        
        const level = elements.logLevel.value;
        const message = elements.logMessage.value.trim();
        const details = elements.logDetails.value.trim();
        const tagsInput = elements.logTags.value.trim();
        
        // 输入验证
        const messageValidation = InputValidator.validateMessage(message);
        if (!messageValidation.valid) {
            ToastManager.show(messageValidation.error, 'error');
            return;
        }
        
        const detailsValidation = InputValidator.validateDetails(details);
        if (!detailsValidation.valid) {
            ToastManager.show(detailsValidation.error, 'error');
            return;
        }
        
        const tags = tagsInput ? 
            tagsInput.split(',').map(t => t.trim()).filter(t => t) : [];
        
        const tagsValidation = InputValidator.validateTags(tags);
        if (!tagsValidation.valid) {
            ToastManager.show(tagsValidation.error, 'error');
            return;
        }
        
        // 构建日志报告
        const logReport = {
            type: `log_${level}`,
            message: message,
            details: details || undefined,
            tags: tags.length > 0 ? tags : undefined,
            level: level,
            timestamp: Date.now(),
            url: window.location.href,
            userAgent: navigator.userAgent,
            sessionId: 'manual_' + Date.now(),
            source: 'manual_report'
        };
        
        // 保存到本地存储（使用安全方法）
        const existing = JSON.parse(
            localStorage.getItem('bugly_reports') || '[]'
        );
        existing.push(logReport);
        
        const result = saveToLocalStorageSafe(
            'bugly_reports', 
            existing.slice(-Config.MAX_REPORTS)
        );
        
        if (!result.success) {
            ToastManager.show('❌ ' + result.error, 'error');
            return;
        }
        
        if (result.reduced) {
            ToastManager.show(
                `⚠️ ${result.warning || '已清理旧数据'}`, 
                'warning'
            );
        }
        
        // 显示成功提示
        ToastManager.show('✅ 日志已成功上报！', 'success');
        
        // 关闭模态框
        this.closeLogModal();
        
        // 刷新显示
        this.refresh();
        
        console.log('[Dashboard] 日志已上报:', logReport);
        
    } catch (e) {
        console.error('[Dashboard] 提交日志失败:', e);
        ToastManager.show('❌ 日志上报失败: ' + e.message, 'error');
    }
}
*/

// ===========================================
// 导出（如果使用模块系统）
// ===========================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        saveToLocalStorageSafe,
        SafeRenderer,
        EventManager,
        ToastManager,
        InputValidator,
        DOMCache,
        Config
    };
}

