// 移动端参数验证工具 v1.0
// 统一的参数验证和错误日志系统

(function() {
    'use strict';
    
    // 日志级别
    const LogLevel = {
        DEBUG: 0,
        INFO: 1,
        WARN: 2,
        ERROR: 3
    };
    
    // 参数验证器
    class ParamValidator {
        constructor(context = 'Unknown') {
            this.context = context;
            this.logLevel = LogLevel.WARN; // 默认只输出警告及以上
        }
        
        /**
         * 验证参数不为空
         * @param {any} value - 待验证的值
         * @param {string} paramName - 参数名称
         * @param {string} functionName - 函数名称
         * @returns {boolean} 验证是否通过
         */
        notNull(value, paramName, functionName) {
            if (value === null || value === undefined) {
                this.logError(functionName, paramName, 'null or undefined', value);
                return false;
            }
            this.logDebug(functionName, paramName, 'passed notNull check');
            return true;
        }
        
        /**
         * 验证参数类型
         * @param {any} value - 待验证的值
         * @param {string} expectedType - 期望的类型 ('string', 'number', 'boolean', 'object', 'function', 'array')
         * @param {string} paramName - 参数名称
         * @param {string} functionName - 函数名称
         * @returns {boolean} 验证是否通过
         */
        isType(value, expectedType, paramName, functionName) {
            let actualType = typeof value;
            
            // 特殊处理数组
            if (expectedType === 'array') {
                if (!Array.isArray(value)) {
                    this.logError(functionName, paramName, `expected array but got ${actualType}`, value);
                    return false;
                }
                this.logDebug(functionName, paramName, 'passed isType check (array)');
                return true;
            }
            
            if (actualType !== expectedType) {
                this.logError(functionName, paramName, `expected ${expectedType} but got ${actualType}`, value);
                return false;
            }
            
            this.logDebug(functionName, paramName, `passed isType check (${expectedType})`);
            return true;
        }
        
        /**
         * 验证参数是DOM元素
         * @param {any} value - 待验证的值
         * @param {string} paramName - 参数名称
         * @param {string} functionName - 函数名称
         * @returns {boolean} 验证是否通过
         */
        isElement(value, paramName, functionName) {
            if (!value || !(value instanceof Element)) {
                this.logError(functionName, paramName, 'expected DOM Element', value);
                return false;
            }
            this.logDebug(functionName, paramName, 'passed isElement check');
            return true;
        }
        
        /**
         * 验证参数是HTMLImageElement
         * @param {any} value - 待验证的值
         * @param {string} paramName - 参数名称
         * @param {string} functionName - 函数名称
         * @returns {boolean} 验证是否通过
         */
        isImage(value, paramName, functionName) {
            if (!value || !(value instanceof HTMLImageElement)) {
                this.logError(functionName, paramName, 'expected HTMLImageElement', value);
                return false;
            }
            this.logDebug(functionName, paramName, 'passed isImage check');
            return true;
        }
        
        /**
         * 验证数字在范围内
         * @param {number} value - 待验证的值
         * @param {number} min - 最小值
         * @param {number} max - 最大值
         * @param {string} paramName - 参数名称
         * @param {string} functionName - 函数名称
         * @returns {boolean} 验证是否通过
         */
        inRange(value, min, max, paramName, functionName) {
            if (typeof value !== 'number' || isNaN(value)) {
                this.logError(functionName, paramName, 'expected number', value);
                return false;
            }
            
            if (value < min || value > max) {
                this.logError(functionName, paramName, `expected value between ${min} and ${max} but got ${value}`, value);
                return false;
            }
            
            this.logDebug(functionName, paramName, `passed inRange check (${min}-${max})`);
            return true;
        }
        
        /**
         * 验证字符串非空
         * @param {string} value - 待验证的值
         * @param {string} paramName - 参数名称
         * @param {string} functionName - 函数名称
         * @returns {boolean} 验证是否通过
         */
        notEmpty(value, paramName, functionName) {
            if (typeof value !== 'string' || value.trim().length === 0) {
                this.logError(functionName, paramName, 'expected non-empty string', value);
                return false;
            }
            this.logDebug(functionName, paramName, 'passed notEmpty check');
            return true;
        }
        
        /**
         * 验证数组非空
         * @param {Array} value - 待验证的值
         * @param {string} paramName - 参数名称
         * @param {string} functionName - 函数名称
         * @returns {boolean} 验证是否通过
         */
        arrayNotEmpty(value, paramName, functionName) {
            if (!Array.isArray(value) || value.length === 0) {
                this.logError(functionName, paramName, 'expected non-empty array', value);
                return false;
            }
            this.logDebug(functionName, paramName, 'passed arrayNotEmpty check');
            return true;
        }
        
        /**
         * 验证对象有特定属性
         * @param {Object} value - 待验证的值
         * @param {string|Array<string>} properties - 必需的属性名
         * @param {string} paramName - 参数名称
         * @param {string} functionName - 函数名称
         * @returns {boolean} 验证是否通过
         */
        hasProperties(value, properties, paramName, functionName) {
            if (!value || typeof value !== 'object') {
                this.logError(functionName, paramName, 'expected object', value);
                return false;
            }
            
            const requiredProps = Array.isArray(properties) ? properties : [properties];
            const missingProps = requiredProps.filter(prop => !(prop in value));
            
            if (missingProps.length > 0) {
                this.logError(functionName, paramName, `missing properties: ${missingProps.join(', ')}`, value);
                return false;
            }
            
            this.logDebug(functionName, paramName, 'passed hasProperties check');
            return true;
        }
        
        /**
         * 自定义验证
         * @param {any} value - 待验证的值
         * @param {Function} validator - 验证函数，返回true表示验证通过
         * @param {string} errorMessage - 验证失败时的错误消息
         * @param {string} paramName - 参数名称
         * @param {string} functionName - 函数名称
         * @returns {boolean} 验证是否通过
         */
        custom(value, validator, errorMessage, paramName, functionName) {
            try {
                if (!validator(value)) {
                    this.logError(functionName, paramName, errorMessage, value);
                    return false;
                }
                this.logDebug(functionName, paramName, 'passed custom check');
                return true;
            } catch (e) {
                this.logError(functionName, paramName, `custom validator threw error: ${e.message}`, value);
                return false;
            }
        }
        
        /**
         * 验证多个参数（批量验证）
         * @param {Array} validations - 验证配置数组
         * @param {string} functionName - 函数名称
         * @returns {boolean} 所有验证是否都通过
         * 
         * 示例：
         * validator.validateMultiple([
         *   { value: img, type: 'image', name: 'img' },
         *   { value: count, type: 'number', name: 'count', min: 0, max: 100 }
         * ], 'loadImages')
         */
        validateMultiple(validations, functionName) {
            let allPassed = true;
            
            for (const validation of validations) {
                const { value, type, name, min, max, properties, validator, errorMessage } = validation;
                
                // notNull检查
                if (validation.nullable !== true) {
                    if (!this.notNull(value, name, functionName)) {
                        allPassed = false;
                        continue;
                    }
                }
                
                // 类型检查
                if (type) {
                    switch (type) {
                        case 'element':
                            if (!this.isElement(value, name, functionName)) allPassed = false;
                            break;
                        case 'image':
                            if (!this.isImage(value, name, functionName)) allPassed = false;
                            break;
                        case 'array':
                            if (!this.isType(value, 'array', name, functionName)) allPassed = false;
                            break;
                        default:
                            if (!this.isType(value, type, name, functionName)) allPassed = false;
                            break;
                    }
                }
                
                // 范围检查
                if (type === 'number' && (min !== undefined || max !== undefined)) {
                    const minVal = min !== undefined ? min : -Infinity;
                    const maxVal = max !== undefined ? max : Infinity;
                    if (!this.inRange(value, minVal, maxVal, name, functionName)) {
                        allPassed = false;
                    }
                }
                
                // 非空检查
                if (validation.notEmpty && !this.notEmpty(value, name, functionName)) {
                    allPassed = false;
                }
                
                // 数组非空检查
                if (validation.arrayNotEmpty && !this.arrayNotEmpty(value, name, functionName)) {
                    allPassed = false;
                }
                
                // 属性检查
                if (properties && !this.hasProperties(value, properties, name, functionName)) {
                    allPassed = false;
                }
                
                // 自定义验证
                if (validator && !this.custom(value, validator, errorMessage || 'custom validation failed', name, functionName)) {
                    allPassed = false;
                }
            }
            
            return allPassed;
        }
        
        // 日志方法
        logDebug(functionName, paramName, message) {
            if (this.logLevel <= LogLevel.DEBUG) {
                console.debug(`[${this.context}][${functionName}] ✓ ${paramName}: ${message}`);
            }
        }
        
        logInfo(functionName, paramName, message) {
            if (this.logLevel <= LogLevel.INFO) {
                console.log(`[${this.context}][${functionName}] ℹ ${paramName}: ${message}`);
            }
        }
        
        logWarn(functionName, paramName, message, value) {
            if (this.logLevel <= LogLevel.WARN) {
                console.warn(`[${this.context}][${functionName}] ⚠️ ${paramName}: ${message}`, {
                    value,
                    type: typeof value,
                    isArray: Array.isArray(value),
                    isElement: value instanceof Element,
                    timestamp: new Date().toISOString()
                });
            }
        }
        
        logError(functionName, paramName, message, value) {
            if (this.logLevel <= LogLevel.ERROR) {
                console.error(`[${this.context}][${functionName}] ❌ ${paramName}: ${message}`, {
                    value,
                    type: typeof value,
                    isArray: Array.isArray(value),
                    isElement: value instanceof Element,
                    stack: new Error().stack,
                    timestamp: new Date().toISOString()
                });
            }
        }
        
        setLogLevel(level) {
            this.logLevel = level;
        }
    }
    
    // 导出到全局对象
    window.ParamValidator = ParamValidator;
    window.LogLevel = LogLevel;
    
    // 创建全局验证器实例
    window.mobileValidator = new ParamValidator('MobileOptimization');
    
    console.log('[ParamValidator] ✅ 参数验证工具已加载');
    
})();

