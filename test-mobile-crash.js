// Playwright 移动端崩溃压力测试
// 模拟 iPhone Safari 反复访问首页，监控崩溃和性能

const { chromium, devices } = require('playwright');
const fs = require('fs');
const path = require('path');

// 测试配置
const CONFIG = {
    url: 'https://www.tommienotes.com/',
    device: 'iPhone 13 Pro',  // 模拟设备
    iterations: 50,            // 测试次数
    delayBetweenVisits: 2000, // 每次访问间隔（毫秒）
    timeout: 30000,           // 页面加载超时（毫秒）
    screenshotOnError: true,  // 错误时截图
    collectMetrics: true,     // 收集性能指标
    outputDir: './test-results'
};

// 创建输出目录
if (!fs.existsSync(CONFIG.outputDir)) {
    fs.mkdirSync(CONFIG.outputDir, { recursive: true });
}

// 测试结果统计
const stats = {
    total: 0,
    success: 0,
    failed: 0,
    crashed: 0,
    timeouts: 0,
    errors: [],
    metrics: [],
    startTime: Date.now()
};

// 颜色输出
const colors = {
    reset: '\x1b[0m',
    green: '\x1b[32m',
    red: '\x1b[31m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    cyan: '\x1b[36m'
};

function log(message, color = 'reset') {
    console.log(`${colors[color]}${message}${colors.reset}`);
}

// 格式化内存大小
function formatBytes(bytes) {
    if (!bytes) return 'N/A';
    return (bytes / 1024 / 1024).toFixed(2) + ' MB';
}

// 格式化时间
function formatDuration(ms) {
    return (ms / 1000).toFixed(2) + 's';
}

// 收集性能指标
async function collectPerformanceMetrics(page) {
    try {
        const metrics = await page.evaluate(() => {
            const perf = performance.getEntriesByType('navigation')[0];
            const memory = performance.memory;
            
            return {
                // 页面加载时间
                loadTime: perf ? perf.loadEventEnd - perf.fetchStart : 0,
                domReady: perf ? perf.domContentLoadedEventEnd - perf.fetchStart : 0,
                
                // 内存使用
                jsHeapSize: memory ? memory.usedJSHeapSize : 0,
                jsHeapLimit: memory ? memory.jsHeapSizeLimit : 0,
                totalHeapSize: memory ? memory.totalJSHeapSize : 0,
                
                // 资源统计
                resources: performance.getEntriesByType('resource').length,
                
                // 页面信息
                url: window.location.href,
                timestamp: Date.now()
            };
        });
        
        return metrics;
    } catch (e) {
        return { error: e.message };
    }
}

// 检查控制台错误
function setupConsoleMonitoring(page, iteration) {
    const consoleErrors = [];
    
    page.on('console', msg => {
        const type = msg.type();
        const text = msg.text();
        
        if (type === 'error') {
            consoleErrors.push({
                iteration,
                type,
                text,
                timestamp: Date.now()
            });
            log(`  ⚠️  控制台错误: ${text}`, 'yellow');
        }
    });
    
    page.on('pageerror', error => {
        consoleErrors.push({
            iteration,
            type: 'pageerror',
            text: error.message,
            stack: error.stack,
            timestamp: Date.now()
        });
        log(`  ❌ 页面异常: ${error.message}`, 'red');
    });
    
    return consoleErrors;
}

// 单次访问测试
async function testVisit(browser, iteration) {
    const context = await browser.newContext({
        ...devices[CONFIG.device],
        locale: 'zh-CN',
        timezoneId: 'Asia/Shanghai',
        viewport: { width: 390, height: 844 }  // iPhone 13 Pro
    });
    
    const page = await context.newPage();
    const consoleErrors = setupConsoleMonitoring(page, iteration);
    
    const result = {
        iteration,
        success: false,
        crashed: false,
        timeout: false,
        error: null,
        metrics: null,
        consoleErrors: [],
        duration: 0,
        timestamp: Date.now()
    };
    
    const startTime = Date.now();
    
    try {
        log(`\n[${iteration}/${CONFIG.iterations}] 开始访问...`, 'cyan');
        
        // 访问页面
        const response = await page.goto(CONFIG.url, {
            waitUntil: 'networkidle',
            timeout: CONFIG.timeout
        });
        
        result.duration = Date.now() - startTime;
        
        // 检查响应状态
        if (!response.ok()) {
            throw new Error(`HTTP ${response.status()}: ${response.statusText()}`);
        }
        
        log(`  ✓ 页面加载成功 (${formatDuration(result.duration)})`, 'green');
        
        // 等待关键元素
        await page.waitForSelector('body', { timeout: 5000 });
        log(`  ✓ DOM 就绪`, 'green');
        
        // 收集性能指标
        if (CONFIG.collectMetrics) {
            result.metrics = await collectPerformanceMetrics(page);
            
            if (result.metrics && !result.metrics.error) {
                log(`  📊 加载时间: ${formatDuration(result.metrics.loadTime)}`, 'blue');
                log(`  📊 DOM就绪: ${formatDuration(result.metrics.domReady)}`, 'blue');
                log(`  💾 JS堆内存: ${formatBytes(result.metrics.jsHeapSize)} / ${formatBytes(result.metrics.jsHeapLimit)}`, 'blue');
                log(`  📦 资源数: ${result.metrics.resources}`, 'blue');
                
                // 检查内存使用率
                const memoryUsage = result.metrics.jsHeapSize / result.metrics.jsHeapLimit;
                if (memoryUsage > 0.8) {
                    log(`  ⚠️  警告: 内存使用率 ${(memoryUsage * 100).toFixed(1)}%`, 'yellow');
                }
            }
        }
        
        // 检查错误处理工具是否加载
        const toolsLoaded = await page.evaluate(() => {
            return {
                bugly: typeof window.BuglyReporter !== 'undefined',
                errorHandler: typeof window.MobileErrorHandler !== 'undefined',
                validator: typeof window.mobileValidator !== 'undefined',
                safeCall: typeof window.safeCall !== 'undefined'
            };
        });
        
        log(`  🔧 工具加载状态:`, 'blue');
        log(`     Bugly: ${toolsLoaded.bugly ? '✓' : '✗'}`, toolsLoaded.bugly ? 'green' : 'red');
        log(`     ErrorHandler: ${toolsLoaded.errorHandler ? '✓' : '✗'}`, toolsLoaded.errorHandler ? 'green' : 'red');
        log(`     Validator: ${toolsLoaded.validator ? '✓' : '✗'}`, toolsLoaded.validator ? 'green' : 'red');
        log(`     safeCall: ${toolsLoaded.safeCall ? '✓' : '✗'}`, toolsLoaded.safeCall ? 'green' : 'red');
        
        // 模拟用户交互
        await page.waitForTimeout(1000);
        
        // 滚动页面
        await page.evaluate(() => {
            window.scrollTo({ top: 500, behavior: 'smooth' });
        });
        await page.waitForTimeout(500);
        
        await page.evaluate(() => {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
        await page.waitForTimeout(500);
        
        result.success = true;
        result.consoleErrors = consoleErrors;
        stats.success++;
        
    } catch (error) {
        result.error = {
            message: error.message,
            stack: error.stack,
            name: error.name
        };
        result.consoleErrors = consoleErrors;
        result.duration = Date.now() - startTime;
        
        if (error.message.includes('crashed')) {
            result.crashed = true;
            stats.crashed++;
            log(`  ❌ 页面崩溃!`, 'red');
        } else if (error.message.includes('timeout')) {
            result.timeout = true;
            stats.timeouts++;
            log(`  ⏱️  加载超时`, 'yellow');
        } else {
            stats.failed++;
            log(`  ❌ 错误: ${error.message}`, 'red');
        }
        
        stats.errors.push(result);
        
        // 错误时截图
        if (CONFIG.screenshotOnError) {
            try {
                const screenshotPath = path.join(CONFIG.outputDir, `error-${iteration}-${Date.now()}.png`);
                await page.screenshot({ path: screenshotPath, fullPage: true });
                log(`  📸 截图已保存: ${screenshotPath}`, 'yellow');
            } catch (e) {
                // 截图失败忽略
            }
        }
    } finally {
        stats.total++;
        if (result.metrics) {
            stats.metrics.push(result.metrics);
        }
        
        // 清理
        await context.close();
        
        // 延迟下次访问
        if (iteration < CONFIG.iterations) {
            await new Promise(resolve => setTimeout(resolve, CONFIG.delayBetweenVisits));
        }
    }
    
    return result;
}

// 生成测试报告
function generateReport() {
    const duration = Date.now() - stats.startTime;
    const successRate = (stats.success / stats.total * 100).toFixed(2);
    
    log('\n' + '='.repeat(60), 'cyan');
    log('📊 测试报告', 'cyan');
    log('='.repeat(60), 'cyan');
    
    log(`\n⏱️  总耗时: ${formatDuration(duration)}`, 'blue');
    log(`📊 测试次数: ${stats.total}`, 'blue');
    log(`✅ 成功: ${stats.success} (${successRate}%)`, 'green');
    log(`❌ 失败: ${stats.failed}`, stats.failed > 0 ? 'red' : 'green');
    log(`💥 崩溃: ${stats.crashed}`, stats.crashed > 0 ? 'red' : 'green');
    log(`⏱️  超时: ${stats.timeouts}`, stats.timeouts > 0 ? 'yellow' : 'green');
    
    // 性能统计
    if (stats.metrics.length > 0) {
        const avgLoadTime = stats.metrics.reduce((sum, m) => sum + (m.loadTime || 0), 0) / stats.metrics.length;
        const avgMemory = stats.metrics.reduce((sum, m) => sum + (m.jsHeapSize || 0), 0) / stats.metrics.length;
        const maxMemory = Math.max(...stats.metrics.map(m => m.jsHeapSize || 0));
        
        log(`\n📈 性能统计:`, 'cyan');
        log(`  平均加载时间: ${formatDuration(avgLoadTime)}`, 'blue');
        log(`  平均内存使用: ${formatBytes(avgMemory)}`, 'blue');
        log(`  峰值内存使用: ${formatBytes(maxMemory)}`, 'blue');
        
        // 内存趋势
        if (stats.metrics.length >= 10) {
            const firstFive = stats.metrics.slice(0, 5).reduce((sum, m) => sum + (m.jsHeapSize || 0), 0) / 5;
            const lastFive = stats.metrics.slice(-5).reduce((sum, m) => sum + (m.jsHeapSize || 0), 0) / 5;
            const memoryGrowth = ((lastFive - firstFive) / firstFive * 100).toFixed(2);
            
            log(`  内存增长趋势: ${memoryGrowth}%`, memoryGrowth > 50 ? 'red' : 'green');
            
            if (memoryGrowth > 50) {
                log(`  ⚠️  警告: 检测到明显的内存泄漏!`, 'red');
            }
        }
    }
    
    // 错误汇总
    if (stats.errors.length > 0) {
        log(`\n❌ 错误详情:`, 'red');
        stats.errors.forEach((err, i) => {
            log(`  ${i + 1}. [迭代 ${err.iteration}] ${err.error.message}`, 'red');
            if (err.consoleErrors && err.consoleErrors.length > 0) {
                log(`     控制台错误: ${err.consoleErrors.length} 个`, 'yellow');
            }
        });
    }
    
    // 保存详细报告
    const reportPath = path.join(CONFIG.outputDir, `report-${Date.now()}.json`);
    fs.writeFileSync(reportPath, JSON.stringify({
        config: CONFIG,
        stats: stats,
        timestamp: new Date().toISOString()
    }, null, 2));
    
    log(`\n💾 详细报告已保存: ${reportPath}`, 'green');
    
    // 结论
    log('\n' + '='.repeat(60), 'cyan');
    if (stats.crashed > 0) {
        log('❌ 测试结果: 检测到崩溃问题!', 'red');
        return 1;
    } else if (successRate >= 95) {
        log('✅ 测试结果: 通过 (成功率 >= 95%)', 'green');
        return 0;
    } else {
        log('⚠️  测试结果: 需要优化 (成功率 < 95%)', 'yellow');
        return 0;
    }
}

// 主函数
async function main() {
    log('🚀 开始移动端崩溃压力测试', 'cyan');
    log(`📱 设备: ${CONFIG.device}`, 'blue');
    log(`🌐 URL: ${CONFIG.url}`, 'blue');
    log(`🔄 迭代次数: ${CONFIG.iterations}`, 'blue');
    log(`⏱️  间隔时间: ${CONFIG.delayBetweenVisits}ms\n`, 'blue');
    
    const browser = await chromium.launch({
        headless: true,  // 无头模式
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-gpu'
        ]
    });
    
    try {
        // 执行测试
        for (let i = 1; i <= CONFIG.iterations; i++) {
            await testVisit(browser, i);
        }
        
        // 生成报告
        const exitCode = generateReport();
        
        await browser.close();
        process.exit(exitCode);
        
    } catch (error) {
        log(`\n❌ 测试失败: ${error.message}`, 'red');
        console.error(error);
        await browser.close();
        process.exit(1);
    }
}

// 运行测试
main().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
});

