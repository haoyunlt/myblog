#!/usr/bin/env python3
"""
修复Mermaid图表渲染问题
主要解决Hugo将Mermaid代码块标记为language-fallback导致无法正确渲染的问题
"""

import os
import re

def fix_mermaid_rendering():
    """
    在extend_head.html中添加更强的Mermaid检测和渲染逻辑
    """
    extend_head_path = "layouts/partials/extend_head.html"
    
    # 读取当前内容
    with open(extend_head_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找Mermaid脚本部分
    mermaid_script_start = content.find('// 查找并转换所有mermaid代码块')
    mermaid_script_end = content.find('mermaidBlocks.forEach(function(codeElement, index) {')
    
    if mermaid_script_start != -1 and mermaid_script_end != -1:
        # 替换查找逻辑，增强对fallback代码块的处理
        new_detection_logic = '''// 查找并转换所有mermaid代码块（包括fallback类型和错误识别的类型）
    const codeBlocks = document.querySelectorAll('pre code');
    
    // 过滤出真正的mermaid代码块 - 增强检测逻辑
    const mermaidBlocks = Array.from(codeBlocks).filter(function(codeElement) {
        const content = codeElement.textContent.trim();
        const classList = codeElement.classList;
        
        // 检查类名或内容特征
        const hasMermaidClass = classList.contains('language-mermaid');
        const isFallbackWithMermaid = classList.contains('language-fallback') && (
            content.startsWith('graph ') || 
            content.startsWith('sequenceDiagram') || 
            content.startsWith('pie ') ||
            content.startsWith('gantt') ||
            content.startsWith('classDiagram') ||
            content.startsWith('stateDiagram') ||
            content.startsWith('stateDiagram-v2') ||
            content.startsWith('flowchart ') ||
            content.includes('-->') ||
            content.includes('->>') ||
            content.includes('-->>') ||
            content.includes('participant ') ||
            content.includes('class ') && content.includes('{') ||  // classDiagram特征
            content.includes('subgraph ') ||  // graph特征
            content.includes('%%') && content.includes('classDef')  // Mermaid注释和样式
        );
        
        return hasMermaidClass || isFallbackWithMermaid;
    });
    
    console.log('找到Mermaid代码块数量:', mermaidBlocks.length);
    mermaidBlocks.forEach((block, index) => {
        console.log(`Mermaid块 ${index + 1}:`, block.textContent.substring(0, 100) + '...');
    });
    
    '''
        
        # 替换内容
        before_logic = content[:mermaid_script_start]
        after_logic = content[mermaid_script_end:]
        
        new_content = before_logic + new_detection_logic + after_logic
        
        # 写回文件
        with open(extend_head_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ 已更新 {extend_head_path} 的Mermaid检测逻辑")
        return True
    
    print(f"❌ 在 {extend_head_path} 中未找到Mermaid脚本部分")
    return False

def add_mermaid_debug_css():
    """
    添加Mermaid调试CSS样式
    """
    css_path = "assets/css/extended/custom.css"
    
    debug_css = '''
/* Mermaid调试样式 */
.mermaid-container {
    background: #f8f9fa;
    border: 2px solid #dee2e6;
    border-radius: 8px;
    margin: 1rem 0;
    padding: 1rem;
}

.mermaid-container .mermaid {
    text-align: center;
    background: white;
    padding: 1rem;
    border-radius: 4px;
}

/* 调试：显示所有可能的Mermaid代码块 */
pre code.language-fallback {
    position: relative;
}

pre code.language-fallback:before {
    content: "FALLBACK";
    position: absolute;
    top: -20px;
    right: 0;
    background: orange;
    color: white;
    padding: 2px 6px;
    font-size: 10px;
    border-radius: 3px;
}

pre code.language-mermaid:before {
    content: "MERMAID";
    position: absolute;
    top: -20px;
    right: 0;
    background: green;
    color: white;
    padding: 2px 6px;
    font-size: 10px;
    border-radius: 3px;
}
'''
    
    try:
        with open(css_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "Mermaid调试样式" not in content:
            with open(css_path, 'a', encoding='utf-8') as f:
                f.write(debug_css)
            print(f"✅ 已添加Mermaid调试CSS到 {css_path}")
        else:
            print(f"ℹ️ Mermaid调试CSS已存在于 {css_path}")
        
        return True
    except FileNotFoundError:
        print(f"❌ 文件未找到: {css_path}")
        return False

if __name__ == "__main__":
    print("🔧 开始修复Mermaid渲染问题...")
    
    success = True
    
    # 1. 修复JavaScript检测逻辑
    if not fix_mermaid_rendering():
        success = False
    
    # 2. 添加调试CSS
    if not add_mermaid_debug_css():
        success = False
    
    if success:
        print("\n✅ Mermaid渲染修复完成！")
        print("请重新构建并部署网站以查看效果。")
    else:
        print("\n❌ 修复过程中出现错误，请检查上述信息。")
